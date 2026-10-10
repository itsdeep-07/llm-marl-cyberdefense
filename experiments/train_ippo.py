"""Train and evaluate IPPO on real CC4 with sb3-contrib MaskablePPO (Batch 4).

This follows CC4's own example, CybORG/Evaluation/training_example/TrainingSB3.py:

    BlueFlatWrapper -> ParallelEnv shim -> parallel_to_aec -> SB3 action-mask
    wrapper -> ActionMasker -> MaskablePPO

One policy is shared by all five Blue agents (parameter sharing). Each agent
acts on its own observation and its value estimate also uses only that
observation, which is IPPO. The shim and wrapper below are adapted from that
example; the additions are explicit training seeds, exact episode logging, and
reward scaling.

Differences from the example, all deliberate:
  * training episode seeds are explicit and never overlap the evaluation seeds
  * training stops after a fixed number of whole EPISODES (identical to MAPPO), so the
    environment-step budget matches exactly; SB3 agent-steps are not used for the budget
    (one environment step = five agent-steps)
  * per-episode returns are measured from the environment, not from SB3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pettingzoo
from gymnasium.spaces import Space
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from envs.cc4_env import make_env
from experiments.rl_common import (DEFAULT_EVAL_SEED_BASE, EPISODES_PER_UPDATE, RED,
                                   REWARD_SCALE, checkpoint_path, plan_seeds, save_run)

N_EPOCHS = 4
MINIBATCH_TIMESTEPS = 80  # same minibatch as MAPPO: 80 timesteps x 5 agents = 400 samples


class TrainingShim(ParallelEnv):
    """CC4's GenericPzShim plus explicit seeds and exact per-episode logging."""

    metadata = {"render_modes": [], "name": "CybORG v4", "is_parallelizable": True,
                "has_manual_policy": False}

    def __init__(self, env, train_seeds):
        super().__init__()
        self.env = env
        self._seeds = iter(train_seeds)
        self.episode_seed = None
        self.ep_return, self.env_steps, self.episodes = 0.0, 0, []
        self.t0 = time.perf_counter()

    def reset(self, seed=None, *args, **kwargs):
        # SB3 passes seed=None (or its own seed); we always use our planned seed.
        self.episode_seed = next(self._seeds)
        self.ep_return = 0.0
        return self.env.reset(seed=self.episode_seed)

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        self.env_steps += 1
        self.ep_return += float(sum(rewards.values()))  # same convention as the baselines
        if truncated.get("__all__", False) or any(terminated.values()):
            self.episodes.append({
                "episode": len(self.episodes) + 1, "episode_seed": self.episode_seed,
                "env_steps": self.env_steps, "return": self.ep_return,
                "wall_time_s": time.perf_counter() - self.t0,
            })
        return obs, rewards, terminated, truncated, info

    @property
    def agents(self):
        return self.env.agents

    @property
    def possible_agents(self):
        return self.env.possible_agents

    @property
    def action_spaces(self) -> dict[str, Space]:
        return self.env.action_spaces()

    def action_space(self, agent_name) -> Space:
        return self.env.action_space(agent_name)

    @property
    def observation_spaces(self) -> dict[str, Space]:
        return self.env.observation_spaces()

    def observation_space(self, agent_name) -> Space:
        return self.env.observation_space(agent_name)

    @property
    def action_masks(self):
        import torch as T
        return {a: T.tensor(self.env.action_masks[a], dtype=T.bool) for a in self.env.agents}


class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    """From CC4's TrainingSB3.py (itself from the PettingZoo SB3 tutorial), plus reward scaling."""

    def __init__(self, env, reward_scale: float):
        super().__init__(env)
        self.reward_scale = reward_scale

    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        self.observation_space = super().observation_space(self.agent_selection)
        self.action_space = super().action_space(self.agent_selection)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        super().step(action)
        obs, reward, terminated, truncated, info = super().last()
        return obs, reward * self.reward_scale, terminated, truncated, info

    def action_mask(self):
        return self.infos[self.agent_selection]["action_mask"]


def mask_fn(env):
    return env.action_mask()


class EpisodeBudget(BaseCallback):
    """Stop training once the planned number of whole episodes has finished.

    MAPPO trains on exactly the same number of episodes, so both algorithms see
    the same number of environment steps (CC4 episodes are one step shorter than
    the --steps setting, so counting episodes is the exact way to match budgets).
    """

    def __init__(self, shim: TrainingShim, n_episodes: int):
        super().__init__()
        self.shim, self.n_episodes = shim, n_episodes

    def _on_step(self) -> bool:
        return len(self.shim.episodes) < self.n_episodes


def evaluate(model: MaskablePPO, seeds: list, steps: int) -> list:
    """Greedy evaluation on held-out seeds, one fresh environment per seed (like the baselines)."""
    returns = []
    for seed in seeds:
        env = make_env(seed=seed, steps=steps, red=RED)
        obs, info = env.reset(seed=seed)
        names = sorted(env.agents)
        total = 0.0
        while True:
            actions = {}
            for a in names:
                mask = np.asarray(info[a]["action_mask"], dtype=bool)
                action, _ = model.predict(obs[a], action_masks=mask, deterministic=True)
                actions[a] = int(action)
            obs, rewards, term, trunc, info = env.step(actions)
            total += float(sum(rewards.values()))
            if trunc.get("__all__", False) or any(term.values()):
                break
        returns.append(total)
    return returns


def train_one(seed: int, total_steps: int, steps: int, eval_episodes: int,
              eval_seed_base: int, out_dir: str) -> dict:
    n_train, train_seeds, eval_seeds = plan_seeds(seed, total_steps, steps, eval_episodes,
                                                  eval_seed_base)
    base_env = make_env(seed=train_seeds[0], steps=steps, red=RED)
    shim = TrainingShim(base_env, train_seeds)
    n_agents = len(shim.possible_agents)
    env = SB3ActionMaskWrapper(parallel_to_aec(shim), REWARD_SCALE)
    env.reset(seed=seed)  # defines the spaces; required before wrapping, as in the CC4 example
    env = ActionMasker(env, mask_fn)

    n_steps = EPISODES_PER_UPDATE * steps * n_agents  # one update per 4 episodes, like MAPPO
    batch_size = MINIBATCH_TIMESTEPS * n_agents
    model = MaskablePPO(
        MaskableActorCriticPolicy, env,
        learning_rate=3e-4, n_steps=n_steps, batch_size=batch_size, n_epochs=N_EPOCHS,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        seed=seed, verbose=0, device="cpu",
    )
    print(f"[IPPO seed={seed}] agents={n_agents} train_episodes={n_train} "
          f"n_steps={n_steps} batch_size={batch_size}")

    shim.env_steps, shim.episodes, shim.t0 = 0, [], time.perf_counter()
    t0 = time.perf_counter()
    model.learn(total_timesteps=10**9, callback=EpisodeBudget(shim, n_train))
    train_time = time.perf_counter() - t0
    if len(shim.episodes) < n_train:
        raise RuntimeError(f"training ended early at {len(shim.episodes)}/{n_train} episodes")
    shim.episodes = shim.episodes[:n_train]

    for ep in shim.episodes[-3:]:
        print(f"[IPPO seed={seed}] ep {ep['episode']} steps={ep['env_steps']} "
              f"return={ep['return']:.1f}")

    rows = [{"algo": "ippo", "seed": seed, **ep} for ep in shim.episodes]
    trained_env_steps = rows[-1]["env_steps"]
    t1 = time.perf_counter()
    eval_returns = evaluate(model, eval_seeds, steps)
    eval_time = time.perf_counter() - t1

    model.save(checkpoint_path(out_dir, "ippo", seed, "zip"))
    config = {"algo": "ippo", "seed": seed, "total_steps": total_steps, "steps": steps,
              "episodes_per_update": EPISODES_PER_UPDATE, "n_steps": n_steps,
              "batch_size": batch_size, "n_epochs": N_EPOCHS, "reward_scale": REWARD_SCALE,
              "eval_seed_base": eval_seed_base}
    summary = save_run(out_dir, "ippo", seed, rows, eval_seeds, eval_returns, steps,
                       trained_env_steps, train_time, eval_time, eval_seed_base, config)
    print(f"[IPPO seed={seed}] eval mean={summary['mean_reward']:.2f} "
          f"std={summary['std_reward']:.2f} train_time={train_time:.0f}s")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Train/evaluate IPPO (MaskablePPO) on real CC4")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 777])
    p.add_argument("--total-steps", type=int, default=20_000,
                   help="nominal budget = episodes x --steps; identical for IPPO and MAPPO")
    p.add_argument("--steps", type=int, default=100, help="steps per episode")
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--eval-seed-base", type=int, default=DEFAULT_EVAL_SEED_BASE)
    p.add_argument("--output-dir", default="results")
    args = p.parse_args()
    for seed in args.seeds:
        train_one(seed, args.total_steps, args.steps, args.eval_episodes,
                  args.eval_seed_base, args.output_dir)


if __name__ == "__main__":
    main()