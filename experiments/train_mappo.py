"""Train and evaluate MAPPO on the real CC4 environment (Batch 4).

Uses the same budget rule, seeds, held-out evaluation and reward convention as
train_ippo.py (see rl_common.py). Evaluation is greedy on held-out seeds. A
return is the sum of all Blue agents' rewards over the episode, the same
convention as run_baselines.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.mappo import MAPPOAgent
from experiments.rl_common import (DEFAULT_EVAL_SEED_BASE, EPISODES_PER_UPDATE, RED,
                                   REWARD_SCALE, checkpoint_path, plan_seeds, save_run)


def _stack(env_out: dict, names: list, key=None) -> np.ndarray:
    if key is None:
        return np.stack([np.asarray(env_out[a], dtype=np.float32) for a in names])
    return np.stack([np.asarray(env_out[a][key], dtype=bool) for a in names])


def run_episode(env_factory, agent: MAPPOAgent, env_seed: int, steps: int,
                greedy: bool, record: bool):
    """Play one episode. Returns (team_return, n_env_steps, episode_dict_or_None)."""
    env = env_factory(env_seed, steps)
    obs, info = env.reset(seed=env_seed)
    names = sorted(env.agents)
    buf = {k: [] for k in ("obs", "masks", "actions", "log_probs", "rewards", "values")}
    team_return, n_steps, terminated = 0.0, 0, False

    while True:
        obs_arr, mask_arr = _stack(obs, names), _stack(info, names, "action_mask")
        actions, log_probs, values = agent.act(obs_arr, mask_arr, greedy=greedy)
        obs, rewards, term, trunc, info = env.step(
            {a: int(actions[i]) for i, a in enumerate(names)}
        )
        team_return += float(sum(rewards.values()))
        n_steps += 1
        if record:
            buf["obs"].append(obs_arr)
            buf["masks"].append(mask_arr)
            buf["actions"].append(actions)
            buf["log_probs"].append(log_probs)
            buf["rewards"].append(np.array([rewards[a] for a in names], dtype=np.float32))
            buf["values"].append(values)
        terminated = any(term.values())
        if trunc.get("__all__", False) or terminated:
            break

    if not record:
        return team_return, n_steps, None
    episode = {k: np.stack(v) for k, v in buf.items()}
    episode["terminated"] = terminated
    episode["last_value"] = agent.value(_stack(obs, names))
    return team_return, n_steps, episode


def train_one(seed: int, total_steps: int, steps: int, eval_episodes: int,
              eval_seed_base: int, out_dir: str, env_factory: Callable | None = None,
              hyper: dict | None = None) -> dict:
    if env_factory is None:
        from envs.cc4_env import make_env  # late import so tests can use a mock env
        env_factory = lambda s, n: make_env(seed=s, steps=n, red=RED)  # noqa: E731
    hyper = dict(hyper or {})
    hyper.setdefault("reward_scale", REWARD_SCALE)

    n_train, train_seeds, eval_seeds = plan_seeds(seed, total_steps, steps, eval_episodes,
                                                  eval_seed_base)
    train_seeds = train_seeds[:n_train]

    probe = env_factory(train_seeds[0], steps)
    obs0, info0 = probe.reset(seed=train_seeds[0])
    names = sorted(probe.agents)
    obs_dim = int(np.asarray(obs0[names[0]]).shape[0])
    act_dim = int(np.asarray(info0[names[0]]["action_mask"]).shape[0])
    agent = MAPPOAgent(len(names), obs_dim, act_dim, seed=seed, **hyper)
    print(f"[MAPPO seed={seed}] agents={len(names)} obs_dim={obs_dim} act_dim={act_dim} "
          f"train_episodes={n_train}")

    rows, pending, env_steps = [], [], 0
    t0 = time.perf_counter()
    for i, env_seed in enumerate(train_seeds):
        ret, n, ep = run_episode(env_factory, agent, env_seed, steps, greedy=False, record=True)
        env_steps += n
        pending.append(ep)
        row = {"algo": "mappo", "seed": seed, "episode": i + 1, "episode_seed": env_seed,
               "env_steps": env_steps, "return": ret, "wall_time_s": time.perf_counter() - t0}
        if len(pending) == EPISODES_PER_UPDATE or i == n_train - 1:
            stats = agent.update(pending)
            row.update({k: stats[k] for k in ("policy_loss", "value_loss", "entropy")})
            pending = []
            print(f"[MAPPO seed={seed}] ep {i + 1}/{n_train} steps={env_steps} "
                  f"return={ret:.1f} entropy={stats['entropy']:.3f}")
        rows.append(row)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    eval_returns = [run_episode(env_factory, agent, s, steps, greedy=True, record=False)[0]
                    for s in eval_seeds]
    eval_time = time.perf_counter() - t1

    agent.save(checkpoint_path(out_dir, "mappo", seed, "pt"))
    config = {"algo": "mappo", "seed": seed, "total_steps": total_steps, "steps": steps,
              "episodes_per_update": EPISODES_PER_UPDATE, "hyper": hyper,
              "eval_seed_base": eval_seed_base}
    summary = save_run(out_dir, "mappo", seed, rows, eval_seeds, eval_returns, steps,
                       env_steps, train_time, eval_time, eval_seed_base, config)
    print(f"[MAPPO seed={seed}] eval mean={summary['mean_reward']:.2f} "
          f"std={summary['std_reward']:.2f} train_time={train_time:.0f}s")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Train/evaluate MAPPO on real CC4")
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