"""Evaluate all Batch 4 agents on identical held-out CC4 episodes."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.baselines import RandomPolicy, RuleBasedPolicy, SleepPolicy
from envs.cc4_env import make_env
from experiments.rl_common import RED, team_reward


def run_policy(policy, seed: int, steps: int) -> float:
    env = make_env(seed=seed, steps=steps, red=RED)
    obs, info = env.reset(seed=seed)
    policy.reset()
    total = 0.0
    while True:
        actions = {
            agent: policy.action(
                agent, obs[agent], env.action_labels(agent),
                info[agent]["action_mask"], env.hosts(agent), env.subnets(agent)
            )
            for agent in env.agents
        }
        obs, rewards, terminated, truncated, info = env.step(actions)
        total += team_reward(rewards)
        if truncated.get("__all__", False) or any(terminated.values()):
            return total


def evaluate_baselines(seeds: list[int], eval_seed_base: int, episodes: int,
                       steps: int) -> list[dict]:
    factories: dict[str, Callable[[int], object]] = {
        "Sleep": lambda seed: SleepPolicy(),
        "Random": lambda seed: RandomPolicy(seed),
        "RuleBased": lambda seed: RuleBasedPolicy(),
    }
    rows = []
    for model_seed in seeds:
        for episode in range(episodes):
            eval_seed = eval_seed_base + episode
            for name, factory in factories.items():
                rows.append({
                    "agent": name, "model_seed": model_seed,
                    "eval_seed": eval_seed,
                    "return": run_policy(factory(model_seed + episode), eval_seed, steps),
                })
    return rows


def evaluate_ippo(seeds: list[int], eval_seed_base: int, episodes: int,
                  steps: int, checkpoint_dir: str) -> list[dict]:
    from sb3_contrib import MaskablePPO

    rows = []
    for model_seed in seeds:
        path = os.path.join(checkpoint_dir, f"ippo_seed{model_seed}.zip")
        model = MaskablePPO.load(path)
        for episode in range(episodes):
            eval_seed = eval_seed_base + episode
            env = make_env(seed=eval_seed, steps=steps, red=RED)
            obs, info = env.reset(seed=eval_seed)
            total = 0.0
            while True:
                actions = {}
                for agent in sorted(env.agents):
                    mask = np.asarray(info[agent]["action_mask"], dtype=bool)
                    action, _ = model.predict(obs[agent], action_masks=mask, deterministic=True)
                    actions[agent] = int(action)
                obs, rewards, terminated, truncated, info = env.step(actions)
                total += team_reward(rewards)
                if truncated.get("__all__", False) or any(terminated.values()):
                    break
            rows.append({"agent": "IPPO", "model_seed": model_seed,
                         "eval_seed": eval_seed, "return": total})
    return rows


def evaluate_mappo(seeds: list[int], eval_seed_base: int, episodes: int,
                   steps: int, checkpoint_dir: str) -> list[dict]:
    from agents.mappo import MAPPOAgent

    rows = []
    for model_seed in seeds:
        probe = make_env(seed=eval_seed_base, steps=steps, red=RED)
        obs, info = probe.reset(seed=eval_seed_base)
        names = sorted(probe.agents)
        obs_dim = len(obs[names[0]])
        act_dim = len(info[names[0]]["action_mask"])
        agent = MAPPOAgent(len(names), obs_dim, act_dim, seed=model_seed)
        agent.load(os.path.join(checkpoint_dir, f"mappo_seed{model_seed}.pt"))
        for episode in range(episodes):
            eval_seed = eval_seed_base + episode
            env = make_env(seed=eval_seed, steps=steps, red=RED)
            obs, info = env.reset(seed=eval_seed)
            total = 0.0
            while True:
                names = sorted(env.agents)
                obs_arr = np.stack([obs[name] for name in names])
                mask_arr = np.stack([info[name]["action_mask"] for name in names])
                actions, _, _ = agent.act(obs_arr, mask_arr, greedy=True)
                obs, rewards, terminated, truncated, info = env.step(
                    {name: int(actions[i]) for i, name in enumerate(names)}
                )
                total += team_reward(rewards)
                if truncated.get("__all__", False) or any(terminated.values()):
                    break
            rows.append({"agent": "MAPPO", "model_seed": model_seed,
                         "eval_seed": eval_seed, "return": total})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 777])
    parser.add_argument("--eval-seed-base", type=int, default=9_000_000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--output", default="results/tables/heldout_episode_returns.csv")
    args = parser.parse_args()

    rows = evaluate_baselines(args.seeds, args.eval_seed_base, args.episodes, args.steps)
    rows += evaluate_ippo(args.seeds, args.eval_seed_base, args.episodes, args.steps,
                          args.checkpoint_dir)
    rows += evaluate_mappo(args.seeds, args.eval_seed_base, args.episodes, args.steps,
                           args.checkpoint_dir)
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"saved={output} rows={len(rows)}")


if __name__ == "__main__":
    main()
