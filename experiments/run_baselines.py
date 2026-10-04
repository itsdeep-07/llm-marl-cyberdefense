"""Evaluate real CC4 baseline teams and save reproducible summary tables."""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.baselines import RandomPolicy, RuleBasedPolicy, SleepPolicy
from envs.cc4_env import make_env


PolicyFactory = Callable[[int], object]


def _run_episode(seed: int, policy: object, steps: int) -> float:
    env = make_env(seed=seed, steps=steps)
    observation, info = env.reset(seed=seed)
    policy.reset()
    total_reward = 0.0

    while True:
        actions = {
            agent: policy.action(
                agent,
                observation[agent],
                env.action_labels(agent),
                info[agent]["action_mask"],
            )
            for agent in env.agents
        }
        observation, rewards, terminated, truncated, info = env.step(actions)
        total_reward += float(sum(rewards.values()))
        if truncated.get("__all__", False) or any(terminated.values()):
            return total_reward


def evaluate_policy(
    policy_factory: PolicyFactory,
    seed: int,
    episodes: int,
    steps: int,
) -> tuple[float, float]:
    returns = [
        _run_episode(seed=seed + episode, policy=policy_factory(seed + episode), steps=steps)
        for episode in range(episodes)
    ]
    return float(np.mean(returns)), float(np.std(returns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real CC4 baseline evaluations")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", default="results/tables/baselines.csv")
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else [42, 101, 777]
    random.seed(args.seed)
    np.random.seed(args.seed)
    policy_factories: dict[str, PolicyFactory] = {
        "Sleep": lambda seed: SleepPolicy(),
        "Random": lambda seed: RandomPolicy(seed),
        "RuleBased": lambda seed: RuleBasedPolicy(),
    }
    rows = []
    for seed in seeds:
        for name, factory in policy_factories.items():
            mean_reward, std_reward = evaluate_policy(
                factory, seed=seed, episodes=args.episodes, steps=args.steps
            )
            rows.append(
                {
                    "seed": seed,
                    "agent": name,
                    "red": "FiniteStateRedAgent",
                    "episodes": args.episodes,
                    "steps": args.steps,
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                }
            )
            print(
                f"seed={seed} agent={name} episodes={args.episodes} "
                f"mean_reward={mean_reward:.3f} std_reward={std_reward:.3f}"
            )

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
