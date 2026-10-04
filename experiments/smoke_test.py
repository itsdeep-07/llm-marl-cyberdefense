"""Run a small real CC4 rollout with a Sleep team."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.cc4_env import make_env


def run_smoke_test(seed: int, episodes: int, steps: int) -> None:
    for episode in range(episodes):
        episode_seed = seed + episode
        env = make_env(seed=episode_seed, steps=steps)
        observations, infos = env.reset(seed=episode_seed)
        del observations, infos

        red_agents = sorted(
            name
            for name in env.env.environment_controller.agent_interfaces
            if name.startswith("red_")
        )
        total_reward = 0.0

        for step in range(steps):
            actions = {agent: 0 for agent in env.agents}
            _, rewards, terminated, truncated, _ = env.step(actions)
            total_reward += sum(rewards.values())
            red_activity = {
                agent: str(env.env.get_last_action(agent)) for agent in red_agents
            }
            blue_actions = {
                agent: str(env.get_last_action(agent)) for agent in env.agents
            }
            print(
                f"episode={episode} seed={episode_seed} step={step} "
                f"red={red_activity} blue={blue_actions} rewards={rewards}"
            )

            if truncated.get("__all__", False) or any(terminated.values()):
                break

        print(
            f"episode={episode} seed={episode_seed} "
            f"total_reward={total_reward}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test real CC4 rollouts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    run_smoke_test(args.seed, args.episodes, args.steps)


if __name__ == "__main__":
    main()
