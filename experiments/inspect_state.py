"""Print CC4's true state so you can see what rollout.py will log.

Run from the repo root:  python experiments/inspect_state.py --steps 30
"""

from __future__ import annotations

import argparse
import os
import sys
from pprint import pprint

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.cc4_env import make_env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args()

    env = make_env(seed=args.seed, steps=args.steps)
    env.reset(seed=args.seed)
    cyborg = env.env
    red_agents = sorted(
        n for n in cyborg.environment_controller.agent_interfaces
        if n.startswith("red_")
    )
    print("red agents:", red_agents)
    print("number of hosts in get_ip_map():", len(cyborg.get_ip_map()))

    for _ in range(args.steps):
        env.step({a: 0 for a in env.agents})  # everyone Sleeps, as in smoke_test.py

    for red in red_agents:
        state = cyborg.get_agent_state(red)
        print(f"\n=== get_agent_state({red!r}): {len(state)} keys ===")
        for hostname, data in state.items():
            if hostname == "success" or not isinstance(data, dict):
                continue
            print(hostname, "->", [s.get("agent") for s in data.get("Sessions", [])])
    first = next(iter(cyborg.get_agent_state(red_agents[0]).items()))
    print("\nOne raw entry, for reference:")
    pprint(first)


if __name__ == "__main__":
    main()