"""Print real state summaries from a CC4 episode (Batch 5, step 5.1).

No API key needed. Run from the repo root:
    python experiments/inspect_summary.py --seed 42 --print-steps 0 20 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.baselines import RandomPolicy, RuleBasedPolicy, SleepPolicy
from envs.cc4_env import make_env
from llm.state_summary import StateSummarizer

POLICIES = {
    "Sleep": lambda seed: SleepPolicy(),
    "Random": lambda seed: RandomPolicy(seed),
    "RuleBased": lambda seed: RuleBasedPolicy(),
}


def main() -> None:
    p = argparse.ArgumentParser(description="Print CC4 state summaries")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--agent", choices=sorted(POLICIES), default="RuleBased")
    p.add_argument("--red", default="FiniteStateRedAgent")
    p.add_argument("--print-steps", type=int, nargs="+", default=[0, 20, 60])
    args = p.parse_args()

    env = make_env(seed=args.seed, steps=args.steps, red=args.red)
    observation, info = env.reset(seed=args.seed)
    policy = POLICIES[args.agent](args.seed)
    policy.reset()
    summarizer = StateSummarizer()

    step = 0
    while True:
        if step in args.print_steps:
            summary = summarizer.summarize(env, step).to_dict()
            text = json.dumps(summary, indent=2)
            print(f"\n===== step {step} | {len(text)} characters =====")
            print(text)
        actions = {
            a: policy.action(a, observation[a], env.action_labels(a), info[a]["action_mask"])
            for a in env.agents
        }
        observation, rewards, terminated, truncated, info = env.step(actions)
        summarizer.record(env)
        step += 1
        if truncated.get("__all__", False) or any(terminated.values()):
            break
    print(f"\nepisode finished after {step} steps")


if __name__ == "__main__":
    main()
