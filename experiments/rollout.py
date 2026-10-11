"""Save one JSON log per real CC4 episode (Batch 3).

Only values returned by the environment are logged.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.baselines import RandomPolicy, RuleBasedPolicy, SleepPolicy
from envs.cc4_env import make_env

POLICIES = {
    "Sleep": lambda seed: SleepPolicy(),
    "Random": lambda seed: RandomPolicy(seed),
    "RuleBased": lambda seed: RuleBasedPolicy(),
}


def git_commit() -> str:
    """Current commit hash, or 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def snapshot_hosts(env, red_agents) -> dict:
    """Return {hostname: "red_session" | "no_red_session"} from the true state.

    Uses two calls documented in the CC4 'Debugging Tools' tutorial:
      cyborg.get_ip_map()          -> {hostname: ip}, every host/router
      cyborg.get_agent_state(name) -> true state; per-host dict with 'Sessions'
    'red_session' means a Red agent holds a session on that host. It does not
    say whether the privilege is user or root. Check the layout first with
    experiments/inspect_state.py; this raises if the layout looks different.
    """
    cyborg = env.env  # BlueFlatWrapper.env is the CybORG object (see smoke_test.py)
    hosts = {name: "no_red_session" for name in cyborg.get_ip_map()}
    for red in red_agents:
        for hostname, data in cyborg.get_agent_state(red).items():
            if hostname == "success":
                continue
            if hostname not in hosts:
                raise KeyError(f"true state has unknown host {hostname!r}")
            for session in data.get("Sessions", []):
                if str(session.get("agent", "")).startswith("red_"):
                    hosts[hostname] = "red_session"
    return hosts


def run_episode(agent_name: str, seed: int, steps: int, red: str) -> dict:
    env = make_env(seed=seed, steps=steps, red=red)
    observation, info = env.reset(seed=seed)
    policy = POLICIES[agent_name](seed)
    policy.reset()

    # Same red-agent discovery as experiments/smoke_test.py
    red_agents = sorted(
        n for n in env.env.environment_controller.agent_interfaces
        if n.startswith("red_")
    )
    log, cumulative, step = [], 0.0, 0

    while True:
        actions = {
            a: policy.action(a, observation[a], env.action_labels(a),
                             info[a]["action_mask"])
            for a in env.agents
        }
        observation, rewards, terminated, truncated, info = env.step(actions)
        # Same convention as run_baselines.py so totals stay comparable
        reward = float(sum(rewards.values()))
        cumulative += reward
        log.append({
            "step": step,
            "red_activity": {r: str(env.env.get_last_action(r)) for r in red_agents},
            "blue_actions": {a: str(env.get_last_action(a)) for a in env.agents},
            "rewards_per_agent": {a: float(v) for a, v in rewards.items()},
            "reward": reward,
            "cumulative_reward": cumulative,
            "host_state": snapshot_hosts(env, red_agents),
        })
        step += 1
        done = truncated.get("__all__", False) or any(terminated.values())
        if done:
            break

    host_count = len(log[-1]["host_state"])
    return {
        "provenance": {
            "env": "CC4", "agent": agent_name, "red": red, "seed": seed,
            "steps": steps, "git_commit": git_commit(),
            "source": "real_rollout", "host_count": host_count,
        },
        "total_reward": cumulative,
        "steps": log,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Save real CC4 episode logs")
    p.add_argument("--agent", choices=sorted(POLICIES), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--red", default="FiniteStateRedAgent")
    p.add_argument("--output-dir", default="results/episodes")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for ep in range(args.episodes):
        ep_seed = args.seed + ep  # same seeding as run_baselines.py
        data = run_episode(args.agent, ep_seed, args.steps, args.red)
        name = f"episode_{args.agent}_{args.red}_seed{ep_seed}.json"
        with open(os.path.join(args.output_dir, name), "w") as f:
            json.dump(data, f, indent=2)
        print(f"saved={name} total_reward={data['total_reward']:.3f}")


if __name__ == "__main__":
    main()