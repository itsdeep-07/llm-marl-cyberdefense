"""Live smoke test for the LLM layer (Batch 5, step 5.3).

Makes a handful of REAL Gemini calls on real CC4 telemetry and prints exactly
what the orchestrator measured. This only checks that the pipeline works end to
end. It is NOT an evaluation and its numbers must not be reported as results.

Run from the repo root (needs GEMINI_API_KEY in .env):
    python experiments/llm_smoke.py --call-steps 0 20 60
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.baselines import RuleBasedPolicy
from envs.cc4_env import make_env
from llm.orchestrator import LLMOrchestrator
from llm.state_summary import StateSummarizer


def show(label: str, result) -> None:
    print(f"\n--- {label} | step {result.step} ---")
    print(f"source={result.source}  counts_for_performance={result.counts_for_performance}")
    print(f"model={result.model}  prompt_version={result.prompt_version}")
    print(f"attempts={result.attempts}  total_wall_ms={result.total_wall_ms:.0f}")
    print(f"latency_ms={result.latency_ms:.0f}  input_tokens={result.input_tokens}  "
          f"output_tokens={result.output_tokens}  cost_usd={result.cost_usd:.6f}")
    if result.origin_latency_ms is not None:
        print(f"(original API call: latency_ms={result.origin_latency_ms:.0f}, "
              f"cost_usd={result.origin_cost_usd:.6f})")
    if result.error:
        print(f"error={result.error}")
    report = result.advice.incident_report
    print(f"kill_chain_stage={report.kill_chain_stage}")
    print(f"summary={report.executive_summary}")
    for p in result.advice.priorities[:3]:
        print(f"priority {p.priority_score:.2f}  {p.zone}: {p.reasoning}")
    for c in result.advice.conflicts[:2]:
        print(f"conflict {c.agents_involved}: {c.conflict_type}")
    for action in report.mitigation_actions[:3]:
        print(f"mitigation: {action}")


def main() -> None:
    p = argparse.ArgumentParser(description="Live Gemini smoke test on CC4 telemetry")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--call-steps", type=int, nargs="+", default=[0, 20, 60])
    p.add_argument("--thinking-level", default=None,
                   help="try a thinking level for this run only (does not edit the config)")
    args = p.parse_args()

    # raise_on_error=True: any problem stops the test and shows the real error
    orch = LLMOrchestrator(raise_on_error=True)
    if args.thinking_level:
        orch.cfg["thinking_level"] = args.thinking_level
    print(orch)

    env = make_env(seed=args.seed, steps=args.steps, red="FiniteStateRedAgent")
    observation, info = env.reset(seed=args.seed)
    policy = RuleBasedPolicy()
    policy.reset()
    summarizer = StateSummarizer()

    results, step = [], 0
    while step <= max(args.call_steps):
        actions = {
            a: policy.action(a, observation[a], env.action_labels(a), info[a]["action_mask"])
            for a in env.agents
        }
        if step in args.call_steps:
            summary = summarizer.summarize(env, step)
            proposals = {a: env.action_labels(a)[int(actions[a])] for a in actions}
            result = orch.advise(summary, proposals)
            show("live call", result)
            results.append(result)
            if len(results) == 1:  # same input again: must come back from the cache
                show("repeat of the first call", orch.advise(summary, proposals))
        observation, rewards, terminated, truncated, info = env.step(actions)
        summarizer.record(env)
        step += 1

    live = [r for r in results if r.source == "gemini"]
    print(f"\nlive calls: {len(live)}  total cost_usd={sum(r.cost_usd for r in live):.6f}  "
          f"mean latency_ms={sum(r.latency_ms for r in live) / max(1, len(live)):.0f}")


if __name__ == "__main__":
    main()