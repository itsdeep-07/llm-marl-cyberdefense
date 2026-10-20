"""Advisory-mode evaluation (Batch 5, step 5.4).

For each held-out evaluation seed this plays the trained IPPO policy twice:
  1. RL-only: no LLM.
  2. RL + advisory: every --call-every steps the LLM is shown the telemetry and the
     actions the RL agents are about to take, and gives advice. The advice is
     logged and NEVER changes an action.

Because advice cannot change actions, both runs must give the same return. The
script checks that (and checks against your saved evaluation file) to prove the
setup is sound. What it really measures is the advisor: reliability, latency,
cost, and whether its advice agrees with the telemetry it was given.

Only fresh Gemini answers count for performance; cache hits and fallbacks are
counted and excluded. Run from the repo root:

    python experiments/eval_advisory.py --episodes 1          # quick trial
    python experiments/eval_advisory.py --episodes 10         # the evaluation
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3_contrib import MaskablePPO

from envs.cc4_env import make_env
from llm.advisory_metrics import aggregate, call_row
from llm.orchestrator import LLMOrchestrator
from llm.state_summary import StateSummarizer

RED = "FiniteStateRedAgent"


def play(model, seed, steps, orch=None, call_every=20, episode=0, advice_dir=None,
         spend=None, max_cost=None, force_live=False, max_fallbacks=3):
    """Play one episode greedily. If orch is given, ask it for advice every call_every steps."""
    env = make_env(seed=seed, steps=steps, red=RED)
    obs, info = env.reset(seed=seed)
    names = sorted(env.agents)
    summarizer = StateSummarizer()
    total, rows, step = 0.0, [], 0

    while True:
        actions = {}
        for a in names:
            mask = np.asarray(info[a]["action_mask"], dtype=bool)
            action, _ = model.predict(obs[a], action_masks=mask, deterministic=True)
            actions[a] = int(action)

        if orch is not None and step % call_every == 0:
            if spend["usd"] > max_cost:
                raise RuntimeError(f"cost limit ${max_cost:.2f} reached; stopping before more calls")
            summary = summarizer.summarize(env, step)
            proposals = {a: env.action_labels(a)[actions[a]] for a in names}
            result = orch.advise(summary, proposals, force_live=force_live)
            spend["usd"] += result.cost_usd
            spend["fallbacks_in_a_row"] = (spend["fallbacks_in_a_row"] + 1
                                           if result.source == "fallback" else 0)
            summary_dict = summary.to_dict()
            rows.append(call_row(episode, seed, step, summary_dict, result))
            with open(os.path.join(advice_dir, f"seed{seed}_step{step:03d}.json"), "w") as f:
                json.dump({"summary": summary_dict, "proposals": proposals,
                           "result": json.loads(result.model_dump_json())}, f, indent=2)
            print(f"  seed={seed} step={step:3d} source={result.source:8s} "
                  f"attempts={result.attempts} latency_ms={result.latency_ms:7.0f} "
                  f"cost_usd={result.cost_usd:.5f} stage={result.advice.incident_report.kill_chain_stage}"
                  + (f" ERROR={result.error}" if result.error else ""), flush=True)

        if orch is not None and spend["fallbacks_in_a_row"] >= max_fallbacks:
            raise RuntimeError(
                f"{max_fallbacks} fallbacks in a row (last error: {result.error}). The API is "
                "probably out of quota or unreachable; stopping instead of wasting time. "
                "Partial results are saved.")
        obs, rewards, term, trunc, info = env.step(actions)
        summarizer.record(env)
        total += float(sum(rewards.values()))
        step += 1
        if trunc.get("__all__", False) or any(term.values()):
            break
    return total, rows


def main() -> None:
    p = argparse.ArgumentParser(description="Advisory-mode evaluation")
    p.add_argument("--checkpoint", default="results/checkpoints/ippo_seed42.zip")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--call-every", type=int, default=20)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--eval-seed-base", type=int, default=9_000_000)
    p.add_argument("--max-cost-usd", type=float, default=1.0)
    p.add_argument("--max-consecutive-fallbacks", type=int, default=3,
                   help="stop the run after this many fallbacks in a row (quota or outage)")
    p.add_argument("--force-live", action="store_true",
                   help="ignore the cache so every call is a fresh API call")
    p.add_argument("--output-dir", default="results")
    args = p.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    tag = os.path.splitext(os.path.basename(args.checkpoint))[0]  # e.g. ippo_seed42
    tables = os.path.join(args.output_dir, "tables")
    advice_dir = os.path.join(args.output_dir, "advisory")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(advice_dir, exist_ok=True)

    saved = None
    saved_path = os.path.join(tables, f"rl_eval_episodes_{tag}.csv")
    if os.path.exists(saved_path):
        saved = pd.read_csv(saved_path).set_index("eval_seed")["return"].to_dict()

    model = MaskablePPO.load(args.checkpoint, device="cpu")
    orch = LLMOrchestrator(raise_on_error=False)
    print(orch, f"| checkpoint={tag} | thinking_level={orch.cfg['thinking_level']}")

    spend = {"usd": 0.0, "fallbacks_in_a_row": 0}
    calls, episodes = [], []
    try:
        for i in range(args.episodes):
            seed = args.eval_seed_base + i
            print(f"episode {i + 1}/{args.episodes} (seed {seed})", flush=True)
            rl_only, _ = play(model, seed, args.steps)
            advisory, rows = play(model, seed, args.steps, orch=orch,
                                  call_every=args.call_every, episode=i, advice_dir=advice_dir,
                                  spend=spend, max_cost=args.max_cost_usd,
                                  force_live=args.force_live,
                                  max_fallbacks=args.max_consecutive_fallbacks)
            calls.extend(rows)
            episodes.append({
                "episode": i, "eval_seed": seed, "rl_only_return": rl_only,
                "advisory_return": advisory, "identical": abs(rl_only - advisory) < 1e-9,
                "saved_eval_return": saved.get(seed) if saved else None,
                "matches_saved_eval": (abs(rl_only - saved[seed]) < 1e-6) if saved and seed in saved else None,
            })
            print(f"  returns: rl_only={rl_only:.1f} advisory={advisory:.1f} "
                  f"identical={episodes[-1]['identical']} matches_saved={episodes[-1]['matches_saved_eval']}",
                  flush=True)
    finally:
        # write whatever finished, even if a cost limit or an error stopped the run
        if calls:
            pd.DataFrame(calls).to_csv(os.path.join(tables, "advisory_calls.csv"), index=False)
        if episodes:
            pd.DataFrame(episodes).to_csv(os.path.join(tables, "advisory_episodes.csv"), index=False)
            meta = {"checkpoint": tag, "model": orch.model, "prompt_version": orch.prompt_version,
                    "thinking_level": orch.cfg["thinking_level"], "call_every": args.call_every,
                    "steps": args.steps, "eval_seed_base": args.eval_seed_base,
                    "force_live": args.force_live}
            summary = aggregate(calls, episodes, meta)
            pd.DataFrame([summary]).to_csv(os.path.join(tables, "advisory_summary.csv"), index=False)
            print("\n=== advisory summary ===")
            for key, value in summary.items():
                print(f"{key}: {value}")
            print(f"\nsaved tables to {os.path.abspath(tables)} and advice files to "
                  f"{os.path.abspath(advice_dir)}")


if __name__ == "__main__":
    main()