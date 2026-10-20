"""Summarize paired held-out Batch 4 returns with 95% confidence intervals."""

from __future__ import annotations

import argparse
import os

import pandas as pd
from scipy import stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/tables/heldout_episode_returns.csv")
    parser.add_argument("--output", default="results/tables/comparison.csv")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    required = {"agent", "model_seed", "eval_seed", "return"}
    missing = required - set(data.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    sleep = data[data["agent"] == "Sleep"][
        ["model_seed", "eval_seed", "return"]
    ].rename(columns={"return": "sleep_return"})
    if sleep.empty:
        raise SystemExit("No Sleep rows found")

    rows = []
    for agent, current in data.groupby("agent", sort=False):
        joined = current.merge(
            sleep, on=["model_seed", "eval_seed"], validate="one_to_one"
        )
        differences = joined["return"] - joined["sleep_return"]
        ci = stats.t.interval(
            0.95,
            len(differences) - 1,
            loc=float(differences.mean()),
            scale=float(stats.sem(differences)),
        ) if len(differences) > 1 else (float("nan"), float("nan"))
        rows.append({
            "agent": agent,
            "episodes": len(current),
            "mean_return": float(current["return"].mean()),
            "std_across_episodes": float(current["return"].std(ddof=1)),
            "paired_mean_difference_vs_sleep": float(differences.mean()),
            "paired_ci95_low": float(ci[0]),
            "paired_ci95_high": float(ci[1]),
        })

    table = pd.DataFrame(rows).sort_values("mean_return", ascending=False)
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
