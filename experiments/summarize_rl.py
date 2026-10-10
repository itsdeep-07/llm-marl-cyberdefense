"""Build the Batch 4 comparison table: Sleep / Random / RuleBased / IPPO / MAPPO.

Reads only saved result files. Fails loudly if the baselines were not scored on
the same held-out seeds as the RL agents, because the numbers would not be
comparable.

Run the baselines on the held-out seeds first:
    python experiments/run_baselines.py --seed 42 --seed-base 9000000 \
        --episodes 30 --steps 100 --output results/tables/baselines_heldout.csv
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tables-dir", default="results/tables")
    p.add_argument("--baselines", default="results/tables/baselines_heldout.csv")
    p.add_argument("--output", default="results/tables/comparison.csv")
    args = p.parse_args()

    rl_files = sorted(glob.glob(os.path.join(args.tables_dir, "rl_eval_ippo_seed*.csv"))
                      + glob.glob(os.path.join(args.tables_dir, "rl_eval_mappo_seed*.csv")))
    if not rl_files:
        raise SystemExit("No rl_eval_*.csv files found. Run experiments/train_cc4.py first.")
    if not os.path.exists(args.baselines):
        raise SystemExit(f"Missing {args.baselines}. See the docstring for the command to run.")

    rl = pd.concat([pd.read_csv(f) for f in rl_files], ignore_index=True)
    base = pd.read_csv(args.baselines)

    for col in ("eval_seed_base", "episodes", "steps"):
        rl_vals, base_vals = set(rl[col]), set(base[col])
        if rl_vals != base_vals:
            raise SystemExit(f"Mismatch in {col}: RL {rl_vals} vs baselines {base_vals}. "
                             "Re-run so both use identical evaluation settings.")

    rows = []
    for name, grp in base.groupby("agent"):
        rows.append({"agent": name, "n_seeds": len(grp),
                     "mean_reward": grp["mean_reward"].mean(),
                     "std_across_seeds": grp["mean_reward"].std(ddof=0),
                     "mean_episode_std": grp["std_reward"].mean(),
                     "train_env_steps": 0, "train_wall_time_s": 0.0})
    for name, grp in rl.groupby("agent"):
        rows.append({"agent": name, "n_seeds": len(grp),
                     "mean_reward": grp["mean_reward"].mean(),
                     "std_across_seeds": grp["mean_reward"].std(ddof=0),
                     "mean_episode_std": grp["std_reward"].mean(),
                     "train_env_steps": int(grp["train_env_steps"].mean()),
                     "train_wall_time_s": grp["train_wall_time_s"].mean()})

    table = pd.DataFrame(rows).sort_values("mean_reward", ascending=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False))
    print(f"saved={os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()