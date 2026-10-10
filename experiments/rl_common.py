"""Rules and output files shared by train_ippo.py and train_mappo.py.

Keeping these in one place guarantees that IPPO and MAPPO use the same budget
rule, the same held-out evaluation seeds, the same reward convention and the
same output format.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

RED = "FiniteStateRedAgent"
TRAIN_SEED_STRIDE = 100_000     # training episode seed = run_seed * stride + episode index
DEFAULT_EVAL_SEED_BASE = 9_000_000
SPARE_TRAIN_SEEDS = 10          # SB3 may reset a few more times than planned
REWARD_SCALE = 0.1              # both algorithms learn from rewards * 0.1; logs are unscaled
EPISODES_PER_UPDATE = 4

TRAIN_COLUMNS = ["algo", "seed", "episode", "episode_seed", "env_steps", "return",
                 "wall_time_s", "policy_loss", "value_loss", "entropy"]


def plan_seeds(seed: int, total_steps: int, steps: int, eval_episodes: int,
               eval_seed_base: int):
    """Return (n_train_episodes, training_seeds, evaluation_seeds), failing loudly on bad input."""
    if total_steps % steps != 0:
        raise ValueError(f"--total-steps ({total_steps}) must be a multiple of --steps ({steps}) "
                         "so both algorithms train on whole episodes")
    n_train = total_steps // steps
    train = [seed * TRAIN_SEED_STRIDE + i for i in range(n_train + SPARE_TRAIN_SEEDS)]
    evals = [eval_seed_base + i for i in range(eval_episodes)]
    overlap = set(train) & set(evals)
    if overlap:
        raise ValueError(f"evaluation seeds overlap training seeds: {sorted(overlap)[:5]}")
    return n_train, train, evals


def save_run(out_dir: str, algo: str, seed: int, rows: list, eval_seeds: list,
             eval_returns: list, steps: int, total_steps: int, train_time: float,
             eval_time: float, eval_seed_base: int, config: dict) -> dict:
    """Write the training log, evaluation files and config; return the summary row."""
    for sub in ("training", "tables", "checkpoints"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    tag = f"{algo}_seed{seed}"

    train_df = pd.DataFrame(rows).reindex(columns=TRAIN_COLUMNS)
    train_df.to_csv(os.path.join(out_dir, "training", f"{tag}_train.csv"), index=False)
    with open(os.path.join(out_dir, "training", f"{tag}_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    pd.DataFrame({"eval_seed": eval_seeds, "return": eval_returns}).to_csv(
        os.path.join(out_dir, "tables", f"rl_eval_episodes_{tag}.csv"), index=False)

    summary = {
        "seed": seed, "agent": algo.upper(), "red": RED, "episodes": len(eval_returns),
        "steps": steps, "mean_reward": float(np.mean(eval_returns)),
        "std_reward": float(np.std(eval_returns)), "eval_seed_base": eval_seed_base,
        "train_env_steps": int(total_steps), "train_wall_time_s": train_time,
        "eval_wall_time_s": eval_time,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "tables", f"rl_eval_{tag}.csv"),
                                   index=False)
    return summary


def checkpoint_path(out_dir: str, algo: str, seed: int, ext: str) -> str:
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    return os.path.join(out_dir, "checkpoints", f"{algo}_seed{seed}.{ext}")