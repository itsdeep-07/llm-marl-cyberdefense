"""Metrics for the advisory-mode evaluation (Batch 5, step 5.4).

Pure functions over plain data, so they can be tested without CybORG or Gemini.

What these metrics do and do not show:
* Advisory mode never changes an action, so the reward is the same as RL-only by
  construction. The reward columns exist to PROVE that (identical returns), not
  to claim an improvement.
* Quality is measured only against the telemetry the model was given: does the
  advice notice alerts when they exist, point at a zone that actually has an
  alert, and stay quiet when there is none?
* Only fresh Gemini answers (source == "gemini") are used for latency, cost and
  quality. Cache hits and fallbacks are counted but excluded.
"""

from __future__ import annotations

import numpy as np

NO_ATTACK = "None observed"


def call_row(episode: int, seed: int, step: int, summary: dict, result) -> dict:
    """One row per advisor call, combining the telemetry with the measured result."""
    zones = [(z["zone"], z["hosts_alerting"]) for a in summary["agents"] for z in a["zones"]]
    alert_zones = sorted(name for name, count in zones if count > 0)
    ranked = sorted(result.advice.priorities, key=lambda p: -p.priority_score)
    top = ranked[0] if ranked else None
    error = result.error or ""
    return {
        "episode": episode,
        "eval_seed": seed,
        "step": step,
        "mission_phase": summary["mission_phase"],
        "source": result.source,
        "attempts": result.attempts,
        "latency_ms": result.latency_ms,
        "total_wall_ms": result.total_wall_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd,
        "error_type": error.split(":")[0] if error else "",
        "total_alerting_hosts": summary["total_alerting_hosts"],
        "alert_zones": ";".join(alert_zones),
        "kill_chain_stage": result.advice.incident_report.kill_chain_stage,
        "top_zone": top.zone if top else "",
        "top_score": top.priority_score if top else 0.0,
        "top_zone_has_alert": bool(top and top.zone in alert_zones),
    }


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def _rate(rows, predicate):
    return float(np.mean([predicate(r) for r in rows])) if rows else None


def aggregate(calls: list, episodes: list, meta: dict) -> dict:
    """Summary row for the whole run."""
    total = len(calls)
    by_source = {s: sum(1 for r in calls if r["source"] == s)
                 for s in ("gemini", "cache", "fallback")}
    fresh = [r for r in calls if r["source"] == "gemini"]
    with_alerts = [r for r in fresh if r["total_alerting_hosts"] > 0]
    quiet = [r for r in fresh if r["total_alerting_hosts"] == 0]
    latencies = [r["latency_ms"] for r in fresh]

    out = dict(meta)
    out.update({
        "calls_total": total,
        "calls_gemini": by_source["gemini"],
        "calls_cache": by_source["cache"],
        "calls_fallback": by_source["fallback"],
        "fallback_rate": (by_source["fallback"] / total) if total else None,
        "gemini_mean_attempts": _mean([r["attempts"] for r in fresh]),
        "gemini_mean_latency_ms": _mean(latencies),
        "gemini_p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "gemini_mean_total_wall_ms": _mean([r["total_wall_ms"] for r in fresh]),
        "gemini_mean_input_tokens": _mean([r["input_tokens"] for r in fresh]),
        "gemini_mean_output_tokens": _mean([r["output_tokens"] for r in fresh]),
        "gemini_mean_cost_usd": _mean([r["cost_usd"] for r in fresh]),
        "gemini_total_cost_usd": float(sum(r["cost_usd"] for r in fresh)),
        "calls_with_alerts": len(with_alerts),
        "alert_noticed_rate": _rate(with_alerts, lambda r: r["kill_chain_stage"] != NO_ATTACK),
        "top_zone_has_alert_rate": _rate(with_alerts, lambda r: r["top_zone_has_alert"]),
        "calls_without_alerts": len(quiet),
        "false_alarm_rate": _rate(quiet, lambda r: r["kill_chain_stage"] != NO_ATTACK),
        "episodes": len(episodes),
        "rl_only_mean_return": _mean([e["rl_only_return"] for e in episodes]),
        "advisory_mean_return": _mean([e["advisory_return"] for e in episodes]),
        "episodes_identical_return": sum(1 for e in episodes if e["identical"]),
    })
    return out
