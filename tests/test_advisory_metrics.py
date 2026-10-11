"""Tests for llm/advisory_metrics.py.

The results below are HAND-WRITTEN STAND-INS, not Gemini output. They only check
the arithmetic and the rule that cache hits and fallbacks are excluded.

Run from the repo root:  python -m pytest tests/test_advisory_metrics.py -v
"""

import os
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.advisory_metrics import aggregate, call_row  # noqa: E402


def summary(alerting_zone=None):
    zones = [("zone_a", 0), ("zone_b", 0)]
    if alerting_zone:
        zones = [(n, 1 if n == alerting_zone else 0) for n, _ in zones]
    return {
        "step": 20, "mission_phase": "MissionA",
        "total_alerting_hosts": 1 if alerting_zone else 0,
        "agents": [{"agent": "blue_agent_0", "last_actions": [],
                    "zones": [{"zone": n, "hosts_present": 5, "hosts_alerting": c,
                               "alert_hosts": [], "blocked_subnets": []} for n, c in zones]}],
    }


def result(source="gemini", stage="None observed", top="zone_a", score=0.5, latency=1000.0,
           cost=0.002, attempts=1, error=None):
    advice = NS(priorities=[NS(zone=top, priority_score=score)],
                incident_report=NS(kill_chain_stage=stage))
    return NS(source=source, attempts=attempts, latency_ms=latency, total_wall_ms=latency + 5,
              input_tokens=800, output_tokens=500, cost_usd=cost, error=error, advice=advice)


def test_call_row_marks_whether_top_zone_has_the_alert():
    hit = call_row(0, 9_000_000, 20, summary("zone_b"), result(stage="Execution", top="zone_b"))
    miss = call_row(0, 9_000_000, 20, summary("zone_b"), result(stage="Execution", top="zone_a"))
    assert hit["top_zone_has_alert"] is True and hit["alert_zones"] == "zone_b"
    assert miss["top_zone_has_alert"] is False


def test_error_type_is_kept_but_not_the_full_message():
    row = call_row(0, 1, 0, summary(), result(source="fallback", error="ServerError: 503 ..."))
    assert row["error_type"] == "ServerError"


def test_aggregate_uses_only_fresh_gemini_calls_for_latency_cost_and_quality():
    calls = [
        call_row(0, 1, 0, summary(), result(latency=1000, cost=0.002)),                      # quiet, correct
        call_row(0, 1, 20, summary("zone_b"), result(stage="Execution", top="zone_b",
                                                      latency=3000, cost=0.004)),            # alert, correct
        call_row(0, 1, 40, summary("zone_b"), result(stage="None observed", top="zone_a")),  # alert, missed
        call_row(0, 1, 60, summary(), result(stage="Execution")),                             # quiet, false alarm
        call_row(0, 1, 80, summary(), result(source="cache", latency=0.0, cost=0.0)),
        call_row(0, 1, 99, summary(), result(source="fallback", latency=0.0, cost=0.0, attempts=3,
                                              error="ServerError: 503")),
    ]
    episodes = [{"rl_only_return": -500.0, "advisory_return": -500.0, "identical": True},
                {"rl_only_return": -700.0, "advisory_return": -700.0, "identical": True}]
    out = aggregate(calls, episodes, {"model": "m"})

    assert out["model"] == "m"
    assert (out["calls_total"], out["calls_gemini"], out["calls_cache"], out["calls_fallback"]) == (6, 4, 1, 1)
    assert out["fallback_rate"] == pytest.approx(1 / 6)
    assert out["gemini_mean_latency_ms"] == pytest.approx((1000 + 3000 + 1000 + 1000) / 4)
    assert out["gemini_total_cost_usd"] == pytest.approx(0.002 + 0.004 + 0.002 + 0.002)
    assert out["calls_with_alerts"] == 2
    assert out["alert_noticed_rate"] == pytest.approx(0.5)
    assert out["top_zone_has_alert_rate"] == pytest.approx(0.5)
    assert out["calls_without_alerts"] == 2
    assert out["false_alarm_rate"] == pytest.approx(0.5)
    assert out["rl_only_mean_return"] == pytest.approx(-600.0)
    assert out["episodes_identical_return"] == 2


def test_aggregate_handles_a_run_with_no_fresh_calls():
    calls = [call_row(0, 1, 0, summary(), result(source="fallback", error="no_api_key: x"))]
    out = aggregate(calls, [], {})
    assert out["calls_gemini"] == 0 and out["gemini_mean_latency_ms"] is None
    assert out["alert_noticed_rate"] is None and out["fallback_rate"] == 1.0
