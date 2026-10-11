"""Tests for the Batch 5 LLM layer.

These use a FAKE Gemini client and a hand-written telemetry snapshot. They check
the mechanics (labels, measured fields, cost arithmetic, cache key, error
handling, key safety). Nothing here is a real model result and nothing produced
by these tests may be used in any report.

Run from the repo root:  python -m pytest tests/test_llm.py -v
"""

import json
import os
import sys
import time
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.orchestrator import LLMOrchestrator, fallback_advice, load_config, make_cache_key  # noqa: E402
from llm.state_summary import AgentSummary, StateSummary, ZoneSummary  # noqa: E402

SECRET = "AIzaSyFAKE_KEY_FOR_TESTS_0123456789abcd"

SUMMARY = {
    "step": 20, "mission_phase": "Preplanning", "total_alerting_hosts": 1,
    "agents": [
        {"agent": "blue_agent_0", "last_actions": ["[Sleep]"], "zones": [
            {"zone": "restricted_zone_a_subnet", "hosts_present": 14, "hosts_alerting": 1,
             "alert_hosts": ["restricted_zone_a_subnet_user_host_2:process"],
             "blocked_subnets": []}]},
        {"agent": "blue_agent_1", "last_actions": ["[Sleep]"], "zones": [
            {"zone": "operational_zone_a_subnet", "hosts_present": 10, "hosts_alerting": 0,
             "alert_hosts": [], "blocked_subnets": []}]},
    ],
}

GOOD = {
    "priorities": [{"zone": "restricted_zone_a_subnet", "priority_score": 0.8,
                    "reasoning": "One host shows a suspicious process."}],
    "conflicts": [],
    "incident_report": {"executive_summary": "One alerting host in restricted zone A.",
                        "kill_chain_stage": "Execution",
                        "mitigation_actions": ["Analyse the alerting host."]},
}
USAGE = SimpleNamespace(prompt_token_count=1000, candidates_token_count=500,
                        thoughts_token_count=250)


class ApiError(Exception):
    """Looks like google.genai.errors.APIError: it carries an HTTP status code."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class FakeClient:
    """Stands in for google.genai.Client."""

    def __init__(self, text=None, usage=USAGE, error=None, fail_first=0, fail_code=503):
        self.text = json.dumps(GOOD) if text is None else text
        self.usage, self.error, self.calls = usage, error, 0
        self.fail_first, self.fail_code = fail_first, fail_code
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls += 1
        time.sleep(0.01)
        if self.calls <= self.fail_first:
            raise ApiError(self.fail_code, f"{self.fail_code} temporary problem")
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text, usage_metadata=self.usage)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)


def write_config(tmp_path, **overrides):
    cfg = {"model": "test-model", "prompt_version": "v1", "temperature": None,
           "thinking_level": None, "max_output_tokens": None,
           "max_retries": 2, "retry_base_seconds": 0,
           "prices": {"input_usd_per_1m": 1.0, "output_usd_per_1m": 2.0,
                      "valid_through": "2099-12-31"}}
    cfg.update(overrides)
    path = tmp_path / "llm_config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def make(tmp_path, client, **kw):
    return LLMOrchestrator(config_path=write_config(tmp_path), cache_dir=str(tmp_path / "cache"),
                           client=client, env_path=None, **kw)


def test_fresh_answer_is_labelled_measured_and_costed(tmp_path):
    client = FakeClient()
    result = make(tmp_path, client).advise(SUMMARY, {"blue_agent_0": "Sleep"})
    assert result.source == "gemini" and result.counts_for_performance
    assert result.latency_ms >= 10.0                       # measured, includes the fake 10 ms
    assert (result.input_tokens, result.output_tokens) == (1000, 750)  # thinking counted as output
    assert result.cost_usd == pytest.approx((1000 * 1.0 + 750 * 2.0) / 1e6)
    assert result.error is None and len(result.cache_key) == 64
    assert result.attempts == 1 and result.total_wall_ms >= result.latency_ms


def test_second_identical_call_is_a_labelled_cache_hit(tmp_path):
    client = FakeClient()
    orch = make(tmp_path, client)
    first = orch.advise(SUMMARY, {})
    second = orch.advise(SUMMARY, {})
    assert client.calls == 1
    assert second.source == "cache" and not second.counts_for_performance
    assert second.cost_usd == 0.0 and second.input_tokens == 0 and second.attempts == 0
    assert second.origin_cost_usd == pytest.approx(first.cost_usd)
    assert second.origin_latency_ms == pytest.approx(first.latency_ms)


def test_force_live_skips_the_cache(tmp_path):
    client = FakeClient()
    orch = make(tmp_path, client)
    orch.advise(SUMMARY, {})
    again = orch.advise(SUMMARY, {}, force_live=True)
    assert client.calls == 2 and again.source == "gemini"


def test_cache_key_is_sha256_and_depends_on_model_and_prompt_version():
    base = dict(model="m", prompt_version="v1", system_prompt="s", settings={}, payload={"a": 1})
    key = make_cache_key(**base)
    assert len(key) == 64 and int(key, 16) >= 0
    assert key == make_cache_key(**base)
    assert key != make_cache_key(**{**base, "model": "m2"})
    assert key != make_cache_key(**{**base, "prompt_version": "v2"})
    assert key != make_cache_key(**{**base, "payload": {"a": 2}})


def test_invalid_json_becomes_a_labelled_fallback_and_is_not_cached(tmp_path):
    orch = make(tmp_path, FakeClient(text="this is not json"))
    result = orch.advise(SUMMARY, {})
    assert result.source == "fallback" and not result.counts_for_performance
    assert "JSONDecodeError" in result.error
    assert result.cost_usd == 0.0
    assert os.listdir(tmp_path / "cache") == []


def test_raise_on_error_raises(tmp_path):
    orch = make(tmp_path, FakeClient(text="nope"), raise_on_error=True)
    with pytest.raises(Exception):
        orch.advise(SUMMARY, {})


def test_advice_mentioning_unknown_zone_is_rejected(tmp_path):
    bad = json.loads(json.dumps(GOOD))
    bad["priorities"][0]["zone"] = "made_up_zone"
    result = make(tmp_path, FakeClient(text=json.dumps(bad))).advise(SUMMARY, {})
    assert result.source == "fallback" and "unknown zones" in result.error


def test_out_of_range_score_is_rejected(tmp_path):
    bad = json.loads(json.dumps(GOOD))
    bad["priorities"][0]["priority_score"] = 1.7
    assert make(tmp_path, FakeClient(text=json.dumps(bad))).advise(SUMMARY, {}).source == "fallback"


def test_missing_token_usage_is_rejected_not_invented(tmp_path):
    result = make(tmp_path, FakeClient(usage=None)).advise(SUMMARY, {})
    assert result.source == "fallback" and "token usage" in result.error


def test_api_error_is_recorded_not_swallowed(tmp_path):
    result = make(tmp_path, FakeClient(error=RuntimeError("429 rate limited"))).advise(SUMMARY, {})
    assert result.source == "fallback" and "429 rate limited" in result.error


def test_no_api_key_gives_labelled_fallback(tmp_path):
    result = make(tmp_path, None).advise(SUMMARY, {})
    assert result.source == "fallback" and result.error.startswith("no_api_key")


def test_api_key_never_appears_in_errors_cache_or_repr(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    leaky = make(tmp_path, FakeClient(error=RuntimeError(f"bad request, key={SECRET}")))
    result = leaky.advise(SUMMARY, {})
    assert SECRET not in (result.error or "") and "***" in result.error
    assert SECRET not in repr(leaky)

    ok = make(tmp_path, FakeClient())
    ok.advise(SUMMARY, {})
    for name in os.listdir(tmp_path / "cache"):
        assert SECRET not in (tmp_path / "cache" / name).read_text()


def test_expired_prices_are_refused(tmp_path):
    cfg = {"input_usd_per_1m": 1.0, "output_usd_per_1m": 2.0, "valid_through": "2000-01-01"}
    with pytest.raises(ValueError, match="expired"):
        load_config(write_config(tmp_path, prices=cfg))


def test_model_mismatch_with_environment_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "some-other-model")
    with pytest.raises(ValueError, match="LLM_MODEL"):
        load_config(write_config(tmp_path))


def test_config_missing_a_key_is_refused(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"model": "x"}))
    with pytest.raises(ValueError, match="missing"):
        load_config(str(path))


def test_fallback_advice_is_valid_and_ranks_alerting_zones():
    advice = fallback_advice(SUMMARY)
    assert [p.zone for p in advice.priorities] == ["restricted_zone_a_subnet"]
    assert advice.incident_report.kill_chain_stage == "Execution"
    quiet = json.loads(json.dumps(SUMMARY))
    quiet["total_alerting_hosts"] = 0
    quiet["agents"][0]["zones"][0]["hosts_alerting"] = 0
    assert fallback_advice(quiet).incident_report.kill_chain_stage == "None observed"


def test_summary_dataclass_is_accepted(tmp_path):
    summary = StateSummary(
        step=5, mission_phase="Preplanning", total_alerting_hosts=0,
        agents=[AgentSummary(agent="blue_agent_0", last_actions=["[Sleep]"],
                             zones=[ZoneSummary(zone="restricted_zone_a_subnet",
                                                hosts_present=14, hosts_alerting=0)])])
    good = json.loads(json.dumps(GOOD))
    result = make(tmp_path, FakeClient(text=json.dumps(good))).advise(summary, {})
    assert result.source == "gemini" and result.step == 5


def test_temporary_errors_are_retried_and_attempts_recorded(tmp_path):
    client = FakeClient(fail_first=2)  # 503, 503, then success; max_retries is 2
    result = make(tmp_path, client).advise(SUMMARY, {})
    assert client.calls == 3
    assert result.source == "gemini" and result.attempts == 3


def test_retries_are_bounded_and_end_in_a_labelled_fallback(tmp_path):
    client = FakeClient(fail_first=10)
    result = make(tmp_path, client).advise(SUMMARY, {})
    assert client.calls == 3  # 1 try + max_retries (2)
    assert result.source == "fallback" and result.attempts == 3 and "503" in result.error


def test_non_temporary_errors_are_not_retried(tmp_path):
    client = FakeClient(fail_first=10, fail_code=400)
    result = make(tmp_path, client).advise(SUMMARY, {})
    assert client.calls == 1 and result.source == "fallback" and result.attempts == 1


def test_negative_retry_settings_are_refused(tmp_path):
    with pytest.raises(ValueError, match="negative"):
        load_config(write_config(tmp_path, max_retries=-1))