"""LLM advisor for CC4 (Batch 5).

What this module guarantees:
* Every result is labelled with where it came from: "gemini" (fresh API answer),
  "cache" (a stored earlier answer) or "fallback" (simple rule-based advice used
  because the API was unavailable or its answer was invalid).
* Latency is the measured wall time of the API call. Cost is computed from the
  token counts the API reports and the prices in configs/llm_config.yaml.
  Nothing is faked or defaulted; if token usage is missing the answer is rejected.
* The model's JSON is validated through llm/schema.py, and advice that mentions
  zones or agents that are not in the telemetry is rejected.
* Errors are never swallowed: they are recorded in the result's `error` field
  (or raised, when raise_on_error=True). The API key is never logged, stored in
  results, or written to the cache.
* The cache key is a SHA-256 over the model name, prompt version, system prompt,
  generation settings and the exact telemetry and proposals.

Only results with source == "gemini" may be used in model-performance claims.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import time

import yaml
from dotenv import load_dotenv

from llm.prompts import SYSTEM_PROMPT, build_user_prompt
from llm.schema import (
    AdvisoryResult,
    DetectedConflict,
    IncidentReport,
    LLMAdvice,
    ZonePriority,
    validate_against_summary,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "llm_config.yaml")
DEFAULT_CACHE_DIR = os.path.join(REPO_ROOT, ".llm_cache")
DEFAULT_ENV = os.path.join(REPO_ROOT, ".env")
PLACEHOLDER_KEY = "your_gemini_api_key_here"
RETRYABLE_CODES = {429, 500, 503, 504}  # temporary server-side problems only


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def make_cache_key(model: str, prompt_version: str, system_prompt: str, settings: dict,
                   payload: dict) -> str:
    blob = canonical_json({
        "model": model,
        "prompt_version": prompt_version,
        "system_prompt": system_prompt,
        "settings": settings,
        "payload": payload,
    })
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    for key in ("model", "prompt_version", "temperature", "thinking_level",
                "max_output_tokens", "max_retries", "retry_base_seconds", "prices"):
        if key not in cfg:
            raise ValueError(f"{path} is missing required key '{key}'")
    for key in ("input_usd_per_1m", "output_usd_per_1m", "valid_through"):
        if key not in cfg["prices"]:
            raise ValueError(f"{path} is missing prices.{key}")
    if int(cfg["max_retries"]) < 0 or float(cfg["retry_base_seconds"]) < 0:
        raise ValueError("max_retries and retry_base_seconds must not be negative")
    valid_through = datetime.date.fromisoformat(str(cfg["prices"]["valid_through"]))
    if datetime.date.today() > valid_through:
        raise ValueError(
            f"Configured prices expired on {valid_through}. Check the current prices at "
            "https://ai.google.dev/gemini-api/docs/pricing and update configs/llm_config.yaml."
        )
    env_model = os.getenv("LLM_MODEL")
    if env_model and env_model != cfg["model"]:
        raise ValueError(
            f"LLM_MODEL in the environment is '{env_model}' but configs/llm_config.yaml "
            f"has prices for '{cfg['model']}'. Make them match."
        )
    return cfg


def fallback_advice(summary: dict) -> LLMAdvice:
    """Simple rule-based advice from the telemetry alone. Not model output."""
    priorities = []
    for agent in summary["agents"]:
        for zone in agent["zones"]:
            alerting, present = zone["hosts_alerting"], zone["hosts_present"]
            if alerting == 0:
                continue
            score = min(1.0, round(alerting / max(1, min(present, 3)), 2))
            priorities.append(ZonePriority(
                zone=zone["zone"], priority_score=score,
                reasoning=f"{alerting} of {present} hosts show alerts (rule-based fallback).",
            ))
    priorities.sort(key=lambda p: (-p.priority_score, p.zone))
    total = summary["total_alerting_hosts"]
    if total == 0:
        stage = "None observed"
        text = "No alerting hosts in the telemetry (rule-based fallback, not model output)."
        actions = []
    else:
        stage = "Execution"
        text = (f"{total} host(s) show suspicious process or connection alerts "
                "(rule-based fallback, not model output).")
        actions = [f"Investigate the alerting hosts in {priorities[0].zone}."]
    return LLMAdvice(
        priorities=priorities,
        conflicts=[],
        incident_report=IncidentReport(
            executive_summary=text, kill_chain_stage=stage, mitigation_actions=actions),
    )


class LLMOrchestrator:
    def __init__(self, config_path: str = DEFAULT_CONFIG, cache_dir: str = DEFAULT_CACHE_DIR,
                 client=None, cache_enabled: bool = True, raise_on_error: bool = False,
                 env_path: str | None = DEFAULT_ENV) -> None:
        if env_path:
            load_dotenv(env_path)
        self.cfg = load_config(config_path)
        self.model = self.cfg["model"]
        self.prompt_version = self.cfg["prompt_version"]
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_enabled = cache_enabled
        self.raise_on_error = raise_on_error

        # The key is read only to build the client and to scrub it out of error text.
        self._last_attempts = 0
        self._api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if client is not None:
            self._client = client
        elif self._api_key and self._api_key != PLACEHOLDER_KEY:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        else:
            self._client = None

    def __repr__(self) -> str:  # never shows the key
        return f"LLMOrchestrator(model={self.model!r}, live={self._client is not None})"

    # ---- public API -----------------------------------------------------
    def advise(self, summary, proposals: dict | None = None,
               force_live: bool = False) -> AdvisoryResult:
        started = time.perf_counter()
        self._last_attempts = 0
        summary_dict = summary.to_dict() if hasattr(summary, "to_dict") else summary
        proposals = proposals or {}
        settings = {k: self.cfg[k] for k in ("temperature", "thinking_level", "max_output_tokens")}
        key = make_cache_key(self.model, self.prompt_version, SYSTEM_PROMPT, settings,
                             {"summary": summary_dict, "proposals": proposals})

        if self.cache_enabled and not force_live:
            entry = self._read_cache(key, summary_dict)
            if entry is not None:
                return AdvisoryResult(
                    step=summary_dict["step"], source="cache",
                    advice=LLMAdvice.model_validate(entry["advice"]),
                    model=self.model, prompt_version=self.prompt_version, cache_key=key,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    total_wall_ms=(time.perf_counter() - started) * 1000.0, attempts=0,
                    input_tokens=0, output_tokens=0, cost_usd=0.0,
                    origin_latency_ms=entry["latency_ms"], origin_cost_usd=entry["cost_usd"],
                )

        if self._client is None:
            return self._fallback(summary_dict, key, started, "no_api_key",
                                  RuntimeError("no GEMINI_API_KEY configured"))

        try:
            return self._call_api(summary_dict, proposals, key, started)
        except Exception as exc:  # recorded in the result, or re-raised; never silent
            if self.raise_on_error:
                raise
            return self._fallback(summary_dict, key, started, type(exc).__name__, exc)

    # ---- internals ------------------------------------------------------
    def _call_api(self, summary: dict, proposals: dict, key: str,
                  started: float) -> AdvisoryResult:
        config = {
            "system_instruction": SYSTEM_PROMPT,
            "response_mime_type": "application/json",
            "response_schema": LLMAdvice,
        }
        if self.cfg["temperature"] is not None:
            config["temperature"] = self.cfg["temperature"]
        if self.cfg["thinking_level"] is not None:
            config["thinking_config"] = {"thinking_level": self.cfg["thinking_level"]}
        if self.cfg["max_output_tokens"] is not None:
            config["max_output_tokens"] = self.cfg["max_output_tokens"]

        prompt = build_user_prompt(summary, proposals)
        attempt = 0
        while True:
            attempt += 1
            self._last_attempts = attempt
            call_started = time.perf_counter()
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config)
                break
            except Exception as exc:
                temporary = getattr(exc, "code", None) in RETRYABLE_CODES
                if not temporary or attempt > int(self.cfg["max_retries"]):
                    raise
                time.sleep(float(self.cfg["retry_base_seconds"]) * 2 ** (attempt - 1))
        latency_ms = (time.perf_counter() - call_started) * 1000.0

        advice = LLMAdvice.model_validate(json.loads(response.text))
        validate_against_summary(advice, summary)

        usage = response.usage_metadata
        if usage is None or usage.prompt_token_count is None:
            raise ValueError("response has no token usage; cannot compute cost")
        input_tokens = int(usage.prompt_token_count)
        output_tokens = int((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0))
        prices = self.cfg["prices"]
        cost = (input_tokens * prices["input_usd_per_1m"]
                + output_tokens * prices["output_usd_per_1m"]) / 1_000_000.0

        result = AdvisoryResult(
            step=summary["step"], source="gemini", advice=advice, model=self.model,
            prompt_version=self.prompt_version, cache_key=key, latency_ms=latency_ms,
            total_wall_ms=(time.perf_counter() - started) * 1000.0, attempts=attempt,
            input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost,
        )
        if self.cache_enabled:
            self._write_cache(result)
        return result

    def _fallback(self, summary: dict, key: str, started: float, label: str,
                  exc: Exception) -> AdvisoryResult:
        return AdvisoryResult(
            step=summary["step"], source="fallback", advice=fallback_advice(summary),
            model=self.model, prompt_version=self.prompt_version, cache_key=key,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            total_wall_ms=(time.perf_counter() - started) * 1000.0,
            attempts=self._last_attempts,
            input_tokens=0, output_tokens=0, cost_usd=0.0,
            error=f"{label}: {self._scrub(str(exc))}",
        )

    def _scrub(self, text: str) -> str:
        if self._api_key:
            text = text.replace(self._api_key, "***")
        return text[:300]

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def _write_cache(self, result: AdvisoryResult) -> None:
        entry = {
            "cache_key": result.cache_key, "model": result.model,
            "prompt_version": result.prompt_version,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "advice": result.advice.model_dump(),
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms, "cost_usd": result.cost_usd,
        }
        path = self._cache_path(result.cache_key)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(entry, f, indent=2)
        os.replace(tmp, path)

    def _read_cache(self, key: str, summary: dict) -> dict | None:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                entry = json.load(f)
            if (entry["cache_key"], entry["model"], entry["prompt_version"]) != (
                    key, self.model, self.prompt_version):
                raise ValueError("entry does not match its key")
            validate_against_summary(LLMAdvice.model_validate(entry["advice"]), summary)
            entry["latency_ms"], entry["cost_usd"]  # must exist
        except Exception as exc:
            raise ValueError(f"corrupt cache entry {path}: {exc}") from exc
        return entry