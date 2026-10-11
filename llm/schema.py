"""Schemas for the LLM layer (Batch 5).

Two kinds of models live here:

* LLMAdvice and its parts: the ONLY thing Gemini is asked to produce. Every
  model response is validated through these classes. The structure is kept
  simple on purpose; value checks (ranges, lengths, allowed words) are done by
  validators here, not by constraints inside the schema sent to the API.
* AdvisoryResult: what the orchestrator returns. It wraps the advice with
  measured facts (source, latency, tokens, cost) that come from our own code and
  are never produced by the model.

Important: only results with source == "gemini" may be used in claims about
model performance. "cache" and "fallback" results are labelled so they can be
excluded.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, field_validator

KILL_CHAIN_STAGES = [
    "None observed",
    "Reconnaissance",
    "Initial Access",
    "Execution",
    "Lateral Movement",
    "Impact",
]
MAX_TEXT = 600
MAX_ACTIONS = 5


def _check_text(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    if len(value) > MAX_TEXT:
        raise ValueError(f"{name} is longer than {MAX_TEXT} characters")
    return value


class ZonePriority(BaseModel):
    zone: str            # must be one of the zones in the telemetry (checked separately)
    priority_score: float  # 0.0 (ignore) to 1.0 (most urgent)
    reasoning: str

    @field_validator("priority_score")
    @classmethod
    def _score_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("priority_score must be between 0.0 and 1.0")
        return v

    @field_validator("reasoning")
    @classmethod
    def _reasoning(cls, v: str) -> str:
        return _check_text(v, "reasoning")


class DetectedConflict(BaseModel):
    agents_involved: List[str]  # Blue agent names from the telemetry
    conflict_type: str
    resolution: str

    @field_validator("agents_involved")
    @classmethod
    def _agents(cls, v: List[str]) -> List[str]:
        if len(v) < 2:
            raise ValueError("a conflict needs at least two agents")
        return v

    @field_validator("conflict_type", "resolution")
    @classmethod
    def _texts(cls, v: str) -> str:
        return _check_text(v, "conflict text")


class IncidentReport(BaseModel):
    executive_summary: str
    kill_chain_stage: str  # one of KILL_CHAIN_STAGES
    mitigation_actions: List[str]

    @field_validator("executive_summary")
    @classmethod
    def _summary(cls, v: str) -> str:
        return _check_text(v, "executive_summary")

    @field_validator("kill_chain_stage")
    @classmethod
    def _stage(cls, v: str) -> str:
        if v not in KILL_CHAIN_STAGES:
            raise ValueError(f"kill_chain_stage must be one of {KILL_CHAIN_STAGES}")
        return v

    @field_validator("mitigation_actions")
    @classmethod
    def _actions(cls, v: List[str]) -> List[str]:
        if len(v) > MAX_ACTIONS:
            raise ValueError(f"at most {MAX_ACTIONS} mitigation actions")
        for item in v:
            _check_text(item, "mitigation action")
        return v


class LLMAdvice(BaseModel):
    priorities: List[ZonePriority]
    conflicts: List[DetectedConflict]
    incident_report: IncidentReport


def validate_against_summary(advice: LLMAdvice, summary: dict) -> None:
    """Reject advice that mentions zones or agents that are not in the telemetry."""
    zones = {z["zone"] for agent in summary["agents"] for z in agent["zones"]}
    agents = {agent["agent"] for agent in summary["agents"]}
    bad_zones = sorted({p.zone for p in advice.priorities} - zones)
    bad_agents = sorted({a for c in advice.conflicts for a in c.agents_involved} - agents)
    if bad_zones or bad_agents:
        raise ValueError(f"advice mentions unknown zones {bad_zones} or agents {bad_agents}")


class AdvisoryResult(BaseModel):
    step: int
    source: Literal["gemini", "cache", "fallback"]
    advice: LLMAdvice
    model: str
    prompt_version: str
    cache_key: str
    latency_ms: float          # wall time of the successful API call (cache: the lookup time)
    total_wall_ms: float       # whole advise() call, including failed attempts and retry waits
    attempts: int              # API attempts made (0 for cache hits and for no-key fallbacks)
    input_tokens: int          # tokens billed for THIS call (0 for cache and fallback)
    output_tokens: int         # includes thinking tokens, as billed
    cost_usd: float            # cost of THIS call at the configured prices (0 for cache/fallback)
    origin_latency_ms: Optional[float] = None  # cache hits: wall time of the original API call
    origin_cost_usd: Optional[float] = None    # cache hits: cost of the original API call
    error: Optional[str] = None                # why a fallback happened; never contains the API key

    @property
    def counts_for_performance(self) -> bool:
        """Only fresh Gemini answers may be used in model-performance claims."""
        return self.source == "gemini"