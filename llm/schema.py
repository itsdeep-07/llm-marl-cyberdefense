from typing import List, Dict
from pydantic import BaseModel, Field

class SubnetPriority(BaseModel):
    subnet: str = Field(description="Name of the subnet (User, Enterprise, Operational)")
    priority_score: float = Field(ge=0.0, le=1.0, description="Priority urgency score between 0.0 and 1.0")
    reasoning: str = Field(description="Operational rationale for priority score")

class DetectedConflict(BaseModel):
    agents_involved: List[str] = Field(description="Names of conflicting Blue agents")
    conflict_type: str = Field(description="Type of conflict (e.g. Bandwidth Contention, Redundant Action, Conflicting Restore)")
    resolution: str = Field(description="Strategic resolution prescribed by orchestrator")

class IncidentReport(BaseModel):
    executive_summary: str = Field(description="High-level incident summary for CISO / SOC lead")
    kill_chain_stage: str = Field(description="Current MITRE ATT&CK / Cyber Kill Chain phase (Reconnaissance, Initial Access, Lateral Movement, Impact)")
    mitigation_actions: List[str] = Field(description="Recommended tactical steps for Blue defenders")

class OrchestratorOutput(BaseModel):
    step: int = Field(description="Simulation step index")
    priorities: List[SubnetPriority] = Field(description="Priority ranking for each subnet")
    conflicts: List[DetectedConflict] = Field(description="Identified cross-subnet action conflicts")
    incident_report: IncidentReport = Field(description="Automated SOC incident report")
    latency_ms: float = Field(default=0.0, description="LLM inference latency in milliseconds")
    token_cost_usd: float = Field(default=0.0, description="Estimated API token cost in USD")
