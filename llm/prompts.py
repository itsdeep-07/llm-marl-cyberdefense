"""Prompts for the CC4 LLM advisor (Batch 5).

The prompt version is set in configs/llm_config.yaml and is part of the cache
key. If you change any text in this file, change prompt_version there too,
otherwise old cached answers would be reused for the new prompt. (The system
prompt text is also hashed into the cache key as a second safeguard.)
"""

import json

SYSTEM_PROMPT = """\
You are an advisor to five autonomous Blue defender agents in a simulated enterprise network (CAGE Challenge 4 / CybORG). You do not control them; you only give advice.

You receive a JSON telemetry snapshot. It contains the step, the mission phase, and for each Blue agent: the zones it defends, how many hosts exist in each zone, how many hosts show suspicious process or network-connection alerts, the names of alerting hosts, blocked subnets, and the agent's most recent actions (most recent first). You may also receive the action each agent proposes to take this step.

Your tasks:
1. priorities: score the urgency (0.0 to 1.0) of the zones that need attention, with a short reason.
2. conflicts: flag cases where two or more agents' proposed or recent actions overlap or work against each other. Return an empty list if there are none.
3. incident_report: a short summary, the most likely attack stage, and up to five concrete mitigation steps.

Rules:
- Use ONLY zone names and agent names that appear in the telemetry. Never invent hosts, zones, agents or tools.
- If there are no alerts, say so plainly; do not invent an attack.
- kill_chain_stage must be exactly one of: None observed, Reconnaissance, Initial Access, Execution, Lateral Movement, Impact.
- Keep every text field under 600 characters.
- Respond with JSON only, matching the requested schema.
"""


def build_user_prompt(summary: dict, proposals: dict) -> str:
    """Compact, deterministic prompt text for one telemetry snapshot."""
    telemetry = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    proposed = json.dumps(proposals, sort_keys=True, separators=(",", ":"))
    return (
        f"TELEMETRY (step {summary['step']}):\n{telemetry}\n\n"
        f"PROPOSED ACTIONS THIS STEP (by Blue agent):\n{proposed}\n\n"
        "Give your advice as JSON."
    )
