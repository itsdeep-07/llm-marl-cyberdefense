SYSTEM_ORCHESTRATOR_PROMPT = """
You are the Strategic Cyber Defense LLM Orchestrator operating above multiple decentralized autonomous Blue RL agents in a simulated enterprise network (CybORG / CAGE Challenge).
Your role:
1. Prioritize which subnet (User, Enterprise, Operational) urgently requires defensive compute and resources.
2. Flag conflicts across Blue agents (e.g. redundant restores, resource contention, cross-subnet miscoordination).
3. Generate a concise, highly professional SOC Executive Incident Report detailing attacker kill-chain progression and recommended remediations.

Output must strictly adhere to the requested JSON structure. Do NOT hallucinate unobserved hosts or imaginary tools.
"""

def generate_orchestrator_prompt(step: int, host_status: dict, blue_proposals: dict, last_red_action: str = "Unknown") -> str:
    compromised_hosts = [h for h, s in host_status.items() if s == "Compromised"]
    scanned_hosts = [h for h, s in host_status.items() if s == "Scanned"]
    secure_hosts = [h for h, s in host_status.items() if s == "Secure"]
    restored_hosts = [h for h, s in host_status.items() if s == "Restored"]
    
    prompt = f"""
[TACTICAL TELEMETRY SNAPSHOT - STEP {step}]
- Active Compromises ({len(compromised_hosts)}): {compromised_hosts if compromised_hosts else "None"}
- Suspected / Scanned ({len(scanned_hosts)}): {scanned_hosts if scanned_hosts else "None"}
- Remediated / Restored ({len(restored_hosts)}): {restored_hosts if restored_hosts else "None"}
- Healthy Hosts ({len(secure_hosts)}): {secure_hosts}
- Last Detected Red Activity: {last_red_action}

[PROPOSED BLUE ACTIONS BY DECENTRALIZED RL DEFENDERS]
{blue_proposals}

Please analyze this tactical state, resolve any inter-agent conflicts, score subnet priorities, and generate an executive incident report.
"""
    return prompt.strip()
