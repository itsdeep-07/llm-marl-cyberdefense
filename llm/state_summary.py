"""CC4 state summarizer for the LLM layer (Batch 5).

Reads the same simulator state that CC4's BlueFlatWrapper uses to build each
Blue agent's observation (mission phase, each agent's hosts and subnets,
blocked subnets, and per-host suspicious process / network-connection events),
so the LLM is told only what the RL defenders can actually see.

Nothing here hides errors: if the simulator state does not have the expected
shape, the exception propagates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

# Same order as CC4's BlueFlatWrapper._build_comms_policy
MISSION_PHASES = ["Preplanning", "MissionA", "MissionB"]
MAX_ALERT_HOSTS_PER_ZONE = 8


@dataclass
class ZoneSummary:
    zone: str                    # subnet name, e.g. restricted_zone_a_subnet
    hosts_present: int           # non-router hosts that exist in this episode
    hosts_alerting: int          # hosts with suspicious process or connection events
    alert_hosts: list = field(default_factory=list)       # "host:process+connection", capped
    blocked_subnets: list = field(default_factory=list)   # as reported by the simulator


@dataclass
class AgentSummary:
    agent: str
    zones: list = field(default_factory=list)
    last_actions: list = field(default_factory=list)      # most recent first


@dataclass
class StateSummary:
    step: int
    mission_phase: str
    total_alerting_hosts: int
    agents: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _alert_kinds(state, hostname: str) -> list:
    """Which alert kinds a host shows, using the same events as BlueFlatWrapper."""
    if hostname not in state.hosts:
        return []
    events = state.hosts[hostname].events
    kinds = []
    if len(events.old_process_creation + events.process_creation) > 0:
        kinds.append("process")
    if len(events.old_network_connections + events.network_connections) > 0:
        kinds.append("connection")
    return kinds


class StateSummarizer:
    """Builds a StateSummary from a live CC4 environment (the BlueFlatWrapper object)."""

    def __init__(self, history_len: int = 3) -> None:
        self.history_len = history_len
        self.history: dict = {}

    def reset(self) -> None:
        self.history = {}

    def record(self, env) -> None:
        """Call once after every env.step() to remember each Blue agent's latest action."""
        for agent in env.possible_agents:
            queue = self.history.setdefault(agent, deque(maxlen=self.history_len))
            queue.appendleft(str(env.get_last_action(agent)))

    def summarize(self, env, step: int) -> StateSummary:
        state = env.env.environment_controller.state
        agents, total_alerting = [], 0

        for agent in sorted(env.possible_agents):
            hosts = env.hosts(agent)
            zones = []
            for subnet in env.subnets(agent):
                zone_hosts = [h for h in hosts if subnet in h and "router" not in h]
                present = [h for h in zone_hosts if h in state.hosts]
                alerting = []
                for host in sorted(present):
                    kinds = _alert_kinds(state, host)
                    if kinds:
                        alerting.append(f"{host}:{'+'.join(kinds)}")
                total_alerting += len(alerting)
                zones.append(ZoneSummary(
                    zone=subnet,
                    hosts_present=len(present),
                    hosts_alerting=len(alerting),
                    alert_hosts=alerting[:MAX_ALERT_HOSTS_PER_ZONE],
                    blocked_subnets=sorted(state.blocks.get(subnet, [])),
                ))
            agents.append(AgentSummary(
                agent=agent,
                zones=zones,
                last_actions=list(self.history.get(agent, [])),
            ))

        return StateSummary(
            step=step,
            mission_phase=MISSION_PHASES[state.mission_phase],
            total_alerting_hosts=total_alerting,
            agents=agents,
        )
