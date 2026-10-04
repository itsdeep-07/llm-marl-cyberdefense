"""CAGE Challenge 4 environment construction."""

from __future__ import annotations

from typing import Type

from CybORG import CybORG
from CybORG.Agents import (
    EnterpriseGreenAgent,
    FiniteStateRedAgent,
    SleepAgent,
)
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator


RED_AGENTS: dict[str, Type[BaseAgent]] = {
    "FiniteStateRedAgent": FiniteStateRedAgent,
}


def make_env(
    seed: int,
    steps: int,
    red: str = "FiniteStateRedAgent",
) -> BlueFlatWrapper:
    """Create a real CC4 environment with fixed-size observations and masks."""
    try:
        red_agent_class = RED_AGENTS[red]
    except KeyError as exc:
        supported = ", ".join(sorted(RED_AGENTS))
        raise ValueError(f"Unsupported CC4 red agent {red!r}; choose {supported}") from exc

    scenario_generator = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=red_agent_class,
        steps=steps,
    )
    cyborg = CybORG(scenario_generator=scenario_generator, seed=seed)
    return BlueFlatWrapper(cyborg, pad_spaces=True)
