"""Baseline policies for the real CAGE Challenge 4 wrapper."""

from __future__ import annotations

import random
from typing import Iterable, Sequence

import numpy as np


# CC4 README, "Appendix B - Agent observation": S=9 subnets, H=16 hosts,
# with 3S prefix fields followed by H malicious-process fields per subnet.
CC4_SUBNET_COUNT = 9
CC4_MAX_HOSTS_PER_SUBNET = 16
CC4_MESSAGE_COUNT = 4
CC4_MESSAGE_BITS = 8
CC4_SUBNET_PREFIX_LENGTH = 3 * CC4_SUBNET_COUNT
CC4_SUBNET_BLOCK_LENGTH = (
    CC4_SUBNET_PREFIX_LENGTH + 2 * CC4_MAX_HOSTS_PER_SUBNET
)
CC4_MESSAGE_BLOCK_LENGTH = CC4_MESSAGE_COUNT * CC4_MESSAGE_BITS
CC4_MISSION_PHASE_INDEX = 0


def _valid_indices(mask: Sequence[bool]) -> list[int]:
    return [index for index, valid in enumerate(mask) if valid]


def _first_action(
    labels: Sequence[str],
    mask: Sequence[bool],
    action_names: Iterable[str],
) -> int | None:
    valid = set(_valid_indices(mask))
    for name in action_names:
        for index, label in enumerate(labels):
            if index in valid and label.split(maxsplit=1)[0] == name:
                return index
    return None


def _action_for_host(
    labels: Sequence[str],
    mask: Sequence[bool],
    action_name: str,
    host: str,
) -> int | None:
    valid = set(_valid_indices(mask))
    prefix = f"{action_name} {host}"
    for index, label in enumerate(labels):
        if index in valid and label.startswith(prefix):
            return index
    return None


def _alerted_hosts(
    observation: np.ndarray,
    hosts: Sequence[str],
    subnets: Sequence[str],
) -> list[str]:
    """Return hosts with process alerts using the Appendix B vector layout."""
    alerted: list[str] = []
    for subnet_index, subnet in enumerate(subnets):
        subnet_hosts = [
            host for host in hosts if subnet in host and "router" not in host
        ]
        process_start = (
            1
            + subnet_index * CC4_SUBNET_BLOCK_LENGTH
            + CC4_SUBNET_PREFIX_LENGTH
        )
        process_end = process_start + len(subnet_hosts)
        process_events = observation[process_start:process_end]
        alerted.extend(
            host
            for host, has_alert in zip(subnet_hosts, process_events)
            if bool(has_alert)
        )
    return alerted


class SleepPolicy:
    """Select the real Sleep action from each agent's dynamic action space."""

    def reset(self) -> None:
        pass

    def action(
        self,
        agent: str,
        observation: np.ndarray,
        labels: Sequence[str],
        mask: Sequence[bool],
        hosts: Sequence[str] = (),
        subnets: Sequence[str] = (),
    ) -> int:
        del agent, observation, hosts, subnets
        sleep = _first_action(labels, mask, ("Sleep",))
        if sleep is not None:
            return sleep
        valid = _valid_indices(mask)
        if not valid:
            raise RuntimeError("CC4 returned an action space with no valid action")
        return valid[0]


class RandomPolicy(SleepPolicy):
    """Sample uniformly from the currently valid actions."""

    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed)

    def action(
        self,
        agent: str,
        observation: np.ndarray,
        labels: Sequence[str],
        mask: Sequence[bool],
        hosts: Sequence[str] = (),
        subnets: Sequence[str] = (),
    ) -> int:
        del agent, observation, labels, hosts, subnets
        valid = _valid_indices(mask)
        if not valid:
            raise RuntimeError("CC4 returned an action space with no valid action")
        return self._random.choice(valid)


class RuleBasedPolicy(SleepPolicy):
    """Progress through Analyse, Remove, and Restore after a process alert."""

    def __init__(self) -> None:
        self._stages: dict[str, tuple[int, str]] = {}

    def reset(self) -> None:
        self._stages.clear()

    def action(
        self,
        agent: str,
        observation: np.ndarray,
        labels: Sequence[str],
        mask: Sequence[bool],
        hosts: Sequence[str] = (),
        subnets: Sequence[str] = (),
    ) -> int:
        alerted_hosts = _alerted_hosts(observation, hosts, subnets)
        stage, target = self._stages.get(agent, (0, ""))
        if target not in alerted_hosts:
            stage = 0
            target = alerted_hosts[0] if alerted_hosts else ""
        if target:
            action_name = ("Analyse", "Remove", "Restore")[stage]
            action = _action_for_host(labels, mask, action_name, target)
            if action is not None:
                if action_name == "Restore":
                    self._stages.pop(agent, None)
                else:
                    self._stages[agent] = (stage + 1, target)
                return action
        self._stages.pop(agent, None)
        return super().action(agent, observation, labels, mask)
