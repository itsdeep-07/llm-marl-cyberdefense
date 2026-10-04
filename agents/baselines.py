"""Baseline policies for the real CAGE Challenge 4 wrapper."""

from __future__ import annotations

import random
from typing import Iterable, Sequence

import numpy as np


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
    ) -> int:
        del agent, observation
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
    ) -> int:
        del agent, observation, labels
        valid = _valid_indices(mask)
        if not valid:
            raise RuntimeError("CC4 returned an action space with no valid action")
        return self._random.choice(valid)


class RuleBasedPolicy(SleepPolicy):
    """Progress through Analyse, Remove, and Restore after a process alert."""

    _MALICIOUS_PROCESS_START = 3 + (2 * 18)
    _MALICIOUS_PROCESS_SIZE = 2 * 16

    def __init__(self) -> None:
        self._stages: dict[str, int] = {}

    def reset(self) -> None:
        self._stages.clear()

    def action(
        self,
        agent: str,
        observation: np.ndarray,
        labels: Sequence[str],
        mask: Sequence[bool],
    ) -> int:
        process_end = self._MALICIOUS_PROCESS_START + self._MALICIOUS_PROCESS_SIZE
        alert = bool(np.any(observation[self._MALICIOUS_PROCESS_START:process_end]))
        stage = self._stages.get(agent, 0)
        if alert or stage:
            action = _first_action(
                labels,
                mask,
                ("Analyse", "Remove", "Restore")[stage:],
            )
            if action is not None:
                self._stages[agent] = min(stage + 1, 2)
                return action
        self._stages.pop(agent, None)
        return super().action(agent, observation, labels, mask)
