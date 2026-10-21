"""Tests for the live controller using a small interface-compatible fake env."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.baselines import SleepPolicy  # noqa: E402
from live_demo import LiveRun  # noqa: E402


class FakeCybORG:
    def get_ip_map(self):
        return {"alpha_subnet_host": "10.0.0.1"}

    def get_agent_state(self, _agent):
        return {"alpha_subnet_host": {"Sessions": []}}

    def get_last_action(self, agent):
        return f"{agent}: DISCOVER"


class FakeEnv:
    agents = ["blue_agent_0"]

    def __init__(self):
        self.env = FakeCybORG()
        self.t = 0

    def action_labels(self, _agent):
        return ["Sleep"]

    def reset(self, seed=None):
        del seed
        self.t = 0
        return {"blue_agent_0": np.zeros(2)}, {
            "blue_agent_0": {"action_mask": np.array([True])}
        }

    def step(self, actions):
        assert actions == {"blue_agent_0": 0}
        self.t += 1
        done = self.t == 2
        return (
            {"blue_agent_0": np.ones(2)},
            {"blue_agent_0": -1.0},
            {"blue_agent_0": done},
            {"blue_agent_0": done, "__all__": done},
            {"blue_agent_0": {"action_mask": np.array([True])}},
        )

    def get_last_action(self, agent):
        return f"{agent}: Sleep"


def test_live_run_advances_and_records_provenance():
    env = FakeEnv()
    run = LiveRun(
        seed=7,
        steps=2,
        env=env,
        observation=env.reset()[0],
        info=env.reset()[1],
        red_agents=["red_agent_0"],
    )
    run._sleep_policy = SleepPolicy()
    run._action_fn = run._sleep_actions

    first = run.step_once()
    second = run.step_once()

    assert first["step"] == 0
    assert second["episode_done"] is True
    assert second["cumulative_reward"] == -2.0
    assert second["provenance"]["source"] == "live_demo"


def test_live_run_rejects_unverified_policy():
    with pytest.raises(ValueError, match="verified checkpoint"):
        LiveRun.create(seed=1, steps=2, blue_policy="IPPO")
