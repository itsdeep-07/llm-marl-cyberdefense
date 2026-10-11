"""Mechanics tests for MAPPOAgent, the training loop and the shared seed rules.

MOCK ENVIRONMENT -- NOT CYBORG. The environment below only mimics the interface
of BlueFlatWrapper (dict observations, action masks, 5-tuple step) so the
learning code can be checked quickly. Nothing produced here may appear in any
report, table or dashboard.

Run from the repo root:  python -m pytest tests/test_mappo.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.mappo import MAPPOAgent  # noqa: E402
from experiments.rl_common import plan_seeds  # noqa: E402
from experiments.train_mappo import train_one  # noqa: E402

N_AGENTS, OBS_DIM, ACT_DIM = 5, 8, 4


class MockBlueEnv:
    """Reward 1 for choosing the 'target' action shown in the observation."""

    def __init__(self, seed: int, steps: int) -> None:
        self.agents = [f"blue_agent_{i}" for i in range(N_AGENTS)]
        self.steps, self.t = steps, 0
        self.rng = np.random.default_rng(seed)

    def _make(self):
        obs, info, self.targets = {}, {}, {}
        for a in self.agents:
            mask = self.rng.random(ACT_DIM) < 0.8
            mask[0] = True
            target = int(self.rng.choice(np.flatnonzero(mask)))
            vec = np.zeros(OBS_DIM, dtype=np.float32)
            vec[target] = 1.0
            obs[a], info[a], self.targets[a] = vec, {"action_mask": mask}, target
        return obs, info

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        return self._make()

    def step(self, actions):
        rewards = {a: float(actions[a] == self.targets[a]) for a in self.agents}
        self.t += 1
        obs, info = self._make()
        done = self.t >= self.steps
        return (obs, rewards, {a: False for a in self.agents},
                {**{a: done for a in self.agents}, "__all__": done}, info)


def mock_factory(seed, steps):
    return MockBlueEnv(seed, steps)


def _random_batch(seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.random((N_AGENTS, OBS_DIM)).astype(np.float32)
    masks = rng.random((N_AGENTS, ACT_DIM)) < 0.5
    masks[:, 0] = True
    return obs, masks


def test_actions_respect_masks():
    agent = MAPPOAgent(N_AGENTS, OBS_DIM, ACT_DIM, seed=1)
    for s in range(200):
        obs, masks = _random_batch(s)
        actions, _, _ = agent.act(obs, masks)
        assert all(masks[i, actions[i]] for i in range(N_AGENTS))


def test_all_masked_agent_raises():
    agent = MAPPOAgent(N_AGENTS, OBS_DIM, ACT_DIM, seed=1)
    obs, masks = _random_batch()
    masks[2] = False
    with pytest.raises(ValueError):
        agent.act(obs, masks)


def test_critic_uses_the_joint_state():
    obs, _ = _random_batch()
    changed = obs.copy()
    changed[1] += 1.0  # change only agent 1's observation
    agent = MAPPOAgent(N_AGENTS, OBS_DIM, ACT_DIM, seed=3)
    # agent 0's value must react to agent 1's observation (centralized critic)
    assert not np.isclose(agent.value(obs)[0], agent.value(changed)[0])


def test_save_load_roundtrip(tmp_path):
    a = MAPPOAgent(N_AGENTS, OBS_DIM, ACT_DIM, seed=1)
    b = MAPPOAgent(N_AGENTS, OBS_DIM, ACT_DIM, seed=2)
    path = str(tmp_path / "m.pt")
    a.save(path)
    b.load(path)
    obs, _ = _random_batch()
    assert np.allclose(a.value(obs), b.value(obs))
    with pytest.raises(ValueError):
        MAPPOAgent(N_AGENTS, OBS_DIM + 1, ACT_DIM, seed=1).load(path)


def test_seed_rules():
    n, train, ev = plan_seeds(42, 400, 100, 5, 9_000_000)
    assert n == 4 and not set(train) & set(ev)
    with pytest.raises(ValueError):
        plan_seeds(42, 450, 100, 5, 9_000_000)          # not a whole number of episodes
    with pytest.raises(ValueError):
        plan_seeds(42, 400, 100, 5, 42 * 100_000)       # evaluation overlaps training


def test_training_loop_learns_on_mock_env(tmp_path):
    hyper = {"lr_actor": 3e-3, "lr_critic": 3e-3, "reward_scale": 1.0}
    summary = train_one(7, total_steps=6000, steps=20, eval_episodes=10,
                        eval_seed_base=9_000_000, out_dir=str(tmp_path),
                        env_factory=mock_factory, hyper=hyper)
    per_step_per_agent = summary["mean_reward"] / (20 * N_AGENTS)
    assert per_step_per_agent > 0.6, per_step_per_agent  # random valid actions score ~0.3-0.4
    for name in ("training/mappo_seed7_train.csv", "tables/rl_eval_mappo_seed7.csv",
                 "tables/rl_eval_episodes_mappo_seed7.csv", "checkpoints/mappo_seed7.pt"):
        assert os.path.exists(os.path.join(tmp_path, name)), name
    assert summary["train_env_steps"] == 6000