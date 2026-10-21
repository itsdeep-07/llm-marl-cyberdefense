"""In-process live CC4 demonstration controller.

The controller keeps simulator state outside the Streamlit rendering code so
the dashboard can advance one verified environment transition per rerun.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agents.baselines import SleepPolicy
from experiments.rl_common import team_reward


ActionFn = Callable[[dict, dict], dict[str, int]]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _host_snapshot(env: Any, red_agents: list[str]) -> dict[str, str]:
    """Read host state from the documented CC4 debugging interface."""
    cyborg = env.env
    hosts = {name: "no_red_session" for name in cyborg.get_ip_map()}
    for red in red_agents:
        for hostname, data in cyborg.get_agent_state(red).items():
            if hostname == "success":
                continue
            if hostname not in hosts:
                raise RuntimeError(f"CC4 returned an unknown host {hostname!r}")
            if any(
                str(session.get("agent", "")).startswith("red_")
                for session in data.get("Sessions", [])
            ):
                hosts[hostname] = "red_session"
    return hosts


def _first_value(mapping: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _mission_phase(infos: dict) -> str:
    for info in infos.values():
        if not isinstance(info, dict):
            continue
        value = _first_value(info, ("mission_phase", "phase"))
        if value is not None:
            return str(value)
    return "Unavailable"


@dataclass
class LiveRun:
    """Mutable state for one independent live episode."""

    seed: int
    steps: int
    red_strategy: str = "FiniteStateRedAgent"
    blue_policy: str = "Sleep"
    checkpoint: str | None = None
    env: Any = field(default=None, repr=False)
    observation: dict = field(default_factory=dict, repr=False)
    info: dict = field(default_factory=dict, repr=False)
    red_agents: list[str] = field(default_factory=list)
    current_step: int = -1
    episode_done: bool = False
    cumulative_reward: float = 0.0
    frames: list[dict] = field(default_factory=list)
    error_state: str | None = None
    _action_fn: ActionFn | None = field(default=None, repr=False)
    _sleep_policy: SleepPolicy | None = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        seed: int,
        steps: int,
        red_strategy: str = "FiniteStateRedAgent",
        blue_policy: str = "Sleep",
        checkpoint: str | None = None,
    ) -> "LiveRun":
        if steps < 1:
            raise ValueError("live demo episode length must be at least one step")
        if blue_policy not in {"Sleep", "IPPO", "MAPPO"}:
            raise ValueError(f"unsupported live Blue policy {blue_policy!r}")
        if blue_policy != "Sleep" and not checkpoint:
            raise ValueError(
                f"{blue_policy} live execution requires a verified checkpoint path"
            )
        if checkpoint and not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"checkpoint for {blue_policy} was not found: {checkpoint}"
            )

        from envs.cc4_env import make_env

        env = make_env(seed=seed, steps=steps, red=red_strategy)
        observation, info = env.reset(seed=seed)
        run = cls(
            seed=seed,
            steps=steps,
            red_strategy=red_strategy,
            blue_policy=blue_policy,
            checkpoint=checkpoint,
            env=env,
            observation=observation,
            info=info,
            red_agents=sorted(
                name
                for name in env.env.environment_controller.agent_interfaces
                if name.startswith("red_")
            ),
            _sleep_policy=SleepPolicy(),
        )
        if blue_policy == "Sleep":
            run._action_fn = run._sleep_actions
        else:
            from experiments.rollout import load_rl_policy

            run._action_fn = load_rl_policy(
                blue_policy, str(checkpoint), seed, env, steps
            )
            run.observation, run.info = env.reset(seed=seed)
        return run

    def _sleep_actions(self, observation: dict, info: dict) -> dict[str, int]:
        assert self._sleep_policy is not None
        return {
            agent: self._sleep_policy.action(
                agent,
                observation[agent],
                self.env.action_labels(agent),
                info[agent]["action_mask"],
            )
            for agent in self.env.agents
        }

    def step_once(self) -> dict:
        if self.episode_done:
            return self.frames[-1] if self.frames else self.frame()
        if self._action_fn is None:
            raise RuntimeError("live policy is not initialized")

        actions = self._action_fn(self.observation, self.info)
        next_observation, rewards, terminated, truncated, next_info = self.env.step(
            actions
        )
        reward = team_reward(rewards)
        self.current_step += 1
        self.cumulative_reward += reward
        self.observation, self.info = next_observation, next_info
        self.episode_done = bool(
            truncated.get("__all__", False) or any(terminated.values())
        )
        frame = self.frame(
            reward=reward,
            actions={agent: str(self.env.get_last_action(agent)) for agent in self.env.agents},
            red_activity={
                agent: str(self.env.env.get_last_action(agent))
                for agent in self.red_agents
            },
            rewards_per_agent={agent: float(value) for agent, value in rewards.items()},
        )
        self.frames.append(frame)
        return frame

    def frame(
        self,
        reward: float = 0.0,
        actions: dict[str, str] | None = None,
        red_activity: dict[str, str] | None = None,
        rewards_per_agent: dict[str, float] | None = None,
    ) -> dict:
        host_state = _host_snapshot(self.env, self.red_agents) if self.env else {}
        return {
            "step": self.current_step,
            "blue_actions": actions or {},
            "red_activity": red_activity or {},
            "rewards_per_agent": rewards_per_agent or {},
            "reward": reward,
            "cumulative_reward": self.cumulative_reward,
            "host_state": host_state,
            "mission_phase": _mission_phase(self.info),
            "episode_done": self.episode_done,
            "provenance": self.provenance(),
        }

    def provenance(self) -> dict:
        checkpoint_hash = None
        if self.checkpoint:
            path = Path(self.checkpoint)
            if path.exists():
                checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "env": "CC4",
            "source": "live_demo",
            "seed": self.seed,
            "steps": self.steps,
            "red": self.red_strategy,
            "agent": self.blue_policy,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": checkpoint_hash,
            "git_commit": _git_commit(),
            "complete": self.episode_done,
        }

    def save(self, path: str | Path) -> Path:
        if not self.frames:
            raise RuntimeError("cannot save a live run before executing a step")
        output = {
            "provenance": self.provenance(),
            "total_reward": self.cumulative_reward,
            "steps": self.frames,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
        return destination
