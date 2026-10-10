"""MAPPO for the five CC4 Blue agents (Batch 4).

Generalised from the toy 3-agent version:
  * any number of agents; dimensions come from the environment, never hard-coded
  * one shared actor (plus a one-hot agent id) acting on each agent's own
    observation, so execution stays decentralised
  * a centralized critic that sees ALL agents' observations joined together
    (the centralized joint state) plus the agent id, and outputs one value per agent
  * per-agent action masks applied when acting and when computing the loss

The old toy implementation now lives in agents/mappo_toy.py.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

NEG_INF = -1e9


def _mlp(inp: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out),
    )


class MAPPOAgent:
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        hidden: int = 128,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        epochs: int = 4,
        minibatch_size: int = 80,
        reward_scale: float = 0.1,
        max_grad_norm: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.n_agents, self.obs_dim, self.act_dim = n_agents, obs_dim, act_dim
        self.gamma, self.gae_lambda, self.clip = gamma, gae_lambda, clip
        self.entropy_coef, self.value_coef = entropy_coef, value_coef
        self.epochs, self.minibatch_size = epochs, minibatch_size
        self.reward_scale, self.max_grad_norm = reward_scale, max_grad_norm

        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.ids = torch.eye(n_agents)  # [N, N] one-hot agent ids

        self.actor = _mlp(obs_dim + n_agents, hidden, act_dim)
        self.critic = _mlp(n_agents * obs_dim + n_agents, hidden, 1)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    # ---- network inputs -------------------------------------------------
    def _ids(self, batch: int) -> torch.Tensor:
        return self.ids.unsqueeze(0).expand(batch, -1, -1)

    def _actor_input(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs, self._ids(obs.shape[0])], dim=-1)

    def _critic_input(self, obs: torch.Tensor) -> torch.Tensor:
        b, n, d = obs.shape
        joint = obs.reshape(b, 1, n * d).expand(b, n, n * d)  # same joint state for every agent
        return torch.cat([joint, self._ids(b)], dim=-1)

    def _dist(self, obs: torch.Tensor, masks: torch.Tensor) -> Categorical:
        logits = self.actor(self._actor_input(obs)).masked_fill(~masks, NEG_INF)
        return Categorical(logits=logits)

    # ---- acting ---------------------------------------------------------
    @staticmethod
    def _to_tensors(obs, masks):
        o = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        m = torch.as_tensor(np.asarray(masks, dtype=bool)).unsqueeze(0)
        if not bool(m.any(dim=-1).all()):
            raise ValueError("an agent has no valid action in its mask")
        return o, m

    @torch.no_grad()
    def act(self, obs, masks, greedy: bool = False):
        """obs [N, obs_dim], masks [N, act_dim] -> actions, log_probs, values (each [N])."""
        o, m = self._to_tensors(obs, masks)
        dist = self._dist(o, m)
        actions = dist.probs.argmax(dim=-1) if greedy else dist.sample()
        log_probs = dist.log_prob(actions)
        values = self.critic(self._critic_input(o)).squeeze(-1)
        return actions[0].numpy(), log_probs[0].numpy(), values[0].numpy()

    @torch.no_grad()
    def value(self, obs) -> np.ndarray:
        o = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        return self.critic(self._critic_input(o)).squeeze(-1)[0].numpy()

    # ---- learning -------------------------------------------------------
    def _gae(self, ep: dict):
        r = np.asarray(ep["rewards"], dtype=np.float32) * self.reward_scale  # [T, N]
        v = np.asarray(ep["values"], dtype=np.float32)                       # [T, N]
        t_len = len(r)
        if ep["terminated"]:
            last = np.zeros_like(v[0])
        else:
            last = np.asarray(ep["last_value"], dtype=np.float32)
        adv = np.zeros_like(r)
        gae = np.zeros_like(v[0])
        for t in reversed(range(t_len)):
            next_v = last if t == t_len - 1 else v[t + 1]
            delta = r[t] + self.gamma * next_v - v[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            adv[t] = gae
        return adv, adv + v

    def update(self, episodes: list) -> dict:
        """episodes: dicts with obs, masks, actions, log_probs, rewards, values
        (all shaped [T, N, ...]), last_value [N], terminated bool."""
        keys = ("obs", "masks", "actions", "log_probs", "adv", "ret")
        parts = {k: [] for k in keys}
        for ep in episodes:
            adv, ret = self._gae(ep)
            parts["obs"].append(np.asarray(ep["obs"], dtype=np.float32))
            parts["masks"].append(np.asarray(ep["masks"], dtype=bool))
            parts["actions"].append(np.asarray(ep["actions"], dtype=np.int64))
            parts["log_probs"].append(np.asarray(ep["log_probs"], dtype=np.float32))
            parts["adv"].append(adv)
            parts["ret"].append(ret)
        obs = torch.as_tensor(np.concatenate(parts["obs"]))
        masks = torch.as_tensor(np.concatenate(parts["masks"]))
        actions = torch.as_tensor(np.concatenate(parts["actions"]))
        old_lp = torch.as_tensor(np.concatenate(parts["log_probs"]))
        adv = torch.as_tensor(np.concatenate(parts["adv"]))
        ret = torch.as_tensor(np.concatenate(parts["ret"]))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        count = 0
        for _ in range(self.epochs):
            perm = self.rng.permutation(n)
            for start in range(0, n, self.minibatch_size):
                idx = torch.as_tensor(perm[start:start + self.minibatch_size])
                dist = self._dist(obs[idx], masks[idx])
                new_lp = dist.log_prob(actions[idx])
                ratio = torch.exp(new_lp - old_lp[idx])
                s1 = ratio * adv[idx]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[idx]
                policy_loss = -torch.min(s1, s2).mean()
                entropy = dist.entropy().mean()
                values = self.critic(self._critic_input(obs[idx])).squeeze(-1)
                value_loss = ((values - ret[idx]) ** 2).mean()

                loss = policy_loss - self.entropy_coef * entropy + self.value_coef * value_loss
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

                stats["policy_loss"] += float(policy_loss)
                stats["value_loss"] += float(value_loss)
                stats["entropy"] += float(entropy)
                stats["approx_kl"] += float((old_lp[idx] - new_lp).mean())
                count += 1
        return {k: v / max(count, 1) for k, v in stats.items()}

    # ---- persistence ----------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "meta": {"n_agents": self.n_agents, "obs_dim": self.obs_dim,
                         "act_dim": self.act_dim},
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path)
        expected = {"n_agents": self.n_agents, "obs_dim": self.obs_dim, "act_dim": self.act_dim}
        if data["meta"] != expected:
            raise ValueError(f"checkpoint {data['meta']} does not match this model {expected}")
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])