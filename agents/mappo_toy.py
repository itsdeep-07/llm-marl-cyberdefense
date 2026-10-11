import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

class ActorNetwork(nn.Module):
    """Decentralized Policy Network (Actor) operating solely on local observations."""
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)

    def evaluate_actions(self, obs, actions):
        dist = self.forward(obs)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean()
        return action_log_probs, dist_entropy


class CentralizedCriticNetwork(nn.Module):
    """Centralized Value Network (Critic) operating on full network global state."""
    def __init__(self, global_state_dim, hidden_dim=128):
        super(CentralizedCriticNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, global_state):
        return self.net(global_state)


class MAPPOAgent:
    """
    Multi-Agent PPO (MAPPO) with Centralized Training and Decentralized Execution (CTDE).
    Complies with Dec-POMDP formulation from Nguyen & Reddi (IEEE TNNLS 2023).
    """
    def __init__(
        self,
        num_agents=3,
        obs_dim=16,
        act_dim=5,
        global_state_dim=52,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        entropy_coef=0.01,
        value_loss_coef=0.5
    ):
        self.num_agents = num_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

        # Decentralized Actors (one per subnet defender)
        self.actors = nn.ModuleList([
            ActorNetwork(obs_dim, act_dim) for _ in range(num_agents)
        ])
        # Centralized Critic
        self.critic = CentralizedCriticNetwork(global_state_dim)

        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_actions(self, local_obs_list):
        """Decentralized Execution: Actors choose action given local observations."""
        actions = []
        action_log_probs = []
        with torch.no_grad():
            for i, obs in enumerate(local_obs_list):
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                dist = self.actors[i](obs_t)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                actions.append(action.item())
                action_log_probs.append(log_prob.item())
        return actions, action_log_probs

    def get_value(self, global_state):
        """Centralized evaluation using global state."""
        with torch.no_grad():
            state_t = torch.as_tensor(global_state, dtype=torch.float32)
            value = self.critic(state_t)
            return value.item()

    def update(self, rollout_buffer, epochs=4, batch_size=32):
        """
        PPO update step with Generalized Advantage Estimation (GAE) and CTDE value loss.
        """
        obs_buf = torch.as_tensor(np.array(rollout_buffer["obs"]), dtype=torch.float32)
        state_buf = torch.as_tensor(np.array(rollout_buffer["states"]), dtype=torch.float32)
        act_buf = torch.as_tensor(np.array(rollout_buffer["actions"]), dtype=torch.long)
        log_prob_buf = torch.as_tensor(np.array(rollout_buffer["log_probs"]), dtype=torch.float32)
        returns_buf = torch.as_tensor(np.array(rollout_buffer["returns"]), dtype=torch.float32)
        adv_buf = torch.as_tensor(np.array(rollout_buffer["advantages"]), dtype=torch.float32)

        # Normalize advantages
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

        total_steps = len(rollout_buffer["obs"])
        indices = np.arange(total_steps)

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, total_steps, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                b_obs = obs_buf[batch_idx]       # [batch, num_agents, obs_dim]
                b_states = state_buf[batch_idx]  # [batch, global_state_dim]
                b_acts = act_buf[batch_idx]      # [batch, num_agents]
                b_old_lp = log_prob_buf[batch_idx]
                b_returns = returns_buf[batch_idx] # [batch, num_agents]
                b_adv = adv_buf[batch_idx]       # [batch, num_agents]

                # 1. Update Decentralized Actors
                for i in range(self.num_agents):
                    curr_log_probs, entropy = self.actors[i].evaluate_actions(b_obs[:, i], b_acts[:, i])
                    ratios = torch.exp(curr_log_probs - b_old_lp[:, i])
                    
                    surr1 = ratios * b_adv[:, i]
                    surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * b_adv[:, i]
                    actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                    self.actor_optimizers[i].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
                    self.actor_optimizers[i].step()

                # 2. Update Centralized Critic
                values = self.critic(b_states).squeeze(-1)
                # Team return target (mean over agents)
                critic_target = b_returns.mean(dim=1)
                critic_loss = nn.MSELoss()(values, critic_target)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()
