import os
import sys
import argparse
import random
import yaml
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from envs.wrappers import SubnetMARLWrapper
from agents.mappo import MAPPOAgent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_gae(rewards, values, next_value, dones, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimator across trajectory."""
    advantages = []
    gae = 0.0
    values = values + [next_value]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1.0 - dones[step]) - values[step]
        gae = delta + gamma * gae_lambda * (1.0 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns

def train_marl(seed, episodes=100, config_path="configs/marl_config.yaml"):
    set_seed(seed)
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["marl"]

    env = SubnetMARLWrapper(
        num_agents=cfg["num_agents"],
        obs_dim_per_agent=cfg["obs_dim_per_agent"],
        act_dim_per_agent=cfg["act_dim_per_agent"],
        max_steps=cfg["max_steps_per_episode"]
    )

    agent = MAPPOAgent(
        num_agents=cfg["num_agents"],
        obs_dim=cfg["obs_dim_per_agent"],
        act_dim=cfg["act_dim_per_agent"],
        global_state_dim=cfg["global_state_dim"],
        lr_actor=cfg["lr_actor"],
        lr_critic=cfg["lr_critic"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_param=cfg["clip_param"],
        entropy_coef=cfg["entropy_coef"]
    )

    history = []
    print(f"\n[MAPPO] Training Defenders (Seed: {seed}, Episodes: {episodes})...")


    for ep in range(1, episodes + 1):
        local_obs, global_state = env.reset(seed=seed + ep)
        rollout = {
            "obs": [], "states": [], "actions": [],
            "log_probs": [], "rewards": [], "values": [], "dones": []
        }
        
        ep_rewards = np.zeros(cfg["num_agents"])
        done = False

        while not done:
            actions, log_probs = agent.select_actions(local_obs)
            val = agent.get_value(global_state)
            
            next_obs, next_global_state, rewards, done, info = env.step(actions)

            rollout["obs"].append(local_obs)
            rollout["states"].append(global_state)
            rollout["actions"].append(actions)
            rollout["log_probs"].append(log_probs)
            rollout["rewards"].append(rewards)
            rollout["values"].append(val)
            rollout["dones"].append(float(done))

            ep_rewards += rewards
            local_obs = next_obs
            global_state = next_global_state

        # Compute GAE and returns
        next_val = agent.get_value(global_state)
        # Average team reward for team advantage computation
        team_rewards = [r.mean() for r in rollout["rewards"]]
        advs, rets = compute_gae(team_rewards, rollout["values"], next_val, rollout["dones"], cfg["gamma"], cfg["gae_lambda"])
        
        # Expand team advantages/returns across agents for buffer format
        rollout["advantages"] = [np.repeat(a, cfg["num_agents"]) for a in advs]
        rollout["returns"] = [np.repeat(r, cfg["num_agents"]) for r in rets]

        # Update MAPPO networks
        actor_loss, critic_loss = agent.update(rollout, epochs=cfg["epochs_per_update"], batch_size=cfg["batch_size"])

        team_ep_return = ep_rewards.mean()
        history.append({
            "Episode": ep,
            "Team_Return": team_ep_return,
            "Agent_0_Return": ep_rewards[0],
            "Agent_1_Return": ep_rewards[1],
            "Agent_2_Return": ep_rewards[2],
            "Compromised_Hosts": info["compromised_hosts"],
            "Restored_Hosts": info["restored_hosts"],
            "Actor_Loss": actor_loss,
            "Critic_Loss": critic_loss
        })

        if ep % 20 == 0 or ep == episodes:
            print(f"Episode {ep:3d}/{episodes} | Return: {team_ep_return:6.2f} | Compromised: {info['compromised_hosts']} | Restored: {info['restored_hosts']} | Actor Loss: {actor_loss:.4f}")

    # Save model weights and training logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    torch.save({
        "actors": [actor.state_dict() for actor in agent.actors],
        "critic": agent.critic.state_dict()
    }, f"models/mappo_seed_{seed}.pt")
    
    df = pd.DataFrame(history)
    df.to_csv(f"results/mappo_training_seed_{seed}.csv", index=False)
    print(f"[MAPPO] Training completed. Checkpoints saved to models/mappo_seed_{seed}.pt")
    return df


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Agent PPO (MAPPO) Defenders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=60, help="Number of training episodes")
    parser.add_argument("--config", type=str, default="configs/marl_config.yaml", help="Config path")
    args = parser.parse_args()

    train_marl(seed=args.seed, episodes=args.episodes, config_path=args.config)

if __name__ == "__main__":
    main()
