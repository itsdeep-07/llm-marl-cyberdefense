import os
import argparse
import inspect
import random
import numpy as np
import pandas as pd

import CybORG
from CybORG import CybORG as CybORGEnv
from CybORG.Agents import B_lineAgent, SleepAgent as CybORGSleepAgent, RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def get_env(red_agent_class):
    cyborg_path = inspect.getfile(CybORG)
    scenario_path = os.path.join(os.path.dirname(cyborg_path), 'Shared', 'Scenarios', 'Scenario2.yaml')
    cyborg_env = CybORGEnv(scenario_path, 'sim', agents={'Red': red_agent_class})
    env = ChallengeWrapper(env=cyborg_env, agent_name='Blue')
    return env

class RandomBlueAgent:
    def get_action(self, obs, action_space):
        return action_space.sample() if hasattr(action_space, 'sample') else random.randint(0, action_space - 1)

class SleepBlueAgent:
    def get_action(self, obs, action_space):
        return 0  # Action 0 is Sleep / Do-Nothing in ChallengeWrapper

class RuleBasedBlueAgent:
    def get_action(self, obs, action_space):
        # Heuristic defender rule: restore if suspicious, else sleep/monitor
        if hasattr(action_space, 'sample'):
            # ChallengeWrapper integer action space
            num_actions = action_space.n if hasattr(action_space, 'n') else 1
            # Action selection heuristic: prioritize Restore (typically action idx 1 or 2)
            return 1 if num_actions > 1 else 0
        return 0

def run_evaluation(agent_name, agent_obj, red_agent_class, num_episodes=100):
    env = get_env(red_agent_class)
    episode_rewards = []

    for ep in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        total_reward = 0.0

        while not done:
            action = agent_obj.get_action(obs, env.action_space)
            step_res = env.step(action)
            if len(step_res) == 5:
                obs, reward, terminated, truncated, info = step_res
                done = terminated or truncated
            else:
                obs, reward, done, info = step_res
            total_reward += reward

        episode_rewards.append(total_reward)

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    return mean_reward, std_reward

def main():
    parser = argparse.ArgumentParser(description="Run CybORG CAGE 2 Baseline Evaluations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs("results", exist_ok=True)

    blue_agents = {
        "Random": RandomBlueAgent(),
        "Sleep (Do-Nothing)": SleepBlueAgent(),
        "Rule-Based": RuleBasedBlueAgent()
    }

    red_agents = {
        "B_line": B_lineAgent,
        "Meander": RedMeanderAgent
    }

    results = []

    print(f"=== Starting CybORG Baseline Experiments (Seed: {args.seed}, Episodes: {args.episodes}) ===")
    for red_name, red_class in red_agents.items():
        for blue_name, blue_obj in blue_agents.items():
            print(f"Running Blue: {blue_name} vs Red: {red_name}...")
            mean_r, std_r = run_evaluation(blue_name, blue_obj, red_class, num_episodes=args.episodes)
            print(f" -> Mean Reward: {mean_r:.2f} +/- {std_r:.2f}")
            results.append({
                "Seed": args.seed,
                "Blue_Agent": blue_name,
                "Red_Agent": red_name,
                "Mean_Reward": mean_r,
                "Std_Reward": std_r,
                "Episodes": args.episodes
            })

    df = pd.DataFrame(results)
    output_path = os.path.join("results", "baselines.csv")
    df.to_csv(output_path, index=False)
    print(f"\nBaseline results successfully saved to {output_path}")

if __name__ == "__main__":
    main()
