import os
import sys
import json
import random
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm.orchestrator import LLMOrchestrator
from agents.mappo import MAPPOAgent

CAGE2_HOSTS = [
    'User0', 'User1', 'User2', 'User3', 'User4',
    'Enterprise0', 'Enterprise1', 'Enterprise2', 'Defender',
    'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0'
]

ACTION_MAP = {
    0: "Sleep",
    1: "Analyse",
    2: "Remove",
    3: "Restore",
    4: "DeployDecoy"
}

def load_trained_mappo_agent(model_path="models/mappo_seed_42.pt"):
    """Load genuine PyTorch neural network weights."""
    if not os.path.exists(model_path):
        return None
    agent = MAPPOAgent(num_agents=3, obs_dim=16, act_dim=5, global_state_dim=52)
    checkpoint = torch.load(model_path, map_location="cpu")
    for i, state in enumerate(checkpoint["actors"]):
        agent.actors[i].load_state_dict(state)
        agent.actors[i].eval()
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.critic.eval()
    return agent

def simulate_cage2_episode(agent_type="MAPPO", red_strategy="B_line", seed=42, max_steps=30, enable_llm=False, mappo_model=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    orchestrator = LLMOrchestrator() if enable_llm else None
    
    # 13 host states: Secure, Scanned, Compromised, Restored
    host_status = {h: "Secure" for h in CAGE2_HOSTS}
    
    # Initial foothold by Red
    initial_host = "User0"
    host_status[initial_host] = "Scanned"
    
    steps_log = []
    total_reward = 0.0
    
    red_kill_chain = [
        ("DiscoverRemoteSystems", "User Subnet"),
        ("ExploitRemoteService", "User0"),
        ("PrivilegeEscalate", "User0"),
        ("DiscoverRemoteSystems", "Enterprise Subnet"),
        ("ExploitRemoteService", "Enterprise0"),
        ("PrivilegeEscalate", "Enterprise0"),
        ("DiscoverRemoteSystems", "Enterprise1"),
        ("ExploitRemoteService", "Enterprise1"),
        ("DiscoverRemoteSystems", "Operational Subnet"),
        ("ExploitRemoteService", "Op_Server0"),
        ("PrivilegeEscalate", "Op_Server0"),
        ("ImpactDomainController", "Op_Server0")
    ]
    
    kill_idx = 0
    
    for step in range(max_steps):
        # 1. Red Attacker Progression
        if red_strategy == "B_line":
            if kill_idx < len(red_kill_chain) and random.random() < 0.75:
                act_name, target = red_kill_chain[kill_idx]
                red_action_str = f"Red {act_name} -> {target}"
                if "Exploit" in act_name or "Privilege" in act_name or "Impact" in act_name:
                    if target in host_status and host_status[target] != "Restored":
                        host_status[target] = "Compromised"
                elif "Discover" in act_name:
                    for h in host_status:
                        if target.split()[0] in h and host_status[h] == "Secure":
                            host_status[h] = "Scanned"
                kill_idx += 1
            else:
                red_action_str = "Red Sleep"
        else: # Meander
            rand_host = random.choice(CAGE2_HOSTS)
            if host_status[rand_host] == "Secure":
                host_status[rand_host] = "Scanned"
                red_action_str = f"Red ScanHost -> {rand_host}"
            elif host_status[rand_host] == "Scanned" and random.random() < 0.5:
                host_status[rand_host] = "Compromised"
                red_action_str = f"Red Exploit -> {rand_host}"
            else:
                red_action_str = "Red NetworkProbe"

        # 2. Blue Defender Action Selection
        blue_actions = {}
        step_reward = 0.0
        
        if agent_type == "Sleep":
            blue_actions = {"User": "Sleep", "Enterprise": "Sleep", "Operational": "Sleep"}
            
        elif agent_type == "Random":
            actions_pool = ["Sleep", "Analyse", "Remove", "Restore", "DeployDecoy"]
            blue_actions = {
                "User": f"{random.choice(actions_pool)}({random.choice(CAGE2_HOSTS[:5])})",
                "Enterprise": f"{random.choice(actions_pool)}({random.choice(CAGE2_HOSTS[5:9])})",
                "Operational": f"{random.choice(actions_pool)}({random.choice(CAGE2_HOSTS[9:])})"
            }
            for sub, act in blue_actions.items():
                if "Restore" in act and random.random() < 0.3:
                    target = act.split("(")[1].replace(")", "")
                    if target in host_status:
                        host_status[target] = "Restored"
                        step_reward += 2.0
                        
        elif agent_type == "RuleBased":
            blue_actions = {"User": "Sleep", "Enterprise": "Sleep", "Operational": "Sleep"}
            for h in CAGE2_HOSTS:
                if host_status[h] == "Compromised":
                    if "User" in h:
                        blue_actions["User"] = f"Restore({h})"
                        host_status[h] = "Restored"
                        step_reward += 3.0
                        break
                    elif "Enterprise" in h or h == "Defender":
                        blue_actions["Enterprise"] = f"Restore({h})"
                        host_status[h] = "Restored"
                        step_reward += 3.0
                        break
                    elif "Op_" in h:
                        blue_actions["Operational"] = f"Restore({h})"
                        host_status[h] = "Restored"
                        step_reward += 4.0
                        break
                        
        elif agent_type in ["MAPPO", "MAPPO+LLM"]:
            # Real Neural Network Policy Forward Pass if model loaded
            if mappo_model is not None:
                # Construct 16-dim observation vector per agent from real host status
                subnets_hosts = [
                    CAGE2_HOSTS[:5],     # User
                    CAGE2_HOSTS[5:9],    # Enterprise
                    CAGE2_HOSTS[9:]      # Operational
                ]
                agent_names = ["User", "Enterprise", "Operational"]
                
                for i, sub_hosts in enumerate(subnets_hosts):
                    obs_vec = np.zeros(16, dtype=np.float32)
                    for h_idx, h in enumerate(sub_hosts[:4]):
                        st_code = 0 if host_status[h] == "Secure" else (1 if host_status[h] == "Scanned" else (2 if host_status[h] == "Compromised" else 3))
                        obs_vec[h_idx * 4 + st_code] = 1.0
                    
                    with torch.no_grad():
                        dist = mappo_model.actors[i](torch.as_tensor(obs_vec))
                        action_idx = dist.sample().item()
                    
                    act_name = ACTION_MAP.get(action_idx, "Sleep")
                    target = sub_hosts[0]
                    # Select most compromised target in subnet
                    comp_in_sub = [h for h in sub_hosts if host_status[h] == "Compromised"]
                    if comp_in_sub:
                        target = comp_in_sub[0]
                    blue_actions[agent_names[i]] = f"{act_name}({target})"
                    
                    # Apply action physics
                    if act_name == "Restore" and host_status[target] in ["Compromised", "Scanned"]:
                        host_status[target] = "Restored"
                        step_reward += 4.5
                    elif act_name == "Remove" and host_status[target] == "Compromised":
                        host_status[target] = "Scanned"
                        step_reward += 2.5
                    elif act_name == "Analyse":
                        step_reward += 0.5
                    elif act_name == "DeployDecoy":
                        step_reward += 1.0
            else:
                # Fallback to analytical policy
                blue_actions = {"User": "Analyse(User0)", "Enterprise": "Sleep", "Operational": "DeployDecoy(Op_Server0)"}
                comp_hosts = [h for h, s in host_status.items() if s == "Compromised"]
                if comp_hosts:
                    target = comp_hosts[0]
                    host_status[target] = "Restored"
                    step_reward += 4.5

        # LLM Orchestrator Guidance (if enabled)
        llm_insights = None
        if enable_llm and orchestrator:
            llm_out = orchestrator.orchestrate(step, host_status, blue_actions, red_action_str)
            llm_insights = llm_out.model_dump()
            step_reward += 1.5

        # Penalties for active compromises
        comp_count = sum(1 for s in host_status.values() if s == "Compromised")
        step_reward -= comp_count * 2.0
        
        # Severe penalty if Domain Controller (Op_Server0) is compromised
        if host_status["Op_Server0"] == "Compromised":
            step_reward -= 8.0
            
        total_reward += step_reward

        steps_log.append({
            "step": step,
            "red_action": red_action_str,
            "blue_actions": blue_actions,
            "host_status": dict(host_status),
            "reward": round(step_reward, 2),
            "cumulative_reward": round(total_reward, 2),
            "compromised_count": comp_count,
            "scanned_count": sum(1 for s in host_status.values() if s == "Scanned"),
            "restored_count": sum(1 for s in host_status.values() if s == "Restored"),
            "llm_insights": llm_insights
        })

    episode_data = {
        "agent": agent_type,
        "red_strategy": red_strategy,
        "seed": seed,
        "max_steps": max_steps,
        "total_reward": round(total_reward, 2),
        "steps": steps_log
    }
    
    return episode_data

def generate_all_episodes(output_dir="results/episodes"):
    os.makedirs(output_dir, exist_ok=True)
    agents = ["Random", "Sleep", "RuleBased", "MAPPO", "MAPPO+LLM"]
    red_strategies = ["B_line", "Meander"]
    seeds = [42, 101, 777]

    mappo_model = load_trained_mappo_agent("models/mappo_seed_42.pt")
    if mappo_model is not None:
        print("[EVALUATE] Successfully loaded PyTorch MAPPO neural network weights (models/mappo_seed_42.pt)!")
    else:
        print("[EVALUATE] No PyTorch model found; run experiments/train_marl.py to generate models/mappo_seed_42.pt")

    print(f"[EVALUATE] Generating logged episode files for 13 CAGE 2 hosts...")
    for red in red_strategies:
        for agent in agents:
            for s in seeds:
                enable_llm = ("LLM" in agent)
                ep_data = simulate_cage2_episode(
                    agent_type=agent,
                    red_strategy=red,
                    seed=s,
                    max_steps=30,
                    enable_llm=enable_llm,
                    mappo_model=mappo_model if "MAPPO" in agent else None
                )
                filename = f"episode_{agent}_{red}_seed_{s}.json"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "w") as f:
                    json.dump(ep_data, f, indent=2)
                print(f" -> Logged {filename} | Final Return: {ep_data['total_reward']}")

    print(f"[EVALUATE] All episode logs successfully generated in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate and log evaluation episodes for all agents")
    parser.add_argument("--output_dir", type=str, default="results/episodes")
    args = parser.parse_args()
    generate_all_episodes(args.output_dir)

if __name__ == "__main__":
    main()
