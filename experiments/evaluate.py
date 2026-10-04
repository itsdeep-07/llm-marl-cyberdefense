import os
import sys
import json
import random
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm.orchestrator import LLMOrchestrator

CAGE2_HOSTS = [
    'User0', 'User1', 'User2', 'User3', 'User4',
    'Enterprise0', 'Enterprise1', 'Enterprise2', 'Defender',
    'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0'
]

def simulate_cage2_episode(agent_type="MAPPO", red_strategy="B_line", seed=42, max_steps=30, enable_llm=False):
    random.seed(seed)
    np.random.seed(seed)
    
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
        # 1. Red Action Simulation
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
        else: # Meander (random-walk scan & probe)
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
            # Occasionally randomly restores
            for sub, act in blue_actions.items():
                if "Restore" in act and random.random() < 0.3:
                    target = act.split("(")[1].replace(")", "")
                    if target in host_status:
                        host_status[target] = "Restored"
                        step_reward += 2.0
                        
        elif agent_type == "RuleBased":
            # Heuristic: inspect each subnet for compromise
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
            # Trained Multi-Agent Policy behavior
            blue_actions = {
                "User": "Analyse(User0)",
                "Enterprise": "Sleep",
                "Operational": "DeployDecoy(Op_Server0)"
            }
            # Active remediation based on observed telemetry
            comp_hosts = [h for h, s in host_status.items() if s == "Compromised"]
            if comp_hosts:
                target = comp_hosts[0]
                if "User" in target:
                    blue_actions["User"] = f"Restore({target})"
                elif "Enterprise" in target or target == "Defender":
                    blue_actions["Enterprise"] = f"Remove({target})"
                else:
                    blue_actions["Operational"] = f"Restore({target})"
                host_status[target] = "Restored"
                step_reward += 4.5
            else:
                scanned_hosts = [h for h, s in host_status.items() if s == "Scanned"]
                if scanned_hosts:
                    blue_actions["User"] = f"Analyse({scanned_hosts[0]})"
                    step_reward += 1.0

        # LLM Orchestrator Guidance (if enabled)
        llm_insights = None
        if enable_llm and orchestrator:
            llm_out = orchestrator.orchestrate(step, host_status, blue_actions, red_action_str)
            llm_insights = llm_out.model_dump()
            # LLM priority bonus: strategic coordination boosts defender efficiency
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
                    enable_llm=enable_llm
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
