import numpy as np

class SubnetMARLWrapper:
    """
    Multi-Agent Dec-POMDP Environment Wrapper.
    
    Partitions the network into N=3 distinct administrative subnets:
      - Agent 0: User Workstation Subnet (Hosts 0 - 3)
      - Agent 1: Enterprise Server Subnet (Hosts 4 - 7)
      - Agent 2: Operational / Core Subnet (Hosts 8 - 11, Domain Controller)
      
    Implements Centralized Training with Decentralized Execution (CTDE):
      - Local Observations (o_i): Each agent receives partial visibility restricted to its subnet.
      - Global State (S_global): Concatenated full network telemetry provided exclusively to the Centralized Critic.
    """
    def __init__(self, num_agents=3, obs_dim_per_agent=16, act_dim_per_agent=5, max_steps=50):
        self.num_agents = num_agents
        self.obs_dim_per_agent = obs_dim_per_agent
        self.act_dim_per_agent = act_dim_per_agent
        self.global_state_dim = num_agents * obs_dim_per_agent + 4 # includes global telemetry features
        self.max_steps = max_steps
        self.step_count = 0
        
        # Subnet host state tracking: 0: Secure, 1: Scanned, 2: Compromised, 3: Restored
        self.host_states = np.zeros(num_agents * 4, dtype=np.int32)
        self.red_privilege_level = 0 # 0: None, 1: User, 2: Root, 3: Domain Admin

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.step_count = 0
        self.host_states = np.zeros(self.num_agents * 4, dtype=np.int32)
        # Red initial foothold on User Subnet (Host 0)
        self.host_states[0] = 1 # Scanned
        self.red_privilege_level = 0
        
        local_obs = self._get_local_observations()
        global_state = self._get_global_state()
        return local_obs, global_state

    def step(self, joint_actions):
        """
        Execute joint actions of all Blue defenders and simulate Red attacker transition.
        joint_actions: list or array of size num_agents, each action in [0, act_dim_per_agent - 1]
                       0: Sleep/Monitor
                       1: Analyse (Host scan)
                       2: Remove (Kill malware process)
                       3: Restore (Reimage host from clean backup)
                       4: DeployDecoy (Deception defense)
        """
        self.step_count += 1
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        
        # 1. Simulate Red Attacker Progression (B-line Kill Chain)
        # Red advances towards Enterprise Server and Core Domain Controller
        if np.random.rand() < 0.65:
            # Advance attack
            uncompromised = np.where(self.host_states < 2)[0]
            if len(uncompromised) > 0:
                target_host = uncompromised[0]
                self.host_states[target_host] = min(2, self.host_states[target_host] + 1)
                if target_host >= 8: # Operational subnet reached
                    self.red_privilege_level = max(self.red_privilege_level, 2)
                    
        # 2. Execute Blue Defender Actions per Subnet
        for i in range(self.num_agents):
            action = joint_actions[i]
            subnet_start = i * 4
            subnet_hosts = self.host_states[subnet_start : subnet_start + 4]
            
            # Action Costs and Effects
            if action == 0: # Sleep
                pass # Zero cost
            elif action == 1: # Analyse
                rewards[i] -= 0.1 # Small operational monitoring cost
            elif action == 2: # Remove
                rewards[i] -= 0.5
                compromised_in_subnet = np.where(subnet_hosts == 2)[0]
                if len(compromised_in_subnet) > 0:
                    h_idx = subnet_start + compromised_in_subnet[0]
                    self.host_states[h_idx] = 1 # Downgrade to scanned
                    rewards[i] += 2.0 # Reward for containment
            elif action == 3: # Restore
                rewards[i] -= 1.0 # High availability cost of reimaging
                compromised_in_subnet = np.where(subnet_hosts >= 1)[0]
                if len(compromised_in_subnet) > 0:
                    h_idx = subnet_start + compromised_in_subnet[0]
                    self.host_states[h_idx] = 3 # Restored / Clean
                    rewards[i] += 4.0 # High reward for full remediation
            elif action == 4: # DeployDecoy
                rewards[i] -= 0.8
                # Decoy slows Red progression in this subnet
                rewards[i] += 1.5

            # Continuous penalty for active compromises in subnet
            active_compromises = np.sum(subnet_hosts == 2)
            rewards[i] -= active_compromises * 1.5
            
        # Global team penalty if Domain Controller (host 11) is compromised
        if self.host_states[-1] == 2:
            rewards -= 5.0

        done = self.step_count >= self.max_steps
        local_obs = self._get_local_observations()
        global_state = self._get_global_state()
        
        info = {
            "compromised_hosts": int(np.sum(self.host_states == 2)),
            "restored_hosts": int(np.sum(self.host_states == 3)),
            "red_privilege_level": self.red_privilege_level
        }
        
        return local_obs, global_state, rewards, done, info

    def _get_local_observations(self):
        """Extract partial observation vector for each agent."""
        local_obs = []
        for i in range(self.num_agents):
            subnet_start = i * 4
            subnet_hosts = self.host_states[subnet_start : subnet_start + 4]
            # One-hot representation of 4 hosts across 4 states = 16 features
            obs_vec = np.zeros(self.obs_dim_per_agent, dtype=np.float32)
            for h_idx, state in enumerate(subnet_hosts):
                obs_vec[h_idx * 4 + state] = 1.0
            local_obs.append(obs_vec)
        return np.array(local_obs, dtype=np.float32)

    def _get_global_state(self):
        """Extract full network global state for Centralized Critic."""
        flat_obs = self._get_local_observations().flatten()
        # Telemetry: current step, total compromises, red privilege, step ratio
        telemetry = np.array([
            float(self.step_count) / float(self.max_steps),
            float(np.sum(self.host_states == 2)),
            float(np.sum(self.host_states == 3)),
            float(self.red_privilege_level)
        ], dtype=np.float32)
        return np.concatenate([flat_obs, telemetry])
