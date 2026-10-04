import gym
import numpy as np

class CybORGVectorWrapper(gym.Wrapper):
    """
    Wrapper to flatten CybORG observation dicts/tables into fixed 1D numpy float arrays
    suitable for RL algorithms (PPO, DQN, MAPPO).
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]
        return self._format_obs(obs)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        formatted_obs = self._format_obs(obs)
        return formatted_obs, reward, done, info

    def _format_obs(self, obs):
        if isinstance(obs, np.ndarray):
            return obs.astype(np.float32)
        if isinstance(obs, (list, tuple)):
            return np.array(obs, dtype=np.float32)
        if isinstance(obs, dict):
            # Numeric conversion of dictionary observation fields
            flat = []
            for k, v in sorted(obs.items()):
                if isinstance(v, (int, float, bool)):
                    flat.append(float(v))
                elif isinstance(v, (list, np.ndarray)):
                    flat.extend([float(x) for x in np.array(v).flatten()])
            return np.array(flat, dtype=np.float32) if flat else np.zeros(1, dtype=np.float32)
        return np.zeros(1, dtype=np.float32)
