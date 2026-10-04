import random
import numpy as np

class RandomAgent:
    """Agent that selects actions uniformly at random from valid action space."""
    def __init__(self, action_space=None):
        self.action_space = action_space

    def get_action(self, observation, action_space=None):
        space = action_space or self.action_space
        if space is None:
            return 0
        if hasattr(space, 'sample'):
            return space.sample()
        if isinstance(space, list):
            return random.choice(space)
        if isinstance(space, int):
            return random.randint(0, space - 1)
        return 0


class SleepAgent:
    """Agent that always performs a 'Do-Nothing' / Sleep action (index 0)."""
    def __init__(self, sleep_action_index=0):
        self.sleep_action_index = sleep_action_index

    def get_action(self, observation, action_space=None):
        return self.sleep_action_index


class RuleBasedDefenderAgent:
    """
    Simple heuristic rule-based defender agent.
    Monitors host observations for anomaly/compromise signals and prioritizes:
    1. Restore compromised hosts (if detected)
    2. Remove malicious processes / Analyse suspicious activity
    3. Sleep / Do-nothing if environment state appears clean
    """
    def __init__(self, action_space=None):
        self.action_space = action_space

    def get_action(self, observation, action_space=None):
        # Heuristic rules based on observation flags if formatted as array/dict
        if isinstance(observation, dict):
            # Check for compromised hosts in observation dictionary
            for host, info in observation.items():
                if isinstance(info, dict) and info.get('Compromised') == 'User':
                    # Priority 1: Restore or analyze
                    return 1
        # Fallback to default action or random action if space is known
        space = action_space or self.action_space
        if hasattr(space, 'sample'):
            return 0
        return 0
