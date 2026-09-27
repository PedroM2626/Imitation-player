"""
Utilities for the Imitation Learning project.
Helper classes and functions.
"""

import os
import re
from pathlib import Path
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy


def get_last_index(path: str, prefix: str, suffix: str = ".zip") -> int:
    """
    Find the highest index number among files in a directory.
    
    Example: if 'bc_policy0.zip' and 'bc_policy15.zip' exist, returns 15.
    
    Args:
        path: Directory path
        prefix: File prefix (ex: "bc_policy")
        suffix: File suffix (ex: ".zip")
    
    Returns:
        Highest index found, or -1 if there are no files
    """
    last_index = -1
    p = Path(path)
    if not p.exists():
        return last_index
    
    pattern = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(suffix) + r"$")
    for file in p.iterdir():
        if file.is_file():
            m = pattern.search(file.name)
            if m:
                last_index = max(last_index, int(m.group(1)))
    return last_index


class LSTMWrapper:
    """
    Wrapper for policies with an LSTM feature extractor.
    Manages the LSTM state across calls (reset between episodes).
    """
    def __init__(self, policy: ActorCriticPolicy):
        self.policy = policy
        self.lstm_state = None
        self.feature_buffer = []  # Keep the buffer here
        self._last_obs = None
        
        # Check whether the policy has an LSTM
        self.has_lstm = (hasattr(policy, 'features_extractor') and 
                        hasattr(policy.features_extractor, 'lstm'))
    
    def reset(self):
        """Reset the LSTM state (must be called for each new episode)."""
        self.lstm_state = None
        self.feature_buffer = []
        # Call the features_extractor reset if it exists
        if self.has_lstm and hasattr(self.policy, 'features_extractor') and \
           hasattr(self.policy.features_extractor, 'reset_hidden'):
            self.policy.features_extractor.reset_hidden()
    
    def predict(self, observation, deterministic: bool = False):
        """
        Predict actions while keeping the LSTM state.
        
        Args:
            observation: Array with the current observation
            deterministic: If True, always takes the most likely action
        
        Returns:
            Predicted action and state (None for compatibility)
        """
        with th.no_grad():
            # Convert to a tensor if needed
            if not isinstance(observation, th.Tensor):
                obs_tensor = th.as_tensor(observation).float()
            else:
                obs_tensor = observation.float()
            
            # Normalize
            if obs_tensor.max() > 1.0:
                obs_tensor = obs_tensor / 255.0
            
            # Make the prediction through the policy
            # Note: the SB3 policy may have its own LSTM management
            deterministic_float = 1.0 if deterministic else 0.0
            
            # Use the predict method of the base policy
            if hasattr(self.policy, 'predict'):
                action, state = self.policy.predict(observation, deterministic=deterministic)
                return action, state
    
    def __getattr__(self, name):
        """Delegate attribute access to the base policy object."""
        return getattr(self.policy, name)
