"""
Base agent classes for multi-robot reinforcement learning.

This module provides the base agent classes that all specific
agent implementations will inherit from.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import pickle
from typing import Dict, Tuple, List, Optional, Union, Any

from models.qmix_network import QMIXNetwork


class BaseAgent:
    """
    Base class for all agents.
    
    This class provides common functionality for all agents.
    
    Attributes:
        state_dim (int): Dimension of the state space
        action_dim_leader (int): Dimension of the leader's action space
        action_dim_follower1 (int): Dimension of follower1's action space
        action_dim_follower2 (int): Dimension of follower2's action space
        device (str): Device to run the model on (cpu or cuda)
        seed (int): Random seed
    """
    def __init__(self, state_dim: int, action_dim_leader: int, action_dim_follower1: int, 
                 action_dim_follower2: int, device: str = 'cpu', seed: int = 42):
        """
        Initialize the base agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim_leader: Dimension of the leader's action space
            action_dim_follower1: Dimension of follower1's action space
            action_dim_follower2: Dimension of follower2's action space
            device: Device to run the model on (cpu or cuda)
            seed: Random seed
        """
        self.state_dim = state_dim
        self.action_dim_leader = action_dim_leader
        self.action_dim_follower1 = action_dim_follower1
        self.action_dim_follower2 = action_dim_follower2
        self.device = device
        self.seed = seed
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize debug flag
        self.debug = False
    
    def update(self, *args, **kwargs):
        """
        Update the agent's networks.
        
        Should be implemented by subclasses.
        """
        raise NotImplementedError
    
    def act(self, *args, **kwargs):
        """
        Select actions for all robots.
        
        Should be implemented by subclasses.
        """
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        """
        Save the agent's state.
        
        Args:
            path: Directory to save to
        """
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        """
        Load the agent's state.
        
        Args:
            path: Directory to load from
        """
        raise NotImplementedError
    
    def apply_action_mask(self, q_values: torch.Tensor, action_mask: torch.Tensor, 
                        min_value: float = -1e8) -> torch.Tensor:
        """
        Apply an action mask to q-values.
        
        Args:
            q_values: Q-values tensor
            action_mask: Boolean mask tensor (True for valid actions)
            min_value: Value to use for masked (invalid) actions
            
        Returns:
            Masked Q-values tensor with invalid actions set to min_value
        """
        # Convert boolean mask to float
        mask = action_mask.float()
        
        # Apply mask: set invalid actions to min_value
        return q_values * mask + (1 - mask) * min_value
        
    def process_action_mask(self, state_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process action masks from environment state.
        
        Args:
            state_dict: Dictionary containing state information including action masks
            
        Returns:
            Tuple of action masks for leader, follower1, and follower2 as tensors
        """
        if 'action_masks' not in state_dict:
            # If no mask provided, all actions are valid
            leader_mask = torch.ones(self.action_dim_leader, dtype=torch.bool)
            follower1_mask = torch.ones(self.action_dim_follower1, dtype=torch.bool)
            follower2_mask = torch.ones(self.action_dim_follower2, dtype=torch.bool)
        else:
            # Convert numpy masks to torch tensors
            leader_mask = torch.tensor(state_dict['action_masks']['leader'], dtype=torch.bool)
            follower1_mask = torch.tensor(state_dict['action_masks']['follower1'], dtype=torch.bool)
            follower2_mask = torch.tensor(state_dict['action_masks']['follower2'], dtype=torch.bool)
        
        return leader_mask.to(self.device), follower1_mask.to(self.device), follower2_mask.to(self.device)

