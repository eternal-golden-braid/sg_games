"""
Soft Actor-Critic (SAC) implementation for multi-robot coordination.

This module provides the SAC agent implementation that uses maximum entropy
reinforcement learning for efficient exploration in the multi-robot battery disassembly task.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import pickle
from typing import Dict, Tuple, List, Optional, Union, Any

from agents.sg_agent_base import BaseAgent
from models.sac_critic_network import SACCritic
from models.sac_actor_network import SACActor


class StackelbergSACAgent(BaseAgent):
    """
    Agent implementation using Soft Actor-Critic for Stackelberg games with three robots.
    
    This agent uses maximum entropy reinforcement learning for efficient exploration
    and improved policy robustness.
    
    Attributes:
        hidden_size (int): Size of hidden layers
        gamma (float): Discount factor for future rewards
        tau (float): Soft update parameter for target network
        alpha (float): Temperature parameter for entropy regularization
        target_update_interval (int): How often to update the target network
        automatic_entropy_tuning (bool): Whether to automatically tune the entropy temperature
    """
    def __init__(self, state_dim: int, action_dim_leader: int, action_dim_follower1: int, action_dim_follower2: int,
                 hidden_size: int = 64, device: str = 'cpu', learning_rate: float = 3e-4,
                 gamma: float = 0.99, tau: float = 0.005, alpha: float = 0.2,
                 target_update_interval: int = 1, automatic_entropy_tuning: bool = True,
                 seed: int = 42, debug: bool = False):
        """
        Initialize the Stackelberg SAC agent for three robots.
        
        Args:
            state_dim: Dimension of the state space
            action_dim_leader: Dimension of the leader's action space
            action_dim_follower1: Dimension of follower1's action space
            action_dim_follower2: Dimension of follower2's action space
            hidden_size: Size of hidden layers
            device: Device to run the model on (cpu or cuda)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            tau: Soft update parameter for target network
            alpha: Temperature parameter for entropy regularization
            target_update_interval: How often to update the target network
            automatic_entropy_tuning: Whether to automatically tune the entropy temperature
            seed: Random seed
            debug: Whether to print debug information
        """
        super().__init__(state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, device, seed)
        
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.debug = debug
        
        # Initialize actors
        self.leader_actor = SACActor(state_dim, action_dim_leader, hidden_size).to(device)
        self.follower1_actor = SACActor(state_dim, action_dim_follower1, hidden_size).to(device)
        self.follower2_actor = SACActor(state_dim, action_dim_follower2, hidden_size).to(device)
        
        # Initialize critics
        self.leader_critic = SACCritic(state_dim, action_dim_leader, hidden_size).to(device)
        self.follower1_critic = SACCritic(state_dim, action_dim_follower1, hidden_size).to(device)
        self.follower2_critic = SACCritic(state_dim, action_dim_follower2, hidden_size).to(device)
        
        # Initialize target critics
        self.leader_critic_target = SACCritic(state_dim, action_dim_leader, hidden_size).to(device)
        self.follower1_critic_target = SACCritic(state_dim, action_dim_follower1, hidden_size).to(device)
        self.follower2_critic_target = SACCritic(state_dim, action_dim_follower2, hidden_size).to(device)
        
        # Copy weights
        self.leader_critic_target.load_state_dict(self.leader_critic.state_dict())
        self.follower1_critic_target.load_state_dict(self.follower1_critic.state_dict())
        self.follower2_critic_target.load_state_dict(self.follower2_critic.state_dict())
        
        # Initialize optimizers
        self.leader_actor_optimizer = optim.Adam(self.leader_actor.parameters(), lr=learning_rate)
        self.follower1_actor_optimizer = optim.Adam(self.follower1_actor.parameters(), lr=learning_rate)
        self.follower2_actor_optimizer = optim.Adam(self.follower2_actor.parameters(), lr=learning_rate)
        
        self.leader_critic_optimizer = optim.Adam(self.leader_critic.parameters(), lr=learning_rate)
        self.follower1_critic_optimizer = optim.Adam(self.follower1_critic.parameters(), lr=learning_rate)
        self.follower2_critic_optimizer = optim.Adam(self.follower2_critic.parameters(), lr=learning_rate)
        
        # Initialize automatic entropy tuning if enabled
        if automatic_entropy_tuning:
            # Target entropy is -|A| for each agent
            self.leader_target_entropy = -np.log(1.0 / action_dim_leader) * 0.98
            self.follower1_target_entropy = -np.log(1.0 / action_dim_follower1) * 0.98
            self.follower2_target_entropy = -np.log(1.0 / action_dim_follower2) * 0.98
            
            # Initialize log alpha parameters
            self.leader_log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.follower1_log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.follower2_log_alpha = torch.zeros(1, requires_grad=True, device=device)
            
            # Initialize optimizers for alpha
            self.leader_alpha_optimizer = optim.Adam([self.leader_log_alpha], lr=learning_rate)
            self.follower1_alpha_optimizer = optim.Adam([self.follower1_log_alpha], lr=learning_rate)
            self.follower2_alpha_optimizer = optim.Adam([self.follower2_log_alpha], lr=learning_rate)
        
        # Initialize training step counter
        self.t_step = 0
    
    def compute_stackelberg_equilibrium(self, state: np.ndarray, 
                                  action_masks: Optional[Dict[str, np.ndarray]] = None) -> Tuple[int, int, int]:
        """
        Compute Stackelberg equilibrium using the current policy.
        In this hierarchy: Leader -> (Follower1, Follower2)
        
        Args:
            state: Current environment state
            action_masks: Dictionary of action masks for each agent
            
        Returns:
            Leader action, follower1 action, and follower2 action
        """
        # Ensure state is a tensor and properly shaped for network input
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).to(self.device)
            # Add batch dimension if needed
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
        else:
            # If already a tensor, ensure it's on the right device
            state_tensor = state.to(self.device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
        
        # Process action masks if provided
        if action_masks is not None:
            if isinstance(action_masks, dict):
                # If action_masks is a dictionary
                leader_mask = torch.tensor(action_masks['leader'], dtype=torch.bool, device=self.device)
                follower1_mask = torch.tensor(action_masks['follower1'], dtype=torch.bool, device=self.device)
                follower2_mask = torch.tensor(action_masks['follower2'], dtype=torch.bool, device=self.device)
            else:
                # If already processed by another method
                leader_mask, follower1_mask, follower2_mask = action_masks
        else:
            # If no masks provided, all actions are valid
            leader_mask = torch.ones(self.action_dim_leader, dtype=torch.bool, device=self.device)
            follower1_mask = torch.ones(self.action_dim_follower1, dtype=torch.bool, device=self.device)
            follower2_mask = torch.ones(self.action_dim_follower2, dtype=torch.bool, device=self.device)
        
        # Get action probabilities
        with torch.no_grad():
            leader_probs, _, _ = self.leader_actor(state_tensor)
            follower1_probs, _, _ = self.follower1_actor(state_tensor)
            follower2_probs, _, _ = self.follower2_actor(state_tensor)
        
        # Apply action masks by zeroing out invalid action probabilities
        masked_leader_probs = self.apply_action_mask(leader_probs, leader_mask, min_value=0)
        masked_follower1_probs = self.apply_action_mask(follower1_probs, follower1_mask, min_value=0)
        masked_follower2_probs = self.apply_action_mask(follower2_probs, follower2_mask, min_value=0)
        
        # Normalize to ensure valid probability distribution
        leader_probs_sum = masked_leader_probs.sum(dim=1, keepdim=True)
        follower1_probs_sum = masked_follower1_probs.sum(dim=1, keepdim=True)
        follower2_probs_sum = masked_follower2_probs.sum(dim=1, keepdim=True)
        
        # Avoid division by zero
        leader_probs_sum = torch.clamp(leader_probs_sum, min=1e-10)
        follower1_probs_sum = torch.clamp(follower1_probs_sum, min=1e-10)
        follower2_probs_sum = torch.clamp(follower2_probs_sum, min=1e-10)
        
        normalized_leader_probs = masked_leader_probs / leader_probs_sum
        normalized_follower1_probs = masked_follower1_probs / follower1_probs_sum
        normalized_follower2_probs = masked_follower2_probs / follower2_probs_sum
        
        # Create categorical distributions
        leader_dist = Categorical(normalized_leader_probs)
        follower1_dist = Categorical(normalized_follower1_probs)
        follower2_dist = Categorical(normalized_follower2_probs)
        
        # Sample actions
        leader_action_idx = leader_dist.sample().item()
        follower1_action_idx = follower1_dist.sample().item()
        follower2_action_idx = follower2_dist.sample().item()
        
        # Convert from index to actual action (-1 to n-2, where n is action_dim)
        leader_action = leader_action_idx - 1 if leader_action_idx > 0 else -1
        follower1_action = follower1_action_idx - 1 if follower1_action_idx > 0 else -1
        follower2_action = follower2_action_idx - 1 if follower2_action_idx > 0 else -1
        
        return leader_action, follower1_action, follower2_action
    
    def act(self, state: np.ndarray, action_masks: Optional[Dict[str, np.ndarray]] = None, 
            deterministic: bool = False) -> Tuple[int, int, int]:
        """
        Select actions using the current policy.
        
        Args:
            state: Current environment state
            action_masks: Dictionary of action masks for each agent
            deterministic: Whether to select actions deterministically
            
        Returns:
            Leader action, follower1 action, and follower2 action
        """
        # For deterministic action selection, use the Stackelberg equilibrium
        if deterministic:
            return self.compute_stackelberg_equilibrium(state, action_masks)
        
        # Ensure state is a tensor and properly shaped for network input
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).to(self.device)
            # Add batch dimension if needed
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
        else:
            # If already a tensor, ensure it's on the right device
            state_tensor = state.to(self.device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
        
        # Process action masks if provided
        if action_masks is not None:
            if isinstance(action_masks, dict):
                # If action_masks is a dictionary
                leader_mask = torch.tensor(action_masks['leader'], dtype=torch.bool, device=self.device)
                follower1_mask = torch.tensor(action_masks['follower1'], dtype=torch.bool, device=self.device)
                follower2_mask = torch.tensor(action_masks['follower2'], dtype=torch.bool, device=self.device)
            else:
                # If already processed by another method
                leader_mask, follower1_mask, follower2_mask = action_masks
        else:
            # If no masks provided, all actions are valid
            leader_mask = torch.ones(self.action_dim_leader, dtype=torch.bool, device=self.device)
            follower1_mask = torch.ones(self.action_dim_follower1, dtype=torch.bool, device=self.device)
            follower2_mask = torch.ones(self.action_dim_follower2, dtype=torch.bool, device=self.device)
        
        # Get action probabilities
        with torch.no_grad():
            leader_probs, _, _ = self.leader_actor(state_tensor)
            follower1_probs, _, _ = self.follower1_actor(state_tensor)
            follower2_probs, _, _ = self.follower2_actor(state_tensor)
        
        # Apply action masks by zeroing out invalid action probabilities
        masked_leader_probs = self.apply_action_mask(leader_probs, leader_mask, min_value=0)
        masked_follower1_probs = self.apply_action_mask(follower1_probs, follower1_mask, min_value=0)
        masked_follower2_probs = self.apply_action_mask(follower2_probs, follower2_mask, min_value=0)
        
        # Normalize to ensure valid probability distribution
        leader_probs_sum = masked_leader_probs.sum(dim=1, keepdim=True)
        follower1_probs_sum = masked_follower1_probs.sum(dim=1, keepdim=True)
        follower2_probs_sum = masked_follower2_probs.sum(dim=1, keepdim=True)
        
        # Avoid division by zero