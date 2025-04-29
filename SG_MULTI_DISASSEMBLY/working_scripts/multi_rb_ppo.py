"""
Proximal Policy Optimization (PPO) implementation for multi-robot coordination.

This module provides the PPO agent implementation that uses actor-critic
architecture for robust policy learning in the multi-robot battery disassembly task.
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

from agents.base_agent import BaseAgent


class PPONetwork(nn.Module):
    """
    PPO Actor-Critic Network implementation.
    
    This network has separate heads for policy (actor) and value function (critic).
    
    Attributes:
        input_dim (int): Dimension of the input state
        action_dim (int): Dimension of the action space
        hidden_size (int): Size of hidden layers
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int = 64):
        """
        Initialize the PPO network.
        
        Args:
            input_dim: Dimension of the input state
            action_dim: Dimension of the action space
            hidden_size: Size of hidden layers
        """
        super(PPONetwork, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        # Shared feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
            # No activation function, as we'll apply softmax when sampling
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with small random values."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Batch of states [batch_size, state_dim]
        
        Returns:
            Tuple of (action_logits, values)
            - action_logits: Logits for action probabilities [batch_size, action_dim]
            - values: State values [batch_size, 1]
        """
        # Extract features
        features = self.feature_extractor(state)
        
        # Get action logits and values
        action_logits = self.policy_head(features)
        values = self.value_head(features)
        
        return action_logits, values
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities from state.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            
        Returns:
            Action probabilities [batch_size, action_dim]
        """
        action_logits, _ = self.forward(state)
        return F.softmax(action_logits, dim=-1)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get state value.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            
        Returns:
            State values [batch_size, 1]
        """
        _, value = self.forward(state)
        return value
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate an action in a state.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            action: Batch of actions [batch_size]
            
        Returns:
            Tuple of (action_log_probs, values, entropy)
        """
        action_logits, values = self.forward(state)
        
        # Convert logits to a distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        # Get log probabilities for actions
        action_log_probs = dist.log_prob(action)
        
        # Calculate entropy for regularization
        entropy = dist.entropy().mean()
        
        return action_log_probs, values.squeeze(), entropy


class StackelbergPPOAgent(BaseAgent):
    """
    Agent implementation using PPO for Stackelberg games with three robots.
    
    This agent uses actor-critic method with proximal policy optimization for more
    stable policy learning with both discrete actions and continuous state spaces.
    
    Attributes:
        hidden_size (int): Size of hidden layers
        gamma (float): Discount factor for future rewards
        epsilon (float): Clipping parameter for PPO
        ppo_epochs (int): Number of epochs to optimize PPO objective
        gae_lambda (float): GAE lambda parameter
        clip_param (float): PPO clipping parameter
        value_coef (float): Coefficient for value function loss
        entropy_coef (float): Coefficient for entropy regularization
    """
    def __init__(self, state_dim: int, action_dim_leader: int, action_dim_follower1: int, action_dim_follower2: int,
                 hidden_size: int = 64, device: str = 'cpu', learning_rate: float = 3e-4,
                 gamma: float = 0.99, clip_param: float = 0.2, ppo_epochs: int = 10, 
                 gae_lambda: float = 0.95, value_coef: float = 0.5, entropy_coef: float = 0.01,
                 seed: int = 42, debug: bool = False):
        """
        Initialize the Stackelberg PPO agent for three robots.
        
        Args:
            state_dim: Dimension of the state space
            action_dim_leader: Dimension of the leader's action space
            action_dim_follower1: Dimension of follower1's action space
            action_dim_follower2: Dimension of follower2's action space
            hidden_size: Size of hidden layers
            device: Device to run the model on (cpu or cuda)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            clip_param: PPO clipping parameter
            ppo_epochs: Number of epochs to optimize PPO objective
            gae_lambda: GAE lambda parameter
            value_coef: Coefficient for value function loss
            entropy_coef: Coefficient for entropy regularization
            seed: Random seed
            debug: Whether to print debug information
        """
        super().__init__(state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, device, seed)
        
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.debug = debug
        
        # Initialize networks
        self.leader = PPONetwork(state_dim, action_dim_leader, hidden_size).to(device)
        self.follower1 = PPONetwork(state_dim, action_dim_follower1, hidden_size).to(device)
        self.follower2 = PPONetwork(state_dim, action_dim_follower2, hidden_size).to(device)
        
        # Initialize optimizers
        self.leader_optimizer = optim.Adam(self.leader.parameters(), lr=learning_rate)
        self.follower1_optimizer = optim.Adam(self.follower1.parameters(), lr=learning_rate)
        self.follower2_optimizer = optim.Adam(self.follower2.parameters(), lr=learning_rate)
    
    def compute_stackelberg_equilibrium(self, state: np.ndarray, 
                                        action_masks: Optional[Dict[str, np.ndarray]] = None) -> Tuple[int, int, int]:
        """
        Compute Stackelberg equilibrium using the current policies.
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
        
        # Get action probabilities and values
        with torch.no_grad():
            leader_probs = self.leader.get_action_probs(state_tensor)
            follower1_probs = self.follower1.get_action_probs(state_tensor)
            follower2_probs = self.follower2.get_action_probs(state_tensor)
        
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
        
        # Convert to numpy for easier manipulation
        leader_probs_np = normalized_leader_probs.cpu().numpy()
        follower1_probs_np = normalized_follower1_probs.cpu().numpy()
        follower2_probs_np = normalized_follower2_probs.cpu().numpy()
        
        # Choose deterministic actions for equilibrium
        leader_action_idx = np.argmax(leader_probs_np[0])
        follower1_action_idx = np.argmax(follower1_probs_np[0])
        follower2_action_idx = np.argmax(follower2_probs_np[0])
        
        # Convert from index to actual action (-1 to n-2, where n is action_dim)
        leader_action = leader_action_idx - 1 if leader_action_idx > 0 else -1
        follower1_action = follower1_action_idx - 1 if follower1_action_idx > 0 else -1
        follower2_action = follower2_action_idx - 1 if follower2_action_idx > 0 else -1
        
        if self.debug:
            print(f"Leader action: {leader_action}, Follower1 action: {follower1_action}, Follower2 action: {follower2_action}")
        
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
            leader_probs = self.leader.get_action_probs(state_tensor)
            follower1_probs = self.follower1.get_action_probs(state_tensor)
            follower2_probs = self.follower2.get_action_probs(state_tensor)
        
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
    
    def update(self, rollout_data: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Update the policy using PPO algorithm.
        
        Args:
            rollout_data: Dictionary containing rollout data
                - states: List of states
                - leader_actions, follower1_actions, follower2_actions: List of actions
                - leader_rewards, follower1_rewards, follower2_rewards: List of rewards
                - leader_values, follower1_values, follower2_values: List of state values
                - leader_log_probs, follower1_log_probs, follower2_log_probs: List of action log probs
                - dones: List of done flags
                
        Returns:
            Losses for leader, follower1, and follower2
        """
        # Convert data to tensors
        states = torch.tensor(rollout_data['states'], dtype=torch.float).to(self.device)
        leader_actions = torch.tensor(rollout_data['leader_actions'], dtype=torch.long).to(self.device)
        follower1_actions = torch.tensor(rollout_data['follower1_actions'], dtype=torch.long).to(self.device)
        follower2_actions = torch.tensor(rollout_data['follower2_actions'], dtype=torch.long).to(self.device)
        leader_rewards = torch.tensor(rollout_data['leader_rewards'], dtype=torch.float).to(self.device)
        follower1_rewards = torch.tensor(rollout_data['follower1_rewards'], dtype=torch.float).to(self.device)
        follower2_rewards = torch.tensor(rollout_data['follower2_rewards'], dtype=torch.float).to(self.device)
        leader_values = torch.tensor(rollout_data['leader_values'], dtype=torch.float).to(self.device)
        follower1_values = torch.tensor(rollout_data['follower1_values'], dtype=torch.float).to(self.device)
        follower2_values = torch.tensor(rollout_data['follower2_values'], dtype=torch.float).to(self.device)
        leader_log_probs = torch.tensor(rollout_data['leader_log_probs'], dtype=torch.float).to(self.device)
        follower1_log_probs = torch.tensor(rollout_data['follower1_log_probs'], dtype=torch.float).to(self.device)
        follower2_log_probs = torch.tensor(rollout_data['follower2_log_probs'], dtype=torch.float).to(self.device)
        dones = torch.tensor(rollout_data['dones'], dtype=torch.float).to(self.device)
        
        # Compute GAE advantages and returns
        leader_advantages = self.compute_gae(leader_rewards, leader_values, dones)
        follower1_advantages = self.compute_gae(follower1_rewards, follower1_values, dones)
        follower2_advantages = self.compute_gae(follower2_rewards, follower2_values, dones)
        
        # Compute returns (value targets)
        leader_returns = leader_advantages + leader_values
        follower1_returns = follower1_advantages + follower1_values
        follower2_returns = follower2_advantages + follower2_values
        
        # Normalize advantages
        leader_advantages = (leader_advantages - leader_advantages.mean()) / (leader_advantages.std() + 1e-8)
        follower1_advantages = (follower1_advantages - follower1_advantages.mean()) / (follower1_advantages.std() + 1e-8)
        follower2_advantages = (follower2_advantages - follower2_advantages.mean()) / (follower2_advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        leader_losses = []
        follower1_losses = []
        follower2_losses = []
        
        for _ in range(self.ppo_epochs):
            # Evaluate actions with current policy
            leader_new_log_probs, leader_new_values, leader_entropy = self.leader.evaluate(states, leader_actions)
            follower1_new_log_probs, follower1_new_values, follower1_entropy = self.follower1.evaluate(states, follower1_actions)
            follower2_new_log_probs, follower2_new_values, follower2_entropy = self.follower2.evaluate(states, follower2_actions)
            
            # Compute ratios
            leader_ratios = torch.exp(leader_new_log_probs - leader_log_probs)
            follower1_ratios = torch.exp(follower1_new_log_probs - follower1_log_probs)
            follower2_ratios = torch.exp(follower2_new_log_probs - follower2_log_probs)
            
            # Compute surrogate losses
            leader_surr1 = leader_ratios * leader_advantages
            leader_surr2 = torch.clamp(leader_ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * leader_advantages
            
            follower1_surr1 = follower1_ratios * follower1_advantages
            follower1_surr2 = torch.clamp(follower1_ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * follower1_advantages
            
            follower2_surr1 = follower2_ratios * follower2_advantages
            follower2_surr2 = torch.clamp(follower2_ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * follower2_advantages
            
            # Compute value losses
            leader_value_loss = F.mse_loss(leader_new_values, leader_returns)
            follower1_value_loss = F.mse_loss(follower1_new_values, follower1_returns)
            follower2_value_loss = F.mse_loss(follower2_new_values, follower2_returns)
            
            # Compute total losses
            leader_loss = -torch.min(leader_surr1, leader_surr2).mean() + \
                           self.value_coef * leader_value_loss - \
                           self.entropy_coef * leader_entropy
            
            follower1_loss = -torch.min(follower1_surr1, follower1_surr2).mean() + \
                               self.value_coef * follower1_value_loss - \
                               self.entropy_coef * follower1_entropy
            
            follower2_loss = -torch.min(follower2_surr1, follower2_surr2).mean() + \
                               self.value_coef * follower2_value_loss - \
                               self.entropy_coef * follower2_entropy
            
            # Update leader policy
            self.leader_optimizer.zero_grad()
            leader_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.leader.parameters(), 0.5)
            self.leader_optimizer.step()
            
            # Update follower1 policy
            self.follower1_optimizer.zero_grad()
            follower1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.follower1.parameters(), 0.5)
            self.follower1_optimizer.step()
            
            # Update follower2 policy
            self.follower2_optimizer.zero_grad()
            follower2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.follower2.parameters(), 0.5)
            self.follower2_optimizer.step()
            
            leader_losses.append(leader_loss.item())
            follower1_losses.append(follower1_loss.item())
            follower2_losses.append(follower2_loss.item())
        
        return np.mean(leader_losses), np.mean(follower1_losses), np.mean(follower2_losses)
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                   dones: torch.Tensor) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Rewards tensor [batch_size]
            values: Values tensor [batch_size]
            dones: Done flags tensor [batch_size]
            
        Returns:
            Advantages tensor [batch_size]
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        # We need to compute the next value for the last step
        # For simplicity, we'll use the last value as an approximation
        next_value = values[-1]
        
        # Reverse iteration for efficient computation
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
        
        return advantages
    
    def collect_rollout(self, env, num_steps: int = 2048) -> Dict[str, Any]:
        """
        Collect rollout data using the current policy.
        
        Args:
            env: Environment to collect data from
            num_steps: Number of steps to collect
            
        Returns:
            Dictionary containing rollout data
        """
        states = []
        leader_actions = []
        follower1_actions = []
        follower2_actions = []
        leader_rewards = []
        follower1_rewards = []
        follower2_rewards = []
        leader_values = []
        follower1_values = []
        follower2_values = []
        leader_log_probs = []
        follower1_log_probs = []
        follower2_log_probs = []
        dones = []
        
        # Reset environment
        env.reset_env()
        state, _ = env.get_current_state()
        
        for _ in range(num_steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action probabilities and values
            with torch.no_grad():
                leader_logits, leader_value = self.leader(state_tensor)
                follower1_logits, follower1_value = self.follower1(state_tensor)
                follower2_logits, follower2_value = self.follower2(state_tensor)
                
                # Create distributions
                leader_dist = Categorical(F.softmax(leader_logits, dim=-1))
                follower1_dist = Categorical(F.softmax(follower1_logits, dim=-1))
                follower2_dist = Categorical(F.softmax(follower2_logits, dim=-1))
                
                # Sample actions
                leader_action_idx = leader_dist.sample().item()
                follower1_action_idx = follower1_dist.sample().item()
                follower2_action_idx = follower2_dist.sample().item()
                
                # Get log probabilities
                leader_log_prob = leader_dist.log_prob(torch.tensor([leader_action_idx], device=self.device)).item()
                follower1_log_prob = follower1_dist.log_prob(torch.tensor([follower1_action_idx], device=self.device)).item()
                follower2_log_prob = follower2_dist.log_prob(torch.tensor([follower2_action_idx], device=self.device)).item()
            
            # Convert from index to actual action (-1 to n-2, where n is action_dim)
            leader_action = leader_action_idx - 1 if leader_action_idx > 0 else -1
            follower1_action = follower1_action_idx - 1 if follower1_action_idx > 0 else -1
            follower2_action = follower2_action_idx - 1 if follower2_action_idx > 0 else -1
            
            # Get rewards
            leader_reward, follower1_reward, follower2_reward = env.reward(state, leader_action, follower1_action, follower2_action)
            
            # Execute actions
            env.step(leader_action, follower1_action, follower2_action)
            
            # Get next state
            next_state, _ = env.get_current_state()
            
            # Check if done
            done = env.is_done()
            
            # Store data
            states.append(state)
            leader_actions.append(leader_action_idx)  # Store index for easier network evaluation
            follower1_actions.append(follower1_action_idx)
            follower2_actions.append(follower2_action_idx)
            leader_rewards.append(leader_reward)
            follower1_rewards.append(follower1_reward)
            follower2_rewards.append(follower2_reward)
            leader_values.append(leader_value.item())
            follower1_values.append(follower1_value.item())
            follower2_values.append(follower2_value.item())
            leader_log_probs.append(leader_log_prob)
            follower1_log_probs.append(follower1_log_prob)
            follower2_log_probs.append(follower2_log_prob)
            dones.append(done)
            
            # Update state
            state = next_state
            
            # If done, reset environment
            if done:
                env.reset_env()
                state, _ = env.get_current_state()
        
        return {
            'states': states,
            'leader_actions': leader_actions,
            'follower1_actions': follower1_actions,
            'follower2_actions': follower2_actions,
            'leader_rewards': leader_rewards,
            'follower1_rewards': follower1_rewards,
            'follower2_rewards': follower2_rewards,
            'leader_values': leader_values,
            'follower1_values': follower1_values,
            'follower2_values': follower2_values,
            'leader_log_probs': leader_log_probs,
            'follower1_log_probs': follower1_log_probs,
            'follower2_log_probs': follower2_log_probs,
            'dones': dones
        }
    
    def save(self, path: str) -> None:
        """
        Save the agent's state.
        
        Args:
            path: Directory to save to
        """
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.leader.state_dict(), f"{path}/leader.pt")
        torch.save(self.follower1.state_dict(), f"{path}/follower1.pt")
        torch.save(self.follower2.state_dict(), f"{path}/follower2.pt")
        
        params = {
            "hidden_size": self.hidden_size,
            "gamma": self.gamma,
            "clip_param": self.clip_param,
            "ppo_epochs": self.ppo_epochs,
            "gae_lambda": self.gae_lambda,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef
        }
        
        with open(f"{path}/params.pkl", "wb") as f:
            pickle.dump(params, f)
    
    def load(self, path: str) -> None:
        """
        Load the agent's state.
        
        Args:
            path: Directory to load from
        """
        self.leader.load_state_dict(torch.load(f"{path}/leader.pt", map_location=self.device))
        self.follower1.load_state_dict(torch.load(f"{path}/follower1.pt", map_location=self.device))
        self.follower2.load_state_dict(torch.load(f"{path}/follower2.pt", map_location=self.device))
        
        with open(f"{path}/params.pkl", "rb") as f:
            params = pickle.load(f)
            # Check if the loaded model has the same configuration
            if "hidden_size" in params and params["hidden_size"] != self.hidden_size:
                print(f"Warning: Loaded model has hidden size {params['hidden_size']}, but current model has {self.hidden_size}")