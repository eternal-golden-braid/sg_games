"""
Stackelberg SAC Implementation with Optimized Training Loop

This script provides a complete implementation of the Stackelberg Soft Actor-Critic (SAC) agent
for multi-robot coordination, along with an optimized training loop using a staged approach.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import pickle
import random
import time
import matplotlib.pyplot as plt
from collections import deque
from typing import Dict, Tuple, List, Optional, Union, Any
import copy
import json

# Set seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#################################
# Replay Buffer Implementation
#################################

class ReplayBuffer:
    """
    Replay buffer for storing and sampling experiences.
    Includes prioritized experience replay capabilities.
    """
    def __init__(self, capacity, state_dim, action_dim=3, use_per=False, alpha=0.6, beta=0.4):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum size of the buffer
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space (default 3 for leader, follower1, follower2)
            use_per: Whether to use Prioritized Experience Replay
            alpha: PER exponent parameter
            beta: PER importance sampling parameter
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 3), dtype=np.float32)  # Rewards for each agent
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.use_per = use_per
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.0001
        self.epsilon = 1e-5  # Small constant to avoid zero priority
        
        self.ptr = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Rewards received (for all agents)
            next_state: Next state
            done: Whether the episode terminated
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        # Set max priority for new experiences
        if self.use_per:
            # Set max priority for new experiences
            max_priority = np.max(self.priorities) if self.size > 0 else 1.0
            self.priorities[self.ptr] = max_priority
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        Uses prioritized sampling if use_per is True.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of batched experiences
        """
        if self.use_per:
            # PER sampling
            if self.size < batch_size:
                indices = np.random.randint(0, self.size, size=batch_size)
            else:
                # Calculate sampling probabilities
                priorities = self.priorities[:self.size] ** self.alpha
                prob = priorities / (np.sum(priorities) + self.epsilon)
                
                # Sample based on priorities
                indices = np.random.choice(self.size, batch_size, p=prob)
                
                # Compute importance sampling weights
                weights = (self.size * prob[indices]) ** (-self.beta)
                weights /= np.max(weights)
                
                # Increment beta for annealing
                self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            # Uniform sampling
            if self.size < batch_size:
                indices = np.random.randint(0, self.size, size=batch_size)
            else:
                indices = np.random.choice(self.size, batch_size, replace=False)
            weights = np.ones(batch_size)  # All weights are equal
        
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]
        
        # Convert to torch tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities for PER.
        
        Args:
            indices: Indices of sampled experiences
            td_errors: TD errors for each experience
        """
        if self.use_per:
            for idx, td_error in zip(indices, td_errors):
                self.priorities[idx] = abs(td_error) + self.epsilon
    
    def __len__(self):
        return self.size

#################################
# Network Implementations
#################################

class SACStackelbergCritic(nn.Module):
    """
    Centralized critic that takes joint actions as input for Stackelberg games.
    """
    def __init__(self, state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, hidden_size):
        super(SACStackelbergCritic, self).__init__()
        
        # Total action dimension is the sum of all agents' action dimensions
        self.joint_action_dim = 3  # One action per agent
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + self.joint_action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Q2 architecture (for min-Q trick)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + self.joint_action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state, joint_actions):
        """
        Args:
            state: Current state [batch_size, state_dim]
            joint_actions: Joint actions [batch_size, total_action_dim]
        
        Returns:
            q1, q2: Two Q-value estimates
        """
        x = torch.cat([state, joint_actions], dim=1)
        return self.q1(x), self.q2(x)
    
    def q1_value(self, state, joint_actions):
        """Get Q1 value only"""
        x = torch.cat([state, joint_actions], dim=1)
        return self.q1(x)


class SACActor(nn.Module):
    """Actor network for SAC with categorical actions."""
    
    def __init__(self, state_dim, action_dim, hidden_size):
        super(SACActor, self).__init__()
        
        self.action_dim = action_dim
        
        # Actor network with 3 hidden layers for more expressive power
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization for stability
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)
        
    def forward(self, state):
        """
        Forward pass through the actor network.
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            action_logits: Raw action logits
            action_probs: Action probabilities
            action_log_probs: Log probabilities of actions
        """
        # Handle single and batch inputs
        batch_mode = True
        if state.dim() == 1:
            state = state.unsqueeze(0)
            batch_mode = False
        
        x = F.relu(self.fc1(state))
        # Only apply batch norm in batch mode
        if batch_mode and x.size(0) > 1:
            x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_logits = self.fc4(x)
        
        action_probs = F.softmax(action_logits, dim=-1)
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        
        return action_logits, action_probs, action_log_probs


class StackelbergSACAgent:
    """
    Agent implementation using Soft Actor-Critic for Stackelberg games with three robots.
    
    This agent uses maximum entropy reinforcement learning for efficient exploration
    and improved policy robustness. It computes true Stackelberg equilibria using
    bilevel optimization.
    """
    def __init__(self, state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2,
                 hidden_size=128, learning_rate=1e-4, gamma=0.995, tau=0.001, alpha=0.2,
                 target_update_interval=1, automatic_entropy_tuning=True, use_per=False):
        """
        Initialize the Stackelberg SAC agent for three robots.
        
        Args:
            state_dim: Dimension of the state space
            action_dim_leader: Dimension of the leader's action space
            action_dim_follower1: Dimension of follower1's action space
            action_dim_follower2: Dimension of follower2's action space
            hidden_size: Size of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            tau: Soft update parameter for target network
            alpha: Temperature parameter for entropy regularization
            target_update_interval: How often to update the target network
            automatic_entropy_tuning: Whether to automatically tune the entropy temperature
            use_per: Whether to use Prioritized Experience Replay
        """
        self.state_dim = state_dim
        self.action_dim_leader = action_dim_leader
        self.action_dim_follower1 = action_dim_follower1
        self.action_dim_follower2 = action_dim_follower2
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.device = device
        self.use_per = use_per
        
        # Initialize actors
        self.leader_actor = SACActor(state_dim, action_dim_leader, hidden_size).to(device)
        self.follower1_actor = SACActor(state_dim, action_dim_follower1, hidden_size).to(device)
        self.follower2_actor = SACActor(state_dim, action_dim_follower2, hidden_size).to(device)
        
        # Initialize centralized critics for each agent
        self.leader_critic = SACStackelbergCritic(
            state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, hidden_size
        ).to(device)
        
        self.follower1_critic = SACStackelbergCritic(
            state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, hidden_size
        ).to(device)
        
        self.follower2_critic = SACStackelbergCritic(
            state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, hidden_size
        ).to(device)
        
        # Initialize target critics
        self.leader_critic_target = SACStackelbergCritic(
            state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, hidden_size
        ).to(device)
        
        self.follower1_critic_target = SACStackelbergCritic(
            state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, hidden_size
        ).to(device)
        
        self.follower2_critic_target = SACStackelbergCritic(
            state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, hidden_size
        ).to(device)
        
        # Copy weights to target networks
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
            # Using a lower multiplier for more exploitation
            self.leader_target_entropy = -np.log(1.0 / action_dim_leader) * 0.5
            self.follower1_target_entropy = -np.log(1.0 / action_dim_follower1) * 0.5
            self.follower2_target_entropy = -np.log(1.0 / action_dim_follower2) * 0.5
            
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
        
        # Initialize a running average of rewards for adaptive entropy
        self.reward_history = {
            'leader': deque(maxlen=100),
            'follower1': deque(maxlen=100),
            'follower2': deque(maxlen=100)
        }
    
    def idx_to_action(self, idx, action_dim):
        """
        Convert an index to an actual action value.
        Always maps the last index to the idle action (-1).
        
        Args:
            idx: Action index
            action_dim: Dimension of the action space
            
        Returns:
            action: Actual action value
        """
        return -1 if idx == action_dim - 1 else idx

    def action_to_idx(self, action, action_dim):
        """
        Convert an action value to its index.
        Always maps the idle action (-1) to the last index.
        
        Args:
            action: Action value
            action_dim: Dimension of the action space
            
        Returns:
            idx: Action index
        """
        return action_dim - 1 if action == -1 else action
    
    def apply_mask_to_logits(self, logits, mask):
        """
        Applies mask to logits using the -inf + softmax approach
        
        Args:
            logits: Raw action logits
            mask: Boolean mask (True for valid actions)
            
        Returns:
            masked_probs: Probability distribution after masking
        """
        # Apply mask by setting invalid actions to -inf
        masked_logits = logits.clone()
        masked_logits.masked_fill_(~mask, float('-inf'))
        
        # Apply softmax to get probabilities
        probs = F.softmax(masked_logits, dim=-1)
        
        return probs
    
    def create_joint_action_tensor(self, leader_action, follower1_action, follower2_action, batch_size=1):
        """
        Create a joint action tensor for critic input.
        
        Args:
            leader_action: Leader action (can be a tensor or scalar)
            follower1_action: Follower 1 action (can be a tensor or scalar)
            follower2_action: Follower 2 action (can be a tensor or scalar)
            batch_size: Size of batch (only used if inputs are scalars)
            
        Returns:
            joint_action: Joint action tensor
        """
        # Handle different input types
        if isinstance(leader_action, (int, float)):
            # Convert scalars to tensors
            leader_tensor = torch.full((batch_size, 1), leader_action, dtype=torch.float32, device=self.device)
            follower1_tensor = torch.full((batch_size, 1), follower1_action, dtype=torch.float32, device=self.device) 
            follower2_tensor = torch.full((batch_size, 1), follower2_action, dtype=torch.float32, device=self.device)
        else:
            # Ensure tensors have correct dimensions
            leader_tensor = leader_action if leader_action.dim() > 1 else leader_action.unsqueeze(1)
            follower1_tensor = follower1_action if follower1_action.dim() > 1 else follower1_action.unsqueeze(1)
            follower2_tensor = follower2_action if follower2_action.dim() > 1 else follower2_action.unsqueeze(1)
        
        # Concatenate to form joint action
        joint_action = torch.cat([leader_tensor, follower1_tensor, follower2_tensor], dim=1)
        
        return joint_action
    
    def predict_follower_responses(self, state_tensor, leader_action, action_masks=None):
        """
        Efficiently predict follower best responses to a leader action.
        
        Args:
            state_tensor: State tensor
            leader_action: Leader's action
            action_masks: Dictionary of action masks for each agent
            
        Returns:
            Best response actions for follower1 and follower2
        """
        # Process masks if provided
        if action_masks is not None:
            follower1_mask = torch.tensor(action_masks['follower1'], dtype=torch.bool, device=self.device)
            follower2_mask = torch.tensor(action_masks['follower2'], dtype=torch.bool, device=self.device)
        else:
            follower1_mask = torch.ones(self.action_dim_follower1, dtype=torch.bool, device=self.device)
            follower2_mask = torch.ones(self.action_dim_follower2, dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            # Get follower policy distributions
            _, f1_probs, _ = self.follower1_actor(state_tensor)
            _, f2_probs, _ = self.follower2_actor(state_tensor)
            
            # Apply masks
            f1_probs = self.apply_mask_to_logits(f1_probs, follower1_mask)
            f2_probs = self.apply_mask_to_logits(f2_probs, follower2_mask)
            
            # Get most likely actions as best responses
            f1_idx = torch.argmax(f1_probs, dim=1).item()
            f2_idx = torch.argmax(f2_probs, dim=1).item()
            
            # Convert to actual actions
            f1_action = self.idx_to_action(f1_idx, self.action_dim_follower1)
            f2_action = self.idx_to_action(f2_idx, self.action_dim_follower2)
            
            return f1_action, f2_action
    
    def compute_efficient_stackelberg(self, state, action_masks=None):
        """
        Efficient approximation of Stackelberg equilibrium during training.
        More computationally efficient than full bilevel optimization.
        
        Args:
            state: Environment state
            action_masks: Action masks for each agent
            
        Returns:
            Leader action, follower1 action, follower2 action
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Process masks if provided
        if action_masks is not None:
            leader_mask = torch.tensor(action_masks['leader'], dtype=torch.bool, device=self.device)
        else:
            leader_mask = torch.ones(self.action_dim_leader, dtype=torch.bool, device=self.device)
        
        # Get leader's action distribution
        with torch.no_grad():
            leader_logits, _, _ = self.leader_actor(state_tensor)
            
            # Apply mask
            leader_logits = leader_logits.clone()
            leader_logits.masked_fill_(~leader_mask, float('-inf'))
            
            # Get top k leader actions (for computational efficiency)
            k = min(5, leader_logits.size(-1))
            top_leader_vals, top_leader_idx = torch.topk(leader_logits, k)
            
            best_value = float('-inf')
            best_leader_action = None
            best_f1_action = None
            best_f2_action = None
            
            # For each potential leader action
            for i in range(k):
                leader_idx = top_leader_idx[0, i].item()
                leader_action = self.idx_to_action(leader_idx, self.action_dim_leader)
                
                # Get follower best responses
                f1_action, f2_action = self.predict_follower_responses(state_tensor, leader_action, action_masks)
                
                # Evaluate leader's value with these responses
                joint_action = self.create_joint_action_tensor(leader_action, f1_action, f2_action)
                value = self.leader_critic.q1_value(state_tensor, joint_action).item()
                
                if value > best_value:
                    best_value = value
                    best_leader_action = leader_action
                    best_f1_action = f1_action
                    best_f2_action = f2_action
            
            # If no valid action found, choose randomly
            if best_leader_action is None:
                valid_indices = torch.where(leader_mask)[0]
                if len(valid_indices) > 0:
                    random_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))].item()
                    best_leader_action = self.idx_to_action(random_idx, self.action_dim_leader)
                else:
                    best_leader_action = 0  # Fallback
                
                f1_action, f2_action = self.predict_follower_responses(state_tensor, best_leader_action, action_masks)
                best_f1_action = f1_action
                best_f2_action = f2_action
            
            return best_leader_action, best_f1_action, best_f2_action
    
    def compute_stackelberg_equilibrium(self, state, action_masks=None):
        """
        Compute Stackelberg equilibrium using bilevel optimization.
        This is a more computationally expensive but more accurate version.
        
        Args:
            state: Environment state
            action_masks: Action masks for each agent
            
        Returns:
            Leader action, follower1 action, follower2 action
        """
        # Use the efficient approximation during training
        if self.t_step < 10000:
            return self.compute_efficient_stackelberg(state, action_masks)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Process masks
        if action_masks is not None:
            leader_mask = torch.tensor(action_masks['leader'], dtype=torch.bool, device=self.device)
            follower1_mask = torch.tensor(action_masks['follower1'], dtype=torch.bool, device=self.device)
            follower2_mask = torch.tensor(action_masks['follower2'], dtype=torch.bool, device=self.device)
        else:
            leader_mask = torch.ones(self.action_dim_leader, dtype=torch.bool, device=self.device)
            follower1_mask = torch.ones(self.action_dim_follower1, dtype=torch.bool, device=self.device)
            follower2_mask = torch.ones(self.action_dim_follower2, dtype=torch.bool, device=self.device)
        
        # Initialize variables to track best leader action and value
        best_leader_action = -1
        best_leader_value = float('-inf')
        best_follower1_action = -1
        best_follower2_action = -1
        
        with torch.no_grad():
            # For each valid leader action
            for leader_idx in range(self.action_dim_leader):
                # Skip invalid actions
                if not leader_mask[leader_idx]:
                    continue
                    
                # Convert index to actual action (-1 if last index, otherwise index)
                leader_action = self.idx_to_action(leader_idx, self.action_dim_leader)
                
                # Find follower1's best response to this leader action
                best_f1_idx = -1
                best_f1_value = float('-inf')
                
                for f1_idx in range(self.action_dim_follower1):
                    if not follower1_mask[f1_idx]:
                        continue
                        
                    f1_action = self.idx_to_action(f1_idx, self.action_dim_follower1)
                    
                    # Create partial joint action for follower1 evaluation
                    joint_f1_eval = self.create_joint_action_tensor(
                        leader_action, 
                        f1_action, 
                        0  # Placeholder
                    )
                    
                    # Evaluate follower1's Q-value for this action
                    f1_value = self.follower1_critic.q1_value(state_tensor, joint_f1_eval).item()
                    
                    if f1_value > best_f1_value:
                        best_f1_value = f1_value
                        best_f1_idx = f1_idx
                
                # Get actual follower1 action from index
                f1_best_action = self.idx_to_action(best_f1_idx, self.action_dim_follower1)
                
                # Find follower2's best response to leader and follower1's actions
                best_f2_idx = -1
                best_f2_value = float('-inf')
                
                for f2_idx in range(self.action_dim_follower2):
                    if not follower2_mask[f2_idx]:
                        continue
                        
                    f2_action = self.idx_to_action(f2_idx, self.action_dim_follower2)
                    
                    # Create joint action for follower2 evaluation
                    joint_f2_eval = self.create_joint_action_tensor(
                        leader_action,
                        f1_best_action,
                        f2_action
                    )
                    
                    # Evaluate follower2's Q-value for this action
                    f2_value = self.follower2_critic.q1_value(state_tensor, joint_f2_eval).item()
                    
                    if f2_value > best_f2_value:
                        best_f2_value = f2_value
                        best_f2_idx = f2_idx
                
                # Get actual follower2 action from index
                f2_best_action = self.idx_to_action(best_f2_idx, self.action_dim_follower2)
                
                # Create full joint action with both followers' best responses
                joint_leader_eval = self.create_joint_action_tensor(
                    leader_action,
                    f1_best_action,
                    f2_best_action
                )
                
                # Evaluate leader's value with followers' best responses
                leader_value = self.leader_critic.q1_value(state_tensor, joint_leader_eval).item()
                
                # Update best leader action if this gives better value
                if leader_value > best_leader_value:
                    best_leader_value = leader_value
                    best_leader_action = leader_action
                    best_follower1_action = f1_best_action
                    best_follower2_action = f2_best_action
        
        return best_leader_action, best_follower1_action, best_follower2_action
    
    def act(self, state, action_masks=None, deterministic=False, exploration_factor=1.0):
        """
        Select actions using the current policy.
        
        Args:
            state: Current environment state
            action_masks: Dictionary of action masks for each agent
            deterministic: Whether to select actions deterministically
            exploration_factor: Factor to control exploration (0.0-1.0)
            
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
            leader_mask = torch.tensor(action_masks['leader'], dtype=torch.bool, device=self.device)
            follower1_mask = torch.tensor(action_masks['follower1'], dtype=torch.bool, device=self.device)
            follower2_mask = torch.tensor(action_masks['follower2'], dtype=torch.bool, device=self.device)
        else:
            # If no masks provided, all actions are valid
            leader_mask = torch.ones(self.action_dim_leader, dtype=torch.bool, device=self.device)
            follower1_mask = torch.ones(self.action_dim_follower1, dtype=torch.bool, device=self.device)
            follower2_mask = torch.ones(self.action_dim_follower2, dtype=torch.bool, device=self.device)
        
        # Use epsilon-greedy approach with Stackelberg when exploration factor is low
        if random.random() > exploration_factor:
            # Use Stackelberg equilibrium with probability (1 - exploration_factor)
            return self.compute_efficient_stackelberg(state, action_masks)
        
        # Otherwise use stochastic policy
        with torch.no_grad():
            # Get action logits from actors
            leader_logits, _, _ = self.leader_actor(state_tensor)
            follower1_logits, _, _ = self.follower1_actor(state_tensor)
            follower2_logits, _, _ = self.follower2_actor(state_tensor)
            
            # Apply masks using -inf + softmax approach
            leader_probs = self.apply_mask_to_logits(leader_logits, leader_mask)
            follower1_probs = self.apply_mask_to_logits(follower1_logits, follower1_mask)
            follower2_probs = self.apply_mask_to_logits(follower2_logits, follower2_mask)
            
            # Create categorical distributions
            leader_dist = Categorical(leader_probs)
            follower1_dist = Categorical(follower1_probs)
            follower2_dist = Categorical(follower2_probs)
            
            # Sample actions
            leader_idx = leader_dist.sample().item()
            follower1_idx = follower1_dist.sample().item()
            follower2_idx = follower2_dist.sample().item()
            
            # Convert indices to actual action values using consistent mapping
            leader_action = self.idx_to_action(leader_idx, self.action_dim_leader)
            follower1_action = self.idx_to_action(follower1_idx, self.action_dim_follower1)
            follower2_action = self.idx_to_action(follower2_idx, self.action_dim_follower2)
            
            return leader_action, follower1_action, follower2_action
    
    def update_followers_only(self, experiences):
        """
        Update only follower policies, keeping leader policy fixed.
        Used during the first training phase.
        
        Args:
            experiences: Batch of experiences
            
        Returns:
            Dictionary of loss metrics
        """
        # Unpack the experience tuple
        if self.use_per:
            states, actions, rewards, next_states, dones, weights, indices = experiences
        else:
            states, actions, rewards, next_states, dones = experiences
            weights = torch.ones_like(rewards[:, 0]).to(self.device)
            indices = None
        
        # Extract individual actions and rewards for each agent
        leader_actions = actions[:, 0].unsqueeze(1)
        follower1_actions = actions[:, 1].unsqueeze(1)
        follower2_actions = actions[:, 2].unsqueeze(1)
        
        follower1_rewards = rewards[:, 1].unsqueeze(1)
        follower2_rewards = rewards[:, 2].unsqueeze(1)
        
        # Create joint actions for critics
        joint_actions = actions  # Shape: [batch_size, 3]
        
        # Get current alpha values for followers
        if self.automatic_entropy_tuning:
            follower1_alpha = self.follower1_log_alpha.exp()
            follower2_alpha = self.follower2_log_alpha.exp()
        else:
            follower1_alpha = torch.tensor(self.alpha).to(self.device)
            follower2_alpha = torch.tensor(self.alpha).to(self.device)
        
        # Dictionary to track various loss metrics
        metrics = {}
        
        # ==============================
        # Update Follower Critic Networks
        # ==============================
        
        with torch.no_grad():
            # Get leader action distribution for next state
            leader_next_logits, _, _ = self.leader_actor(next_states)
            leader_next_probs = F.softmax(leader_next_logits, dim=-1)
            leader_next_dist = Categorical(leader_next_probs)
            leader_next_actions_idx = leader_next_dist.sample()
            
            # Convert to actual action values
            leader_next_actions = torch.tensor([
                self.idx_to_action(idx.item(), self.action_dim_leader) 
                for idx in leader_next_actions_idx
            ], device=self.device).unsqueeze(1)
            
            # Get follower action distributions for next state
            _, follower1_next_probs, follower1_next_log_probs = self.follower1_actor(next_states)
            _, follower2_next_probs, follower2_next_log_probs = self.follower2_actor(next_states)
            
            # Sample follower actions
            follower1_next_dist = Categorical(follower1_next_probs)
            follower2_next_dist = Categorical(follower2_next_probs)
            
            follower1_next_actions_idx = follower1_next_dist.sample()
            follower2_next_actions_idx = follower2_next_dist.sample()
            
            # Convert to actual action values
            follower1_next_actions = torch.tensor([
                self.idx_to_action(idx.item(), self.action_dim_follower1) 
                for idx in follower1_next_actions_idx
            ], device=self.device).unsqueeze(1)
            
            follower2_next_actions = torch.tensor([
                self.idx_to_action(idx.item(), self.action_dim_follower2) 
                for idx in follower2_next_actions_idx
            ], device=self.device).unsqueeze(1)
            
            # Create joint next actions
            next_joint_actions = torch.cat([
                leader_next_actions, follower1_next_actions, follower2_next_actions
            ], dim=1)
            
            # Get log probs of selected actions
            follower1_next_log_prob = torch.gather(
                follower1_next_log_probs, 1, 
                torch.tensor([
                    self.action_to_idx(a.item(), self.action_dim_follower1) 
                    for a in follower1_next_actions
                ], device=self.device).unsqueeze(1)
            )
            
            follower2_next_log_prob = torch.gather(
                follower2_next_log_probs, 1, 
                torch.tensor([
                    self.action_to_idx(a.item(), self.action_dim_follower2) 
                    for a in follower2_next_actions
                ], device=self.device).unsqueeze(1)
            )
            
            # Compute target Q-values for followers
            follower1_q1_next, follower1_q2_next = self.follower1_critic_target(next_states, next_joint_actions)
            follower2_q1_next, follower2_q2_next = self.follower2_critic_target(next_states, next_joint_actions)
            
            # Take the minimum of two Q-values
            follower1_q_next = torch.min(follower1_q1_next, follower1_q2_next)
            follower2_q_next = torch.min(follower2_q1_next, follower2_q2_next)
            
            # Add entropy regularization
            follower1_q_target = follower1_rewards + (1 - dones) * self.gamma * (follower1_q_next - follower1_alpha * follower1_next_log_prob)
            follower2_q_target = follower2_rewards + (1 - dones) * self.gamma * (follower2_q_next - follower2_alpha * follower2_next_log_prob)
        
        # Compute current Q-values
        follower1_q1, follower1_q2 = self.follower1_critic(states, joint_actions)
        follower2_q1, follower2_q2 = self.follower2_critic(states, joint_actions)
        
        # Compute losses with PER weights if using PER
        follower1_q1_loss = (weights * F.mse_loss(follower1_q1, follower1_q_target, reduction='none')).mean()
        follower1_q2_loss = (weights * F.mse_loss(follower1_q2, follower1_q_target, reduction='none')).mean()
        follower1_critic_loss = follower1_q1_loss + follower1_q2_loss
        
        follower2_q1_loss = (weights * F.mse_loss(follower2_q1, follower2_q_target, reduction='none')).mean()
        follower2_q2_loss = (weights * F.mse_loss(follower2_q2, follower2_q_target, reduction='none')).mean()
        follower2_critic_loss = follower2_q1_loss + follower2_q2_loss
        
        # Update follower critic networks
        self.follower1_critic_optimizer.zero_grad()
        follower1_critic_loss.backward()
        self.follower1_critic_optimizer.step()
        
        self.follower2_critic_optimizer.zero_grad()
        follower2_critic_loss.backward()
        self.follower2_critic_optimizer.step()
        
        # Save critic loss metrics
        metrics['follower1_critic_loss'] = follower1_critic_loss.item()
        metrics['follower2_critic_loss'] = follower2_critic_loss.item()
        
        # ==============================
        # Update Follower Actor Networks
        # ==============================
        
        # Update Follower1's Actor
        _, follower1_probs, follower1_log_probs = self.follower1_actor(states)
        follower1_dist = Categorical(follower1_probs)
        follower1_entropy = follower1_dist.entropy().unsqueeze(1)
        
        # Evaluate follower1's Q-value
        joint_actions_f1 = joint_actions.clone()
        follower1_q1_pi, _ = self.follower1_critic(states, joint_actions_f1)
        
        # Compute follower1's actor loss
        follower1_actor_loss = (follower1_alpha * follower1_log_probs.gather(
            1, torch.tensor([
                self.action_to_idx(a.item(), self.action_dim_follower1) 
                for a in follower1_actions
            ], device=self.device).unsqueeze(1)
        ) - follower1_q1_pi).mean()
        
        # Update follower1's actor
        self.follower1_actor_optimizer.zero_grad()
        follower1_actor_loss.backward()
        self.follower1_actor_optimizer.step()
        
        # Update Follower2's Actor
        _, follower2_probs, follower2_log_probs = self.follower2_actor(states)
        follower2_dist = Categorical(follower2_probs)
        follower2_entropy = follower2_dist.entropy().unsqueeze(1)
        
        # Evaluate follower2's Q-value
        joint_actions_f2 = joint_actions.clone()
        follower2_q1_pi, _ = self.follower2_critic(states, joint_actions_f2)
        
        # Compute follower2's actor loss
        follower2_actor_loss = (follower2_alpha * follower2_log_probs.gather(
            1, torch.tensor([
                self.action_to_idx(a.item(), self.action_dim_follower2) 
                for a in follower2_actions
            ], device=self.device).unsqueeze(1)
        ) - follower2_q1_pi).mean()
        
        # Update follower2's actor
        self.follower2_actor_optimizer.zero_grad()
        follower2_actor_loss.backward()
        self.follower2_actor_optimizer.step()
        
        # Save actor loss metrics
        metrics['follower1_actor_loss'] = follower1_actor_loss.item()
        metrics['follower2_actor_loss'] = follower2_actor_loss.item()
        
        # ==============================
        # Update Entropy Coefficients
        # ==============================
        
        if self.automatic_entropy_tuning:
            # Follower1 entropy tuning
            follower1_entropy_loss = -(self.follower1_log_alpha * (follower1_entropy + self.follower1_target_entropy).detach()).mean()
            self.follower1_alpha_optimizer.zero_grad()
            follower1_entropy_loss.backward()
            self.follower1_alpha_optimizer.step()
            
            # Follower2 entropy tuning
            follower2_entropy_loss = -(self.follower2_log_alpha * (follower2_entropy + self.follower2_target_entropy).detach()).mean()
            self.follower2_alpha_optimizer.zero_grad()
            follower2_entropy_loss.backward()
            self.follower2_alpha_optimizer.step()
            
            # Save entropy metrics
            metrics['follower1_alpha'] = follower1_alpha.item()
            metrics['follower2_alpha'] = follower2_alpha.item()
            metrics['follower1_entropy'] = follower1_entropy.mean().item()
            metrics['follower2_entropy'] = follower2_entropy.mean().item()
        
        # ==============================
        # Update Target Networks
        # ==============================
        
        # Soft update follower target networks
        for target_param, local_param in zip(self.follower1_critic_target.parameters(), self.follower1_critic.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            
        for target_param, local_param in zip(self.follower2_critic_target.parameters(), self.follower2_critic.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
        # Update PER priorities if using PER
        if self.use_per and indices is not None:
            with torch.no_grad():
                follower1_td_error = torch.abs(follower1_q1 - follower1_q_target).cpu().numpy().flatten()
                follower2_td_error = torch.abs(follower2_q1 - follower2_q_target).cpu().numpy().flatten()
                # Use max td_error among followers
                td_errors = np.maximum(follower1_td_error, follower2_td_error)
                replay_buffer.update_priorities(indices, td_errors)
        
        self.t_step += 1
        
        return metrics
    
    def update_leader_only(self, experiences):
        """
        Update only leader policy, keeping follower policies fixed.
        Used during the second training phase.
        
        Args:
            experiences: Batch of experiences
            
        Returns:
            Dictionary of loss metrics
        """
        # Unpack the experience tuple
        if self.use_per:
            states, actions, rewards, next_states, dones, weights, indices = experiences
        else:
            states, actions, rewards, next_states, dones = experiences
            weights = torch.ones_like(rewards[:, 0]).to(self.device)
            indices = None
        
        # Extract individual actions and rewards for each agent
        leader_actions = actions[:, 0].unsqueeze(1)
        leader_rewards = rewards[:, 0].unsqueeze(1)
        
        # Create joint actions for critics
        joint_actions = actions  # Shape: [batch_size, 3]
        
        # Get current alpha value for leader
        if self.automatic_entropy_tuning:
            leader_alpha = self.leader_log_alpha.exp()
        else:
            leader_alpha = torch.tensor(self.alpha).to(self.device)
        
        # Dictionary to track various loss metrics
        metrics = {}
        
        # ==============================
        # Update Leader Critic Network
        # ==============================
        
        with torch.no_grad():
            # Get leader action distribution for next state
            _, leader_next_probs, leader_next_log_probs = self.leader_actor(next_states)
            leader_next_dist = Categorical(leader_next_probs)
            leader_next_actions_idx = leader_next_dist.sample()
            
            # Convert to actual action values
            leader_next_actions = torch.tensor([
                self.idx_to_action(idx.item(), self.action_dim_leader) 
                for idx in leader_next_actions_idx
            ], device=self.device).unsqueeze(1)
            
            # Get follower action distributions for next state (keep fixed)
            _, follower1_next_probs, _ = self.follower1_actor(next_states)
            _, follower2_next_probs, _ = self.follower2_actor(next_states)
            
            # Sample follower actions
            follower1_next_dist = Categorical(follower1_next_probs)
            follower2_next_dist = Categorical(follower2_next_probs)
            
            follower1_next_actions_idx = follower1_next_dist.sample()
            follower2_next_actions_idx = follower2_next_dist.sample()
            
            # Convert to actual action values
            follower1_next_actions = torch.tensor([
                self.idx_to_action(idx.item(), self.action_dim_follower1) 
                for idx in follower1_next_actions_idx
            ], device=self.device).unsqueeze(1)
            
            follower2_next_actions = torch.tensor([
                self.idx_to_action(idx.item(), self.action_dim_follower2) 
                for idx in follower2_next_actions_idx
            ], device=self.device).unsqueeze(1)
            
            # Create joint next actions
            next_joint_actions = torch.cat([
                leader_next_actions, follower1_next_actions, follower2_next_actions
            ], dim=1)
            
            # Get log probs of selected leader actions
            leader_next_log_prob = torch.gather(
                leader_next_log_probs, 1, 
                torch.tensor([
                    self.action_to_idx(a.item(), self.action_dim_leader) 
                    for a in leader_next_actions
                ], device=self.device).unsqueeze(1)
            )
            
            # Compute target Q-value for leader
            leader_q1_next, leader_q2_next = self.leader_critic_target(next_states, next_joint_actions)
            
            # Take the minimum of two Q-values
            leader_q_next = torch.min(leader_q1_next, leader_q2_next)
            
            # Add entropy regularization
            leader_q_target = leader_rewards + (1 - dones) * self.gamma * (leader_q_next - leader_alpha * leader_next_log_prob)
        
        # Compute current Q-values
        leader_q1, leader_q2 = self.leader_critic(states, joint_actions)
        
        # Compute losses with PER weights if using PER
        leader_q1_loss = (weights * F.mse_loss(leader_q1, leader_q_target, reduction='none')).mean()
        leader_q2_loss = (weights * F.mse_loss(leader_q2, leader_q_target, reduction='none')).mean()
        leader_critic_loss = leader_q1_loss + leader_q2_loss
        
        # Update leader critic network
        self.leader_critic_optimizer.zero_grad()
        leader_critic_loss.backward()
        self.leader_critic_optimizer.step()
        
        # Save critic loss metrics
        metrics['leader_critic_loss'] = leader_critic_loss.item()
        
        # ==============================
        # Update Leader Actor Network
        # ==============================
        
        # Get leader policy distribution
        _, leader_probs, leader_log_probs = self.leader_actor(states)
        leader_dist = Categorical(leader_probs)
        leader_entropy = leader_dist.entropy().unsqueeze(1)
        
        # For each state, predict follower best responses to leader actions
        # This allows leader to learn optimal Stackelberg strategy
        batch_size = states.size(0)
        leader_stackelberg_q = torch.zeros(batch_size, 1, device=self.device)
        
        for i in range(batch_size):
            state_i = states[i:i+1]
            leader_action_i = leader_actions[i].item()
            
            # Predict follower responses using their current policies
            f1_action, f2_action = self.predict_follower_responses(state_i, leader_action_i)
            
            # Create joint action with predicted follower responses
            joint_action_i = self.create_joint_action_tensor(leader_action_i, f1_action, f2_action)
            
            # Evaluate leader's Q-value
            leader_stackelberg_q[i] = self.leader_critic.q1_value(state_i, joint_action_i)
        
        # Compute leader's actor loss with predicted follower responses
        leader_actor_loss = (leader_alpha * leader_log_probs.gather(
            1, torch.tensor([
                self.action_to_idx(a.item(), self.action_dim_leader) 
                for a in leader_actions
            ], device=self.device).unsqueeze(1)
        ) - leader_stackelberg_q).mean()
        
        # Update leader's actor
        self.leader_actor_optimizer.zero_grad()
        leader_actor_loss.backward()
        self.leader_actor_optimizer.step()
        
        # Save actor loss metrics
        metrics['leader_actor_loss'] = leader_actor_loss.item()
        
        # ==============================
        # Update Entropy Coefficient
        # ==============================
        
        if self.automatic_entropy_tuning:
            # Leader entropy tuning
            leader_entropy_loss = -(self.leader_log_alpha * (leader_entropy + self.leader_target_entropy).detach()).mean()
            self.leader_alpha_optimizer.zero_grad()
            leader_entropy_loss.backward()
            self.leader_alpha_optimizer.step()
            
            # Save entropy metrics
            metrics['leader_alpha'] = leader_alpha.item()
            metrics['leader_entropy'] = leader_entropy.mean().item()
        
        # ==============================
        # Update Target Network
        # ==============================
        
        # Soft update leader target network
        for target_param, local_param in zip(self.leader_critic_target.parameters(), self.leader_critic.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
        # Update PER priorities if using PER
        if self.use_per and indices is not None:
            with torch.no_grad():
                leader_td_error = torch.abs(leader_q1 - leader_q_target).cpu().numpy().flatten()
                replay_buffer.update_priorities(indices, leader_td_error)
        
        self.t_step += 1
        
        return metrics
    
    def update(self, experiences, followers_importance=1.0, leader_importance=1.0):
        """
        Update all agent networks using a batch of experiences.
        
        Args:
            experiences: Batch of experiences
            followers_importance: Importance weight for follower updates
            leader_importance: Importance weight for leader updates
            
        Returns:
            Dictionary of loss metrics
        """
        # Unpack the experience tuple
        if self.use_per:
            states, actions, rewards, next_states, dones, weights, indices = experiences
        else:
            states, actions, rewards, next_states, dones = experiences
            weights = torch.ones_like(rewards[:, 0]).to(self.device)
            indices = None
        
        # Extract individual actions for each agent
        leader_actions = actions[:, 0].unsqueeze(1)
        follower1_actions = actions[:, 1].unsqueeze(1)
        follower2_actions = actions[:, 2].unsqueeze(1)
        
        # Create joint actions for critics
        joint_actions = actions  # Shape: [batch_size, 3]
        
        # Increment the update step counter
        self.t_step += 1
        
        # Get current alpha values for each agent
        if self.automatic_entropy_tuning:
            leader_alpha = self.leader_log_alpha.exp()
            follower1_alpha = self.follower1_log_alpha.exp()
            follower2_alpha = self.follower2_log_alpha.exp()
        else:
            leader_alpha = torch.tensor(self.alpha).to(self.device)
            follower1_alpha = torch.tensor(self.alpha).to(self.device)
            follower2_alpha = torch.tensor(self.alpha).to(self.device)
        
        # Dictionary to track various loss metrics
        metrics = {}
        
        # ==============================
        # 1. Update Critic Networks
        # ==============================
        
        # Get next actions and log probs from the current policy
        with torch.no_grad():
            # Get action distributions for the next state
            _, leader_next_probs, leader_next_log_probs = self.leader_actor(next_states)
            _, follower1_next_probs, follower1_next_log_probs = self.follower1_actor(next_states)
            _, follower2_next_probs, follower2_next_log_probs = self.follower2_actor(next_states)
            
            # Sample actions from the distributions
            leader_next_dist = Categorical(leader_next_probs)
            follower1_next_dist = Categorical(follower1_next_probs)
            follower2_next_dist = Categorical(follower2_next_probs)
            
            leader_next_actions_idx = leader_next_dist.sample()
            follower1_next_actions_idx = follower1_next_dist.sample()
            follower2_next_actions_idx = follower2_next_dist.sample()
            
            # Convert indices to actual action values
            leader_next_actions = torch.tensor([
                self.idx_to_action(idx.item(), self.action_dim_leader) 
                for idx in leader_next_actions_idx
            ], device=self.device).unsqueeze(1)
            
            follower1_next_actions = torch.tensor([
                self.idx_to_action(idx.item(), self.action_dim_follower1) 
                for idx in follower1_next_actions_idx
            ], device=self.device).unsqueeze(1)
            
            follower2_next_actions = torch.tensor([
                self.idx_to_action(idx.item(), self.action_dim_follower2) 
                for idx in follower2_next_actions_idx
            ], device=self.device).unsqueeze(1)
            
            # Create joint next actions
            next_joint_actions = torch.cat([
                leader_next_actions, follower1_next_actions, follower2_next_actions
            ], dim=1)
            
            # Compute log probabilities of the sampled actions
            leader_next_log_prob = torch.gather(
                leader_next_log_probs, 1, 
                torch.tensor([
                    self.action_to_idx(a.item(), self.action_dim_leader) 
                    for a in leader_next_actions
                ], device=self.device).unsqueeze(1)
            )
            
            follower1_next_log_prob = torch.gather(
                follower1_next_log_probs, 1, 
                torch.tensor([
                    self.action_to_idx(a.item(), self.action_dim_follower1) 
                    for a in follower1_next_actions
                ], device=self.device).unsqueeze(1)
            )
            
            follower2_next_log_prob = torch.gather(
                follower2_next_log_probs, 1, 
                torch.tensor([
                    self.action_to_idx(a.item(), self.action_dim_follower2) 
                    for a in follower2_next_actions
                ], device=self.device).unsqueeze(1)
            )
            
            # Compute Q-value targets for each agent using their respective target critics
            leader_q1_next, leader_q2_next = self.leader_critic_target(next_states, next_joint_actions)
            follower1_q1_next, follower1_q2_next = self.follower1_critic_target(next_states, next_joint_actions)
            follower2_q1_next, follower2_q2_next = self.follower2_critic_target(next_states, next_joint_actions)
            
            # Take the minimum of the two Q-values (to mitigate overestimation)
            leader_q_next = torch.min(leader_q1_next, leader_q2_next)
            follower1_q_next = torch.min(follower1_q1_next, follower1_q2_next)
            follower2_q_next = torch.min(follower2_q1_next, follower2_q2_next)
            
            # Extract individual rewards
            leader_rewards = rewards[:, 0].unsqueeze(1)
            follower1_rewards = rewards[:, 1].unsqueeze(1)
            follower2_rewards = rewards[:, 2].unsqueeze(1)
            
            # Add entropy term to the target
            leader_q_target = leader_rewards + (1 - dones) * self.gamma * (leader_q_next - leader_alpha * leader_next_log_prob)
            follower1_q_target = follower1_rewards + (1 - dones) * self.gamma * (follower1_q_next - follower1_alpha * follower1_next_log_prob)
            follower2_q_target = follower2_rewards + (1 - dones) * self.gamma * (follower2_q_next - follower2_alpha * follower2_next_log_prob)
        
        # Compute current Q-values
        leader_q1, leader_q2 = self.leader_critic(states, joint_actions)
        follower1_q1, follower1_q2 = self.follower1_critic(states, joint_actions)
        follower2_q1, follower2_q2 = self.follower2_critic(states, joint_actions)
        
        # Compute MSE loss between current Q-values and targets, weighted by PER weights if applicable
        leader_q1_loss = (weights * F.mse_loss(leader_q1, leader_q_target, reduction='none')).mean()
        leader_q2_loss = (weights * F.mse_loss(leader_q2, leader_q_target, reduction='none')).mean()
        leader_critic_loss = (leader_q1_loss + leader_q2_loss) * leader_importance
        
        follower1_q1_loss = (weights * F.mse_loss(follower1_q1, follower1_q_target, reduction='none')).mean()
        follower1_q2_loss = (weights * F.mse_loss(follower1_q2, follower1_q_target, reduction='none')).mean()
        follower1_critic_loss = (follower1_q1_loss + follower1_q2_loss) * followers_importance
        
        follower2_q1_loss = (weights * F.mse_loss(follower2_q1, follower2_q_target, reduction='none')).mean()
        follower2_q2_loss = (weights * F.mse_loss(follower2_q2, follower2_q_target, reduction='none')).mean()
        follower2_critic_loss = (follower2_q1_loss + follower2_q2_loss) * followers_importance
        
        # Update critic networks
        self.leader_critic_optimizer.zero_grad()
        leader_critic_loss.backward()
        self.leader_critic_optimizer.step()
        
        self.follower1_critic_optimizer.zero_grad()
        follower1_critic_loss.backward()
        self.follower1_critic_optimizer.step()
        
        self.follower2_critic_optimizer.zero_grad()
        follower2_critic_loss.backward()
        self.follower2_critic_optimizer.step()
        
        # Save critic loss metrics
        metrics['leader_critic_loss'] = leader_critic_loss.item()
        metrics['follower1_critic_loss'] = follower1_critic_loss.item()
        metrics['follower2_critic_loss'] = follower2_critic_loss.item()
        
        # ==============================
        # 2. Update Actor Networks (considering Stackelberg hierarchy)
        # ==============================
        
        # For Stackelberg hierarchy, update actors in reverse order:
        # First followers, then leader
        
        # Update Follower2's Actor
        _, follower2_probs, follower2_log_probs = self.follower2_actor(states)
        follower2_dist = Categorical(follower2_probs)
        follower2_entropy = follower2_dist.entropy().unsqueeze(1)
        
        # Evaluate follower2's Q-value with these actions
        follower2_joint_actions = joint_actions.clone()
        follower2_q1_pi, _ = self.follower2_critic(states, follower2_joint_actions)
        
        # Compute follower2's actor loss
        follower2_log_prob = torch.gather(
            follower2_log_probs, 1, 
            torch.tensor([
                self.action_to_idx(a.item(), self.action_dim_follower2) 
                for a in follower2_actions
            ], device=self.device).unsqueeze(1)
        )
        follower2_actor_loss = ((follower2_alpha * follower2_log_prob - follower2_q1_pi) * weights).mean() * followers_importance
        
        # Update follower2's actor
        self.follower2_actor_optimizer.zero_grad()
        follower2_actor_loss.backward()
        self.follower2_actor_optimizer.step()
        
        # Update Follower1's Actor
        _, follower1_probs, follower1_log_probs = self.follower1_actor(states)
        follower1_dist = Categorical(follower1_probs)
        follower1_entropy = follower1_dist.entropy().unsqueeze(1)
        
        # Evaluate follower1's Q-value
        follower1_joint_actions = joint_actions.clone()
        follower1_q1_pi, _ = self.follower1_critic(states, follower1_joint_actions)
        
        # Compute follower1's actor loss
        follower1_log_prob = torch.gather(
            follower1_log_probs, 1, 
            torch.tensor([
                self.action_to_idx(a.item(), self.action_dim_follower1) 
                for a in follower1_actions
            ], device=self.device).unsqueeze(1)
        )
        follower1_actor_loss = ((follower1_alpha * follower1_log_prob - follower1_q1_pi) * weights).mean() * followers_importance
        
        # Update follower1's actor
        self.follower1_actor_optimizer.zero_grad()
        follower1_actor_loss.backward()
        self.follower1_actor_optimizer.step()
        
        # Update Leader's Actor with Stackelberg considerations
        _, leader_probs, leader_log_probs = self.leader_actor(states)
        leader_dist = Categorical(leader_probs)
        leader_entropy = leader_dist.entropy().unsqueeze(1)
        
        # For leader, consider follower responses to leader actions
        batch_size = states.size(0)
        leader_stackelberg_q = torch.zeros(batch_size, 1, device=self.device)
        
        for i in range(batch_size):
            state_i = states[i:i+1]
            leader_action_i = leader_actions[i].item()
            
            # Predict follower responses using their current policies
            f1_action, f2_action = self.predict_follower_responses(state_i, leader_action_i)
            
            # Create joint action with predicted responses
            joint_action_i = self.create_joint_action_tensor(leader_action_i, f1_action, f2_action)
            
            # Evaluate leader's Q-value
            leader_stackelberg_q[i] = self.leader_critic.q1_value(state_i, joint_action_i)
        
        # Compute leader's actor loss
        leader_log_prob = torch.gather(
            leader_log_probs, 1, 
            torch.tensor([
                self.action_to_idx(a.item(), self.action_dim_leader) 
                for a in leader_actions
            ], device=self.device).unsqueeze(1)
        )
        leader_actor_loss = ((leader_alpha * leader_log_prob - leader_stackelberg_q) * weights).mean() * leader_importance
        
        # Update leader's actor
        self.leader_actor_optimizer.zero_grad()
        leader_actor_loss.backward()
        self.leader_actor_optimizer.step()
        
        # Save actor loss metrics
        metrics['leader_actor_loss'] = leader_actor_loss.item()
        metrics['follower1_actor_loss'] = follower1_actor_loss.item()
        metrics['follower2_actor_loss'] = follower2_actor_loss.item()
        
        # ==============================
        # 3. Update Entropy Coefficients (if automatic tuning is enabled)
        # ==============================
        
        if self.automatic_entropy_tuning:
            # Leader entropy tuning
            leader_entropy_loss = -(self.leader_log_alpha * (leader_entropy + self.leader_target_entropy).detach()).mean()
            self.leader_alpha_optimizer.zero_grad()
            leader_entropy_loss.backward()
            self.leader_alpha_optimizer.step()
            
            # Follower1 entropy tuning
            follower1_entropy_loss = -(self.follower1_log_alpha * (follower1_entropy + self.follower1_target_entropy).detach()).mean()
            self.follower1_alpha_optimizer.zero_grad()
            follower1_entropy_loss.backward()
            self.follower1_alpha_optimizer.step()
            
            # Follower2 entropy tuning
            follower2_entropy_loss = -(self.follower2_log_alpha * (follower2_entropy + self.follower2_target_entropy).detach()).mean()
            self.follower2_alpha_optimizer.zero_grad()
            follower2_entropy_loss.backward()
            self.follower2_alpha_optimizer.step()
            
            # Save entropy metrics
            metrics['leader_alpha'] = leader_alpha.item()
            metrics['follower1_alpha'] = follower1_alpha.item()
            metrics['follower2_alpha'] = follower2_alpha.item()
            metrics['leader_entropy'] = leader_entropy.mean().item()
            metrics['follower1_entropy'] = follower1_entropy.mean().item()
            metrics['follower2_entropy'] = follower2_entropy.mean().item()
        
        # ==============================
        # 4. Update Target Networks
        # ==============================
        
        # Only update target networks periodically
        if self.t_step % self.target_update_interval == 0:
            # Soft update of the target networks' parameters
            # θ_target = τ*θ_local + (1 - τ)*θ_target
            for target_param, local_param in zip(self.leader_critic_target.parameters(), self.leader_critic.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
                
            for target_param, local_param in zip(self.follower1_critic_target.parameters(), self.follower1_critic.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
                
            for target_param, local_param in zip(self.follower2_critic_target.parameters(), self.follower2_critic.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
        # Update PER priorities if using PER
        if self.use_per and indices is not None:
            with torch.no_grad():
                leader_td_error = torch.abs(leader_q1 - leader_q_target).cpu().numpy().flatten()
                follower1_td_error = torch.abs(follower1_q1 - follower1_q_target).cpu().numpy().flatten()
                follower2_td_error = torch.abs(follower2_q1 - follower2_q_target).cpu().numpy().flatten()
                
                # Use max td_error among all agents
                td_errors = np.maximum(np.maximum(leader_td_error, follower1_td_error), follower2_td_error)
                replay_buffer.update_priorities(indices, td_errors)
        
        return metrics
    
    def adjust_entropy_targets(self, avg_reward):
        """
        Adjust entropy targets based on recent performance.
        
        Args:
            avg_reward: Average rewards across all agents
        """
        if not self.automatic_entropy_tuning:
            return
        
        # Adjust multiplier based on performance
        if avg_reward < -30:  # Very poor performance, increase exploration
            multiplier = 1.0  # Higher entropy
        elif avg_reward < -20:
            multiplier = 0.75
        elif avg_reward < -10:
            multiplier = 0.5
        else:
            multiplier = 0.25  # Lower entropy (more exploitation)
        
        # Set new entropy targets
        self.leader_target_entropy = -np.log(1.0 / self.action_dim_leader) * multiplier
        self.follower1_target_entropy = -np.log(1.0 / self.action_dim_follower1) * multiplier
        self.follower2_target_entropy = -np.log(1.0 / self.action_dim_follower2) * multiplier
    
    def record_rewards(self, rewards):
        """
        Record rewards for adaptive entropy adjustment.
        
        Args:
            rewards: List of rewards [leader_reward, follower1_reward, follower2_reward]
        """
        self.reward_history['leader'].append(rewards[0])
        self.reward_history['follower1'].append(rewards[1])
        self.reward_history['follower2'].append(rewards[2])
        
        # Adjust entropy targets if we have enough history
        if len(self.reward_history['leader']) >= 10:
            avg_leader = sum(self.reward_history['leader']) / len(self.reward_history['leader'])
            avg_follower1 = sum(self.reward_history['follower1']) / len(self.reward_history['follower1'])
            avg_follower2 = sum(self.reward_history['follower2']) / len(self.reward_history['follower2'])
            
            # Use average of all agents
            avg_reward = (avg_leader + avg_follower1 + avg_follower2) / 3
            
            # Adjust entropy targets
            self.adjust_entropy_targets(avg_reward)
    
    def save(self, path):
        """Save agent models to the specified path."""
        checkpoint = {
            'leader_actor': self.leader_actor.state_dict(),
            'follower1_actor': self.follower1_actor.state_dict(),
            'follower2_actor': self.follower2_actor.state_dict(),
            'leader_critic': self.leader_critic.state_dict(),
            'follower1_critic': self.follower1_critic.state_dict(),
            'follower2_critic': self.follower2_critic.state_dict(),
            'leader_critic_target': self.leader_critic_target.state_dict(),
            'follower1_critic_target': self.follower1_critic_target.state_dict(),
            'follower2_critic_target': self.follower2_critic_target.state_dict(),
            't_step': self.t_step,
        }
        
        if self.automatic_entropy_tuning:
            checkpoint.update({
                'leader_log_alpha': self.leader_log_alpha,
                'follower1_log_alpha': self.follower1_log_alpha,
                'follower2_log_alpha': self.follower2_log_alpha,
                'leader_target_entropy': self.leader_target_entropy,
                'follower1_target_entropy': self.follower1_target_entropy,
                'follower2_target_entropy': self.follower2_target_entropy,
            })
        
        torch.save(checkpoint, path)
        print(f"Agent saved to {path}")
        
    def load(self, path):
        """Load agent models from the specified path."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.leader_actor.load_state_dict(checkpoint['leader_actor'])
        self.follower1_actor.load_state_dict(checkpoint['follower1_actor'])
        self.follower2_actor.load_state_dict(checkpoint['follower2_actor'])
        
        self.leader_critic.load_state_dict(checkpoint['leader_critic'])
        self.follower1_critic.load_state_dict(checkpoint['follower1_critic'])
        self.follower2_critic.load_state_dict(checkpoint['follower2_critic'])
        
        self.leader_critic_target.load_state_dict(checkpoint['leader_critic_target'])
        self.follower1_critic_target.load_state_dict(checkpoint['follower1_critic_target'])
        self.follower2_critic_target.load_state_dict(checkpoint['follower2_critic_target'])
        
        if 't_step' in checkpoint:
            self.t_step = checkpoint['t_step']
        
        if self.automatic_entropy_tuning:
            if 'leader_log_alpha' in checkpoint:
                self.leader_log_alpha = checkpoint['leader_log_alpha']
                self.follower1_log_alpha = checkpoint['follower1_log_alpha']
                self.follower2_log_alpha = checkpoint['follower2_log_alpha']
            
            if 'leader_target_entropy' in checkpoint:
                self.leader_target_entropy = checkpoint['leader_target_entropy']
                self.follower1_target_entropy = checkpoint['follower1_target_entropy']
                self.follower2_target_entropy = checkpoint['follower2_target_entropy']
        
        print(f"Agent loaded from {path}")

#################################
# Battery Disassembly Environment
#################################

class BatteryDisassemblyEnv:
    """
    Environment class for the battery disassembly task with three robots.
    This environment models a workstation with a battery module and three robots:
    - Franka robot (Leader): Equipped with a two-finger gripper for unbolting operations
    - UR10 robot (Follower 1): Equipped with vacuum suction for sorting and pick-and-place
    - Kuka robot (Follower 2): Equipped with specialized tools for casing and connections
    """
    def __init__(self, parameters):
        """
        Initialize the battery disassembly environment.
        
        Parameters:
        - parameters: Dictionary containing environment parameters
        """
        self.rng = np.random.default_rng(parameters['seed'])
        self.task_id = parameters['task_id']
        
        # Load the task board and properties
        self.task_board, self.task_prop = self.task_reader(self.task_id)
        self.curr_board = np.copy(self.task_board)
        
        # Define robot properties
        self.franka_pos = np.array([0.5, -0.3, 0.5])   # Base position of Franka robot (Leader)
        self.ur10_pos = np.array([-0.5, -0.3, 0.5])    # Base position of UR10 robot (Follower 1)
        self.kuka_pos = np.array([0.0, -0.5, 0.5])     # Base position of Kuka robot (Follower 2)
        
        # Define workspace properties
        self.battery_pos = np.array([0.0, 0.0, 0.1])  # Position of the battery module
        self.bin_positions = {
            'screws': np.array([0.3, 0.4, 0.1]),
            'cells': np.array([-0.3, 0.4, 0.1]),
            'casings': np.array([0.0, 0.5, 0.1]),
            'connectors': np.array([0.3, -0.4, 0.1])  # New bin for connectors
        }
        
        # Task completion tracking
        self.completed_tasks = []
        
        # Robot states
        self.franka_state = {'position': self.franka_pos, 'gripper_open': True, 'holding': None}
        self.ur10_state = {'position': self.ur10_pos, 'suction_active': False, 'holding': None}
        self.kuka_state = {'position': self.kuka_pos, 'tool_active': False, 'holding': None}
        
        # Task timing and resource tracking
        self.time_step = 0
        self.max_time_steps = parameters.get('max_time_steps', 100)
        
        # Robot kinematic constraints
        self.franka_workspace_radius = 0.8
        self.ur10_workspace_radius = 1.0
        self.kuka_workspace_radius = 0.9
        
        # Task failure probabilities (uncertainty modeling)
        self.franka_failure_prob = parameters.get('franka_failure_prob', 0.1)
        self.ur10_failure_prob = parameters.get('ur10_failure_prob', 0.1)
        self.kuka_failure_prob = parameters.get('kuka_failure_prob', 0.1)
    
    def reset(self):
        """
        Reset the environment and return the initial state.
        
        Returns:
            Initial state (first row of the task board)
        """
        self.reset_env()
        initial_state, _ = self.get_current_state()
        return initial_state
    
    def step_wrapper(self, action):
        """
        Wrapper method for the step function to handle tuple actions.
        
        Args:
            action: Tuple of (leader_action, follower1_action, follower2_action)
            
        Returns:
            next_state: Next state
            rewards: Rewards for each agent
            done: Whether the episode is done
            info: Additional information
        """
        # Unpack the joint action
        leader_action, follower1_action, follower2_action = action
        
        # Get current state before taking a step
        curr_state, _ = self.get_current_state()
        
        # Calculate rewards based on current state and actions
        rewards = self.reward(curr_state, leader_action, follower1_action, follower2_action)
        
        # Execute the step in the environment using the original step method
        self.step(leader_action, follower1_action, follower2_action)
        
        # Get next state after the step
        next_state, _ = self.get_current_state()
        
        # Check if the episode is done
        done = self.is_done()
        
        # Additional info (can be used for debugging or monitoring)
        info = {
            'time_step': self.time_step,
            'completed_tasks': self.completed_tasks.copy() if hasattr(self, 'completed_tasks') else []
        }
        
        return next_state, rewards, done, info
        
    def task_reader(self, task_id):
        """
        Read the task information from the configuration files.
        Extended for three-robot scenario with more task types.
        """
        # Task board represents the spatial arrangement of components to be disassembled
        # 0: Empty space
        # 1-4: Top screws (requires unbolting by Franka)
        # 5-8: Side screws (requires unbolting by Franka)
        # 9-12: Battery cells (requires pick-and-place by UR10)
        # 13-16: Casing components (requires specialized tools by Kuka)
        # 17-20: Connectors (requires collaborative effort between UR10 and Kuka)
        # 21-22: Complex assemblies (requires all three robots)
        task_board = np.array([
            [1, 2, 3, 4],
            [9, 10, 11, 12],
            [17, 18, 19, 20],
            [5, 6, 7, 8],
            [13, 14, 15, 16],
            [21, 21, 22, 22]
        ])
        
        # Task properties define the characteristics of each task
        # type 1: Leader-specific tasks (unbolting by Franka)
        # type 2: Follower1-specific tasks (pick-and-place by UR10)
        # type 3: Follower2-specific tasks (casing work by Kuka)
        # type 4: Collaborative tasks between Follower1 and Follower2
        # type 5: Tasks requiring Leader and one Follower
        # type 6: Complex tasks requiring all three robots
        
        # Create a type array matching the size of the largest task ID
        max_task_id = np.max(task_board)
        type_array = np.zeros(max_task_id + 1, dtype=int)
        
        # Assign task types
        type_array[1:9] = 1      # Franka tasks (screws)
        type_array[9:13] = 2     # UR10 tasks (battery cells)
        type_array[13:17] = 3    # Kuka tasks (casing)
        type_array[17:21] = 4    # UR10 + Kuka collaborative tasks (connectors)
        type_array[21:23] = 6    # All three robots (complex assemblies)
        
        # Success probabilities for each robot on different task types
        l_succ = np.zeros(max_task_id + 1)
        f1_succ = np.zeros(max_task_id + 1)
        f2_succ = np.zeros(max_task_id + 1)
        
        # Set success probabilities based on task types
        # Type 1: Leader (Franka) tasks
        l_succ[type_array == 1] = 0.9
        f1_succ[type_array == 1] = 0.0
        f2_succ[type_array == 1] = 0.0
        
        # Type 2: Follower 1 (UR10) tasks
        l_succ[type_array == 2] = 0.0
        f1_succ[type_array == 2] = 0.9
        f2_succ[type_array == 2] = 0.0
        
        # Type 3: Follower 2 (Kuka) tasks
        l_succ[type_array == 3] = 0.0
        f1_succ[type_array == 3] = 0.0
        f2_succ[type_array == 3] = 0.9
        
        # Type 4: Follower 1 + Follower 2 collaborative tasks
        l_succ[type_array == 4] = 0.0
        f1_succ[type_array == 4] = 0.7
        f2_succ[type_array == 4] = 0.7
        
        # Type 5: Leader + Follower collaborative tasks (not in this board)
        
        # Type 6: All three robots collaborative tasks
        l_succ[type_array == 6] = 0.7
        f1_succ[type_array == 6] = 0.7
        f2_succ[type_array == 6] = 0.7
        
        # Shape indicates the physical size/complexity (affects timing)
        shape_array = np.ones(max_task_id + 1, dtype=int)
        shape_array[0] = 0  # Empty space has no shape
        shape_array[type_array == 6] = 3  # Complex tasks have larger shape value
        
        task_prop = {
            'type': type_array,
            'shape': shape_array,
            'l_succ': l_succ,
            'f1_succ': f1_succ,
            'f2_succ': f2_succ
        }
        
        return task_board, task_prop

    def get_task_info(self):
        """
        Get task information for initializing the learning algorithms.
        """
        info = {}
        info['task_id'] = self.task_id
        info['dims'] = self.task_board.shape[1]
        info['dimAl'] = self.task_board.shape[1] + 1   # +1 for "do nothing" action
        info['dimAf1'] = self.task_board.shape[1] + 1  # +1 for "do nothing" action
        info['dimAf2'] = self.task_board.shape[1] + 1  # +1 for "do nothing" action
        info['dimal'] = 1
        info['dimaf1'] = 1
        info['dimaf2'] = 1
        info['task_prop'] = self.task_prop
        return info

    def get_current_state(self):
        """
        Get the current state of the environment.
        
        Returns:
        - First row of the current board (simplified state representation)
        - Complete current board
        """
        return np.copy(self.curr_board[0, :]), np.copy(self.curr_board)
    
    def set_env(self, board):
        """
        Set the environment to a specific board configuration.
        """
        self.curr_board = np.copy(board)
    
    def reset_env(self):
        """
        Reset the environment to the initial state.
        """
        self.curr_board = np.copy(self.task_board)
        self.completed_tasks = []
        self.time_step = 0
        self.franka_state = {'position': self.franka_pos, 'gripper_open': True, 'holding': None}
        self.ur10_state = {'position': self.ur10_pos, 'suction_active': False, 'holding': None}
        self.kuka_state = {'position': self.kuka_pos, 'tool_active': False, 'holding': None}
    
    def step(self, al, af1, af2):
        """
        Execute one step in the environment based on all three robots' actions.
        
        Parameters:
        - al: Leader action (Franka robot)
        - af1: Follower 1 action (UR10 robot)
        - af2: Follower 2 action (Kuka robot)
        """
        # Simulate if task is completed by the leader (Franka)
        if al == -1:
            tl, tl_done = 0, False  # Leader does nothing
        else:
            tl = self.curr_board[0, al]
            if tl == 0:
                tl_done = False  # Task already completed or invalid
            else:
                # Check if task is within Franka's capabilities and workspace
                if self.is_task_feasible(tl, 'leader'):
                    tl_done = True if self.rng.uniform() < self.task_prop['l_succ'][tl] else False
                else:
                    tl_done = False
        
        # Simulate if task is completed by follower 1 (UR10)
        if af1 == -1:
            tf1, tf1_done = 0, False  # Follower 1 does nothing
        else:
            tf1 = self.curr_board[0, af1]
            if tf1 == 0:
                tf1_done = False  # Task already completed or invalid
            else:
                # Check if task is within UR10's capabilities and workspace
                if self.is_task_feasible(tf1, 'follower1'):
                    tf1_done = True if self.rng.uniform() < self.task_prop['f1_succ'][tf1] else False
                else:
                    tf1_done = False
        
        # Simulate if task is completed by follower 2 (Kuka)
        if af2 == -1:
            tf2, tf2_done = 0, False  # Follower 2 does nothing
        else:
            tf2 = self.curr_board[0, af2]
            if tf2 == 0:
                tf2_done = False  # Task already completed or invalid
            else:
                # Check if task is within Kuka's capabilities and workspace
                if self.is_task_feasible(tf2, 'follower2'):
                    tf2_done = True if self.rng.uniform() < self.task_prop['f2_succ'][tf2] else False
                else:
                    tf2_done = False
        
        # Update the task board based on the simulated results
        self.update_board(tl, tl_done, tf1, tf1_done, tf2, tf2_done)
        
        # Update robot positions based on actions
        if tl_done or al != -1:
            self.update_robot_position('leader', al)
        
        if tf1_done or af1 != -1:
            self.update_robot_position('follower1', af1)
            
        if tf2_done or af2 != -1:
            self.update_robot_position('follower2', af2)
        
        # Increment time step
        self.time_step += 1
    
    def is_task_feasible(self, task_id, robot):
        """
        Check if a task is feasible for the given robot based on capabilities and workspace constraints.
        
        Parameters:
        - task_id: ID of the task to check
        - robot: 'leader', 'follower1', or 'follower2'
        
        Returns:
        - Boolean indicating if the task is feasible
        """
        # Check robot capability based on task type
        task_type = self.task_prop['type'][task_id]
        
        if robot == 'leader':
            # Leader can do type 1 tasks, and participate in type 5 & 6 collaborative tasks
            if task_type not in [1, 5, 6]:
                return False
                
            # Check if Franka can reach the task
            task_pos = self.get_task_position(task_id)
            if task_pos is None:
                return False
            dist = np.linalg.norm(task_pos - self.franka_state['position'])
            return dist <= self.franka_workspace_radius
            
        elif robot == 'follower1':
            # Follower1 can do type 2 tasks, and participate in type 4, 5 (with leader), & 6 collaborative tasks
            if task_type not in [2, 4, 5, 6]:
                return False
                
            # Check if UR10 can reach the task
            task_pos = self.get_task_position(task_id)
            if task_pos is None:
                return False
            dist = np.linalg.norm(task_pos - self.ur10_state['position'])
            return dist <= self.ur10_workspace_radius
            
        elif robot == 'follower2':
            # Follower2 can do type 3 tasks, and participate in type 4, 5 (with leader), & 6 collaborative tasks
            if task_type not in [3, 4, 5, 6]:
                return False
                
            # Check if Kuka can reach the task
            task_pos = self.get_task_position(task_id)
            if task_pos is None:
                return False
            dist = np.linalg.norm(task_pos - self.kuka_state['position'])
            return dist <= self.kuka_workspace_radius
    
    def get_task_position(self, task_id):
        """
        Get the 3D position of a task based on its ID.
        
        In a realistic scenario, this would map task IDs to actual 
        positions on the battery module.
        """
        # Find the task coordinates in the board
        coords = np.argwhere(self.curr_board == task_id)
        if len(coords) == 0:
            return None
        
        row, col = coords[0]
        
        # Map the 2D coordinates to 3D positions relative to the battery position
        x = self.battery_pos[0] + (col - self.curr_board.shape[1]/2) * 0.1
        y = self.battery_pos[1] + (row - self.curr_board.shape[0]/2) * 0.1
        z = self.battery_pos[2] + 0.05  # Slight offset from battery surface
        
        return np.array([x, y, z])
    
    def update_robot_position(self, robot, action):
        """
        Update the position of a robot based on its action.
        
        Parameters:
        - robot: 'leader', 'follower1', or 'follower2'
        - action: The robot's action
        """
        if action == -1:
            # No movement for "do nothing" action
            return
        
        task_pos = self.get_task_position(self.curr_board[0, action])
        if task_pos is None:
            # No valid task position
            return
        
        if robot == 'leader':
            # Move Franka to the task position
            self.franka_state['position'] = task_pos
            # Update gripper state based on task type
            task_id = self.curr_board[0, action]
            if task_id > 0:
                task_type = self.task_prop['type'][task_id]
                self.franka_state['gripper_open'] = task_type != 1  # Close gripper for unbolting
                self.franka_state['holding'] = task_id if task_type == 1 else None
        
        elif robot == 'follower1':
            # Move UR10 to the task position
            self.ur10_state['position'] = task_pos
            # Update suction state based on task type
            task_id = self.curr_board[0, action]
            if task_id > 0:
                task_type = self.task_prop['type'][task_id]
                self.ur10_state['suction_active'] = task_type == 2  # Activate suction for pick-and-place
                self.ur10_state['holding'] = task_id if task_type == 2 else None
        
        elif robot == 'follower2':
            # Move Kuka to the task position
            self.kuka_state['position'] = task_pos
            # Update tool state based on task type
            task_id = self.curr_board[0, action]
            if task_id > 0:
                task_type = self.task_prop['type'][task_id]
                self.kuka_state['tool_active'] = task_type == 3  # Activate tool for casing work
                self.kuka_state['holding'] = task_id if task_type == 3 else None
    
    def update_board(self, tl, tl_done, tf1, tf1_done, tf2, tf2_done):
        """
        Update the task board based on completed tasks.
        
        Parameters:
        - tl: Leader's task ID
        - tl_done: Whether leader's task was completed
        - tf1: Follower1's task ID
        - tf1_done: Whether follower1's task was completed
        - tf2: Follower2's task ID
        - tf2_done: Whether follower2's task was completed
        """
        # Handle single robot tasks
        if tl != 0 and tl_done and self.task_prop['type'][tl] == 1:
            idx = np.where(self.curr_board[0] == tl)[0]
            self.curr_board[0, idx] = 0
            self.completed_tasks.append(tl)
            
        if tf1 != 0 and tf1_done and self.task_prop['type'][tf1] == 2:
            idx = np.where(self.curr_board[0] == tf1)[0]
            self.curr_board[0, idx] = 0
            self.completed_tasks.append(tf1)
            
        if tf2 != 0 and tf2_done and self.task_prop['type'][tf2] == 3:
            idx = np.where(self.curr_board[0] == tf2)[0]
            self.curr_board[0, idx] = 0
            self.completed_tasks.append(tf2)
        
        # Handle collaborative tasks between the two followers (type 4)
        if tf1 == tf2 and tf1 != 0:
            task_type = self.task_prop['type'][tf1]
            if task_type == 4 and tf1_done and tf2_done:
                idx = np.where(self.curr_board[0] == tf1)[0]
                self.curr_board[0, idx] = 0
                self.completed_tasks.append(tf1)
        
        # Handle three-robot collaborative tasks (type 6)
        if tl == tf1 and tf1 == tf2 and tl != 0:
            task_type = self.task_prop['type'][tl]
            if task_type == 6 and tl_done and tf1_done and tf2_done:
                idx = np.where(self.curr_board[0] == tl)[0]
                self.curr_board[0, idx] = 0
                self.completed_tasks.append(tl)
        
        # Update subsequent rows (task dependencies)
        for i in range(self.task_board.shape[0] - 1):
            curr_row, next_row = self.curr_board[i, :], self.curr_board[i+1, :]
            
            # Find tasks that may drop from the next row
            task_list = []
            idx = np.where(curr_row == 0)[0]
            for j in idx:
                task_id = next_row[j]
                if task_id != 0 and task_id not in task_list:    # task 0 does not count
                    task_list.append(task_id)

            # Check for tasks that can now be accessed
            mod_flag = False
            for ti in task_list:
                idx = np.where(next_row == ti)[0]
                if np.all(curr_row[idx] == 0):
                    curr_row[idx] = ti
                    next_row[idx] = 0
                    mod_flag = True
            
            # If no modifications were made, no need to update future rows
            if not mod_flag:
                break
    
    def reward(self, s, al, af1, af2):
        """
        Calculate rewards for all three robots based on their actions.
        
        Parameters:
        - s: Current state (first row of the board)
        - al: Leader's action
        - af1: Follower1's action
        - af2: Follower2's action
        
        Returns:
        - rl, rf1, rf2: Rewards for leader, follower1, and follower2
        """
        # Determine task IDs corresponding to the actions
        tl = 0 if al == -1 else s[al]
        tf1 = 0 if af1 == -1 else s[af1]
        tf2 = 0 if af2 == -1 else s[af2]
        
        # Initialize rewards
        rl, rf1, rf2 = 0, 0, 0
        
        # All robots idle
        if tl == 0 and tf1 == 0 and tf2 == 0:
            if al == -1 and af1 == -1 and af2 == -1:
                rl, rf1, rf2 = -0.5, -0.5, -0.5  # All idle (slight penalty)
            else:
                # One or more robots attempting empty tasks
                rl = 0 if al == -1 else -1
                rf1 = 0 if af1 == -1 else -1
                rf2 = 0 if af2 == -1 else -1
        
        # Check for collaborative task situations
        
        # Two-robot follower collaboration (type 4)
        if tf1 == tf2 and tf1 != 0 and self.task_prop['type'][tf1] == 4:
            rf1, rf2 = 2, 2  # Higher reward for collaborative task
            rl = 0 if al == -1 else -1  # Leader should not interfere
        
        # Three-robot collaboration (type 6)
        elif tl == tf1 and tf1 == tf2 and tl != 0 and self.task_prop['type'][tl] == 6:
            rl, rf1, rf2 = 3, 3, 3  # Highest reward for full collaboration
        
        # Handle non-collaborative or individual tasks
        else:
            # Process leader's reward
            if tl == 0:
                rl = 0 if al == -1 else -1  # Either idle or empty task
            elif self.task_prop['type'][tl] == 1:
                rl = 1  # Leader-appropriate task
            else:
                rl = -1  # Leader-inappropriate task
            
            # Process follower1's reward
            if tf1 == 0:
                rf1 = 0 if af1 == -1 else -1  # Either idle or empty task
            elif self.task_prop['type'][tf1] == 2:
                rf1 = 1  # Follower1-appropriate task
            else:
                rf1 = -1  # Follower1-inappropriate task
                
            # Process follower2's reward
            if tf2 == 0:
                rf2 = 0 if af2 == -1 else -1  # Either idle or empty task
            elif self.task_prop['type'][tf2] == 3:
                rf2 = 1  # Follower2-appropriate task
            else:
                rf2 = -1  # Follower2-inappropriate task
        
        # Additional penalties for task conflicts (non-collaborative tasks)
        if tl == tf1 and tl != 0 and self.task_prop['type'][tl] not in [5, 6]:
            rl -= 1
            rf1 -= 1
            
        if tl == tf2 and tl != 0 and self.task_prop['type'][tl] not in [5, 6]:
            rl -= 1
            rf2 -= 1
            
        if tf1 == tf2 and tf1 != 0 and self.task_prop['type'][tf1] != 4:
            rf1 -= 1
            rf2 -= 1
        
        return float(rl), float(rf1), float(rf2)
    
    def is_done(self):
        """
        Check if the task is complete (all tasks processed).
        """
        return np.all(self.curr_board == 0)
    
    def render(self, ax=None, mode='human'):
        """
        Render the current state of the environment.
        
        Parameters:
        - ax: Matplotlib axis for rendering
        - mode: Rendering mode ('human' for visualization, 'rgb_array' for image)
        
        Returns:
        - Matplotlib axis with the rendered environment
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        # Clear previous plot
        ax.clear()
        
        # Plot workstation surface
        x = np.linspace(-0.8, 0.8, 10)
        y = np.linspace(-0.6, 0.6, 10)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        ax.plot_surface(X, Y, Z, alpha=0.3, color='gray')
        
        # Plot battery module
        battery_size = 0.2
        ax.bar3d(
            self.battery_pos[0] - battery_size/2,
            self.battery_pos[1] - battery_size/2,
            0,
            battery_size, battery_size, self.battery_pos[2],
            color='silver', alpha=0.7
        )
        
        # Plot bins
        bin_size = 0.1
        for bin_name, bin_pos in self.bin_positions.items():
            ax.bar3d(
                bin_pos[0] - bin_size/2,
                bin_pos[1] - bin_size/2,
                0,
                bin_size, bin_size, bin_pos[2],
                color='lightblue', alpha=0.5
            )
            ax.text(bin_pos[0], bin_pos[1], bin_pos[2] + 0.02, bin_name, 
                   horizontalalignment='center', verticalalignment='bottom')
        
        # Visualize tasks on the battery
        for i in range(self.task_board.shape[0]):
            for j in range(self.task_board.shape[1]):
                task_id = self.curr_board[i, j]
                if task_id > 0:
                    task_pos = self.get_task_position(task_id)
                    if task_pos is not None:
                        task_type = self.task_prop['type'][task_id]
                        
                        # Color based on task type
                        if task_type == 1:
                            color = 'blue'        # Leader tasks
                        elif task_type == 2:
                            color = 'green'       # Follower1 tasks
                        elif task_type == 3:
                            color = 'orange'      # Follower2 tasks
                        elif task_type == 4:
                            color = 'purple'      # Follower1 + Follower2 collaborative tasks
                        elif task_type == 5:
                            color = 'magenta'     # Leader + Follower collaborative tasks
                        else:  # type 6
                            color = 'red'         # All three robots
                        
                        # Plot task markers
                        ax.scatter(
                            task_pos[0], task_pos[1], task_pos[2],
                            color=color, s=100, alpha=0.8
                        )
                        
                        ax.text(
                            task_pos[0], task_pos[1], task_pos[2] + 0.02,
                            f'T{task_id}', 
                            horizontalalignment='center',
                            verticalalignment='bottom'
                        )
        
        # Plot robots
        # Franka robot (leader)
        self._plot_robot(
            ax, self.franka_state['position'], 
            'Franka (L)', 'blue', 
            self.franka_state['gripper_open'], 
            self.franka_state['holding']
        )
        
        # UR10 robot (follower1)
        self._plot_robot(
            ax, self.ur10_state['position'], 
            'UR10 (F1)', 'green', 
            not self.ur10_state['suction_active'], 
            self.ur10_state['holding']
        )
        
        # Kuka robot (follower2)
        self._plot_robot(
            ax, self.kuka_state['position'], 
            'Kuka (F2)', 'orange', 
            not self.kuka_state['tool_active'], 
            self.kuka_state['holding']
        )
        
        # Add legend explaining task types
        ax.text(0.7, 0.5, 0.1, 'Task Types:', fontweight='bold')
        ax.scatter(0.7, 0.45, 0.1, color='blue', s=50)
        ax.text(0.75, 0.45, 0.1, 'Type 1: Leader (Unbolting)')
        
        ax.scatter(0.7, 0.4, 0.1, color='green', s=50)
        ax.text(0.75, 0.4, 0.1, 'Type 2: Follower1 (Pick & Place)')
        
        ax.scatter(0.7, 0.35, 0.1, color='orange', s=50)
        ax.text(0.75, 0.35, 0.1, 'Type 3: Follower2 (Casing)')
        
        ax.scatter(0.7, 0.3, 0.1, color='purple', s=50)
        ax.text(0.75, 0.3, 0.1, 'Type 4: F1 + F2 Collaborative')
        
        ax.scatter(0.7, 0.25, 0.1, color='red', s=50)
        ax.text(0.75, 0.25, 0.1, 'Type 6: All Three Robots')
        
        # Set plot limits and labels
        ax.set_xlim([-0.8, 0.8])
        ax.set_ylim([-0.6, 0.7])
        ax.set_zlim([0, 0.8])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Battery Disassembly Simulation - Time Step: {self.time_step}')
        
        # Show completed tasks
        if self.completed_tasks:
            completed_str = 'Completed: ' + ', '.join([f'T{t}' for t in self.completed_tasks[-5:]])
            if len(self.completed_tasks) > 5:
                completed_str += f'... (+{len(self.completed_tasks)-5} more)'
            ax.text2D(0.05, 0.95, completed_str, transform=ax.transAxes)
        
        plt.draw()
        return ax
    
    def _plot_robot(self, ax, position, name, color, gripper_open, holding):
        """
        Plot a robot in the environment.
        
        Parameters:
        - ax: Matplotlib axis
        - position: Robot position [x, y, z]
        - name: Robot name
        - color: Robot color
        - gripper_open: Boolean indicating if gripper/tool is inactive
        - holding: ID of the task being held (if any)
        """
        # Plot robot base
        ax.bar3d(
            position[0] - 0.05,
            position[1] - 0.05,
            0,
            0.1, 0.1, position[2] - 0.1,
            color=color, alpha=0.5
        )
        
        # Plot end-effector
        ax.scatter(
            position[0], position[1], position[2],
            color=color, s=150, alpha=0.8
        )
        
        # Show robot name
        ax.text(
            position[0], position[1], position[2] + 0.05,
            name, 
            horizontalalignment='center',
            verticalalignment='bottom'
        )
        
        # Indicate tool state
        tool_state = "Inactive" if gripper_open else "Active"
        ax.text(
            position[0], position[1], position[2] - 0.05,
            tool_state, 
            horizontalalignment='center',
            verticalalignment='top',
            fontsize=8
        )
        
        # Indicate held object
        if holding is not None:
            ax.text(
                position[0], position[1], position[2] - 0.1,
                f"Holding: T{holding}", 
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=8
            )
        
        # Plot a line from the base to the end-effector
        ax.plot(
            [position[0], position[0]],
            [position[1], position[1]],
            [0, position[2]],
            color=color, linewidth=2, alpha=0.6
        )

#################################
# Training Functions
#################################

def fill_buffer(env, agent, buffer, min_size=5000):
    """
    Pre-fill replay buffer with initial experiences.
    
    Args:
        env: Environment
        agent: Agent
        buffer: Replay buffer
        min_size: Minimum buffer size to collect
    """
    print(f"Pre-filling replay buffer to {min_size} experiences...")
    state = env.reset()
    
    collected = 0
    while len(buffer) < min_size:
        # Random exploration initially
        leader_action = np.random.randint(-1, env.task_board.shape[1] - 1)
        follower1_action = np.random.randint(-1, env.task_board.shape[1] - 1)
        follower2_action = np.random.randint(-1, env.task_board.shape[1] - 1)
        
        action = (leader_action, follower1_action, follower2_action)
        next_state, rewards, done, _ = env.step_wrapper(action)
        
        # Add to buffer
        buffer.add(state, action, rewards, next_state, done)
        
        state = next_state if not done else env.reset()
        
        collected += 1
        if collected % 500 == 0:
            print(f"Collected {collected} experiences, buffer size: {len(buffer)}")
    
    print(f"Buffer prefilled with {len(buffer)} experiences.")

def train_phase1_followers(env, agent, buffer, n_episodes=100, batch_size=64, update_freq=4):
    """
    Phase 1: Train followers with fixed leader policy.
    
    Args:
        env: Environment
        agent: Agent
        buffer: Replay buffer
        n_episodes: Number of episodes to train
        batch_size: Batch size for updates
        update_freq: How often to update networks (every N steps)
    
    Returns:
        Dictionary of training metrics
    """
    print("\n===== Phase 1: Training followers with fixed leader policy =====")
    
    # Save initial leader policy
    torch.save(agent.leader_actor.state_dict(), "initial_leader.pth")
    
    metrics = {
        'episode_rewards': [],
        'episode_steps': [],
        'follower1_losses': [],
        'follower2_losses': []
    }
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = [0, 0, 0]
        episode_steps = 0
        
        done = False
        while not done:
            # Sample leader action according to fixed policy
            with torch.no_grad():
                leader_logits, _, _ = agent.leader_actor(torch.FloatTensor(state).unsqueeze(0).to(device))
                leader_probs = F.softmax(leader_logits, dim=-1)
                leader_dist = Categorical(leader_probs)
                leader_idx = leader_dist.sample().item()
                leader_action = agent.idx_to_action(leader_idx, agent.action_dim_leader)
            
            # Sample follower actions according to current policies
            action_masks = None  # Add masks here if your environment provides them
            _, f1_action, f2_action = agent.act(state, action_masks, deterministic=False, exploration_factor=0.9)
            
            # Take step in environment
            action = (leader_action, f1_action, f2_action)
            next_state, rewards, done, _ = env.step_wrapper(action)
            
            # Add to buffer
            buffer.add(state, action, rewards, next_state, done)
            
            # Record rewards
            episode_reward = [r + rew for r, rew in zip(episode_reward, rewards)]
            
            # Update followers only, every update_freq steps
            if len(buffer) > batch_size and episode_steps % update_freq == 0:
                experiences = buffer.sample(batch_size)
                update_metrics = agent.update_followers_only(experiences)
                metrics['follower1_losses'].append(update_metrics['follower1_actor_loss'])
                metrics['follower2_losses'].append(update_metrics['follower2_actor_loss'])
            
            state = next_state
            episode_steps += 1
            
            # Terminate if episode is too long
            if episode_steps >= env.max_time_steps:
                done = True
        
        # Record episode metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_steps'].append(episode_steps)
        
        # Log progress
        if episode % 10 == 0 or episode == n_episodes - 1:
            print(f"Phase 1 - Episode {episode}/{n_episodes}: "
                  f"Leader={episode_reward[0]:.1f}, "
                  f"Follower1={episode_reward[1]:.1f}, "
                  f"Follower2={episode_reward[2]:.1f}, "
                  f"Steps={episode_steps}")
    
    print("Phase 1 complete!")
    return metrics

def train_phase2_leader(env, agent, buffer, n_episodes=100, batch_size=64, update_freq=4):
    """
    Phase 2: Train leader with trained follower policies.
    
    Args:
        env: Environment
        agent: Agent
        buffer: Replay buffer
        n_episodes: Number of episodes to train
        batch_size: Batch size for updates
        update_freq: How often to update networks (every N steps)
    
    Returns:
        Dictionary of training metrics
    """
    print("\n===== Phase 2: Training leader with trained follower policies =====")
    
    # Save follower policies
    torch.save(agent.follower1_actor.state_dict(), "trained_follower1.pth")
    torch.save(agent.follower2_actor.state_dict(), "trained_follower2.pth")
    
    metrics = {
        'episode_rewards': [],
        'episode_steps': [],
        'leader_losses': []
    }
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = [0, 0, 0]
        episode_steps = 0
        
        done = False
        while not done:
            # Sample leader action according to current policy
            leader_action, _, _ = agent.act(state, deterministic=False, exploration_factor=0.8)
            
            # Get follower actions using their fixed policies
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                # Sample from follower1's policy
                _, f1_probs, _ = agent.follower1_actor(state_tensor)
                f1_dist = Categorical(f1_probs)
                f1_idx = f1_dist.sample().item()
                f1_action = agent.idx_to_action(f1_idx, agent.action_dim_follower1)
                
                # Sample from follower2's policy
                _, f2_probs, _ = agent.follower2_actor(state_tensor)
                f2_dist = Categorical(f2_probs)
                f2_idx = f2_dist.sample().item()
                f2_action = agent.idx_to_action(f2_idx, agent.action_dim_follower2)
            
            # Take step in environment
            action = (leader_action, f1_action, f2_action)
            next_state, rewards, done, _ = env.step_wrapper(action)
            
            # Add to buffer
            buffer.add(state, action, rewards, next_state, done)
            
            # Record rewards
            episode_reward = [r + rew for r, rew in zip(episode_reward, rewards)]
            
            # Update leader only, every update_freq steps
            if len(buffer) > batch_size and episode_steps % update_freq == 0:
                experiences = buffer.sample(batch_size)
                update_metrics = agent.update_leader_only(experiences)
                metrics['leader_losses'].append(update_metrics['leader_actor_loss'])
            
            state = next_state
            episode_steps += 1
            
            # Terminate if episode is too long
            if episode_steps >= env.max_time_steps:
                done = True
        
        # Record episode metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_steps'].append(episode_steps)
        
        # Record rewards for entropy adjustment
        agent.record_rewards(episode_reward)
        
        # Log progress
        if episode % 10 == 0 or episode == n_episodes - 1:
            print(f"Phase 2 - Episode {episode}/{n_episodes}: "
                  f"Leader={episode_reward[0]:.1f}, "
                  f"Follower1={episode_reward[1]:.1f}, "
                  f"Follower2={episode_reward[2]:.1f}, "
                  f"Steps={episode_steps}")
    
    print("Phase 2 complete!")
    return metrics

def train_phase3_joint(env, agent, buffer, n_episodes=200, batch_size=64, update_freq=4):
    """
    Phase 3: Train all agents together with Stackelberg hierarchy.
    
    Args:
        env: Environment
        agent: Agent
        buffer: Replay buffer
        n_episodes: Number of episodes to train
        batch_size: Batch size for updates
        update_freq: How often to update networks (every N steps)
    
    Returns:
        Dictionary of training metrics
    """
    print("\n===== Phase 3: Joint training with Stackelberg hierarchy =====")
    
    metrics = {
        'episode_rewards': [],
        'episode_steps': [],
        'entropy_values': []
    }
    
    # Initialize exploration factor (will decay over time)
    exploration_factor = 1.0
    exploration_decay = 0.995
    min_exploration = 0.1
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = [0, 0, 0]
        episode_steps = 0
        
        done = False
        while not done:
            # Use Stackelberg with probability (1 - exploration_factor)
            action_masks = None  # Add masks here if your environment provides them
            if random.random() > exploration_factor:
                # Use deterministic Stackelberg equilibrium
                leader_action, f1_action, f2_action = agent.compute_stackelberg_equilibrium(state, action_masks)
            else:
                # Use stochastic policies
                leader_action, f1_action, f2_action = agent.act(state, action_masks, deterministic=False)
            
            # Take step in environment
            action = (leader_action, f1_action, f2_action)
            next_state, rewards, done, _ = env.step_wrapper(action)
            
            # Add to buffer
            buffer.add(state, action, rewards, next_state, done)
            
            # Record rewards
            episode_reward = [r + rew for r, rew in zip(episode_reward, rewards)]
            
            # Update all agents with hierarchical importance
            if len(buffer) > batch_size and episode_steps % update_freq == 0:
                experiences = buffer.sample(batch_size)
                # Gradually increase leader importance over followers
                leader_importance = min(1.0, 0.5 + episode / n_episodes)
                follower_importance = 1.0
                agent.update(experiences, follower_importance, leader_importance)
            
            state = next_state
            episode_steps += 1
            
            # Terminate if episode is too long
            if episode_steps >= env.max_time_steps:
                done = True
        
        # Decay exploration factor
        exploration_factor = max(min_exploration, exploration_factor * exploration_decay)
        
        # Record episode metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_steps'].append(episode_steps)
        
        # Record entropy values
        if agent.automatic_entropy_tuning:
            entropy_values = {
                'leader_alpha': agent.leader_log_alpha.exp().item(),
                'follower1_alpha': agent.follower1_log_alpha.exp().item(),
                'follower2_alpha': agent.follower2_log_alpha.exp().item()
            }
            metrics['entropy_values'].append(entropy_values)
        
        # Record rewards for entropy adjustment
        agent.record_rewards(episode_reward)
        
        # Save model every 50 episodes
        if episode % 50 == 0:
            agent.save(f"stackelberg_sac_phase3_ep{episode}.pth")
        
        # Log progress
        if episode % 10 == 0 or episode == n_episodes - 1:
            print(f"Phase 3 - Episode {episode}/{n_episodes}: "
                  f"Leader={episode_reward[0]:.1f}, "
                  f"Follower1={episode_reward[1]:.1f}, "
                  f"Follower2={episode_reward[2]:.1f}, "
                  f"Steps={episode_steps}, "
                  f"Exploration={exploration_factor:.2f}")
    
    print("Phase 3 complete!")
    return metrics

def evaluate_agent(env, agent, n_episodes=10):
    """
    Evaluate agent performance by running deterministic episodes.
    
    Args:
        env: Environment
        agent: Agent
        n_episodes: Number of evaluation episodes
    
    Returns:
        Average rewards and steps
    """
    print("\n===== Evaluating Agent Performance =====")
    
    total_rewards = [0, 0, 0]
    total_steps = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = [0, 0, 0]
        episode_steps = 0
        
        done = False
        while not done:
            # Always use deterministic Stackelberg equilibrium for evaluation
            leader_action, f1_action, f2_action = agent.compute_stackelberg_equilibrium(state)
            
            # Take step in environment
            action = (leader_action, f1_action, f2_action)
            next_state, rewards, done, _ = env.step_wrapper(action)
            
            # Record rewards
            episode_reward = [r + rew for r, rew in zip(episode_reward, rewards)]
            
            state = next_state
            episode_steps += 1
            
            # Terminate if episode is too long
            if episode_steps >= env.max_time_steps:
                done = True
        
        # Record episode metrics
        total_rewards = [tr + er for tr, er in zip(total_rewards, episode_reward)]
        total_steps += episode_steps
        
        # Log progress
        print(f"Eval Episode {episode}: "
              f"Leader={episode_reward[0]:.1f}, "
              f"Follower1={episode_reward[1]:.1f}, "
              f"Follower2={episode_reward[2]:.1f}, "
              f"Steps={episode_steps}")
    
    # Calculate averages
    avg_rewards = [r / n_episodes for r in total_rewards]
    avg_steps = total_steps / n_episodes
    
    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"Average Rewards - Leader: {avg_rewards[0]:.1f}, "
          f"Follower1: {avg_rewards[1]:.1f}, "
          f"Follower2: {avg_rewards[2]:.1f}")
    print(f"Average Steps: {avg_steps:.1f}")
    
    return avg_rewards, avg_steps

def plot_training_results(metrics_phase1, metrics_phase2, metrics_phase3, save_path="training_results.png"):
    """
    Plot training results from all phases.
    
    Args:
        metrics_phase1: Metrics from phase 1
        metrics_phase2: Metrics from phase 2
        metrics_phase3: Metrics from phase 3
        save_path: Path to save the plot
    """
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot phase 1 rewards (followers training)
    phase1_episodes = len(metrics_phase1['episode_rewards'])
    phase1_follower1 = [r[1] for r in metrics_phase1['episode_rewards']]
    phase1_follower2 = [r[2] for r in metrics_phase1['episode_rewards']]
    
    axs[0].plot(range(phase1_episodes), phase1_follower1, label='Follower 1')
    axs[0].plot(range(phase1_episodes), phase1_follower2, label='Follower 2')
    axs[0].set_title('Phase 1: Followers Training')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot phase 2 rewards (leader training)
    phase2_episodes = len(metrics_phase2['episode_rewards'])
    phase2_leader = [r[0] for r in metrics_phase2['episode_rewards']]
    
    axs[1].plot(range(phase2_episodes), phase2_leader, label='Leader')
    axs[1].set_title('Phase 2: Leader Training')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Reward')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot phase 3 rewards (joint training)
    phase3_episodes = len(metrics_phase3['episode_rewards'])
    phase3_leader = [r[0] for r in metrics_phase3['episode_rewards']]
    phase3_follower1 = [r[1] for r in metrics_phase3['episode_rewards']]
    phase3_follower2 = [r[2] for r in metrics_phase3['episode_rewards']]
    
    axs[2].plot(range(phase3_episodes), phase3_leader, label='Leader')
    axs[2].plot(range(phase3_episodes), phase3_follower1, label='Follower 1')
    axs[2].plot(range(phase3_episodes), phase3_follower2, label='Follower 2')
    axs[2].set_title('Phase 3: Joint Training with Stackelberg Hierarchy')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Reward')
    axs[2].legend()
    axs[2].grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training results plot saved to {save_path}")

def main():
    """Main training function that runs all phases."""
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Environment parameters
    env_params = {
        'seed': RANDOM_SEED,
        'task_id': 0,
        'max_time_steps': 40
    }
    
    # Create environment
    env = BatteryDisassemblyEnv(env_params)
    
    # Get state and action dimensions from the environment
    state_dim = env.task_board.shape[1]  # Number of columns in the task board
    action_dim = state_dim + 1  # +1 for idle action (-1)
    
    # Create agent
    agent = StackelbergSACAgent(
        state_dim=state_dim,
        action_dim_leader=action_dim,
        action_dim_follower1=action_dim,
        action_dim_follower2=action_dim,
        hidden_size=128,
        learning_rate=1e-4,
        gamma=0.995,
        tau=0.001,
        alpha=0.2,
        target_update_interval=1,
        automatic_entropy_tuning=True,
        use_per=True
    )
    
    # Create replay buffer
    buffer_capacity = 100000
    replay_buffer = ReplayBuffer(
        capacity=buffer_capacity,
        state_dim=state_dim,
        action_dim=3,
        use_per=True,
        alpha=0.6,
        beta=0.4
    )
    
    # Training parameters
    batch_size = 64
    update_freq = 4
    phase1_episodes = 100  # Followers training
    phase2_episodes = 100  # Leader training
    phase3_episodes = 300  # Joint training
    
    # Fill buffer with initial experiences
    fill_buffer(env, agent, replay_buffer, min_size=5000)
    
    # Phase 1: Train followers with fixed leader policy
    metrics_phase1 = train_phase1_followers(
        env, agent, replay_buffer, 
        n_episodes=phase1_episodes, 
        batch_size=batch_size, 
        update_freq=update_freq
    )
    
    # Phase 2: Train leader with trained follower policies
    metrics_phase2 = train_phase2_leader(
        env, agent, replay_buffer, 
        n_episodes=phase2_episodes, 
        batch_size=batch_size, 
        update_freq=update_freq
    )
    
    # Phase 3: Train all agents together with Stackelberg hierarchy
    metrics_phase3 = train_phase3_joint(
        env, agent, replay_buffer, 
        n_episodes=phase3_episodes, 
        batch_size=batch_size, 
        update_freq=update_freq
    )
    
    # Save final model
    agent.save("stackelberg_sac_final.pth")
    
    # Evaluate agent
    avg_rewards, avg_steps = evaluate_agent(env, agent, n_episodes=20)
    
    # Plot training results
    plot_training_results(metrics_phase1, metrics_phase2, metrics_phase3, save_path="results/training_results.png")
    
    # Save metrics
    all_metrics = {
        'phase1': metrics_phase1,
        'phase2': metrics_phase2,
        'phase3': metrics_phase3,
        'evaluation': {
            'avg_rewards': avg_rewards,
            'avg_steps': avg_steps
        }
    }
    
    with open('results/training_metrics.json', 'w') as f:
        # Convert some values to lists for JSON serialization
        serializable_metrics = copy.deepcopy(all_metrics)
        # Handle non-serializable items
        if 'entropy_values' in serializable_metrics['phase3']:
            serializable_metrics['phase3']['entropy_values'] = [
                {k: float(v) for k, v in ev.items()} 
                for ev in serializable_metrics['phase3']['entropy_values']
            ]
        json.dump(serializable_metrics, f, indent=4)
    
    print("Training complete! Results saved to 'results' directory.")

if __name__ == "__main__":
    main()
