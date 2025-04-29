import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import random
import pickle



class RecurrentGaussianPolicy(nn.Module):
    """
    Recurrent Gaussian Policy implementation for SAC.
    This outputs a mean and standard deviation for action sampling.
    """
    def __init__(self, input_dim, action_dim, hidden_size=64, lstm_layers=1, log_std_min=-20, log_std_max=2):
        super(RecurrentGaussianPolicy, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        
        # LSTM layer for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with small random values."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, state, hidden_state=None):
        """
        Forward pass through the network.
        
        Parameters:
        - state: Batch of state sequences [batch_size, seq_len, state_dim]
        - hidden_state: Initial hidden state for LSTM
        
        Returns:
        - mean: Mean of the action distribution
        - log_std: Log standard deviation of the action distribution
        - hidden_state: Final hidden state
        """
        # Extract features
        features = self.feature_extractor(state)
        
        # Pass through LSTM
        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(features)
        else:
            lstm_out, hidden_state = self.lstm(features, hidden_state)
        
        # Get mean and log_std
        mean = self.mean_layer(lstm_out)
        log_std = self.log_std_layer(lstm_out)
        
        # Clamp log_std to ensure numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std, hidden_state
    
    def sample(self, state, hidden_state=None, deterministic=False, last_timestep=False):
        """
        Sample actions from the policy.
        
        Parameters:
        - state: Current state [batch_size, seq_len, state_dim] or [state_dim]
        - hidden_state: Initial hidden state for LSTM
        - deterministic: Whether to sample deterministically
        - last_timestep: Whether to return only the last timestep's action
        
        Returns:
        - Sampled actions
        - Log probabilities
        - Entropy
        - Final hidden state
        """
        # Ensure state is a tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        # Add batch dimension if needed
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Add sequence dimension if needed
        if len(state.shape) == 2:
            state = state.unsqueeze(1)
        
        # Get mean and log_std from policy
        mean, log_std, hidden_state = self.forward(state, hidden_state)
        
        # Convert log_std to std (ensuring positivity)
        std = torch.exp(log_std)
        
        # If deterministic, use mean as action
        if deterministic:
            actions = mean
            log_probs = torch.zeros_like(mean)
            entropy = torch.zeros_like(mean)
        else:
            # Sample from normal distribution
            normal = torch.distributions.Normal(mean, std)
            actions = normal.rsample()  # Reparameterization trick
            log_probs = normal.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = normal.entropy().sum(dim=-1, keepdim=True)
        
        # Apply tanh to bound actions
        actions = torch.tanh(actions)
        
        # Adjust log probabilities for tanh
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        # If the caller explicitly asks for only the last step (e.g. for deployment),
        # then crop; otherwise keep the full sequence
        if last_timestep and mean.size(1) > 1:
            actions = actions[:, -1:, :]
            log_probs = log_probs[:, -1:, :]
            entropy = entropy[:, -1:, :]
        
        return actions, log_probs, entropy, hidden_state
    
    def _update_actor_and_alpha(self, policy, critic1, critic2, policy_optimizer,
                                alpha, log_alpha, alpha_optimizer, target_entropy,
                                state_batch, batch_size):
        """
        Update actor (policy) and alpha (temperature) parameters.
        """
        # Sample actions from current policy
        sampled_actions, log_probs, entropy, _ = policy.sample(
            state_batch, deterministic=False, last_timestep=False)
        
        # Get Q values from critics
        q1, _ = critic1(state_batch, sampled_actions)
        q2, _ = critic2(state_batch, sampled_actions)
        q = torch.min(q1, q2)
        
        # Calculate policy loss
        policy_loss = (alpha * log_probs - q).mean()
        
        # Optimize policy
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        
        # Update alpha if auto-tuning is enabled
        if self.auto_entropy_tuning:
            alpha_loss = -(log_alpha * (log_probs + target_entropy).detach()).mean()
            
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            
            alpha = log_alpha.exp()
        
        return policy_loss.item(), alpha.item()


class RecurrentQNetwork(nn.Module):
    """
    Recurrent Q-Network implementation for SAC.
    This evaluates state-action pairs.
    """
    def __init__(self, input_dim, action_dim, hidden_size=64, lstm_layers=1):
        super(RecurrentQNetwork, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        
        # LSTM layer for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Output layer for Q-value
        self.output_layer = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with small random values."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, state, action, hidden_state=None):
        """
        Forward pass through the network.
        
        Parameters:
        - state: Batch of state sequences [batch_size, seq_len, state_dim]
        - action: Batch of action sequences [batch_size, seq_len, action_dim]
        - hidden_state: Initial hidden state for LSTM
        
        Returns:
        - Q-values
        - Final hidden state
        """
        # Debug prints
        print(f"Forward - state shape: {state.shape}")
        print(f"Forward - action shape: {action.shape}")
        
        batch_size, seq_len, _ = state.shape
        
        # Concatenate state and action
        state_action = torch.cat([state, action], dim=2)
        print(f"Forward - state_action shape before reshape: {state_action.shape}")
        
        # Calculate the expected size
        expected_size = batch_size * seq_len * (self.input_dim + action.shape[-1])
        actual_size = state_action.numel()
        print(f"Forward - Expected size: {expected_size}, Actual size: {actual_size}")
        
        # Reshape for feature extraction
        state_action_reshaped = state_action.reshape(-1, state_action.shape[-1])
        print(f"Forward - state_action_reshaped shape: {state_action_reshaped.shape}")
        
        # Create feature extractor with correct input dimensions
        feature_extractor = nn.Sequential(
            nn.Linear(state_action_reshaped.shape[-1], self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU()
        ).to(state_action_reshaped.device)
        
        # Initialize weights for the new feature extractor
        for name, param in feature_extractor.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        # Extract features
        features = feature_extractor(state_action_reshaped)
        features = features.view(batch_size, seq_len, self.hidden_size)
        print(f"Forward - features shape: {features.shape}")
        
        # Pass through LSTM
        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(features)
        else:
            lstm_out, hidden_state = self.lstm(features, hidden_state)
        
        # Generate Q-values
        q_values = self.output_layer(lstm_out)
        print(f"Forward - q_values shape: {q_values.shape}")
        
        return q_values, hidden_state


class StackelbergThreeRobotSACAgent:
    """
    Agent implementation using Recurrent Soft Actor-Critic for Stackelberg games with three robots.
    """
    def __init__(self, state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, 
                 hidden_size=64, sequence_length=8, device='cpu', learning_rate=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, auto_entropy_tuning=True, seed=42):
        """
        Initialize the Stackelberg SAC agent for three robots.
        """
        self.state_dim = state_dim
        self.action_dim_leader = action_dim_leader
        self.action_dim_follower1 = action_dim_follower1
        self.action_dim_follower2 = action_dim_follower2
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize networks for each agent
        # Leader networks
        self.leader_policy = RecurrentGaussianPolicy(
            state_dim, action_dim_leader, hidden_size).to(device)
        
        # Leader critics need to see all actions
        total_action_dim = action_dim_leader + action_dim_follower1 + action_dim_follower2
        self.leader_critic1 = RecurrentQNetwork(
            state_dim, total_action_dim, hidden_size).to(device)
        self.leader_critic2 = RecurrentQNetwork(
            state_dim, total_action_dim, hidden_size).to(device)
        self.leader_target_critic1 = RecurrentQNetwork(
            state_dim, total_action_dim, hidden_size).to(device)
        self.leader_target_critic2 = RecurrentQNetwork(
            state_dim, total_action_dim, hidden_size).to(device)
        
        # Follower1 networks
        self.follower1_policy = RecurrentGaussianPolicy(
            state_dim, action_dim_follower1, hidden_size).to(device)
        self.follower1_critic1 = RecurrentQNetwork(
            state_dim, action_dim_follower1, hidden_size).to(device)
        self.follower1_critic2 = RecurrentQNetwork(
            state_dim, action_dim_follower1, hidden_size).to(device)
        self.follower1_target_critic1 = RecurrentQNetwork(
            state_dim, action_dim_follower1, hidden_size).to(device)
        self.follower1_target_critic2 = RecurrentQNetwork(
            state_dim, action_dim_follower1, hidden_size).to(device)
        
        # Follower2 networks
        self.follower2_policy = RecurrentGaussianPolicy(
            state_dim, action_dim_follower2, hidden_size).to(device)
        self.follower2_critic1 = RecurrentQNetwork(
            state_dim, action_dim_follower2, hidden_size).to(device)
        self.follower2_critic2 = RecurrentQNetwork(
            state_dim, action_dim_follower2, hidden_size).to(device)
        self.follower2_target_critic1 = RecurrentQNetwork(
            state_dim, action_dim_follower2, hidden_size).to(device)
        self.follower2_target_critic2 = RecurrentQNetwork(
            state_dim, action_dim_follower2, hidden_size).to(device)
        
        # Copy target networks
        self._copy_target_networks()
        
        # Initialize optimizers
        self.leader_policy_optimizer = optim.Adam(self.leader_policy.parameters(), lr=learning_rate)
        self.leader_critic_optimizer = optim.Adam(
            list(self.leader_critic1.parameters()) + list(self.leader_critic2.parameters()),
            lr=learning_rate)
        
        self.follower1_policy_optimizer = optim.Adam(self.follower1_policy.parameters(), lr=learning_rate)
        self.follower1_critic_optimizer = optim.Adam(
            list(self.follower1_critic1.parameters()) + list(self.follower1_critic2.parameters()),
            lr=learning_rate)
        
        self.follower2_policy_optimizer = optim.Adam(self.follower2_policy.parameters(), lr=learning_rate)
        self.follower2_critic_optimizer = optim.Adam(
            list(self.follower2_critic1.parameters()) + list(self.follower2_critic2.parameters()),
            lr=learning_rate)
        
        # Initialize entropy tuning
        if auto_entropy_tuning:
            self.target_entropy_leader = -torch.prod(torch.Tensor([action_dim_leader])).item()
            self.target_entropy_follower1 = -torch.prod(torch.Tensor([action_dim_follower1])).item()
            self.target_entropy_follower2 = -torch.prod(torch.Tensor([action_dim_follower2])).item()
            
            self.log_alpha_leader = torch.zeros(1, requires_grad=True, device=device)
            self.log_alpha_follower1 = torch.zeros(1, requires_grad=True, device=device)
            self.log_alpha_follower2 = torch.zeros(1, requires_grad=True, device=device)
            
            self.alpha_leader = self.log_alpha_leader.exp()
            self.alpha_follower1 = self.log_alpha_follower1.exp()
            self.alpha_follower2 = self.log_alpha_follower2.exp()
            
            self.alpha_optimizer_leader = optim.Adam([self.log_alpha_leader], lr=learning_rate)
            self.alpha_optimizer_follower1 = optim.Adam([self.log_alpha_follower1], lr=learning_rate)
            self.alpha_optimizer_follower2 = optim.Adam([self.log_alpha_follower2], lr=learning_rate)
        
        # Initialize hidden states
        self.reset_hidden_states()
    
    def _copy_target_networks(self):
        """Copy parameters from online networks to target networks."""
        self.leader_target_critic1.load_state_dict(self.leader_critic1.state_dict())
        self.leader_target_critic2.load_state_dict(self.leader_critic2.state_dict())
        self.follower1_target_critic1.load_state_dict(self.follower1_critic1.state_dict())
        self.follower1_target_critic2.load_state_dict(self.follower1_critic2.state_dict())
        self.follower2_target_critic1.load_state_dict(self.follower2_critic1.state_dict())
        self.follower2_target_critic2.load_state_dict(self.follower2_critic2.state_dict())
    
    def reset_hidden_states(self):
        """Reset hidden states for all networks."""
        self.leader_policy_hidden = None
        self.follower1_policy_hidden = None
        self.follower2_policy_hidden = None
        self.leader_critic_hidden = None
        self.follower1_critic_hidden = None
        self.follower2_critic_hidden = None
    
    def compute_stackelberg_equilibrium(self, state, deterministic=False):
        """
        Compute Stackelberg equilibrium using the current policies.
        In the Stackelberg hierarchy: Leader -> (Follower1, Follower2)
        
        Parameters:
        - state: Current environment state
        - deterministic: Whether to use deterministic actions
        
        Returns:
        - leader_action, follower1_action, follower2_action: Equilibrium actions
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Leader acts first
        print(f"State shape: {state_tensor.shape}")
        leader_action, _, _, self.leader_policy_hidden = self.leader_policy.sample(
            state_tensor, self.leader_policy_hidden, deterministic)
        
        print(f"Leader action type: {type(leader_action)}")
        print(f"Leader action shape: {leader_action.shape}")
        print(f"Leader action values: {leader_action}")
        
        # Followers observe leader's action and react
        follower1_action, _, _, self.follower1_policy_hidden = self.follower1_policy.sample(
            state_tensor, self.follower1_policy_hidden, deterministic)
        
        follower2_action, _, _, self.follower2_policy_hidden = self.follower2_policy.sample(
            state_tensor, self.follower2_policy_hidden, deterministic)
        
        # Convert to numpy arrays
        leader_action = leader_action.detach().cpu().numpy()
        follower1_action = follower1_action.detach().cpu().numpy()
        follower2_action = follower2_action.detach().cpu().numpy()
        
        print(f"Numpy leader action type: {type(leader_action)}")
        print(f"Numpy leader action shape: {leader_action.shape}")
        print(f"Numpy leader action values: {leader_action}")
        
        # Apply action space mapping if needed (e.g., from continuous to discrete)
        try:
            leader_action_discrete = self._continuous_to_discrete(leader_action, self.action_dim_leader)
            follower1_action_discrete = self._continuous_to_discrete(follower1_action, self.action_dim_follower1)
            follower2_action_discrete = self._continuous_to_discrete(follower2_action, self.action_dim_follower2)
            
            print(f"Discrete leader action: {leader_action_discrete}")
            
            return leader_action_discrete, follower1_action_discrete, follower2_action_discrete
        except Exception as e:
            print(f"Error in continuous_to_discrete: {e}")
            print(f"Debug info - leader_action: {leader_action}, type: {type(leader_action)}, shape: {leader_action.shape}")
            raise
    
    def _continuous_to_discrete(self, continuous_action, num_discrete_actions):
        """
        Map continuous actions to discrete action space.
        
        Parameters:
        - continuous_action: Continuous action from policy [-1, 1]
        - num_discrete_actions: Number of discrete actions available
        
        Returns:
        - Discrete action index (-1 to num_discrete_actions-2)
        """
        # Debug info
        print(f"continuous_action in _continuous_to_discrete: {continuous_action}")
        print(f"type: {type(continuous_action)}, shape: {continuous_action.shape if hasattr(continuous_action, 'shape') else 'no shape'}")
        
        # Handle different types of input
        try:
            # Scale from [-1, 1] to [0, num_discrete_actions-1]
            scaled_action = (continuous_action + 1) / 2 * (num_discrete_actions - 1)
            
            # Convert to a scalar regardless of the input shape
            if isinstance(scaled_action, np.ndarray):
                # If it's a multi-dimensional array, flatten it and take the first element
                flat_action = scaled_action.flatten()[0]
            else:
                # Already a scalar
                flat_action = scaled_action
            
            # Round to nearest integer and convert to int
            discrete_idx = int(round(float(flat_action)))
            
            # Clip to valid range
            discrete_idx = max(0, min(discrete_idx, num_discrete_actions - 1))
            
            # Convert to action space format (-1 to num_discrete_actions-2)
            return discrete_idx - 1
        except Exception as e:
            print(f"Exception in _continuous_to_discrete: {e}")
            print(f"continuous_action: {continuous_action}")
            print(f"num_discrete_actions: {num_discrete_actions}")
            # Default to "do nothing" action if conversion fails
            return -1
    
    def _discrete_to_continuous(self, discrete_action, num_discrete_actions):
        """
        Map discrete actions to continuous action space for critics.
        
        Parameters:
        - discrete_action: Discrete action index (-1 to num_discrete_actions-2)
        - num_discrete_actions: Number of discrete actions available
        
        Returns:
        - Continuous action tensor [-1, 1]
        """
        # Convert from action space format to [0, num_discrete_actions-1]
        idx = discrete_action + 1
        
        # Scale to [-1, 1]
        continuous = 2 * (idx / (num_discrete_actions - 1)) - 1
        
        return continuous
    
    def act(self, state, deterministic=False):
        """
        Get actions from all agents for the current state.
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Disable gradient tracking for inference
        with torch.no_grad():
            # Get actions from each agent's policy
            leader_action, _, _, self.leader_policy_hidden = self.leader_policy.sample(
                state_tensor, self.leader_policy_hidden, deterministic, last_timestep=True)
            follower1_action, _, _, self.follower1_policy_hidden = self.follower1_policy.sample(
                state_tensor, self.follower1_policy_hidden, deterministic, last_timestep=True)
            follower2_action, _, _, self.follower2_policy_hidden = self.follower2_policy.sample(
                state_tensor, self.follower2_policy_hidden, deterministic, last_timestep=True)
        
        # Convert actions to numpy arrays (now safe since gradients are disabled)
        leader_action = leader_action.squeeze().cpu().numpy()
        follower1_action = follower1_action.squeeze().cpu().numpy()
        follower2_action = follower2_action.squeeze().cpu().numpy()
        
        # Map continuous actions to discrete space
        leader_action_discrete = self._continuous_to_discrete(leader_action, self.action_dim_leader)
        follower1_action_discrete = self._continuous_to_discrete(follower1_action, self.action_dim_follower1)
        follower2_action_discrete = self._continuous_to_discrete(follower2_action, self.action_dim_follower2)
        
        return leader_action_discrete, follower1_action_discrete, follower2_action_discrete
    
    def update(self, experiences, batch_size):
        """
        Update all networks using a batch of experiences.
        
        Parameters:
        - experiences: List of sequences of (state, a_leader, a_follower1, a_follower2, r_leader, r_follower1, r_follower2, next_state, done)
        - batch_size: Batch size for training
        
        Returns:
        - Dictionary of losses for monitoring
        """
        # Process experiences
        state_batch, action_leader_batch, action_follower1_batch, action_follower2_batch, \
        reward_leader_batch, reward_follower1_batch, reward_follower2_batch, \
        next_state_batch, done_batch = self._process_experiences(experiences)
        
        # Update critics
        leader_critic_loss = self._update_critic(
            self.leader_critic1, self.leader_critic2,
            self.leader_target_critic1, self.leader_target_critic2,
            self.leader_critic_optimizer,
            state_batch, action_leader_batch, action_follower1_batch, action_follower2_batch,
            reward_leader_batch, next_state_batch, done_batch
        )
        
        follower1_critic_loss = self._update_critic(
            self.follower1_critic1, self.follower1_critic2,
            self.follower1_target_critic1, self.follower1_target_critic2,
            self.follower1_critic_optimizer,
            state_batch, action_leader_batch, action_follower1_batch, action_follower2_batch,
            reward_follower1_batch, next_state_batch, done_batch
        )
        
        follower2_critic_loss = self._update_critic(
            self.follower2_critic1, self.follower2_critic2,
            self.follower2_target_critic1, self.follower2_target_critic2,
            self.follower2_critic_optimizer,
            state_batch, action_leader_batch, action_follower1_batch, action_follower2_batch,
            reward_follower2_batch, next_state_batch, done_batch
        )
        
        # Update actor and alpha for each agent
        leader_policy_loss, leader_alpha_loss = self._update_actor_and_alpha(
            self.leader_policy, self.leader_critic1, self.leader_critic2,
            self.leader_policy_optimizer, 
            self.alpha_leader, self.log_alpha_leader, self.alpha_optimizer_leader,
            self.target_entropy_leader if self.auto_entropy_tuning else None,
            state_batch, batch_size
        )
        
        follower1_policy_loss, follower1_alpha_loss = self._update_actor_and_alpha(
            self.follower1_policy, self.follower1_critic1, self.follower1_critic2,
            self.follower1_policy_optimizer,
            self.alpha_follower1, self.log_alpha_follower1, self.alpha_optimizer_follower1,
            self.target_entropy_follower1 if self.auto_entropy_tuning else None,
            state_batch, batch_size
        )
        
        follower2_policy_loss, follower2_alpha_loss = self._update_actor_and_alpha(
            self.follower2_policy, self.follower2_critic1, self.follower2_critic2,
            self.follower2_policy_optimizer,
            self.alpha_follower2, self.log_alpha_follower2, self.alpha_optimizer_follower2,
            self.target_entropy_follower2 if self.auto_entropy_tuning else None,
            state_batch, batch_size
        )
        
        # Update target networks
        self._soft_update_target_networks()
        
        # Update alpha values if using automatic entropy tuning
        if self.auto_entropy_tuning:
            self.alpha_leader = self.log_alpha_leader.exp().detach()
            self.alpha_follower1 = self.log_alpha_follower1.exp().detach()
            self.alpha_follower2 = self.log_alpha_follower2.exp().detach()
        
        # Return all losses for monitoring
        loss_dict = {
            'leader_critic_loss': leader_critic_loss,
            'leader_policy_loss': leader_policy_loss,
            'leader_alpha_loss': leader_alpha_loss if self.auto_entropy_tuning else 0,
            'follower1_critic_loss': follower1_critic_loss,
            'follower1_policy_loss': follower1_policy_loss,
            'follower1_alpha_loss': follower1_alpha_loss if self.auto_entropy_tuning else 0,
            'follower2_critic_loss': follower2_critic_loss,
            'follower2_policy_loss': follower2_policy_loss,
            'follower2_alpha_loss': follower2_alpha_loss if self.auto_entropy_tuning else 0,
            'alpha_leader': self.alpha_leader.item() if isinstance(self.alpha_leader, torch.Tensor) else self.alpha_leader,
            'alpha_follower1': self.alpha_follower1.item() if isinstance(self.alpha_follower1, torch.Tensor) else self.alpha_follower1,
            'alpha_follower2': self.alpha_follower2.item() if isinstance(self.alpha_follower2, torch.Tensor) else self.alpha_follower2
        }
        
        return loss_dict
    
    def _process_experiences(self, experiences):
        """
        Process a batch of experiences into tensors.
        """
        states = []
        actions_leader = []
        actions_follower1 = []
        actions_follower2 = []
        rewards_leader = []
        rewards_follower1 = []
        rewards_follower2 = []
        next_states = []
        dones = []
        
        # Process each sequence
        for sequence in experiences:
            states_seq = []
            actions_leader_seq = []
            actions_follower1_seq = []
            actions_follower2_seq = []
            rewards_leader_seq = []
            rewards_follower1_seq = []
            rewards_follower2_seq = []
            next_states_seq = []
            dones_seq = []
            
            for exp in sequence:
                s, a_l, a_f1, a_f2, r_l, r_f1, r_f2, next_s, done = exp
                # Convert discrete actions to continuous for the SAC network
                a_l_cont = self._discrete_to_continuous(a_l, self.action_dim_leader)
                a_f1_cont = self._discrete_to_continuous(a_f1, self.action_dim_follower1)
                a_f2_cont = self._discrete_to_continuous(a_f2, self.action_dim_follower2)
                
                states_seq.append(s)
                actions_leader_seq.append(a_l_cont)
                actions_follower1_seq.append(a_f1_cont)
                actions_follower2_seq.append(a_f2_cont)
                rewards_leader_seq.append(r_l)
                rewards_follower1_seq.append(r_f1)
                rewards_follower2_seq.append(r_f2)
                next_states_seq.append(next_s)
                dones_seq.append(done)
            
            states.append(states_seq)
            actions_leader.append(actions_leader_seq)
            actions_follower1.append(actions_follower1_seq)
            actions_follower2.append(actions_follower2_seq)
            rewards_leader.append(rewards_leader_seq)
            rewards_follower1.append(rewards_follower1_seq)
            rewards_follower2.append(rewards_follower2_seq)
            next_states.append(next_states_seq)
            dones.append(dones_seq)
        
        # Convert to numpy arrays first, then to tensors to avoid the warning
        states = np.array(states, dtype=np.float32)
        actions_leader = np.array(actions_leader, dtype=np.float32).reshape(len(experiences), len(experiences[0]), 1)
        actions_follower1 = np.array(actions_follower1, dtype=np.float32).reshape(len(experiences), len(experiences[0]), 1)
        actions_follower2 = np.array(actions_follower2, dtype=np.float32).reshape(len(experiences), len(experiences[0]), 1)
        rewards_leader = np.array(rewards_leader, dtype=np.float32).reshape(len(experiences), len(experiences[0]), 1)
        rewards_follower1 = np.array(rewards_follower1, dtype=np.float32).reshape(len(experiences), len(experiences[0]), 1)
        rewards_follower2 = np.array(rewards_follower2, dtype=np.float32).reshape(len(experiences), len(experiences[0]), 1)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32).reshape(len(experiences), len(experiences[0]), 1)
        
        # Convert numpy arrays to tensors
        states = torch.from_numpy(states).to(self.device)
        actions_leader = torch.from_numpy(actions_leader).to(self.device)
        actions_follower1 = torch.from_numpy(actions_follower1).to(self.device)
        actions_follower2 = torch.from_numpy(actions_follower2).to(self.device)
        rewards_leader = torch.from_numpy(rewards_leader).to(self.device)
        rewards_follower1 = torch.from_numpy(rewards_follower1).to(self.device)
        rewards_follower2 = torch.from_numpy(rewards_follower2).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)

        return states, actions_leader, actions_follower1, actions_follower2, \
            rewards_leader, rewards_follower1, rewards_follower2, \
            next_states, dones
    
    def _update_critic(self, critic1, critic2, target_critic1, target_critic2, critic_optimizer,
                   state_batch, action_leader_batch, action_follower1_batch, action_follower2_batch,
                   reward_batch, next_state_batch, done_batch):
        """
        Update critic networks for one agent.
        """
        batch_size, seq_len = state_batch.shape[0], state_batch.shape[1]
        
        with torch.no_grad():
            # Sample next actions from each agent's policy
            # We need to flatten the batch and sequence dimensions for sampling
            next_state_reshaped = next_state_batch.reshape(-1, next_state_batch.shape[-1])
            
            # Sample actions from flattened states
            next_leader_action, next_leader_log_prob, _, _ = self.leader_policy.sample(
                next_state_reshaped.unsqueeze(0))
            next_follower1_action, next_follower1_log_prob, _, _ = self.follower1_policy.sample(
                next_state_reshaped.unsqueeze(0))
            next_follower2_action, next_follower2_log_prob, _, _ = self.follower2_policy.sample(
                next_state_reshaped.unsqueeze(0))
            
            # Get the shape of the actions
            leader_action_shape = next_leader_action.shape
            follower1_action_shape = next_follower1_action.shape
            follower2_action_shape = next_follower2_action.shape
            
            # Print debug info for shapes
            print(f"Next leader action shape: {leader_action_shape}")
            print(f"Next follower1 action shape: {follower1_action_shape}")
            print(f"Next follower2 action shape: {follower2_action_shape}")
            
            # Perform proper reshaping based on the actual dimensions
            # The reshape should preserve the last dimension (action features)
            # and distribute elements across batch and sequence dimensions
            total_seq_items = batch_size * seq_len
            
            # Adjust reshaping based on the actual dimensions we have
            if len(leader_action_shape) == 3 and leader_action_shape[0] == 1:
                # If shape is [1, N, action_dim], we might need to repeat or reshape
                if leader_action_shape[1] == total_seq_items:
                    # This means we have one action per sequence item, just need to reshape
                    next_leader_action = next_leader_action.squeeze(0).reshape(batch_size, seq_len, -1)
                    next_follower1_action = next_follower1_action.squeeze(0).reshape(batch_size, seq_len, -1)
                    next_follower2_action = next_follower2_action.squeeze(0).reshape(batch_size, seq_len, -1)
                    
                    # Also reshape log probs
                    next_leader_log_prob = next_leader_log_prob.squeeze(0).reshape(batch_size, seq_len, -1)
                    next_follower1_log_prob = next_follower1_log_prob.squeeze(0).reshape(batch_size, seq_len, -1)
                    next_follower2_log_prob = next_follower2_log_prob.squeeze(0).reshape(batch_size, seq_len, -1)
                else:
                    # We need to handle the mismatch - this could be tricky
                    # For now, let's repeat the actions to match the sequence length
                    repeat_factor = total_seq_items // leader_action_shape[1] if leader_action_shape[1] > 0 else 1
                    if repeat_factor > 1:
                        next_leader_action = next_leader_action.repeat(1, repeat_factor, 1)
                        next_follower1_action = next_follower1_action.repeat(1, repeat_factor, 1)
                        next_follower2_action = next_follower2_action.repeat(1, repeat_factor, 1)
                        
                        next_leader_log_prob = next_leader_log_prob.repeat(1, repeat_factor, 1)
                        next_follower1_log_prob = next_follower1_log_prob.repeat(1, repeat_factor, 1)
                        next_follower2_log_prob = next_follower2_log_prob.repeat(1, repeat_factor, 1)
                    
                    # Then reshape
                    next_leader_action = next_leader_action.squeeze(0)[:total_seq_items].reshape(batch_size, seq_len, -1)
                    next_follower1_action = next_follower1_action.squeeze(0)[:total_seq_items].reshape(batch_size, seq_len, -1)
                    next_follower2_action = next_follower2_action.squeeze(0)[:total_seq_items].reshape(batch_size, seq_len, -1)
                    
                    next_leader_log_prob = next_leader_log_prob.squeeze(0)[:total_seq_items].reshape(batch_size, seq_len, -1)
                    next_follower1_log_prob = next_follower1_log_prob.squeeze(0)[:total_seq_items].reshape(batch_size, seq_len, -1)
                    next_follower2_log_prob = next_follower2_log_prob.squeeze(0)[:total_seq_items].reshape(batch_size, seq_len, -1)
            else:
                # If we have a completely different shape, we need a different approach
                # This is a fallback that just repeats the first action across all sequence items
                action_dim_leader = next_leader_action.shape[-1]
                action_dim_follower1 = next_follower1_action.shape[-1]
                action_dim_follower2 = next_follower2_action.shape[-1]
                
                # Create tensors with the right shape
                next_leader_action = next_leader_action.flatten()[:action_dim_leader].repeat(batch_size, seq_len, 1)
                next_follower1_action = next_follower1_action.flatten()[:action_dim_follower1].repeat(batch_size, seq_len, 1)
                next_follower2_action = next_follower2_action.flatten()[:action_dim_follower2].repeat(batch_size, seq_len, 1)
                
                # Same for log probs
                next_leader_log_prob = next_leader_log_prob.flatten()[:1].repeat(batch_size, seq_len, 1)
                next_follower1_log_prob = next_follower1_log_prob.flatten()[:1].repeat(batch_size, seq_len, 1)
                next_follower2_log_prob = next_follower2_log_prob.flatten()[:1].repeat(batch_size, seq_len, 1)
        
        # Determine which agent's critic we're updating
        if critic1 == self.leader_critic1:
            # Leader's critic sees all actions
            actions_combined = torch.cat([action_leader_batch, action_follower1_batch, action_follower2_batch], dim=2)
            next_actions_combined = torch.cat([next_leader_action, next_follower1_action, next_follower2_action], dim=2)
            next_log_prob = next_leader_log_prob
            alpha = self.alpha_leader
        elif critic1 == self.follower1_critic1:
            # Follower1's critic only sees its own actions
            actions_combined = action_follower1_batch
            next_actions_combined = next_follower1_action
            next_log_prob = next_follower1_log_prob
            alpha = self.alpha_follower1
        else:  # Follower2's critic
            # Follower2's critic only sees its own actions
            actions_combined = action_follower2_batch
            next_actions_combined = next_follower2_action
            next_log_prob = next_follower2_log_prob
            alpha = self.alpha_follower2
        
        with torch.no_grad():
            # Get target Q values
            target_q1, _ = target_critic1(next_state_batch, next_actions_combined)
            target_q2, _ = target_critic2(next_state_batch, next_actions_combined)
            
            # Take minimum Q value for stability
            target_q = torch.min(target_q1, target_q2)
            
            # Add entropy term
            target_q = target_q - alpha * next_log_prob
            
            # Calculate target value
            target_value = reward_batch + (1.0 - done_batch) * self.gamma * target_q
        
        # Get current Q values
        current_q1, _ = critic1(state_batch, actions_combined)
        current_q2, _ = critic2(state_batch, actions_combined)
        
        # Calculate critic loss
        critic_loss = F.mse_loss(current_q1, target_value.detach()) + \
                     F.mse_loss(current_q2, target_value.detach())
        
        # Optimize the critic
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        return critic_loss.item()
    
    def _soft_update_target_networks(self):
        """
        Soft update of target networks using the tau parameter:
        θ_target = τ * θ_online + (1 - τ) * θ_target
        """
        # Update leader's target networks
        for target_param, param in zip(self.leader_target_critic1.parameters(), self.leader_critic1.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        
        for target_param, param in zip(self.leader_target_critic2.parameters(), self.leader_critic2.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        
        # Update follower1's target networks
        for target_param, param in zip(self.follower1_target_critic1.parameters(), self.follower1_critic1.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        
        for target_param, param in zip(self.follower1_target_critic2.parameters(), self.follower1_critic2.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        
        # Update follower2's target networks
        for target_param, param in zip(self.follower2_target_critic1.parameters(), self.follower2_critic1.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        
        for target_param, param in zip(self.follower2_target_critic2.parameters(), self.follower2_critic2.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, path):
        """
        Save all networks and parameters.
        
        Parameters:
        - path: Directory to save to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save policy networks
        torch.save(self.leader_policy.state_dict(), f"{path}/leader_policy.pt")
        torch.save(self.follower1_policy.state_dict(), f"{path}/follower1_policy.pt")
        torch.save(self.follower2_policy.state_dict(), f"{path}/follower2_policy.pt")
        
        # Save critic networks
        torch.save(self.leader_critic1.state_dict(), f"{path}/leader_critic1.pt")
        torch.save(self.leader_critic2.state_dict(), f"{path}/leader_critic2.pt")
        torch.save(self.follower1_critic1.state_dict(), f"{path}/follower1_critic1.pt")
        torch.save(self.follower1_critic2.state_dict(), f"{path}/follower1_critic2.pt")
        torch.save(self.follower2_critic1.state_dict(), f"{path}/follower2_critic1.pt")
        torch.save(self.follower2_critic2.state_dict(), f"{path}/follower2_critic2.pt")
        
        # Save target networks
        torch.save(self.leader_target_critic1.state_dict(), f"{path}/leader_target_critic1.pt")
        torch.save(self.leader_target_critic2.state_dict(), f"{path}/leader_target_critic2.pt")
        torch.save(self.follower1_target_critic1.state_dict(), f"{path}/follower1_target_critic1.pt")
        torch.save(self.follower1_target_critic2.state_dict(), f"{path}/follower1_target_critic2.pt")
        torch.save(self.follower2_target_critic1.state_dict(), f"{path}/follower2_target_critic1.pt")
        torch.save(self.follower2_target_critic2.state_dict(), f"{path}/follower2_target_critic2.pt")
        
        # Save alpha parameters
        if self.auto_entropy_tuning:
            alpha_params = {
                'log_alpha_leader': self.log_alpha_leader.item(),
                'log_alpha_follower1': self.log_alpha_follower1.item(),
                'log_alpha_follower2': self.log_alpha_follower2.item()
            }
        else:
            alpha_params = {
                'alpha_leader': self.alpha_leader,
                'alpha_follower1': self.alpha_follower1,
                'alpha_follower2': self.alpha_follower2
            }
        
        with open(f"{path}/alpha_params.pkl", "wb") as f:
            pickle.dump(alpha_params, f)
    
    def load(self, path):
        """
        Load all networks and parameters.
        
        Parameters:
        - path: Directory to load from
        """
        # Load policy networks
        self.leader_policy.load_state_dict(torch.load(f"{path}/leader_policy.pt"))
        self.follower1_policy.load_state_dict(torch.load(f"{path}/follower1_policy.pt"))
        self.follower2_policy.load_state_dict(torch.load(f"{path}/follower2_policy.pt"))
        
        # Load critic networks
        self.leader_critic1.load_state_dict(torch.load(f"{path}/leader_critic1.pt"))
        self.leader_critic2.load_state_dict(torch.load(f"{path}/leader_critic2.pt"))
        self.follower1_critic1.load_state_dict(torch.load(f"{path}/follower1_critic1.pt"))
        self.follower1_critic2.load_state_dict(torch.load(f"{path}/follower1_critic2.pt"))
        self.follower2_critic1.load_state_dict(torch.load(f"{path}/follower2_critic1.pt"))
        self.follower2_critic2.load_state_dict(torch.load(f"{path}/follower2_critic2.pt"))
        
        # Load target networks
        self.leader_target_critic1.load_state_dict(torch.load(f"{path}/leader_target_critic1.pt"))
        self.leader_target_critic2.load_state_dict(torch.load(f"{path}/leader_target_critic2.pt"))
        self.follower1_target_critic1.load_state_dict(torch.load(f"{path}/follower1_target_critic1.pt"))
        self.follower1_target_critic2.load_state_dict(torch.load(f"{path}/follower1_target_critic2.pt"))
        self.follower2_target_critic1.load_state_dict(torch.load(f"{path}/follower2_target_critic1.pt"))
        self.follower2_target_critic2.load_state_dict(torch.load(f"{path}/follower2_target_critic2.pt"))
        
        # Load alpha parameters
        with open(f"{path}/alpha_params.pkl", "rb") as f:
            alpha_params = pickle.load(f)
        
        if self.auto_entropy_tuning:
            self.log_alpha_leader = torch.tensor(alpha_params['log_alpha_leader'], 
                                                requires_grad=True, device=self.device)
            self.log_alpha_follower1 = torch.tensor(alpha_params['log_alpha_follower1'], 
                                                   requires_grad=True, device=self.device)
            self.log_alpha_follower2 = torch.tensor(alpha_params['log_alpha_follower2'], 
                                                   requires_grad=True, device=self.device)
            
            # Initialize alpha optimizers with loaded parameters
            self.alpha_optimizer_leader = optim.Adam([self.log_alpha_leader], lr=3e-4)
            self.alpha_optimizer_follower1 = optim.Adam([self.log_alpha_follower1], lr=3e-4)
            self.alpha_optimizer_follower2 = optim.Adam([self.log_alpha_follower2], lr=3e-4)
            
            # Update alpha values
            self.alpha_leader = self.log_alpha_leader.exp().detach()
            self.alpha_follower1 = self.log_alpha_follower1.exp().detach()
            self.alpha_follower2 = self.log_alpha_follower2.exp().detach()
        else:
            self.alpha_leader = alpha_params['alpha_leader']
            self.alpha_follower1 = alpha_params['alpha_follower1']
            self.alpha_follower2 = alpha_params['alpha_follower2']


class SequenceReplayBufferSAC:
    """
    Replay buffer for storing and sampling sequences of experiences for SAC.
    Includes done flags and handles recurrent state sequences.
    """
    def __init__(self, buffer_size, sequence_length, state_dim, batch_size, seed):
        self.buffer_size = buffer_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.buffer = []
        self.episode_buffer = []
    
    def __len__(self):
        """Get the current size of the buffer."""
        return len(self.buffer)
    
    def add(self, experience):
        """
        Add an experience to the episode buffer.
        
        Parameters:
        - experience: Experience to add [state, a_leader, a_follower1, a_follower2, 
                                         r_leader, r_follower1, r_follower2, next_state, done]
        """
        self.episode_buffer.append(experience)
    
    def end_episode(self):
        """
        End the current episode and transfer sequences to the main buffer.
        """
        if len(self.episode_buffer) == 0:
            return
        
        # Add overlapping sequences from the episode to the buffer
        for i in range(max(1, len(self.episode_buffer) - self.sequence_length + 1)):
            sequence = self.episode_buffer[i:i+self.sequence_length]
            if len(sequence) < self.sequence_length:
                # Pad shorter sequences
                padding = [sequence[-1]] * (self.sequence_length - len(sequence))
                # Ensure done flag is True for padding
                if len(sequence) > 0:
                    padding_exp = list(sequence[-1])
                    padding_exp[-1] = True  # Set done flag to True
                    padding = [tuple(padding_exp)] * (self.sequence_length - len(sequence))
                sequence.extend(padding)
            
            self.buffer.append(sequence)
            
            # Maintain buffer size
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)
        
        self.episode_buffer = []
    
    def sample(self, batch_size=None):
        """
        Sample a batch of sequences from the buffer.
        
        Parameters:
        - batch_size: Size of batch to sample (uses default if None)
        
        Returns:
        - Batch of sequence experiences
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer contains {len(self.buffer)} sequences, but requested batch size is {batch_size}")
        
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    


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




class StackelbergThreeRobotSACSimulation:
    """
    Main simulation class for the Stackelberg game using SAC with three robots.
    """
    def __init__(self, parameters):
        """
        Initialize the simulation.
        
        Parameters:
        - parameters: Dictionary containing simulation parameters
        """
        self.env = BatteryDisassemblyEnv(parameters)
        self.device = parameters.get('device', 'cpu')
        
        # Extract environment information
        env_info = self.env.get_task_info()
        state_dim = env_info['dims']
        action_dim_leader = env_info['dimAl']
        action_dim_follower1 = env_info['dimAf1']
        action_dim_follower2 = env_info['dimAf2']
        
        # Initialize agent
        self.agent = StackelbergThreeRobotSACAgent(
            state_dim=state_dim,
            action_dim_leader=action_dim_leader,
            action_dim_follower1=action_dim_follower1,
            action_dim_follower2=action_dim_follower2,
            hidden_size=parameters.get('hidden_size', 64),
            sequence_length=parameters.get('sequence_length', 8),
            device=self.device,
            learning_rate=parameters.get('learning_rate', 3e-4),
            gamma=parameters.get('gamma', 0.99),
            tau=parameters.get('tau', 0.005),
            alpha=parameters.get('alpha', 0.2),
            auto_entropy_tuning=parameters.get('auto_entropy_tuning', True),
            seed=parameters.get('seed', 42)
        )
        
        # Initialize replay buffer
        self.buffer = SequenceReplayBufferSAC(
            buffer_size=parameters.get('buffer_size', 10000),
            sequence_length=parameters.get('sequence_length', 8),
            state_dim=state_dim,
            batch_size=parameters.get('batch_size', 32),
            seed=parameters.get('seed', 42)
        )
        
        # Training parameters
        self.n_episodes = parameters.get('episode_size', 1000)
        self.n_steps_per_episode = parameters.get('step_per_episode', 40)
        self.batch_size = parameters.get('batch_size', 32)
        
        # Statistics tracking
        self.training_stats = {
            'leader_rewards': [],
            'follower1_rewards': [],
            'follower2_rewards': [],
            'completion_steps': [],
            'completion_rates': [],
            'leader_critic_losses': [],
            'leader_policy_losses': [],
            'follower1_critic_losses': [],
            'follower1_policy_losses': [],
            'follower2_critic_losses': [],
            'follower2_policy_losses': [],
            'alpha_leader': [],
            'alpha_follower1': [],
            'alpha_follower2': []
        }
    
    def generate_initial_buffer(self, n_episodes=10):
        """
        Generate initial experiences using random actions.
        
        Parameters:
        - n_episodes: Number of episodes to generate
        """
        print("Generating initial experiences...")
        
        for episode in range(n_episodes):
            self.env.reset_env()
            state, _ = self.env.get_current_state()
            
            for step in range(self.n_steps_per_episode):
                # Choose random actions
                leader_action = np.random.randint(-1, self.env.task_board.shape[1])
                follower1_action = np.random.randint(-1, self.env.task_board.shape[1])
                follower2_action = np.random.randint(-1, self.env.task_board.shape[1])
                
                # Get rewards and update environment
                leader_reward, follower1_reward, follower2_reward = self.env.reward(
                    state, leader_action, follower1_action, follower2_action)
                self.env.step(leader_action, follower1_action, follower2_action)
                next_state, _ = self.env.get_current_state()
                
                # Check if done
                done = self.env.is_done()
                
                # Store experience with done flag
                experience = (state, leader_action, follower1_action, follower2_action, 
                             leader_reward, follower1_reward, follower2_reward, next_state, done)
                self.buffer.add(experience)
                
                # Check if done
                if done:
                    break
                
                state = next_state
            
            # End episode in buffer
            self.buffer.end_episode()
        
        print(f"Initial buffer size: {len(self.buffer)}")
    
    def train(self, n_episodes=None, render_interval=None):
        """
        Train the agents using SAC.
        
        Parameters:
        - n_episodes: Number of episodes to train (uses default if None)
        - render_interval: How often to render an episode (None for no rendering)
        
        Returns:
        - Training statistics
        """
        if n_episodes is None:
            n_episodes = self.n_episodes
        
        # Generate initial experiences if buffer is empty
        if len(self.buffer) < self.batch_size:
            self.generate_initial_buffer()
        
        print(f"Starting training for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            # Reset environment and agent hidden states
            self.env.reset_env()
            self.agent.reset_hidden_states()
            state, _ = self.env.get_current_state()
            
            episode_leader_reward = 0
            episode_follower1_reward = 0
            episode_follower2_reward = 0
            episode_losses = {
                'leader_critic_losses': [],
                'leader_policy_losses': [],
                'follower1_critic_losses': [],
                'follower1_policy_losses': [],
                'follower2_critic_losses': [],
                'follower2_policy_losses': [],
                'alpha_leader': [],
                'alpha_follower1': [],
                'alpha_follower2': []
            }
            steps = 0
            
            # Create figure for rendering if needed
            if render_interval is not None and episode % render_interval == 0:
                render = True
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                plt.ion()  # Turn on interactive mode
            else:
                render = False
            
            # Run the episode
            for step in range(self.n_steps_per_episode):
                # Select actions using current policy
                leader_action, follower1_action, follower2_action = self.agent.act(state)
                
                # Get rewards and update environment
                leader_reward, follower1_reward, follower2_reward = self.env.reward(
                    state, leader_action, follower1_action, follower2_action)
                self.env.step(leader_action, follower1_action, follower2_action)
                next_state, _ = self.env.get_current_state()
                
                # Check if done
                done = self.env.is_done()
                
                # Store experience with done flag
                experience = (state, leader_action, follower1_action, follower2_action, 
                             leader_reward, follower1_reward, follower2_reward, next_state, done)
                self.buffer.add(experience)
                
                # Update statistics
                episode_leader_reward += leader_reward
                episode_follower1_reward += follower1_reward
                episode_follower2_reward += follower2_reward
                steps += 1
                
                # Update networks if enough experiences are available
                if len(self.buffer) >= self.batch_size:
                    experiences = self.buffer.sample(self.batch_size)
                    losses = self.agent.update(experiences, self.batch_size)
                    
                    # Store losses
                    for key, value in losses.items():
                        if key in episode_losses:
                            episode_losses[key].append(value)
                
                # Render if requested
                if render:
                    self.env.render(ax)
                    plt.draw()
                    plt.pause(0.1)  # Short pause to update display
                
                # End episode if done
                if done:
                    break
                
                # Update state
                state = next_state
            
            # End episode in buffer
            self.buffer.end_episode()
            
            if render:
                plt.ioff()  # Turn off interactive mode
            
            # Store episode statistics
            self.training_stats['leader_rewards'].append(episode_leader_reward)
            self.training_stats['follower1_rewards'].append(episode_follower1_reward)
            self.training_stats['follower2_rewards'].append(episode_follower2_reward)
            self.training_stats['completion_steps'].append(steps)
            self.training_stats['completion_rates'].append(float(done))
            
            # Store average losses
            if episode_losses['leader_critic_losses']:
                self.training_stats['leader_critic_losses'].append(np.mean(episode_losses['leader_critic_losses']))
                self.training_stats['leader_policy_losses'].append(np.mean(episode_losses['leader_policy_losses']))
                self.training_stats['follower1_critic_losses'].append(np.mean(episode_losses['follower1_critic_losses']))
                self.training_stats['follower1_policy_losses'].append(np.mean(episode_losses['follower1_policy_losses']))
                self.training_stats['follower2_critic_losses'].append(np.mean(episode_losses['follower2_critic_losses']))
                self.training_stats['follower2_policy_losses'].append(np.mean(episode_losses['follower2_policy_losses']))
                self.training_stats['alpha_leader'].append(np.mean(episode_losses['alpha_leader']))
                self.training_stats['alpha_follower1'].append(np.mean(episode_losses['alpha_follower1']))
                self.training_stats['alpha_follower2'].append(np.mean(episode_losses['alpha_follower2']))
            
            # Print progress
            if episode % 10 == 0 or (n_episodes > 100 and episode % 50 == 0):
                print(f"Episode {episode}/{n_episodes}: "
                      f"Leader Reward = {episode_leader_reward:.2f}, "
                      f"Follower1 Reward = {episode_follower1_reward:.2f}, "
                      f"Follower2 Reward = {episode_follower2_reward:.2f}, "
                      f"Steps = {steps}, "
                      f"Alpha L/F1/F2 = {self.agent.alpha_leader:.3f}/{self.agent.alpha_follower1:.3f}/{self.agent.alpha_follower2:.3f}")
            
            # Save checkpoint for long training runs
            if n_episodes >= 1000 and episode > 0 and episode % 200 == 0:
                print(f"Saving checkpoint at episode {episode}...")
                self.agent.save(f"checkpoints/three_robot_sac_episode_{episode}")
                
                # Save training statistics
                checkpoint = {
                    'episode': episode,
                    'training_stats': self.training_stats
                }
                
                try:
                    os.makedirs('checkpoints', exist_ok=True)
                    with open(f'checkpoints/three_robot_sac_stats_ep{episode}.pkl', 'wb') as f:
                        pickle.dump(checkpoint, f)
                except Exception as e:
                    print(f"Warning: Could not save checkpoint: {e}")
        
        print("Training complete!")
        
        # Save final model
        self.agent.save("checkpoints/three_robot_sac_final")
        
        return self.training_stats
    
    def evaluate(self, n_episodes=10, render=False):
        """
        Evaluate the trained agents.
        
        Parameters:
        - n_episodes: Number of episodes to evaluate
        - render: Whether to render the evaluation episodes
        
        Returns:
        - Evaluation statistics
        """
        eval_stats = {
            'leader_rewards': [],
            'follower1_rewards': [],
            'follower2_rewards': [],
            'completion_steps': [],
            'completion_rates': []
        }
        
        print(f"Evaluating for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            # Reset environment and agent hidden states
            self.env.reset_env()
            self.agent.reset_hidden_states()
            state, _ = self.env.get_current_state()
            
            episode_leader_reward = 0
            episode_follower1_reward = 0
            episode_follower2_reward = 0
            steps = 0
            
            # Create figure for rendering if needed
            if render:
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                plt.ion()  # Turn on interactive mode
            
            # Run the episode
            for step in range(self.n_steps_per_episode):
                # Select actions using current policy (deterministic for evaluation)
                leader_action, follower1_action, follower2_action = self.agent.act(state, deterministic=True)
                
                # Get rewards and update environment
                leader_reward, follower1_reward, follower2_reward = self.env.reward(
                    state, leader_action, follower1_action, follower2_action)
                self.env.step(leader_action, follower1_action, follower2_action)
                next_state, _ = self.env.get_current_state()
                
                # Update statistics
                episode_leader_reward += leader_reward
                episode_follower1_reward += follower1_reward
                episode_follower2_reward += follower2_reward
                steps += 1
                
                # Render if requested
                if render:
                    self.env.render(ax)
                    plt.draw()
                    plt.pause(0.2)  # Longer pause to view the simulation
                
                # Check if episode is done
                if self.env.is_done():
                    break
                
                # Update state
                state = next_state
            
            if render:
                plt.ioff()  # Turn off interactive mode
            
            # Store episode statistics
            eval_stats['leader_rewards'].append(episode_leader_reward)
            eval_stats['follower1_rewards'].append(episode_follower1_reward)
            eval_stats['follower2_rewards'].append(episode_follower2_reward)
            eval_stats['completion_steps'].append(steps)
            eval_stats['completion_rates'].append(float(self.env.is_done()))
            
            print(f"Evaluation Episode {episode+1}/{n_episodes}: "
                  f"Leader Reward = {episode_leader_reward:.2f}, "
                  f"Follower1 Reward = {episode_follower1_reward:.2f}, "
                  f"Follower2 Reward = {episode_follower2_reward:.2f}, "
                  f"Steps = {steps}")
        
        # Calculate averages
        avg_leader_reward = np.mean(eval_stats['leader_rewards'])
        avg_follower1_reward = np.mean(eval_stats['follower1_rewards'])
        avg_follower2_reward = np.mean(eval_stats['follower2_rewards'])
        avg_steps = np.mean(eval_stats['completion_steps'])
        completion_rate = np.mean(eval_stats['completion_rates']) * 100
        
        print(f"Evaluation Results (over {n_episodes} episodes):")
        print(f"Average Leader Reward: {avg_leader_reward:.2f}")
        print(f"Average Follower1 Reward: {avg_follower1_reward:.2f}")
        print(f"Average Follower2 Reward: {avg_follower2_reward:.2f}")
        print(f"Average Steps: {avg_steps:.2f}")
        print(f"Task Completion Rate: {completion_rate:.1f}%")
        
        return eval_stats
    
    def visualize_training_stats(self):
        """
        Visualize the training statistics.
        
        Returns:
        - Matplotlib figure with training plots
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        
        # Plot rewards
        axes[0, 0].plot(self.training_stats['leader_rewards'], label='Leader')
        axes[0, 0].plot(self.training_stats['follower1_rewards'], label='Follower1')
        axes[0, 0].plot(self.training_stats['follower2_rewards'], label='Follower2')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Rewards per Episode')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot completion steps
        axes[0, 1].plot(self.training_stats['completion_steps'])
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Completion Steps per Episode')
        axes[0, 1].grid(True)
        
        # Plot completion rates (using a moving average)
        window_size = min(50, len(self.training_stats['completion_rates']))
        completion_rates = np.array(self.training_stats['completion_rates'])
        moving_avg = np.convolve(completion_rates, np.ones(window_size)/window_size, mode='valid')
        axes[0, 2].plot(moving_avg * 100)
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Completion Rate (%)')
        axes[0, 2].set_title(f'Task Completion Rate (Moving Avg, Window={window_size})')
        axes[0, 2].grid(True)
        
        # Plot critic losses
        if self.training_stats.get('leader_critic_losses'):
            axes[1, 0].plot(self.training_stats['leader_critic_losses'], label='Leader')
            axes[1, 0].plot(self.training_stats['follower1_critic_losses'], label='Follower1')
            axes[1, 0].plot(self.training_stats['follower2_critic_losses'], label='Follower2')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Critic Loss')
            axes[1, 0].set_title('Critic Losses')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot policy losses
        if self.training_stats.get('leader_policy_losses'):
            axes[1, 1].plot(self.training_stats['leader_policy_losses'], label='Leader')
            axes[1, 1].plot(self.training_stats['follower1_policy_losses'], label='Follower1')
            axes[1, 1].plot(self.training_stats['follower2_policy_losses'], label='Follower2')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Policy Loss')
            axes[1, 1].set_title('Policy Losses')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # Plot alpha values
        if self.training_stats.get('alpha_leader'):
            axes[1, 2].plot(self.training_stats['alpha_leader'], label='Leader')
            axes[1, 2].plot(self.training_stats['alpha_follower1'], label='Follower1')
            axes[1, 2].plot(self.training_stats['alpha_follower2'], label='Follower2')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Alpha')
            axes[1, 2].set_title('Entropy Coefficient (Alpha)')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        # Plot cumulative rewards
        cum_leader_rewards = np.cumsum(self.training_stats['leader_rewards'])
        cum_follower1_rewards = np.cumsum(self.training_stats['follower1_rewards'])
        cum_follower2_rewards = np.cumsum(self.training_stats['follower2_rewards'])
        axes[2, 0].plot(cum_leader_rewards, label='Leader')
        axes[2, 0].plot(cum_follower1_rewards, label='Follower1')
        axes[2, 0].plot(cum_follower2_rewards, label='Follower2')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Cumulative Reward')
        axes[2, 0].set_title('Cumulative Rewards')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # Plot moving average rewards
        window_size = min(50, len(self.training_stats['leader_rewards']))
        rewards_l = np.array(self.training_stats['leader_rewards'])
        rewards_f1 = np.array(self.training_stats['follower1_rewards'])
        rewards_f2 = np.array(self.training_stats['follower2_rewards'])
        
        moving_avg_l = np.convolve(rewards_l, np.ones(window_size)/window_size, mode='valid')
        moving_avg_f1 = np.convolve(rewards_f1, np.ones(window_size)/window_size, mode='valid')
        moving_avg_f2 = np.convolve(rewards_f2, np.ones(window_size)/window_size, mode='valid')
        
        axes[2, 1].plot(moving_avg_l, label='Leader')
        axes[2, 1].plot(moving_avg_f1, label='Follower1')
        axes[2, 1].plot(moving_avg_f2, label='Follower2')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Average Reward')
        axes[2, 1].set_title(f'Moving Average Rewards (Window={window_size})')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        # Plot reward distribution (histogram)
        axes[2, 2].hist(rewards_l, alpha=0.5, label='Leader', bins=20)
        axes[2, 2].hist(rewards_f1, alpha=0.5, label='Follower1', bins=20)
        axes[2, 2].hist(rewards_f2, alpha=0.5, label='Follower2', bins=20)
        axes[2, 2].set_xlabel('Reward')
        axes[2, 2].set_ylabel('Frequency')
        axes[2, 2].set_title('Reward Distribution')
        axes[2, 2].legend()
        axes[2, 2].grid(True)
        
        plt.tight_layout()
        plt.show()
        return fig


def run_three_robot_sac_simulation():
    """
    Run a simulation using the three-robot SAC implementation.
    """
    # Define simulation parameters
    parameters = {
        'task_id': 1,
        'seed': 42,
        'device': 'cpu',
        'batch_size': 32,
        'buffer_size': 10000,
        'sequence_length': 8,
        'episode_size': 1000,
        'step_per_episode': 40,
        'max_time_steps': 100,
        'franka_failure_prob': 0.1,
        'ur10_failure_prob': 0.1,
        'kuka_failure_prob': 0.1,
        'hidden_size': 64,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'alpha': 0.2,
        'auto_entropy_tuning': True
    }
    
    # Create the simulation
    sim = StackelbergThreeRobotSACSimulation(parameters)
    
    # Train the agent
    print("Starting three-robot SAC training with 1000 episodes...")
    train_stats = sim.train(n_episodes=1000, render_interval=None)
    
    # Evaluate the trained policy
    eval_stats = sim.evaluate(n_episodes=5, render=True)
    
    # Visualize training statistics
    fig = sim.visualize_training_stats()
    
    return sim


if __name__ == "__main__":
    # Set a specific backend that should be more stable
    import matplotlib
    matplotlib.use('TkAgg')  # You can also try 'Agg' for non-interactive use
    
    # Display backend information
    print(f"Using matplotlib backend: {matplotlib.get_backend()}")
    
    try:
        # Run SAC simulation
        sac_sim = run_three_robot_sac_simulation()
        
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()