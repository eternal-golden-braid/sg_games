"""
Rainbow DQN implementation for multi-robot coordination.

This module provides the Rainbow DQN agent implementation that combines multiple
improvements to DQN for more efficient and stable learning in the multi-robot task.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from typing import Dict, Tuple, List, Optional, Union, Any

from agents.sg_agent_base import BaseAgent
from models.noisy_linear_network import NoisyLinear
from models.rainbow_dqn_network import RainbowNetwork
from models.qmix_network import QMIXNetwork


class StackelbergRainbowAgent(BaseAgent):
    """
    Agent implementation using Rainbow DQN for Stackelberg games with three robots.
    
    This agent combines multiple improvements to DQN:
    - Double Q-learning
    - Prioritized Experience Replay
    - Dueling architecture
    - Multi-step learning
    - Distributional RL
    - Noisy Networks
    
    Attributes:
        hidden_size (int): Size of hidden layers
        n_atoms (int): Number of atoms in the distribution
        v_min (float): Minimum support value
        v_max (float): Maximum support value
        gamma (float): Discount factor for future rewards
        tau (float): Soft update parameter for target network
        update_every (int): How often to update the target network
        prioritized_replay (bool): Whether to use prioritized experience replay
        alpha (float): Priority exponent for PER
        beta (float): Initial importance sampling weight exponent for PER
        noisy (bool): Whether to use noisy networks for exploration
        n_step (int): Number of steps for multi-step learning
    """
    def __init__(self, state_dim: int, action_dim_leader: int, action_dim_follower1: int, action_dim_follower2: int,
                 hidden_size: int = 64, n_atoms: int = 51, v_min: float = -10, v_max: float = 10,
                 device: str = 'cpu', learning_rate: float = 1e-4, gamma: float = 0.99,
                 tau: float = 0.01, update_every: int = 10, prioritized_replay: bool = True,
                 alpha: float = 0.6, beta: float = 0.4, noisy: bool = True, n_step: int = 3,
                 seed: int = 42, debug: bool = False):
        """
        Initialize the Stackelberg Rainbow agent for three robots.
        
        Args:
            state_dim: Dimension of the state space
            action_dim_leader: Dimension of the leader's action space
            action_dim_follower1: Dimension of follower1's action space
            action_dim_follower2: Dimension of follower2's action space
            hidden_size: Size of hidden layers
            n_atoms: Number of atoms in the distribution
            v_min: Minimum support value
            v_max: Maximum support value
            device: Device to run the model on (cpu or cuda)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            tau: Soft update parameter for target network
            update_every: How often to update the target network
            prioritized_replay: Whether to use prioritized experience replay
            alpha: Priority exponent for PER
            beta: Initial importance sampling weight exponent for PER
            noisy: Whether to use noisy networks for exploration
            n_step: Number of steps for multi-step learning
            seed: Random seed
            debug: Whether to print debug information
        """
        super().__init__(state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, device, seed)
        
        self.hidden_size = hidden_size
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.prioritized_replay = prioritized_replay
        self.alpha = alpha
        self.beta = beta
        self.noisy = noisy
        self.n_step = n_step
        self.debug = debug
        
        # Initialize support for the distribution
        self.support = torch.linspace(v_min, v_max, n_atoms).to(device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Initialize networks
        self.leader_online = RainbowNetwork(
            state_dim, action_dim_leader, hidden_size, n_atoms, v_min, v_max, noisy).to(device)
        self.leader_target = RainbowNetwork(
            state_dim, action_dim_leader, hidden_size, n_atoms, v_min, v_max, noisy).to(device)
        
        self.follower1_online = RainbowNetwork(
            state_dim, action_dim_follower1, hidden_size, n_atoms, v_min, v_max, noisy).to(device)
        self.follower1_target = RainbowNetwork(
            state_dim, action_dim_follower1, hidden_size, n_atoms, v_min, v_max, noisy).to(device)
        
        self.follower2_online = RainbowNetwork(
            state_dim, action_dim_follower2, hidden_size, n_atoms, v_min, v_max, noisy).to(device)
        self.follower2_target = RainbowNetwork(
            state_dim, action_dim_follower2, hidden_size, n_atoms, v_min, v_max, noisy).to(device)
        
        # Initialize target networks with same weights as online networks
        self.leader_target.load_state_dict(self.leader_online.state_dict())
        self.follower1_target.load_state_dict(self.follower1_online.state_dict())
        self.follower2_target.load_state_dict(self.follower2_online.state_dict())
        
        # Initialize optimizers
        self.leader_optimizer = optim.Adam(self.leader_online.parameters(), lr=learning_rate)
        self.follower1_optimizer = optim.Adam(self.follower1_online.parameters(), lr=learning_rate)
        self.follower2_optimizer = optim.Adam(self.follower2_online.parameters(), lr=learning_rate)
        
        # Initialize training step counter
        self.t_step = 0
    
    def compute_stackelberg_equilibrium(self, state: np.ndarray, 
                                  action_masks: Optional[Dict[str, np.ndarray]] = None) -> Tuple[int, int, int]:
        """
        Compute Stackelberg equilibrium using the current networks.
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
        
        # Get Q-values for all possible actions
        with torch.no_grad():
            leader_q_values = self.leader_online.get_q_values(state_tensor)
            follower1_q_values = self.follower1_online.get_q_values(state_tensor)
            follower2_q_values = self.follower2_online.get_q_values(state_tensor)
        
        # Apply action masks
        leader_q_values = self.apply_action_mask(leader_q_values, leader_mask)
        follower1_q_values = self.apply_action_mask(follower1_q_values, follower1_mask)
        follower2_q_values = self.apply_action_mask(follower2_q_values, follower2_mask)
        
        # Convert to numpy for easier manipulation
        leader_q = leader_q_values.cpu().numpy()
        follower1_q = follower1_q_values.cpu().numpy()
        follower2_q = follower2_q_values.cpu().numpy()
        
        # Make sure q values are the right shape (batch_size, action_dim)
        if len(leader_q.shape) == 1:
            leader_q = leader_q.reshape(1, -1)
        if len(follower1_q.shape) == 1:
            follower1_q = follower1_q.reshape(1, -1)
        if len(follower2_q.shape) == 1:
            follower2_q = follower2_q.reshape(1, -1)
        
        # Convert masks to numpy
        leader_mask_np = leader_mask.cpu().numpy()
        follower1_mask_np = follower1_mask.cpu().numpy()
        follower2_mask_np = follower2_mask.cpu().numpy()
        
        # Find the best action for the leader by simulating followers' responses
        best_leader_value = float('-inf')
        leader_se_action = 0
        follower1_se_action = 0
        follower2_se_action = 0
        
        # Loop through all possible leader actions
        for a_l in range(self.action_dim_leader):
            if not leader_mask_np[a_l]:
                continue  # Skip invalid leader actions
            
            # Initialize with a suboptimal solution
            f1_action, f2_action = 0, 0
            
            # Simple iterative best response for the followers' subgame
            for _ in range(5):  # Few iterations usually converge
                # Follower 1's best response to current follower 2's action
                valid_f1_actions = np.where(follower1_mask_np)[0]
                if len(valid_f1_actions) > 0:
                    f1_best_response = valid_f1_actions[np.argmax(follower1_q[0, valid_f1_actions])]
                else:
                    f1_best_response = 0
                
                # Follower 2's best response to updated follower 1's action
                valid_f2_actions = np.where(follower2_mask_np)[0]
                if len(valid_f2_actions) > 0:
                    f2_best_response = valid_f2_actions[np.argmax(follower2_q[0, valid_f2_actions])]
                else:
                    f2_best_response = 0
                
                # Update actions
                if f1_action == f1_best_response and f2_action == f2_best_response:
                    break  # Equilibrium reached
                    
                f1_action, f2_action = f1_best_response, f2_best_response
            
            # Evaluate leader's utility with this followers' equilibrium
            leader_value = leader_q[0, a_l]
            
            if leader_value > best_leader_value:
                best_leader_value = leader_value
                leader_se_action = a_l
                follower1_se_action = f1_action
                follower2_se_action = f2_action
        
        # Convert from index to actual action (-1 to n-2, where n is action_dim)
        leader_action = leader_se_action - 1 if leader_se_action > 0 else -1
        follower1_action = follower1_se_action - 1 if follower1_se_action > 0 else -1
        follower2_action = follower2_se_action - 1 if follower2_se_action > 0 else -1
        
        if self.debug:
            print(f"Leader action: {leader_action}, Follower1 action: {follower1_action}, Follower2 action: {follower2_action}")
        
        return leader_action, follower1_action, follower2_action
    
    def act(self, state: np.ndarray, action_masks: Optional[Dict[str, np.ndarray]] = None) -> Tuple[int, int, int]:
        """
        Select actions using the current policy.
        
        Args:
            state: Current environment state
            action_masks: Dictionary of action masks for each agent
            
        Returns:
            Leader action, follower1 action, and follower2 action
        """
        # When using noisy networks, we don't need epsilon-greedy exploration
        # Compute Stackelberg equilibrium directly
        return self.compute_stackelberg_equilibrium(state, action_masks)
    
    def save(self, path: str) -> None:
        """
        Save the agent's state.
        
        Args:
            path: Directory to save to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save network state dicts
        torch.save(self.leader_online.state_dict(), f"{path}/leader_online.pt")
        torch.save(self.leader_target.state_dict(), f"{path}/leader_target.pt")
        torch.save(self.follower1_online.state_dict(), f"{path}/follower1_online.pt")
        torch.save(self.follower1_target.state_dict(), f"{path}/follower1_target.pt")
        torch.save(self.follower2_online.state_dict(), f"{path}/follower2_online.pt")
        torch.save(self.follower2_target.state_dict(), f"{path}/follower2_target.pt")
        
        # Save parameters
        params = {
            "hidden_size": self.hidden_size,
            "n_atoms": self.n_atoms,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "gamma": self.gamma,
            "tau": self.tau,
            "update_every": self.update_every,
            "prioritized_replay": self.prioritized_replay,
            "alpha": self.alpha,
            "beta": self.beta,
            "noisy": self.noisy,
            "n_step": self.n_step,
            "t_step": self.t_step
        }
        
        with open(f"{path}/params.pkl", "wb") as f:
            pickle.dump(params, f)

    def project_distribution(self, next_distribution, rewards, dones, gamma):
        """Project categorical distribution onto support for distributional RL."""
        batch_size = rewards.shape[0]
        projected_distribution = torch.zeros(batch_size, self.n_atoms, device=self.device)
        
        for i in range(batch_size):
            if dones[i]:
                # Terminal state handling
                tz = torch.clamp(rewards[i], self.v_min, self.v_max)
                b = ((tz - self.v_min) / self.delta_z).floor().long()
                l = ((tz - self.v_min) / self.delta_z) - b.float()
                b = torch.clamp(b, 0, self.n_atoms - 1)
                projected_distribution[i, b] += (1 - l)
                if b < self.n_atoms - 1:
                    projected_distribution[i, b + 1] += l
            else:
                # Non-terminal state Bellman update
                for j in range(self.n_atoms):
                    tz = torch.clamp(rewards[i] + gamma * self.support[j], self.v_min, self.v_max)
                    b = ((tz - self.v_min) / self.delta_z).floor().long()
                    l = ((tz - self.v_min) / self.delta_z) - b.float()
                    b = torch.clamp(b, 0, self.n_atoms - 1)
                    projected_distribution[i, b] += (1 - l) * next_distribution[i, j]
                    if b < self.n_atoms - 1:
                        projected_distribution[i, b + 1] += l * next_distribution[i, j]
        
        return projected_distribution

    def soft_update(self, online_model, target_model):
        """Soft update target network parameters."""
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data)
    
    def update(self, experiences: Tuple[List[Tuple], np.ndarray, np.ndarray]) -> Tuple[float, float, float]:
        """
        Update the networks using a batch of experiences.
        
        Args:
            experiences: Tuple containing:
                - List of experience tuples (s, a_l, a_f1, a_f2, r_l, r_f1, r_f2, s_next, done)
                - Indices of the sampled experiences
                - Importance sampling weights
                
        Returns:
            Losses for leader, follower1, and follower2
        """
        samples, indices, weights = experiences
        
        # Process experiences
        states = []
        leader_actions = []
        follower1_actions = []
        follower2_actions = []
        leader_rewards = []
        follower1_rewards = []
        follower2_rewards = []
        next_states = []
        dones = []
        
        for exp in samples:
            s, a_l, a_f1, a_f2, r_l, r_f1, r_f2, s_next, done = exp
            
            # Convert actions to indices (add 1 to handle -1 actions)
            a_l_idx = a_l + 1
            a_f1_idx = a_f1 + 1
            a_f2_idx = a_f2 + 1
            
            states.append(s)
            leader_actions.append(a_l_idx)
            follower1_actions.append(a_f1_idx)
            follower2_actions.append(a_f2_idx)
            leader_rewards.append(r_l)
            follower1_rewards.append(r_f1)
            follower2_rewards.append(r_f2)
            next_states.append(s_next)
            dones.append(done)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        leader_actions = torch.tensor(leader_actions, dtype=torch.long).to(self.device)
        follower1_actions = torch.tensor(follower1_actions, dtype=torch.long).to(self.device)
        follower2_actions = torch.tensor(follower2_actions, dtype=torch.long).to(self.device)
        leader_rewards = torch.tensor(leader_rewards, dtype=torch.float).to(self.device)
        follower1_rewards = torch.tensor(follower1_rewards, dtype=torch.float).to(self.device)
        follower2_rewards = torch.tensor(follower2_rewards, dtype=torch.float).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float).to(self.device).unsqueeze(1)
        
        batch_size = states.shape[0]
        
        # Reset noisy layers
        self.leader_online.reset_noise()
        self.leader_target.reset_noise()
        self.follower1_online.reset_noise()
        self.follower1_target.reset_noise()
        self.follower2_online.reset_noise()
        self.follower2_target.reset_noise()
        
        # Calculate TD errors for prioritized replay
        leader_td_errors = []
        follower1_td_errors = []
        follower2_td_errors = []
        
        # Update Leader network
        # Get current distributions
        leader_dist = self.leader_online(states)
        
        # Get target distributions using Double DQN
        with torch.no_grad():
            # Select actions using online network
            next_leader_q = self.leader_online.get_q_values(next_states)
            next_leader_actions = next_leader_q.argmax(dim=1)
            
            # Get distributional targets using target network
            next_leader_dist = self.leader_target(next_states)
            
            # Extract target distribution for selected actions
            next_leader_dist_selected = next_leader_dist[torch.arange(batch_size), next_leader_actions]
            
            # Project distribution
            leader_target_dist = self.project_distribution(
                next_leader_dist_selected, leader_rewards, dones, self.gamma**self.n_step)
        
        # Extract current distribution for taken actions
        leader_dist_selected = leader_dist[torch.arange(batch_size), leader_actions]
        
        # Calculate cross-entropy loss
        leader_loss = -(leader_target_dist * torch.log(leader_dist_selected + 1e-10)).sum(dim=1)
        
        # Apply importance sampling weights if using prioritized replay
        if self.prioritized_replay:
            leader_loss = (leader_loss * weights).mean()
        else:
            leader_loss = leader_loss.mean()
        
        # Update network
        self.leader_optimizer.zero_grad()
        leader_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.leader_online.parameters(), 1.0)
        self.leader_optimizer.step()
        
        # Calculate TD errors for PER
        leader_td_errors = torch.abs(leader_dist_selected - leader_target_dist).sum(dim=1).detach().cpu().numpy()
        
        # Similar updates for Follower1 and Follower2 networks
        # Follower1 update
        follower1_dist = self.follower1_online(states)
        
        with torch.no_grad():
            next_follower1_q = self.follower1_online.get_q_values(next_states)
            next_follower1_actions = next_follower1_q.argmax(dim=1)
            
            next_follower1_dist = self.follower1_target(next_states)
            next_follower1_dist_selected = next_follower1_dist[torch.arange(batch_size), next_follower1_actions]
            
            follower1_target_dist = self.project_distribution(
                next_follower1_dist_selected, follower1_rewards, dones, self.gamma**self.n_step)
        
        follower1_dist_selected = follower1_dist[torch.arange(batch_size), follower1_actions]
        
        follower1_loss = -(follower1_target_dist * torch.log(follower1_dist_selected + 1e-10)).sum(dim=1)
        
        if self.prioritized_replay:
            follower1_loss = (follower1_loss * weights).mean()
        else:
            follower1_loss = follower1_loss.mean()
        
        self.follower1_optimizer.zero_grad()
        follower1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.follower1_online.parameters(), 1.0)
        self.follower1_optimizer.step()
        
        follower1_td_errors = torch.abs(follower1_dist_selected - follower1_target_dist).sum(dim=1).detach().cpu().numpy()
        
        # Follower2 update
        follower2_dist = self.follower2_online(states)
        
        with torch.no_grad():
            next_follower2_q = self.follower2_online.get_q_values(next_states)
            next_follower2_actions = next_follower2_q.argmax(dim=1)
            
            next_follower2_dist = self.follower2_target(next_states)
            next_follower2_dist_selected = next_follower2_dist[torch.arange(batch_size), next_follower2_actions]
            
            follower2_target_dist = self.project_distribution(
                next_follower2_dist_selected, follower2_rewards, dones, self.gamma**self.n_step)
        
        follower2_dist_selected = follower2_dist[torch.arange(batch_size), follower2_actions]
        
        follower2_loss = -(follower2_target_dist * torch.log(follower2_dist_selected + 1e-10)).sum(dim=1)
        
        if self.prioritized_replay:
            follower2_loss = (follower2_loss * weights).mean()
        else:
            follower2_loss = follower2_loss.mean()
        
        self.follower2_optimizer.zero_grad()
        follower2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.follower2_online.parameters(), 1.0)
        self.follower2_optimizer.step()
        
        follower2_td_errors = torch.abs(follower2_dist_selected - follower2_target_dist).sum(dim=1).detach().cpu().numpy()
        
        # Soft update of target networks
        self.t_step += 1
        if self.t_step % self.update_every == 0:
            self.soft_update(self.leader_online, self.leader_target)
            self.soft_update(self.follower1_online, self.follower1_target)
            self.soft_update(self.follower2_online, self.follower2_target)
        
        # Compute mean TD errors for each sequence (for prioritized replay)
        mean_td_errors = (leader_td_errors + follower1_td_errors + follower2_td_errors) / 3.0
        
        return mean_td_errors, leader_loss.item(), follower1_loss.item(), follower2_loss.item()