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
import random
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
    - Additional exploration strategies
    - Reward normalization
    
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
                 hidden_size: int = 128, n_atoms: int = 51, v_min: float = -50, v_max: float = 10,
                 device: str = 'cpu', learning_rate: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, update_every: int = 1, prioritized_replay: bool = True,
                 alpha: float = 0.4, beta: float = 0.6, beta_annealing: float = 0.001,
                 noisy: bool = True, n_step: int = 3, initial_epsilon: float = 0.5,
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.01,
                 seed: int = 42, debug: bool = True):
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
            beta_annealing: Rate at which beta is annealed towards 1
            noisy: Whether to use noisy networks for exploration
            n_step: Number of steps for multi-step learning
            initial_epsilon: Initial epsilon for epsilon-greedy exploration
            epsilon_decay: Decay rate for epsilon
            min_epsilon: Minimum value for epsilon
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
        self.beta_annealing = beta_annealing
        self.noisy = noisy
        self.n_step = n_step
        self.debug = debug
        
        # Epsilon greedy parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Tracking statistics for rewards
        self.reward_stats = {
            "leader": {"count": 0, "mean": 0, "std": 1, "min": 0, "max": 0},
            "follower1": {"count": 0, "mean": 0, "std": 1, "min": 0, "max": 0},
            "follower2": {"count": 0, "mean": 0, "std": 1, "min": 0, "max": 0}
        }
        
        # For logging and debugging
        self.training_stats = {
            "q_values": [],
            "losses": [],
            "actions": [],
            "rewards": [],
            "td_errors": []
        }
        
        # Initialize support for the distribution
        self.support = torch.linspace(v_min, v_max, n_atoms).to(device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Initialize networks with larger hidden layers
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
        
        # Initialize optimizers with a higher learning rate
        self.leader_optimizer = optim.Adam(self.leader_online.parameters(), lr=learning_rate)
        self.follower1_optimizer = optim.Adam(self.follower1_online.parameters(), lr=learning_rate)
        self.follower2_optimizer = optim.Adam(self.follower2_online.parameters(), lr=learning_rate)
        
        # Learning rate schedulers
        self.leader_scheduler = optim.lr_scheduler.StepLR(self.leader_optimizer, step_size=100, gamma=0.5)
        self.follower1_scheduler = optim.lr_scheduler.StepLR(self.follower1_optimizer, step_size=100, gamma=0.5)
        self.follower2_scheduler = optim.lr_scheduler.StepLR(self.follower2_optimizer, step_size=100, gamma=0.5)
        
        # Initialize training step counter
        self.t_step = 0
        self.episode_count = 0
    
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
            # Reset noise if using noisy networks for consistent evaluation
            if self.noisy:
                self.leader_online.reset_noise()
                self.follower1_online.reset_noise()
                self.follower2_online.reset_noise()
                
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
        
        return leader_action, follower1_action, follower2_action
    
    def simplified_act(self, state: np.ndarray, action_masks: Optional[Dict[str, np.ndarray]] = None) -> Tuple[int, int, int]:
        """
        Simplified action selection that uses greedy action selection for each agent without
        trying to compute the Stackelberg equilibrium. Useful for debugging.
        
        Args:
            state: Current environment state
            action_masks: Dictionary of action masks for each agent
            
        Returns:
            Leader action, follower1 action, and follower2 action
        """
        # Ensure state is a tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).to(self.device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
        else:
            state_tensor = state.to(self.device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
        
        # Process action masks if provided
        if action_masks is not None:
            if isinstance(action_masks, dict):
                leader_mask = torch.tensor(action_masks['leader'], dtype=torch.bool, device=self.device)
                follower1_mask = torch.tensor(action_masks['follower1'], dtype=torch.bool, device=self.device)
                follower2_mask = torch.tensor(action_masks['follower2'], dtype=torch.bool, device=self.device)
            else:
                leader_mask, follower1_mask, follower2_mask = action_masks
        else:
            leader_mask = torch.ones(self.action_dim_leader, dtype=torch.bool, device=self.device)
            follower1_mask = torch.ones(self.action_dim_follower1, dtype=torch.bool, device=self.device)
            follower2_mask = torch.ones(self.action_dim_follower2, dtype=torch.bool, device=self.device)
        
        # Get Q-values
        with torch.no_grad():
            if self.noisy:
                self.leader_online.reset_noise()
                self.follower1_online.reset_noise()
                self.follower2_online.reset_noise()
                
            leader_q = self.leader_online.get_q_values(state_tensor)
            follower1_q = self.follower1_online.get_q_values(state_tensor)
            follower2_q = self.follower2_online.get_q_values(state_tensor)
        
        # Apply masks
        leader_q_masked = leader_q.clone()
        follower1_q_masked = follower1_q.clone()
        follower2_q_masked = follower2_q.clone()
        
        # Set masked actions to very negative values
        leader_q_masked[0, ~leader_mask] = float('-inf')
        follower1_q_masked[0, ~follower1_mask] = float('-inf')
        follower2_q_masked[0, ~follower2_mask] = float('-inf')
        
        # Select greedy actions
        leader_action = leader_q_masked[0].argmax().item() - 1
        follower1_action = follower1_q_masked[0].argmax().item() - 1
        follower2_action = follower2_q_masked[0].argmax().item() - 1
        
        return leader_action, follower1_action, follower2_action
    
    def act(self, state: np.ndarray, action_masks: Optional[Dict[str, np.ndarray]] = None, 
            use_simplified: bool = False) -> Tuple[int, int, int]:
        """
        Select actions using the current policy with epsilon-greedy exploration.
        
        Args:
            state: Current environment state
            action_masks: Dictionary of action masks for each agent
            use_simplified: Whether to use the simplified action selection
            
        Returns:
            Leader action, follower1 action, and follower2 action
        """
        # Random exploration based on epsilon
        if random.random() < self.epsilon:
            # Random exploration - select random valid actions
            if action_masks is not None:
                if isinstance(action_masks, dict):
                    valid_leader_actions = np.where(action_masks['leader'])[0]
                    valid_follower1_actions = np.where(action_masks['follower1'])[0]
                    valid_follower2_actions = np.where(action_masks['follower2'])[0]
                else:
                    valid_leader_actions = np.where(action_masks[0].cpu().numpy())[0]
                    valid_follower1_actions = np.where(action_masks[1].cpu().numpy())[0]
                    valid_follower2_actions = np.where(action_masks[2].cpu().numpy())[0]
            else:
                valid_leader_actions = np.arange(self.action_dim_leader)
                valid_follower1_actions = np.arange(self.action_dim_follower1)
                valid_follower2_actions = np.arange(self.action_dim_follower2)
            
            # Select random valid actions
            leader_idx = np.random.choice(valid_leader_actions)
            follower1_idx = np.random.choice(valid_follower1_actions)
            follower2_idx = np.random.choice(valid_follower2_actions)
            
            # Convert from index to actual action
            leader_action = leader_idx - 1 if leader_idx > 0 else -1
            follower1_action = follower1_idx - 1 if follower1_idx > 0 else -1
            follower2_action = follower2_idx - 1 if follower2_idx > 0 else -1
            
            return leader_action, follower1_action, follower2_action
        
        # Use current policy
        if use_simplified:
            return self.simplified_act(state, action_masks)
        else:
            return self.compute_stackelberg_equilibrium(state, action_masks)
    
    def decay_epsilon(self):
        """Decay epsilon for epsilon-greedy exploration."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        if self.debug:
            print(f"Epsilon decayed to {self.epsilon:.4f}")
    
    def update_reward_stats(self, reward_type: str, rewards: np.ndarray):
        """
        Update reward statistics for normalization.
        
        Args:
            reward_type: Type of reward ('leader', 'follower1', or 'follower2')
            rewards: Array of rewards
        """
        stats = self.reward_stats[reward_type]
        old_count = stats["count"]
        stats["count"] += len(rewards)
        
        if old_count == 0:
            stats["mean"] = np.mean(rewards)
            stats["std"] = np.std(rewards) + 1e-8
            stats["min"] = np.min(rewards)
            stats["max"] = np.max(rewards)
        else:
            # Update running statistics
            delta = np.mean(rewards) - stats["mean"]
            stats["mean"] += delta * len(rewards) / stats["count"]
            stats["std"] = np.sqrt(((stats["std"] ** 2) * old_count + np.std(rewards) ** 2 * len(rewards)) / stats["count"]) + 1e-8
            stats["min"] = min(stats["min"], np.min(rewards))
            stats["max"] = max(stats["max"], np.max(rewards))
    
    def normalize_reward(self, reward_type: str, reward: float) -> float:
        """
        Normalize a reward value based on collected statistics.
        
        Args:
            reward_type: Type of reward ('leader', 'follower1', or 'follower2')
            reward: Raw reward value
            
        Returns:
            Normalized reward value
        """
        stats = self.reward_stats[reward_type]
        
        if stats["count"] < 100:  # Use raw rewards until we have enough data
            return reward
        
        # Clip to reasonable range
        normalized = (reward - stats["mean"]) / stats["std"]
        return np.clip(normalized, -10, 10)
    
    def apply_action_mask(self, q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply action mask to Q-values.
        
        Args:
            q_values: Q-values tensor
            mask: Boolean mask tensor
            
        Returns:
            Masked Q-values tensor
        """
        masked_q = q_values.clone()
        masked_q[0, ~mask] = float('-inf')
        return masked_q
    
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
            "beta_annealing": self.beta_annealing,
            "noisy": self.noisy,
            "n_step": self.n_step,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "t_step": self.t_step,
            "episode_count": self.episode_count,
            "reward_stats": self.reward_stats
        }
        
        with open(f"{path}/params.pkl", "wb") as f:
            pickle.dump(params, f)
        
        # Save training statistics
        with open(f"{path}/training_stats.pkl", "wb") as f:
            pickle.dump(self.training_stats, f)
    
    def load(self, path: str) -> None:
        """
        Load the agent's state.
        
        Args:
            path: Directory to load from
        """
        # Load network state dicts
        self.leader_online.load_state_dict(torch.load(f"{path}/leader_online.pt"))
        self.leader_target.load_state_dict(torch.load(f"{path}/leader_target.pt"))
        self.follower1_online.load_state_dict(torch.load(f"{path}/follower1_online.pt"))
        self.follower1_target.load_state_dict(torch.load(f"{path}/follower1_target.pt"))
        self.follower2_online.load_state_dict(torch.load(f"{path}/follower2_online.pt"))
        self.follower2_target.load_state_dict(torch.load(f"{path}/follower2_target.pt"))
        
        # Load parameters
        with open(f"{path}/params.pkl", "rb") as f:
            params = pickle.load(f)
        
        self.hidden_size = params["hidden_size"]
        self.n_atoms = params["n_atoms"]
        self.v_min = params["v_min"]
        self.v_max = params["v_max"]
        self.gamma = params["gamma"]
        self.tau = params["tau"]
        self.update_every = params["update_every"]
        self.prioritized_replay = params["prioritized_replay"]
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.beta_annealing = params.get("beta_annealing", 0.001)
        self.noisy = params["noisy"]
        self.n_step = params["n_step"]
        self.epsilon = params.get("epsilon", 0.01)
        self.epsilon_decay = params.get("epsilon_decay", 0.995)
        self.min_epsilon = params.get("min_epsilon", 0.01)
        self.t_step = params["t_step"]
        self.episode_count = params.get("episode_count", 0)
        self.reward_stats = params.get("reward_stats", self.reward_stats)
        
        # Load training statistics if available
        try:
            with open(f"{path}/training_stats.pkl", "rb") as f:
                self.training_stats = pickle.load(f)
        except FileNotFoundError:
            print("Training statistics file not found. Using empty statistics.")
            pass
    
    def project_distribution(self, next_distribution, rewards, dones, gamma):
        """
        Project categorical distribution onto support for distributional RL.
        
        Args:
            next_distribution: Next state distribution
            rewards: Rewards
            dones: Done flags
            gamma: Discount factor
            
        Returns:
            Projected distribution
        """
        batch_size = rewards.shape[0]
        projected_distribution = torch.zeros(batch_size, self.n_atoms, device=self.device)
        
        # Ensure rewards are within a reasonable range for numerical stability
        rewards = torch.clamp(rewards, min=self.v_min / 10, max=self.v_max / 10)
        
        for i in range(batch_size):
            if dones[i]:
                # Terminal state handling
                tz = torch.clamp(rewards[i], self.v_min, self.v_max)
                b = ((tz - self.v_min) / self.delta_z).floor().long()
                l = ((tz - self.v_min) / self.delta_z) - b.float()
                b = torch.clamp(b, 0, self.n_atoms - 1)
                
                # Handle potential numerical issues
                l = torch.clamp(l, 0, 1)
                
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
                    
                    # Handle potential numerical issues
                    l = torch.clamp(l, 0, 1)
                    
                    projected_distribution[i, b] += (1 - l) * next_distribution[i, j]
                    if b < self.n_atoms - 1:
                        projected_distribution[i, b + 1] += l * next_distribution[i, j]
        
        # Ensure the distribution sums to 1
        proj_sum = projected_distribution.sum(dim=1, keepdim=True)
        projected_distribution = projected_distribution / (proj_sum + 1e-10)
        
        return projected_distribution
    
    def soft_update(self, online_model, target_model):
        """
        Soft update target network parameters.
        
        Args:
            online_model: Online network
            target_model: Target network
        """
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data)
    
    def log_training_stats(self, episode, states, actions, rewards, losses, td_errors):
        """
        Log detailed training statistics for debugging.
        
        Args:
            episode: Current episode
            states: Batch of states
            actions: Actions
            rewards: Rewards
            losses: Losses
            td_errors: TD errors
        """
        if not self.debug:
            return
        
        # Only log detailed stats every 10 episodes to match PPO style output
        # and avoid cluttering the console with too much information
        if episode % 10 == 0:
            leader_q = self.leader_online.get_q_values(states).mean().item()
            follower1_q = self.follower1_online.get_q_values(states).mean().item()
            follower2_q = self.follower2_online.get_q_values(states).mean().item()
            
            # Convert actions to numpy for bincount
            leader_actions = actions[0].cpu().numpy() + 1  # +1 to handle -1 actions
            follower1_actions = actions[1].cpu().numpy() + 1
            follower2_actions = actions[2].cpu().numpy() + 1
            
            # Store stats for later analysis but don't print them
            # to keep console output clean and match PPO style
            self.training_stats["q_values"].append((leader_q, follower1_q, follower2_q))
            self.training_stats["losses"].append((losses[0], losses[1], losses[2]))
            self.training_stats["actions"].append((leader_actions, follower1_actions, follower2_actions))
            
            leader_rewards = rewards[0].cpu().numpy() if isinstance(rewards[0], torch.Tensor) else rewards[0]
            follower1_rewards = rewards[1].cpu().numpy() if isinstance(rewards[1], torch.Tensor) else rewards[1]
            follower2_rewards = rewards[2].cpu().numpy() if isinstance(rewards[2], torch.Tensor) else rewards[2]
            
            # Calculate mean rewards
            mean_leader_reward = np.mean(leader_rewards) if isinstance(leader_rewards, np.ndarray) else leader_rewards
            mean_follower1_reward = np.mean(follower1_rewards) if isinstance(follower1_rewards, np.ndarray) else follower1_rewards
            mean_follower2_reward = np.mean(follower2_rewards) if isinstance(follower2_rewards, np.ndarray) else follower2_rewards
            
            self.training_stats["rewards"].append((mean_leader_reward, mean_follower1_reward, mean_follower2_reward))
            self.training_stats["td_errors"].append(np.mean(td_errors))
    
    def update(self, experiences: Tuple[List[Tuple], np.ndarray, np.ndarray]) -> Tuple[np.ndarray, float, float, float]:
        """
        Update the networks using a batch of experiences.
        
        Args:
            experiences: Tuple containing:
                - List of experience tuples (s, a_l, a_f1, a_f2, r_l, r_f1, r_f2, s_next, done)
                - Indices of the sampled experiences
                - Importance sampling weights
                
        Returns:
            TD errors, leader loss, follower1 loss, and follower2 loss
        """
        # Print the rewards from the first few experiences for debugging
        if self.debug and len(experiences[0]) > 0:
            sample_size = min(5, len(experiences[0]))
            print("\nSample Rewards from Batch:")
            for i in range(sample_size):
                r_l = experiences[0][i][4]  # 5th element in experience tuple is leader reward
                r_f1 = experiences[0][i][5]  # 6th element is follower1 reward
                r_f2 = experiences[0][i][6]  # 7th element is follower2 reward
                print(f"  Sample {i}: Leader = {r_l:.2f}, Follower1 = {r_f1:.2f}, Follower2 = {r_f2:.2f}")
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
            
            # Update reward statistics for normalization
            self.update_reward_stats('leader', np.array([r_l]))
            self.update_reward_stats('follower1', np.array([r_f1]))
            self.update_reward_stats('follower2', np.array([r_f2]))
            
            # Normalize rewards if we have enough data
            if self.reward_stats['leader']['count'] > 100:
                r_l = self.normalize_reward('leader', r_l)
                r_f1 = self.normalize_reward('follower1', r_f1)
                r_f2 = self.normalize_reward('follower2', r_f2)
            
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
        
        # Anneal beta for prioritized replay
        if self.prioritized_replay:
            self.beta = min(1.0, self.beta + self.beta_annealing)
        
        # Reset noisy layers
        if self.noisy:
            self.leader_online.reset_noise()
            self.leader_target.reset_noise()
            self.follower1_online.reset_noise()
            self.follower1_target.reset_noise()
            self.follower2_online.reset_noise()
            self.follower2_target.reset_noise()
        
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
        
        # Calculate cross-entropy loss with added stability
        leader_loss = -(leader_target_dist * torch.log(leader_dist_selected + 1e-10)).sum(dim=1)
        
        # Apply importance sampling weights if using prioritized replay
        if self.prioritized_replay:
            leader_loss = (leader_loss * weights).mean()
        else:
            leader_loss = leader_loss.mean()
        
        # Update network
        self.leader_optimizer.zero_grad()
        leader_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.leader_online.parameters(), 10.0)  # Higher clip value
        self.leader_optimizer.step()
        
        # Calculate TD errors for PER
        leader_td_errors = torch.abs(leader_dist_selected - leader_target_dist).sum(dim=1).detach().cpu().numpy()
        
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
        torch.nn.utils.clip_grad_norm_(self.follower1_online.parameters(), 10.0)  # Higher clip value
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
        torch.nn.utils.clip_grad_norm_(self.follower2_online.parameters(), 10.0)  # Higher clip value
        self.follower2_optimizer.step()
        
        follower2_td_errors = torch.abs(follower2_dist_selected - follower2_target_dist).sum(dim=1).detach().cpu().numpy()
        
        # Soft update of target networks (more frequent updates)
        self.t_step += 1
        if self.t_step % self.update_every == 0:
            self.soft_update(self.leader_online, self.leader_target)
            self.soft_update(self.follower1_online, self.follower1_target)
            self.soft_update(self.follower2_online, self.follower2_target)
        
        # Step the learning rate schedulers occasionally
        if self.t_step % 1000 == 0:
            self.leader_scheduler.step()
            self.follower1_scheduler.step()
            self.follower2_scheduler.step()
            
            if self.debug:
                print(f"Learning rates updated: {self.leader_optimizer.param_groups[0]['lr']:.6f}")
        
        # Compute mean TD errors for each sequence (for prioritized replay)
        mean_td_errors = (leader_td_errors + follower1_td_errors + follower2_td_errors) / 3.0
        
        # Log training statistics
        self.log_training_stats(
            self.episode_count,
            states,
            (leader_actions, follower1_actions, follower2_actions),
            (leader_rewards, follower1_rewards, follower2_rewards),
            (leader_loss.item(), follower1_loss.item(), follower2_loss.item()),
            mean_td_errors
        )
        
        return mean_td_errors, leader_loss.item(), follower1_loss.item(), follower2_loss.item()
    
    def end_episode(self, episode_rewards=None, steps=None, current_episode=None):
        """
        Update tracking variables at the end of an episode and log episode rewards.
        
        Args:
            episode_rewards: Tuple of (leader_reward, follower1_reward, follower2_reward) 
                            for the completed episode
            steps: Number of steps taken in the episode
            current_episode: Current episode number (if provided externally)
        """
        if current_episode is not None:
            episode_num = current_episode
        else:
            episode_num = self.episode_count
            self.episode_count += 1
            
        self.decay_epsilon()
        
        # Log episode rewards if provided - using PPO style formatting
        if episode_rewards is not None:
            leader_reward, follower1_reward, follower2_reward = episode_rewards
            
            # Format exactly like PPO output
            if steps is not None:
                print(f"Episode {episode_num}/200: Leader Reward = {leader_reward:.2f}, "
                      f"Follower1 Reward = {follower1_reward:.2f}, "
                      f"Follower2 Reward = {follower2_reward:.2f}, "
                      f"Steps = {steps}")
            else:
                print(f"Episode {episode_num}/200: Leader Reward = {leader_reward:.2f}, "
                      f"Follower1 Reward = {follower1_reward:.2f}, "
                      f"Follower2 Reward = {follower2_reward:.2f}")
            
            # Update reward statistics with the episode rewards
            self.update_reward_stats('leader', np.array([leader_reward]))
            self.update_reward_stats('follower1', np.array([follower1_reward]))
            self.update_reward_stats('follower2', np.array([follower2_reward]))