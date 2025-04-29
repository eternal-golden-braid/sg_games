"""
Stackelberg DDQN with Prioritized Experience Replay for Three-Robot Coordination
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import pickle
from collections import deque
import torch.nn.functional as F
from scipy import stats

# Import the environment class
from battery_disassembly_env import BatteryDisassemblyEnv


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for more efficient sampling
    """
    def __init__(self, buffer_size, batch_size, alpha=0.6, beta=0.4, beta_increment=0.001, seed=42):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling weight exponent
        self.beta_increment = beta_increment  # Beta increment per sampling
        self.rng = np.random.default_rng(seed)
        
        # Initialize buffer and priorities
        self.buffer = []
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def __len__(self):
        """Get the current size of the buffer."""
        return self.size
    
    def add(self, experience, priority=None):
        """
        Add an experience to the buffer with priority.
        
        Parameters:
        - experience: Experience to add [state, a_leader, a_follower1, a_follower2, r_leader, r_follower1, r_follower2, next_state]
        - priority: Priority of the experience (if None, max priority is used)
        """
        if priority is None:
            priority = 1.0 if self.size == 0 else np.max(self.priorities[:self.size])
        
        # If buffer is not full
        if self.size < self.buffer_size:
            self.buffer.append(experience)
            self.size += 1
        else:
            # Replace an old experience
            self.buffer[self.position] = experience
        
        # Update priority
        self.priorities[self.position] = priority
        
        # Update position
        self.position = (self.position + 1) % self.buffer_size
    
    def sample(self, batch_size=None):
        """
        Sample a batch of experiences from the buffer based on priorities.
        
        Parameters:
        - batch_size: Size of batch to sample (uses default if None)
        
        Returns:
        - Batch of experiences, indices of sampled experiences, and importance sampling weights
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if self.size < batch_size:
            raise ValueError(f"Buffer contains {self.size} experiences, but requested batch size is {batch_size}")
        
        # Calculate sampling probabilities from priorities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= np.sum(probabilities)
        
        # Sample indices based on probabilities
        indices = self.rng.choice(self.size, batch_size, replace=False, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)  # Normalize weights
        
        # Increment beta for next sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get sampled experiences
        samples = [self.buffer[idx] for idx in indices]
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities of experiences.
        
        Parameters:
        - indices: Indices of experiences to update
        - priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network implementation for DDQN.
    Separates state value and advantage streams for better value estimation.
    """
    def __init__(self, input_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, hidden_size=64):
        super(DuelingQNetwork, self).__init__()
        self.input_dim = input_dim
        self.action_dim_leader = action_dim_leader
        self.action_dim_follower1 = action_dim_follower1
        self.action_dim_follower2 = action_dim_follower2
        self.hidden_size = hidden_size
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        
        # Value stream (state value)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream (action advantages)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, action_dim_leader * action_dim_follower1 * action_dim_follower2)
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
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Parameters:
        - state: Batch of states [batch_size, state_dim]
        
        Returns:
        - Q-values for all action combinations
        """
        # Extract features
        features = self.feature_extractor(state)
        
        # Compute state value
        value = self.value_stream(features)
        
        # Compute action advantages
        advantages = self.advantage_stream(features)
        advantages = advantages.view(-1, self.action_dim_leader, self.action_dim_follower1, self.action_dim_follower2)
        
        # Combine value and advantages (dueling architecture)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + (
            advantages - advantages.mean(dim=(1, 2, 3), keepdim=True))
        
        return q_values


class StackelbergDDQNPERAgent:
    """
    Agent implementation using Double DQN with Prioritized Experience Replay for Stackelberg games with three robots.
    """
    def __init__(self, state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, 
                 hidden_size=64, device='cpu', learning_rate=1e-4,
                 gamma=0.9, epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01,
                 tau=0.01, update_every=10, buffer_size=10000, batch_size=32, 
                 alpha=0.6, beta=0.4, seed=42):
        """
        Initialize the Stackelberg DDQN agent with PER for three robots.
        
        Parameters:
        - state_dim: Dimension of the state space
        - action_dim_leader: Dimension of the leader's action space
        - action_dim_follower1: Dimension of follower1's action space
        - action_dim_follower2: Dimension of follower2's action space
        - hidden_size: Hidden layer size in the network
        - device: Device to run the model on (cpu or cuda)
        - learning_rate: Learning rate for optimizer
        - gamma: Discount factor for future rewards
        - epsilon: Exploration rate
        - epsilon_decay: Rate at which epsilon decays over time
        - epsilon_min: Minimum value for epsilon
        - tau: Soft update parameter for target network
        - update_every: How often to update the target network
        - buffer_size: Size of replay buffer
        - batch_size: Size of mini-batch for training
        - alpha: Priority exponent for PER
        - beta: Initial importance sampling weight exponent for PER
        - seed: Random seed
        """
        self.state_dim = state_dim
        self.action_dim_leader = action_dim_leader
        self.action_dim_follower1 = action_dim_follower1
        self.action_dim_follower2 = action_dim_follower2
        self.hidden_size = hidden_size
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.update_every = update_every
        self.seed = seed
        self.batch_size = batch_size
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize leader and follower networks
        self.leader_online = DuelingQNetwork(
            state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, 
            hidden_size).to(device)
        self.leader_target = DuelingQNetwork(
            state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, 
            hidden_size).to(device)
        self.follower1_online = DuelingQNetwork(
            state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, 
            hidden_size).to(device)
        self.follower1_target = DuelingQNetwork(
            state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, 
            hidden_size).to(device)
        self.follower2_online = DuelingQNetwork(
            state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, 
            hidden_size).to(device)
        self.follower2_target = DuelingQNetwork(
            state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, 
            hidden_size).to(device)
        
        # Initialize target networks with same weights as online networks
        self.leader_target.load_state_dict(self.leader_online.state_dict())
        self.follower1_target.load_state_dict(self.follower1_online.state_dict())
        self.follower2_target.load_state_dict(self.follower2_online.state_dict())
        
        # Initialize optimizers
        self.leader_optimizer = optim.Adam(self.leader_online.parameters(), lr=learning_rate)
        self.follower1_optimizer = optim.Adam(self.follower1_online.parameters(), lr=learning_rate)
        self.follower2_optimizer = optim.Adam(self.follower2_online.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = PrioritizedReplayBuffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            alpha=alpha,
            beta=beta,
            seed=seed
        )
        
        # Initialize training step counter
        self.t_step = 0
    
    def compute_stackelberg_equilibrium(self, state):
        """
        Compute Stackelberg equilibrium using the current Q-networks.
        In this hierarchy: Leader -> (Follower1, Follower2)
        
        Parameters:
        - state: Current environment state
        
        Returns:
        - leader_action, follower1_action, follower2_action: Equilibrium actions
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values for all possible action combinations
        leader_q_values = self.leader_online(state_tensor)
        follower1_q_values = self.follower1_online(state_tensor)
        follower2_q_values = self.follower2_online(state_tensor)
        
        # Convert to numpy for easier manipulation
        leader_q = leader_q_values.detach().cpu().numpy()[0]
        follower1_q = follower1_q_values.detach().cpu().numpy()[0]
        follower2_q = follower2_q_values.detach().cpu().numpy()[0]
        
        # For each potential leader action, compute the Nash equilibrium between the followers
        best_leader_value = float('-inf')
        leader_se_action = 0
        follower1_se_action = 0
        follower2_se_action = 0
        
        for a_l in range(self.action_dim_leader):
            # For this leader action, find the equilibrium between followers
            # This is a simpler subgame where each follower responds to the leader and the other follower
            
            # Initialize with a suboptimal solution
            f1_action, f2_action = 0, 0
            
            # Simple iterative best response for the followers' subgame
            # In practice, this could be replaced with a more sophisticated equilibrium solver
            for _ in range(10):  # Few iterations usually converge
                # Follower 1's best response to current follower 2's action
                f1_best_response = np.argmax(follower1_q[a_l, :, f2_action])
                
                # Follower 2's best response to updated follower 1's action
                f2_best_response = np.argmax(follower2_q[a_l, f1_best_response, :])
                
                # Update actions
                if f1_action == f1_best_response and f2_action == f2_best_response:
                    break  # Equilibrium reached
                    
                f1_action, f2_action = f1_best_response, f2_best_response
            
            # Evaluate leader's utility with this followers' equilibrium
            leader_value = leader_q[a_l, f1_action, f2_action]
            
            if leader_value > best_leader_value:
                best_leader_value = leader_value
                leader_se_action = a_l
                follower1_se_action = f1_action
                follower2_se_action = f2_action
        
        # Convert from index to actual action (-1 to n-2, where n is action_dim)
        return leader_se_action - 1, follower1_se_action - 1, follower2_se_action - 1
    
    def act(self, state, epsilon=None):
        """
        Select actions according to epsilon-greedy policy.
        
        Parameters:
        - state: Current environment state
        - epsilon: Exploration rate (uses default if None)
        
        Returns:
        - leader_action, follower1_action, follower2_action: Selected actions
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # With probability epsilon, select random actions
        if np.random.random() < epsilon:
            leader_action = np.random.randint(-1, self.action_dim_leader - 1)
            follower1_action = np.random.randint(-1, self.action_dim_follower1 - 1)
            follower2_action = np.random.randint(-1, self.action_dim_follower2 - 1)
            return leader_action, follower1_action, follower2_action
        
        # Otherwise, compute and return Stackelberg equilibrium actions
        return self.compute_stackelberg_equilibrium(state)
    
    def step(self, state, action_leader, action_f1, action_f2, reward_leader, reward_f1, reward_f2, next_state, done):
        """
        Process a step of experience and learn from it.
        
        Parameters:
        - state: Current state
        - action_leader, action_f1, action_f2: Actions taken by each robot
        - reward_leader, reward_f1, reward_f2: Rewards received
        - next_state: Next state
        - done: Whether the episode is done
        """
        # Convert actions to indices (add 1 to handle -1 actions)
        action_leader_idx = action_leader + 1
        action_f1_idx = action_f1 + 1
        action_f2_idx = action_f2 + 1
        
        # Create experience tuple
        experience = (
            state, 
            action_leader_idx, action_f1_idx, action_f2_idx, 
            reward_leader, reward_f1, reward_f2, 
            next_state, 
            done
        )
        
        # Add experience to memory with max priority (will be updated during learning)
        self.memory.add(experience)
        
        # Learn from experiences when enough samples are available
        self.t_step += 1
        if len(self.memory) > self.batch_size and self.t_step % self.update_every == 0:
            self.learn()
    
    def learn(self):
        """
        Update the Q-networks using a batch of experiences from prioritized replay buffer.
        """
        # Sample a batch of experiences from memory
        experiences, indices, weights = self.memory.sample(self.batch_size)
        
        # Convert experiences to tensors
        states = torch.tensor([exp[0] for exp in experiences], dtype=torch.float).to(self.device)
        actions_leader = torch.tensor([exp[1] for exp in experiences], dtype=torch.long).to(self.device)
        actions_f1 = torch.tensor([exp[2] for exp in experiences], dtype=torch.long).to(self.device)
        actions_f2 = torch.tensor([exp[3] for exp in experiences], dtype=torch.long).to(self.device)
        rewards_leader = torch.tensor([exp[4] for exp in experiences], dtype=torch.float).to(self.device)
        rewards_f1 = torch.tensor([exp[5] for exp in experiences], dtype=torch.float).to(self.device)
        rewards_f2 = torch.tensor([exp[6] for exp in experiences], dtype=torch.float).to(self.device)
        next_states = torch.tensor([exp[7] for exp in experiences], dtype=torch.float).to(self.device)
        dones = torch.tensor([exp[8] for exp in experiences], dtype=torch.float).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float).to(self.device)
        
        # Compute current Q values
        current_q_leader = self.leader_online(states)
        current_q_f1 = self.follower1_online(states)
        current_q_f2 = self.follower2_online(states)
        
        # Gather Q-values for the actions taken
        batch_indices = torch.arange(self.batch_size)
        q_leader = current_q_leader[batch_indices, actions_leader, actions_f1, actions_f2]
        q_f1 = current_q_f1[batch_indices, actions_leader, actions_f1, actions_f2]
        q_f2 = current_q_f2[batch_indices, actions_leader, actions_f1, actions_f2]
        
        # Compute next state target values using Double DQN
        with torch.no_grad():
            # Get best actions from online networks
            next_q_leader_online = self.leader_online(next_states)
            next_q_f1_online = self.follower1_online(next_states)
            next_q_f2_online = self.follower2_online(next_states)
            
            # Compute Stackelberg equilibrium for each sample in batch
            next_leader_actions = []
            next_f1_actions = []
            next_f2_actions = []
            
            for b in range(self.batch_size):
                # For each batch sample, compute Stackelberg equilibrium
                # This is a similar process to compute_stackelberg_equilibrium but for a single batch sample
                best_leader_value = float('-inf')
                best_leader_action = 0
                best_f1_action = 0
                best_f2_action = 0
                
                # Get Q-values for this batch sample
                leader_q = next_q_leader_online[b].detach().cpu().numpy()
                f1_q = next_q_f1_online[b].detach().cpu().numpy()
                f2_q = next_q_f2_online[b].detach().cpu().numpy()
                
                for a_l in range(self.action_dim_leader):
                    # Find equilibrium between followers for this leader action
                    f1_action, f2_action = 0, 0
                    
                    # Iterative best response
                    for _ in range(5):
                        f1_best_response = np.argmax(f1_q[a_l, :, f2_action])
                        f2_best_response = np.argmax(f2_q[a_l, f1_best_response, :])
                        
                        if f1_action == f1_best_response and f2_action == f2_best_response:
                            break
                            
                        f1_action, f2_action = f1_best_response, f2_best_response
                    
                    # Evaluate leader's utility
                    leader_value = leader_q[a_l, f1_action, f2_action]
                    
                    if leader_value > best_leader_value:
                        best_leader_value = leader_value
                        best_leader_action = a_l
                        best_f1_action = f1_action
                        best_f2_action = f2_action
                
                next_leader_actions.append(best_leader_action)
                next_f1_actions.append(best_f1_action)
                next_f2_actions.append(best_f2_action)
            
            # Convert to tensors
            next_leader_actions = torch.tensor(next_leader_actions, device=self.device)
            next_f1_actions = torch.tensor(next_f1_actions, device=self.device)
            next_f2_actions = torch.tensor(next_f2_actions, device=self.device)
            
            # Get Q-values from target networks for selected actions
            next_q_leader_target = self.leader_target(next_states)
            next_q_f1_target = self.follower1_target(next_states)
            next_q_f2_target = self.follower2_target(next_states)
            
            # Extract target Q-values
            target_q_leader = next_q_leader_target[batch_indices, next_leader_actions, next_f1_actions, next_f2_actions]
            target_q_f1 = next_q_f1_target[batch_indices, next_leader_actions, next_f1_actions, next_f2_actions]
            target_q_f2 = next_q_f2_target[batch_indices, next_leader_actions, next_f1_actions, next_f2_actions]
            
            # Compute target values
            target_leader = rewards_leader + (1 - dones) * self.gamma * target_q_leader
            target_f1 = rewards_f1 + (1 - dones) * self.gamma * target_q_f1
            target_f2 = rewards_f2 + (1 - dones) * self.gamma * target_q_f2
        
        # Compute TD errors for PER priority updates
        td_error_leader = torch.abs(q_leader - target_leader).detach().cpu().numpy()
        td_error_f1 = torch.abs(q_f1 - target_f1).detach().cpu().numpy()
        td_error_f2 = torch.abs(q_f2 - target_f2).detach().cpu().numpy()
        
        # Total TD error (used for priority updates)
        total_td_error = td_error_leader + td_error_f1 + td_error_f2
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, total_td_error + 1e-5)  # Small constant to avoid zero priority
        
        # Compute weighted MSE loss
        leader_loss = (weights * F.mse_loss(q_leader, target_leader, reduction='none')).mean()
        f1_loss = (weights * F.mse_loss(q_f1, target_f1, reduction='none')).mean()
        f2_loss = (weights * F.mse_loss(q_f2, target_f2, reduction='none')).mean()
        
        # Update online networks
        # Leader
        self.leader_optimizer.zero_grad()
        leader_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.leader_online.parameters(), 1.0)
        self.leader_optimizer.step()
        
        # Follower 1
        self.follower1_optimizer.zero_grad()
        f1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.follower1_online.parameters(), 1.0)
        self.follower1_optimizer.step()
        
        # Follower 2
        self.follower2_optimizer.zero_grad()
        f2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.follower2_online.parameters(), 1.0)
        self.follower2_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.leader_online, self.leader_target)
        self.soft_update(self.follower1_online, self.follower1_target)
        self.soft_update(self.follower2_online, self.follower2_target)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return leader_loss.item(), f1_loss.item(), f2_loss.item()
    
    def soft_update(self, online_model, target_model):
        """
        Soft update of target network parameters.
        θ_target = τ*θ_online + (1 - τ)*θ_target
        
        Parameters:
        - online_model: Online network
        - target_model: Target network
        """
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, path):
        """
        Save the agent's state.
        
        Parameters:
        - path: Directory to save to
        """
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.leader_online.state_dict(), f"{path}/leader_online.pt")
        torch.save(self.leader_target.state_dict(), f"{path}/leader_target.pt")
        torch.save(self.follower1_online.state_dict(), f"{path}/follower1_online.pt")
        torch.save(self.follower1_target.state_dict(), f"{path}/follower1_target.pt")
        torch.save(self.follower2_online.state_dict(), f"{path}/follower2_online.pt")
        torch.save(self.follower2_target.state_dict(), f"{path}/follower2_target.pt")
        
        params = {
            "epsilon": self.epsilon,
            "t_step": self.t_step
        }
        
        with open(f"{path}/params.pkl", "wb") as f:
            pickle.dump(params, f)
    
    def load(self, path):
        """
        Load the agent's state.
        
        Parameters:
        - path: Directory to load from
        """
        self.leader_online.load_state_dict(torch.load(f"{path}/leader_online.pt"))
        self.leader_target.load_state_dict(torch.load(f"{path}/leader_target.pt"))
        self.follower1_online.load_state_dict(torch.load(f"{path}/follower1_online.pt"))
        self.follower1_target.load_state_dict(torch.load(f"{path}/follower1_target.pt"))
        self.follower2_online.load_state_dict(torch.load(f"{path}/follower2_online.pt"))
        self.follower2_target.load_state_dict(torch.load(f"{path}/follower2_target.pt"))
        
        with open(f"{path}/params.pkl", "rb") as f:
            params = pickle.load(f)
            self.epsilon = params["epsilon"]
            self.t_step = params["t_step"]


class StackelbergDDQNPERSimulation:
    """
    Simulation class for the Stackelberg game using DDQN with PER for three robots.
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
        self.agent = StackelbergDDQNPERAgent(
            state_dim=state_dim,
            action_dim_leader=action_dim_leader,
            action_dim_follower1=action_dim_follower1,
            action_dim_follower2=action_dim_follower2,
            hidden_size=parameters.get('hidden_size', 64),
            device=self.device,
            learning_rate=parameters.get('learning_rate', 1e-4),
            gamma=parameters.get('gamma', 0.9),
            epsilon=parameters.get('epsilon', 0.1),
            epsilon_decay=parameters.get('epsilon_decay', 0.995),
            epsilon_min=parameters.get('epsilon_min', 0.01),
            tau=parameters.get('tau', 0.01),
            update_every=parameters.get('update_every', 10),
            buffer_size=parameters.get('buffer_size', 10000),
            batch_size=parameters.get('batch_size', 32),
            alpha=parameters.get('alpha', 0.6),
            beta=parameters.get('beta', 0.4),
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
            'leader_losses': [],
            'follower1_losses': [],
            'follower2_losses': []
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
                
                # Check if episode is done
                done = self.env.is_done()
                
                # Store experience
                self.agent.step(state, leader_action, follower1_action, follower2_action,
                               leader_reward, follower1_reward, follower2_reward, next_state, done)
                
                # Check if done
                if done:
                    break
                
                state = next_state
        
        print(f"Initial buffer size: {len(self.agent.memory)}")
    
    def train(self, n_episodes=None, render_interval=None):
        """
        Train the agents using DDQN with PER.
        
        Parameters:
        - n_episodes: Number of episodes to train (uses default if None)
        - render_interval: How often to render an episode (None for no rendering)
        
        Returns:
        - Training statistics
        """
        if n_episodes is None:
            n_episodes = self.n_episodes
        
        # Generate initial experiences if buffer is empty
        if len(self.agent.memory) < self.batch_size:
            self.generate_initial_buffer()
        
        print(f"Starting training for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            # Reset environment and agent hidden states
            self.env.reset_env()
            state, _ = self.env.get_current_state()
            
            episode_leader_reward = 0
            episode_follower1_reward = 0
            episode_follower2_reward = 0
            episode_leader_losses = []
            episode_follower1_losses = []
            episode_follower2_losses = []
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
                
                # Check if episode is done
                done = self.env.is_done()
                
                # Store experience and learn
                self.agent.step(state, leader_action, follower1_action, follower2_action,
                               leader_reward, follower1_reward, follower2_reward, next_state, done)
                
                # Update statistics
                episode_leader_reward += leader_reward
                episode_follower1_reward += follower1_reward
                episode_follower2_reward += follower2_reward
                steps += 1
                
                # Render if requested
                if render:
                    self.env.render(ax)
                    plt.draw()
                    plt.pause(0.1)  # Short pause to update display
                
                # Check if episode is done
                if done:
                    break
                
                # Update state
                state = next_state
            
            if render:
                plt.ioff()  # Turn off interactive mode
                plt.close()
            
            # Store episode statistics
            self.training_stats['leader_rewards'].append(episode_leader_reward)
            self.training_stats['follower1_rewards'].append(episode_follower1_reward)
            self.training_stats['follower2_rewards'].append(episode_follower2_reward)
            self.training_stats['completion_steps'].append(steps)
            self.training_stats['completion_rates'].append(float(done))
            
            # Print progress
            if episode % 10 == 0 or (n_episodes > 100 and episode % 50 == 0):
                print(f"Episode {episode}/{n_episodes}: "
                      f"Leader Reward = {episode_leader_reward:.2f}, "
                      f"Follower1 Reward = {episode_follower1_reward:.2f}, "
                      f"Follower2 Reward = {episode_follower2_reward:.2f}, "
                      f"Steps = {steps}, "
                      f"Epsilon = {self.agent.epsilon:.3f}")
            
            # Save checkpoint for long training runs
            if n_episodes >= 1000 and episode > 0 and episode % 200 == 0:
                print(f"Saving checkpoint at episode {episode}...")
                self.agent.save(f"checkpoints/ddqn_per_episode_{episode}")
                
                # Save training statistics
                checkpoint = {
                    'episode': episode,
                    'leader_rewards': self.training_stats['leader_rewards'],
                    'follower1_rewards': self.training_stats['follower1_rewards'],
                    'follower2_rewards': self.training_stats['follower2_rewards'],
                    'completion_steps': self.training_stats['completion_steps'],
                    'completion_rates': self.training_stats['completion_rates'],
                    'leader_losses': self.training_stats['leader_losses'],
                    'follower1_losses': self.training_stats['follower1_losses'],
                    'follower2_losses': self.training_stats['follower2_losses']
                }
                
                try:
                    os.makedirs('checkpoints', exist_ok=True)
                    with open(f'checkpoints/ddqn_per_stats_ep{episode}.pkl', 'wb') as f:
                        pickle.dump(checkpoint, f)
                except Exception as e:
                    print(f"Warning: Could not save checkpoint: {e}")
        
        print("Training complete!")
        
        # Save final model
        self.agent.save("checkpoints/ddqn_per_final")
        
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
        
        # Set agent to evaluation mode (epsilon=0)
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0
        
        for episode in range(n_episodes):
            # Reset environment
            self.env.reset_env()
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
                # Select actions using current policy (no exploration)
                leader_action, follower1_action, follower2_action = self.agent.act(state, epsilon=0)
                
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
                done = self.env.is_done()
                if done:
                    break
                
                # Update state
                state = next_state
            
            if render:
                plt.ioff()  # Turn off interactive mode
                plt.close()
            
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
        
        # Restore agent's original epsilon
        self.agent.epsilon = original_epsilon
        
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
        fig, axes = plt.subplots(3, 2, figsize=(14, 16))
        
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
        axes[1, 0].plot(moving_avg * 100)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Completion Rate (%)')
        axes[1, 0].set_title(f'Task Completion Rate (Moving Avg, Window={window_size})')
        axes[1, 0].grid(True)
        
        # Plot cumulative rewards
        cum_leader_rewards = np.cumsum(self.training_stats['leader_rewards'])
        cum_follower1_rewards = np.cumsum(self.training_stats['follower1_rewards'])
        cum_follower2_rewards = np.cumsum(self.training_stats['follower2_rewards'])
        axes[1, 1].plot(cum_leader_rewards, label='Leader')
        axes[1, 1].plot(cum_follower1_rewards, label='Follower1')
        axes[1, 1].plot(cum_follower2_rewards, label='Follower2')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Cumulative Reward')
        axes[1, 1].set_title('Cumulative Rewards')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Plot losses if available
        if self.training_stats.get('leader_losses'):
            axes[2, 0].plot(self.training_stats['leader_losses'], label='Leader')
            axes[2, 0].plot(self.training_stats['follower1_losses'], label='Follower1')
            axes[2, 0].plot(self.training_stats['follower2_losses'], label='Follower2')
            axes[2, 0].set_xlabel('Episode')
            axes[2, 0].set_ylabel('Loss')
            axes[2, 0].set_title('TD Losses')
            axes[2, 0].legend()
            axes[2, 0].grid(True)
            
            # Plot exploration decay
            episodes = np.arange(len(self.training_stats['leader_losses']))
            epsilon_values = 0.1 * np.power(0.995, episodes)
            axes[2, 1].plot(episodes, epsilon_values)
            axes[2, 1].set_xlabel('Episode')
            axes[2, 1].set_ylabel('Epsilon')
            axes[2, 1].set_title('Exploration Rate Decay')
            axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        return fig


def run_stackelberg_ddqn_per_simulation(n_episodes=200, render_interval=None):
    """
    Run a simulation using the Stackelberg DDQN with PER implementation.
    
    Parameters:
    - n_episodes: Number of episodes to train
    - render_interval: How often to render an episode (None for no rendering)
    
    Returns:
    - Simulation object and evaluation statistics
    """
    # Define simulation parameters
    parameters = {
        'task_id': 1,
        'seed': 42,
        'device': 'cpu',
        'batch_size': 32,
        'buffer_size': 10000,
        'update_every': 10,
        'episode_size': n_episodes,
        'step_per_episode': 40,
        'max_time_steps': 100,
        'franka_failure_prob': 0.1,
        'ur10_failure_prob': 0.1,
        'kuka_failure_prob': 0.1,
        'hidden_size': 64,
        'learning_rate': 1e-4,
        'gamma': 0.9,
        'epsilon': 0.1,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'tau': 0.01,
        'alpha': 0.6,  # PER priority exponent
        'beta': 0.4    # PER importance sampling exponent
    }
    
    # Create the simulation
    sim = StackelbergDDQNPERSimulation(parameters)
    
    # Train the agent
    print(f"Starting Stackelberg DDQN with PER training with {n_episodes} episodes...")
    train_stats = sim.train(n_episodes=n_episodes, render_interval=render_interval)
    
    # Evaluate the trained policy
    eval_stats = sim.evaluate(n_episodes=5, render=False)
    
    # Visualize training statistics
    fig = sim.visualize_training_stats()
    
    return sim, eval_stats


if __name__ == "__main__":
    # Run the simulation
    simulation, eval_results = run_stackelberg_ddqn_per_simulation(n_episodes=200)