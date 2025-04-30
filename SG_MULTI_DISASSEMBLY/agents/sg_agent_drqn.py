import torch
import torch.optim as optim
import numpy as np
import os
import pickle
from models.recurrent_q_network import RecurrentQNetwork

class StackelbergThreeRobotDRQNAgent:
    """
    Agent implementation using Deep Recurrent Q-Networks for Stackelberg games with three robots.
    """
    def __init__(self, state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, 
                 hidden_size=64, sequence_length=8, device='cpu', learning_rate=1e-4,
                 gamma=0.9, epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01,
                 tau=0.01, update_every=10, seed=42):
        """
        Initialize the Stackelberg DRQN agent for three robots.
        
        Parameters:
        - state_dim: Dimension of the state space
        - action_dim_leader: Dimension of the leader's action space
        - action_dim_follower1: Dimension of follower1's action space
        - action_dim_follower2: Dimension of follower2's action space
        - hidden_size: Hidden layer size in the recurrent network
        - sequence_length: Length of sequences for training
        - device: Device to run the model on (cpu or cuda)
        - learning_rate: Learning rate for optimizer
        - gamma: Discount factor for future rewards
        - epsilon: Exploration rate
        - epsilon_decay: Rate at which epsilon decays over time
        - epsilon_min: Minimum value for epsilon
        - tau: Soft update parameter for target network
        - update_every: How often to update the target network
        - seed: Random seed
        """
        self.state_dim = state_dim
        self.action_dim_leader = action_dim_leader
        self.action_dim_follower1 = action_dim_follower1
        self.action_dim_follower2 = action_dim_follower2
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.update_every = update_every
        self.seed = seed
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize leader and follower networks
        self.leader_online = RecurrentQNetwork(
            state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, 
            hidden_size).to(device)
        self.leader_target = RecurrentQNetwork(
            state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, 
            hidden_size).to(device)
        self.follower1_online = RecurrentQNetwork(
            state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, 
            hidden_size).to(device)
        self.follower1_target = RecurrentQNetwork(
            state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, 
            hidden_size).to(device)
        self.follower2_online = RecurrentQNetwork(
            state_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, 
            hidden_size).to(device)
        self.follower2_target = RecurrentQNetwork(
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
        
        # Initialize hidden states
        self.leader_hidden = None
        self.follower1_hidden = None
        self.follower2_hidden = None
        
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
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Get Q-values for all possible action combinations
        leader_q_values, self.leader_hidden = self.leader_online.get_q_values(
            state_tensor, self.leader_hidden)
        follower1_q_values, self.follower1_hidden = self.follower1_online.get_q_values(
            state_tensor, self.follower1_hidden)
        follower2_q_values, self.follower2_hidden = self.follower2_online.get_q_values(
            state_tensor, self.follower2_hidden)
        
        # Convert to numpy for easier manipulation
        leader_q = leader_q_values.detach().cpu().numpy()
        follower1_q = follower1_q_values.detach().cpu().numpy()
        follower2_q = follower2_q_values.detach().cpu().numpy()
        
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
    
    def reset_hidden_states(self):
        """Reset the hidden states for all agents."""
        self.leader_hidden = None
        self.follower1_hidden = None
        self.follower2_hidden = None
    
    def compute_stackelberg_equilibrium_from_q_values(self, leader_q_values, follower1_q_values, follower2_q_values):
        """
        Compute Stackelberg equilibrium using pre-computed Q-values.
        
        Parameters:
        - leader_q_values: Leader's Q-values tensor
        - follower1_q_values: Follower1's Q-values tensor
        - follower2_q_values: Follower2's Q-values tensor
        
        Returns:
        - leader_action, follower1_action, follower2_action: Equilibrium actions
        """
        # Convert Q-values to numpy for easier manipulation
        leader_q = leader_q_values.detach().cpu().numpy()
        follower1_q = follower1_q_values.detach().cpu().numpy()
        follower2_q = follower2_q_values.detach().cpu().numpy()
        
        # For each potential leader action, compute the Nash equilibrium between the followers
        best_leader_value = float('-inf')
        leader_se_action = 0
        follower1_se_action = 0
        follower2_se_action = 0
        
        for a_l in range(self.action_dim_leader):
            # For this leader action, find the equilibrium between followers
            # Initialize with a suboptimal solution
            f1_action, f2_action = 0, 0
            
            # Simple iterative best response for the followers' subgame
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
        """Select actions with proper state handling."""
        # Handle raw state from environment
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).to(self.device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims
            elif len(state_tensor.shape) == 2:
                state_tensor = state_tensor.unsqueeze(0)  # Add batch dim
        else:
            # Already a tensor
            state_tensor = state.to(self.device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
            elif len(state_tensor.shape) == 2:
                state_tensor = state_tensor.unsqueeze(0)
        
        # Get action using epsilon-greedy approach
        if epsilon is None:
            epsilon = self.epsilon
        
        if np.random.random() < epsilon:
            leader_action = np.random.randint(-1, self.action_dim_leader - 1)
            follower1_action = np.random.randint(-1, self.action_dim_follower1 - 1)
            follower2_action = np.random.randint(-1, self.action_dim_follower2 - 1)
            return leader_action, follower1_action, follower2_action
        
        # Use policy to select actions
        leader_q_values, self.leader_hidden = self.leader_online.get_q_values(state_tensor, self.leader_hidden)
        follower1_q_values, self.follower1_hidden = self.follower1_online.get_q_values(state_tensor, self.follower1_hidden)
        follower2_q_values, self.follower2_hidden = self.follower2_online.get_q_values(state_tensor, self.follower2_hidden)
        
        # Compute Stackelberg equilibrium
        return self.compute_stackelberg_equilibrium_from_q_values(leader_q_values, follower1_q_values, follower2_q_values)
    
    def update(self, experiences):
        """
        Update the Q-networks using a batch of experiences.
        
        Parameters:
        - experiences: List of (state, action, reward, next_state, done) tuples
        """
        # Convert experiences to tensors
        states = []
        leader_actions = []
        follower1_actions = []
        follower2_actions = []
        leader_rewards = []
        follower1_rewards = []
        follower2_rewards = []
        next_states = []
        dones = []
        
        # Process each sequence of experiences
        for sequence in experiences:
            seq_states = []
            seq_leader_actions = []
            seq_follower1_actions = []
            seq_follower2_actions = []
            seq_leader_rewards = []
            seq_follower1_rewards = []
            seq_follower2_rewards = []
            seq_next_states = []
            seq_dones = []
            
            for exp in sequence:
                s, a_l, a_f1, a_f2, r_l, r_f1, r_f2, s_next = exp
                # Convert actions to indices (add 1 to handle -1 actions)
                a_l_idx = a_l + 1
                a_f1_idx = a_f1 + 1
                a_f2_idx = a_f2 + 1
                # Determine if state is terminal
                done = np.all(s_next == 0)
                
                seq_states.append(s)
                seq_leader_actions.append(a_l_idx)
                seq_follower1_actions.append(a_f1_idx)
                seq_follower2_actions.append(a_f2_idx)
                seq_leader_rewards.append(r_l)
                seq_follower1_rewards.append(r_f1)
                seq_follower2_rewards.append(r_f2)
                seq_next_states.append(s_next)
                seq_dones.append(done)
            
            states.append(seq_states)
            leader_actions.append(seq_leader_actions)
            follower1_actions.append(seq_follower1_actions)
            follower2_actions.append(seq_follower2_actions)
            leader_rewards.append(seq_leader_rewards)
            follower1_rewards.append(seq_follower1_rewards)
            follower2_rewards.append(seq_follower2_rewards)
            next_states.append(seq_next_states)
            dones.append(seq_dones)
        
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
        
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Compute Q-values for current states
        leader_q_values, _ = self.leader_online(states)
        follower1_q_values, _ = self.follower1_online(states)
        follower2_q_values, _ = self.follower2_online(states)
        
        # Reshape for easier indexing
        leader_q_values = leader_q_values.view(batch_size, seq_len, 
                                              self.action_dim_leader, 
                                              self.action_dim_follower1, 
                                              self.action_dim_follower2)
        follower1_q_values = follower1_q_values.view(batch_size, seq_len, 
                                                    self.action_dim_leader, 
                                                    self.action_dim_follower1, 
                                                    self.action_dim_follower2)
        follower2_q_values = follower2_q_values.view(batch_size, seq_len, 
                                                    self.action_dim_leader, 
                                                    self.action_dim_follower1, 
                                                    self.action_dim_follower2)
        
        # Gather Q-values for taken actions
        # This is more complex with 3 dimensions of actions
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, seq_len)
        seq_indices = torch.arange(seq_len).view(1, -1).expand(batch_size, -1)
        
        # Extract Q-values for the actual actions taken
        leader_q = leader_q_values[batch_indices, seq_indices, 
                                   leader_actions, follower1_actions, follower2_actions]
        follower1_q = follower1_q_values[batch_indices, seq_indices, 
                                         leader_actions, follower1_actions, follower2_actions]
        follower2_q = follower2_q_values[batch_indices, seq_indices, 
                                         leader_actions, follower1_actions, follower2_actions]
        
        # Compute next state targets (Stackelberg equilibrium values)
        with torch.no_grad():
            # Initialize target values with rewards
            leader_targets = leader_rewards.clone()
            follower1_targets = follower1_rewards.clone()
            follower2_targets = follower2_rewards.clone()
            
            # For non-terminal states, add discounted future value
            non_terminal_mask = 1 - dones
            
            # Compute target Q-values for next states
            next_leader_q_values, _ = self.leader_target(next_states)
            next_follower1_q_values, _ = self.follower1_target(next_states)
            next_follower2_q_values, _ = self.follower2_target(next_states)
            
            # Reshape for easier processing
            next_leader_q_values = next_leader_q_values.view(batch_size, seq_len, 
                                                           self.action_dim_leader, 
                                                           self.action_dim_follower1, 
                                                           self.action_dim_follower2)
            next_follower1_q_values = next_follower1_q_values.view(batch_size, seq_len, 
                                                                 self.action_dim_leader, 
                                                                 self.action_dim_follower1, 
                                                                 self.action_dim_follower2)
            next_follower2_q_values = next_follower2_q_values.view(batch_size, seq_len, 
                                                                 self.action_dim_leader, 
                                                                 self.action_dim_follower1, 
                                                                 self.action_dim_follower2)
            
            # For each batch and sequence step, compute Stackelberg equilibrium value
            for b in range(batch_size):
                for s in range(seq_len):
                    if not dones[b, s]:
                        # For each potential leader action, find follower equilibrium
                        best_leader_value = float('-inf')
                        best_follower1_value = 0
                        best_follower2_value = 0
                        
                        for a_l in range(self.action_dim_leader):
                            # Find equilibrium between followers for this leader action
                            f1_action, f2_action = 0, 0
                            
                            # Iterative best response for follower equilibrium
                            for _ in range(5):
                                f1_best_response = torch.argmax(next_follower1_q_values[b, s, a_l, :, f2_action])
                                f2_best_response = torch.argmax(next_follower2_q_values[b, s, a_l, f1_best_response, :])
                                
                                if f1_action == f1_best_response and f2_action == f2_best_response:
                                    break
                                    
                                f1_action, f2_action = f1_best_response, f2_best_response
                            
                            # Evaluate leader's utility
                            leader_value = next_leader_q_values[b, s, a_l, f1_action, f2_action]
                            
                            if leader_value > best_leader_value:
                                best_leader_value = leader_value
                                best_follower1_value = next_follower1_q_values[b, s, a_l, f1_action, f2_action]
                                best_follower2_value = next_follower2_q_values[b, s, a_l, f1_action, f2_action]
                        
                        # Add discounted future value to rewards
                        leader_targets[b, s] += self.gamma * best_leader_value
                        follower1_targets[b, s] += self.gamma * best_follower1_value
                        follower2_targets[b, s] += self.gamma * best_follower2_value
        
        # Compute loss for all agents
        leader_loss = torch.nn.functional.mse_loss(leader_q, leader_targets)
        follower1_loss = torch.nn.functional.mse_loss(follower1_q, follower1_targets)
        follower2_loss = torch.nn.functional.mse_loss(follower2_q, follower2_targets)
        
        # Optimize leader network
        self.leader_optimizer.zero_grad()
        leader_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.leader_online.parameters(), 1)  # Gradient clipping
        self.leader_optimizer.step()
        
        # Optimize follower1 network
        self.follower1_optimizer.zero_grad()
        follower1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.follower1_online.parameters(), 1)  # Gradient clipping
        self.follower1_optimizer.step()
        
        # Optimize follower2 network
        self.follower2_optimizer.zero_grad()
        follower2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.follower2_online.parameters(), 1)  # Gradient clipping
        self.follower2_optimizer.step()
        
        # Soft update target networks
        self.t_step += 1
        if self.t_step % self.update_every == 0:
            self.soft_update(self.leader_online, self.leader_target)
            self.soft_update(self.follower1_online, self.follower1_target)
            self.soft_update(self.follower2_online, self.follower2_target)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return leader_loss.item(), follower1_loss.item(), follower2_loss.item()
    
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