import torch
import torch.nn as nn
from typing import List


class QMIXNetwork(nn.Module):
    """
    QMIX network for value factorization.
    
    This network combines individual agent Q-values into a joint Q-value
    while maintaining monotonicity to enable efficient training.
    
    Attributes:
        state_dim (int): Dimension of the state space
        hidden_size (int): Size of hidden layers
    """
    def __init__(self, state_dim: int, hidden_size: int = 64):
        """
        Initialize the QMIX network.
        
        Args:
            state_dim: Dimension of the state space
            hidden_size: Size of hidden layers
        """
        super(QMIXNetwork, self).__init__()
        
        # Hypernetwork that generates weights for the first layer of mixing network
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)  # 3 agents: leader, follower1, follower2
        )
        
        # Hypernetwork that generates weights for the second layer of mixing network
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Output a single weight for the final layer
        )
        
        # Hypernetwork that generates bias for the first layer of mixing network
        self.hyper_b1 = nn.Linear(state_dim, 1)
        
        # Hypernetwork that generates bias for the second layer of mixing network
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, agent_q_values: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the QMIX network.
        
        Args:
            agent_q_values: Q-values from individual agents [batch_size, 3]
            states: Global states [batch_size, state_dim]
            
        Returns:
            Joint Q-values [batch_size, 1]
        """
        # Get batch size
        batch_size = agent_q_values.shape[0]
        
        # Generate weights and biases from hypernetworks
        w1 = self.hyper_w1(states).view(batch_size, 1, 3)   # [batch_size, 1, 3]
        b1 = self.hyper_b1(states).view(batch_size, 1, 1)   # [batch_size, 1, 1]
        
        # Ensure weights are positive for monotonicity
        w1 = torch.abs(w1)
        
        # First layer of mixing network
        hidden = torch.bmm(w1, agent_q_values.unsqueeze(2)).view(batch_size, 1) + b1.view(batch_size, 1)
        hidden = torch.relu(hidden)
        
        # Generate weights and biases for the second layer
        w2 = torch.abs(self.hyper_w2(states)).view(batch_size, 1, 1)   # [batch_size, 1, 1]
        b2 = self.hyper_b2(states).view(batch_size, 1)                # [batch_size, 1]
        
        # Second layer of mixing network
        q_total = torch.bmm(w2, hidden.unsqueeze(2)).view(batch_size, 1) + b2
        
        return q_total
        
    def get_joint_q_value(self, agent_q_values: List[torch.Tensor], states: torch.Tensor) -> torch.Tensor:
        """
        Calculate joint Q-value from individual agent Q-values.
        
        Args:
            agent_q_values: List of Q-values from individual agents [leader, follower1, follower2]
            states: Global states [batch_size, state_dim]
            
        Returns:
            Joint Q-values
        """
        # Stack agent Q-values along dim 1 (agent dimension)
        # Handle the case where q_values might have different shapes
        batch_size = agent_q_values[0].shape[0]
        
        # Reshape if needed (for sequence data)
        reshaped_q_values = []
        for q in agent_q_values:
            if len(q.shape) > 2:  # If q has sequence dimension
                reshaped_q_values.append(q.reshape(batch_size, -1))
            else:
                reshaped_q_values.append(q)
        
        # Stack along agent dimension
        agent_qs = torch.stack(reshaped_q_values, dim=1)  # [batch_size, 3]
        
        # Reshape states if needed
        if len(states.shape) > 2:  # If states has sequence dimension
            states = states.reshape(batch_size, -1)
        
        # Forward pass through the mixing network
        return self.forward(agent_qs, states)