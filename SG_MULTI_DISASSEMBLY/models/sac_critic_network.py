import torch
import torch.nn as nn


class SACCritic(nn.Module):
    """
    Soft Actor-Critic Critic Network (Q-function).
    
    This network estimates the action-value function (Q-values).
    
    Attributes:
        input_dim (int): Dimension of the input state
        action_dim (int): Dimension of the action space
        hidden_size (int): Size of hidden layers
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int = 64):
        """
        Initialize the SAC critic network.
        
        Args:
            input_dim: Dimension of the input state
            action_dim: Dimension of the action space
            hidden_size: Size of hidden layers
        """
        super(SACCritic, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        # Q1 Network
        self.q1_network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        
        # Q2 Network (for reducing overestimation bias)
        self.q2_network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        
        # Initialize weights
        self._init_weights(self.q1_network)
        self._init_weights(self.q2_network)
    
    def _init_weights(self, network):
        """Initialize network weights with small random values."""
        for m in network:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Batch of states [batch_size, state_dim]
        
        Returns:
            Tuple of (q1_values, q2_values)
            - q1_values: Q-values from first network [batch_size, action_dim]
            - q2_values: Q-values from second network [batch_size, action_dim]
        """
        q1 = self.q1_network(state)
        q2 = self.q2_network(state)
        
        return q1, q2