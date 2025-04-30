import torch
import torch.nn as nn
from typing import Tuple
from torch.distributions import Categorical
import torch.nn.functional as F

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