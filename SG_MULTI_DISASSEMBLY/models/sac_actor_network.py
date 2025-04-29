import torch
import torch.nn as nn

class SACActor(nn.Module):
    """
    Soft Actor-Critic Actor Network (Policy).
    
    This network outputs a probability distribution over discrete actions.
    
    Attributes:
        input_dim (int): Dimension of the input state
        action_dim (int): Dimension of the action space
        hidden_size (int): Size of hidden layers
        log_std_min (float): Minimum log standard deviation
        log_std_max (float): Maximum log standard deviation
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int = 64,
                 log_std_min: float = -20, log_std_max: float = 2):
        """
        Initialize the SAC actor network.
        
        Args:
            input_dim: Dimension of the input state
            action_dim: Dimension of the action space
            hidden_size: Size of hidden layers
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super(SACActor, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        
        # Policy head
        self.policy_head = nn.Linear(hidden_size, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with small random values."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Batch of states [batch_size, state_dim]
        
        Returns:
            Tuple of (action_probs, log_probs, entropy)
            - action_probs: Action probabilities [batch_size, action_dim]
            - log_probs: Log probabilities [batch_size, action_dim]
            - entropy: Entropy of the distribution [batch_size]
        """
        # Extract features
        features = self.feature_extractor(state)
        
        # Get logits
        logits = self.policy_head(features)
        
        # Convert to probabilities
        action_probs = F.softmax(logits, dim=-1)
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Compute entropy
        entropy = -torch.sum(action_probs * log_probs, dim=-1)
        
        return action_probs, log_probs, entropy
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from the policy.
        
        Args:
            state: State tensor [batch_size, state_dim]
            deterministic: Whether to select actions deterministically
            
        Returns:
            Tuple of (action_indices, log_probs, entropy)
            - action_indices: Selected action indices [batch_size]
            - log_probs: Log probabilities of selected actions [batch_size]
            - entropy: Entropy of the distribution [batch_size]
        """
        # Get action probabilities and log probabilities
        action_probs, log_probs, entropy = self.forward(state)
        
        if deterministic:
            # Select action with highest probability
            action_indices = torch.argmax(action_probs, dim=-1)
        else:
            # Sample from the distribution
            dist = Categorical(action_probs)
            action_indices = dist.sample()
        
        # Get log probabilities of selected actions
        selected_log_probs = log_probs.gather(1, action_indices.unsqueeze(-1)).squeeze(-1)
        
        return action_indices, selected_log_probs, entropy