import torch
import torch.nn as nn
import torch.nn.functional as F


class RainbowNetwork(nn.Module):
    """
    Rainbow DQN Network implementation.
    
    This network combines various improvements:
    - Dueling architecture (separate value and advantage streams)
    - Noisy networks for exploration
    - Distributional RL (categorical approach)
    
    Attributes:
        input_dim (int): Dimension of the input state
        action_dim (int): Dimension of the action space
        hidden_size (int): Size of hidden layers
        n_atoms (int): Number of atoms in the distribution
        v_min (float): Minimum support value
        v_max (float): Maximum support value
        noisy (bool): Whether to use noisy networks
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int = 64,
                n_atoms: int = 51, v_min: float = -10, v_max: float = 10, noisy: bool = True):
        """
        Initialize the Rainbow DQN network.
        
        Args:
            input_dim: Dimension of the input state
            action_dim: Dimension of the action space
            hidden_size: Size of hidden layers
            n_atoms: Number of atoms in the distribution
            v_min: Minimum support value
            v_max: Maximum support value
            noisy: Whether to use noisy networks
        """
        super(RainbowNetwork, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.hidden_size = hidden_size
        self.noisy = noisy
        
        # Initialize support for the distribution
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Feature extraction layers
        if noisy:
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.LeakyReLU(),
                NoisyLinear(hidden_size, hidden_size),
                nn.LeakyReLU()
            )
            
            # Value stream (state value)
            self.value_stream = nn.Sequential(
                NoisyLinear(hidden_size, hidden_size // 2),
                nn.LeakyReLU(),
                NoisyLinear(hidden_size // 2, n_atoms)
            )
            
            # Advantage stream (action advantages)
            self.advantage_stream = nn.Sequential(
                NoisyLinear(hidden_size, hidden_size // 2),
                nn.LeakyReLU(),
                NoisyLinear(hidden_size // 2, action_dim * n_atoms)
            )
        else:
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
                nn.Linear(hidden_size // 2, n_atoms)
            )
            
            # Advantage stream (action advantages)
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LeakyReLU(),
                nn.Linear(hidden_size // 2, action_dim * n_atoms)
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with small random values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Batch of states [batch_size, state_dim]
        
        Returns:
            Action distribution [batch_size, action_dim, n_atoms]
        """
        batch_size = state.shape[0]
        
        # Extract features
        features = self.feature_extractor(state)
        
        # Compute value distribution
        value_dist = self.value_stream(features).view(batch_size, 1, self.n_atoms)
        
        # Compute advantage distribution
        advantage_dist = self.advantage_stream(features).view(batch_size, self.action_dim, self.n_atoms)
        
        # Combine using dueling architecture
        q_dist = value_dist + advantage_dist - advantage_dist.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probabilities
        q_dist = F.softmax(q_dist, dim=2)
        
        return q_dist
    
    def reset_noise(self):
        """Reset the noise in all noisy layers."""
        if not self.noisy:
            return
        
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Convert probability distributions to Q-values.
        
        Args:
            state: Batch of states [batch_size, state_dim]
        
        Returns:
            Q-values [batch_size, action_dim]
        """
        # Get probability distributions
        dist = self.forward(state)
        
        # Expected value: Sum(p_i * z_i) for each action
        q_values = torch.sum(dist * self.support.expand_as(dist), dim=2)
        
        return q_values