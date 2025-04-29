import torch
import torch.nn as nn

class RecurrentQNetwork(nn.Module):
    """
    Deep Recurrent Q-Network implementation using LSTM.
    This handles temporal dependencies through recurrent layers.
    """
    def __init__(self, input_dim, action_dim_leader, action_dim_follower1, action_dim_follower2, hidden_size=64, lstm_layers=1):
        super(RecurrentQNetwork, self).__init__()
        self.input_dim = input_dim
        self.action_dim_leader = action_dim_leader        # Leader action space size
        self.action_dim_follower1 = action_dim_follower1  # Follower 1 action space size
        self.action_dim_follower2 = action_dim_follower2  # Follower 2 action space size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        
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
        
        # Output layer for Q-values - now needs to handle combinations of three robots' actions
        self.output_layer = nn.Linear(hidden_size, action_dim_leader * action_dim_follower1 * action_dim_follower2)
        
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
        - Q-values for all action combinations
        - Final hidden state
        """
        batch_size, seq_len, _ = state.shape
        
        # Extract features
        features = self.feature_extractor(state.view(-1, self.input_dim))
        features = features.view(batch_size, seq_len, self.hidden_size)
        
        # Pass through LSTM
        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(features)
        else:
            lstm_out, hidden_state = self.lstm(features, hidden_state)
        
        # Generate Q-values
        q_values = self.output_layer(lstm_out)
        
        return q_values, hidden_state
    
    def get_q_values(self, state, hidden_state=None):
        """
        Get Q-values for a single state.
        
        Parameters:
        - state: Single state tensor [state_dim]
        - hidden_state: Hidden state for LSTM
        
        Returns:
        - Q-values for all action combinations
        - Updated hidden state
        """
        # Add batch and sequence dimensions if not present
        if len(state.shape) == 1:
            state = state.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
        elif len(state.shape) == 2:
            state = state.unsqueeze(0)  # [1, seq_len, state_dim]
        
        # Forward pass
        q_values, new_hidden_state = self.forward(state, hidden_state)
        
        # Return last timestep's Q-values, reshaped to 3D tensor [leader, follower1, follower2]
        last_q = q_values[:, -1, :]
        return last_q.view(self.action_dim_leader, self.action_dim_follower1, self.action_dim_follower2), new_hidden_state 