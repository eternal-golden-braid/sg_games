import numpy as np

class SequenceReplayBuffer:
    """
    Replay buffer for storing and sampling sequences of experiences.
    """
    def __init__(self, buffer_size, sequence_length, state_dim, batch_size, seed):
        self.buffer_size = buffer_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.buffer = []
        self.episode_buffer = []
    
    def __len__(self):
        """Get the current size of the buffer."""
        return len(self.buffer)
    
    def add(self, experience):
        """
        Add an experience to the episode buffer.
        
        Parameters:
        - experience: Experience to add [state, a_leader, a_follower1, a_follower2, r_leader, r_follower1, r_follower2, next_state]
        """
        self.episode_buffer.append(experience)
    
    def end_episode(self):
        """
        End the current episode and transfer sequences to the main buffer.
        """
        if len(self.episode_buffer) == 0:
            return
        
        # Add overlapping sequences from the episode to the buffer
        for i in range(max(1, len(self.episode_buffer) - self.sequence_length + 1)):
            sequence = self.episode_buffer[i:i+self.sequence_length]
            if len(sequence) < self.sequence_length:
                # Pad shorter sequences
                padding = [sequence[-1]] * (self.sequence_length - len(sequence))
                sequence.extend(padding)
            
            self.buffer.append(sequence)
            
            # Maintain buffer size
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)
        
        self.episode_buffer = []
    
    def sample(self, batch_size=None):
        """
        Sample a batch of sequences from the buffer.
        
        Parameters:
        - batch_size: Size of batch to sample (uses default if None)
        
        Returns:
        - Batch of sequence experiences
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer contains {len(self.buffer)} sequences, but requested batch size is {batch_size}")
        
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices] 