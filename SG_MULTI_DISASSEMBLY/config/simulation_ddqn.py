import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import os
from typing import List
from environments.battery_disassembly_env import BatteryDisassemblyEnv
from agents.sg_agent_ddqn import StackelbergRainbowAgent
from utils.replay_buffer import SequenceReplayBuffer

class StackelbergRainbowSimulation:
    """
    Simulation class for the Stackelberg game using Rainbow DQN with three robots.
    """
    def __init__(self, parameters):
        """Initialize the simulation."""
        self.env = BatteryDisassemblyEnv(parameters)
        self.device = parameters.get('device', 'cpu')
        
        # Extract environment information
        env_info = self.env.get_task_info()
        state_dim = env_info['dims']
        action_dim_leader = env_info['dimAl']
        action_dim_follower1 = env_info['dimAf1']
        action_dim_follower2 = env_info['dimAf2']
        
        # Initialize agent
        self.agent = StackelbergRainbowAgent(
            state_dim=state_dim,
            action_dim_leader=action_dim_leader,
            action_dim_follower1=action_dim_follower1,
            action_dim_follower2=action_dim_follower2,
            hidden_size=parameters.get('hidden_size', 64),
            n_atoms=parameters.get('n_atoms', 51),
            v_min=parameters.get('v_min', -10),
            v_max=parameters.get('v_max', 10),
            device=self.device,
            learning_rate=parameters.get('learning_rate', 1e-4),
            gamma=parameters.get('gamma', 0.99),
            tau=parameters.get('tau', 0.01),
            update_every=parameters.get('update_every', 10),
            prioritized_replay=parameters.get('prioritized_replay', True),
            alpha=parameters.get('alpha', 0.6),
            beta=parameters.get('beta', 0.4),
            noisy=parameters.get('noisy', True),
            n_step=parameters.get('n_step', 3),
            seed=parameters.get('seed', 42)
        )
        
        # Initialize replay buffer
        self.batch_size = parameters.get('batch_size', 32)
        self.buffer = SequenceReplayBuffer(
            buffer_size=parameters.get('buffer_size', 10000),
            sequence_length=parameters.get('sequence_length', 8),
            state_dim=state_dim,
            batch_size=self.batch_size,
            seed=parameters.get('seed', 42)
        )
        
        # Training parameters
        self.n_episodes = parameters.get('episode_size', 1000)
        self.n_steps_per_episode = parameters.get('step_per_episode', 40)
        
        # Statistics tracking
        self.training_stats = self._initialize_stats()
    
    def _initialize_buffer(self, buffer_size, batch_size, state_dim, seed):
        """Initialize appropriate replay buffer for Rainbow."""
        from collections import namedtuple
        import random
        
        # Define experience tuple type
        Experience = namedtuple("Experience", field_names=["state", "action_l", "action_f1", "action_f2", 
                                                         "reward_l", "reward_f1", "reward_f2", "next_state", "done"])
        
        # Create simple buffer class for Rainbow with prioritization
        class PrioritizedReplayBuffer:
            def __init__(self, buffer_size, batch_size, seed, alpha=0.6, beta=0.4):
                self.memory = []
                self.priorities = []
                self.buffer_size = buffer_size
                self.batch_size = batch_size
                self.alpha = alpha  # Priority exponent
                self.beta = beta    # Importance sampling weight
                self.beta_increment = 0.001
                self.position = 0
                self.random = random.Random(seed)
            
            def add(self, experience):
                # Add experience to memory
                max_priority = max(self.priorities) if self.memory else 1.0
                
                if len(self.memory) < self.buffer_size:
                    self.memory.append(experience)
                    self.priorities.append(max_priority)
                else:
                    self.memory[self.position] = experience
                    self.priorities[self.position] = max_priority
                
                self.position = (self.position + 1) % self.buffer_size
            
            def sample(self, batch_size=None):
                if batch_size is None:
                    batch_size = self.batch_size
                
                if len(self.memory) < batch_size:
                    return [], [], []
                
                # Calculate sampling probabilities
                priorities = np.array(self.priorities)
                p = priorities ** self.alpha
                p = p / np.sum(p)
                
                # Sample indices based on priorities
                indices = np.random.choice(len(self.memory), batch_size, p=p, replace=False)
                
                # Calculate importance sampling weights
                weights = (len(self.memory) * p[indices]) ** (-self.beta)
                weights = weights / np.max(weights)
                
                # Increment beta for next time
                self.beta = min(1.0, self.beta + self.beta_increment)
                
                # Get sampled experiences
                experiences = [self.memory[idx] for idx in indices]
                
                return experiences, indices, weights
            
            def update_priorities(self, indices, priorities):
                for idx, priority in zip(indices, priorities):
                    self.priorities[idx] = priority
            
            def __len__(self):
                return len(self.memory)
        
        return PrioritizedReplayBuffer(buffer_size, batch_size, seed)
    
    def _initialize_stats(self):
        """Initialize statistics tracking."""
        return {
            'leader_rewards': [],
            'follower1_rewards': [],
            'follower2_rewards': [],
            'completion_steps': [],
            'completion_rates': [],
            'leader_losses': [],
            'follower1_losses': [],
            'follower2_losses': []
        }
    
    def train(self, n_episodes=None, render_interval=None):
        """
        Train the agent using Rainbow DQN.
        
        Parameters:
        - n_episodes: Number of episodes to train (uses default if None)
        - render_interval: How often to render an episode (None for no rendering)
        
        Returns:
        - Training statistics
        """
        if n_episodes is None:
            n_episodes = self.n_episodes
        
        # Generate initial experiences if buffer is empty
        if len(self.buffer) < self.batch_size:
            self._generate_initial_buffer(10)
        
        print(f"Starting training for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            # Reset environment
            self.env.reset_env()
            state, _ = self.env.get_current_state()
            
            episode_leader_reward = 0
            episode_follower1_reward = 0
            episode_follower2_reward = 0
            episode_losses = []
            steps = 0
            
            # Create figure for rendering if needed
            if render_interval is not None and episode % render_interval == 0:
                render = True
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                plt.ion()
            else:
                render = False
            
            # Run the episode
            for step in range(self.n_steps_per_episode):
                # Select actions
                leader_action, follower1_action, follower2_action = self.agent.act(state)
                
                # Get rewards
                leader_reward, follower1_reward, follower2_reward = self.env.reward(
                    state, leader_action, follower1_action, follower2_action)
                
                # Execute actions
                self.env.step(leader_action, follower1_action, follower2_action)
                next_state, _ = self.env.get_current_state()
                
                # Check if done
                done = self.env.is_done()
                
                # Store experience
                experience = (state, leader_action, follower1_action, follower2_action, 
                             leader_reward, follower1_reward, follower2_reward, next_state, done)
                self.buffer.add(experience)
                
                # Update statistics
                episode_leader_reward += leader_reward
                episode_follower1_reward += follower1_reward
                episode_follower2_reward += follower2_reward
                steps += 1
                
                # Update networks
                if len(self.buffer) >= self.batch_size:
                    experiences, indices, weights = self.buffer.sample(self.batch_size)
                    td_errors, leader_loss, follower1_loss, follower2_loss = self.agent.update(
                        (experiences, indices, weights))
                    
                    # Update priorities in buffer
                    self.buffer.update_priorities(indices, td_errors + 1e-5)  # Small constant for stability
                    
                    episode_losses.append((leader_loss, follower1_loss, follower2_loss))
                
                # Render if requested
                if render:
                    self.env.render(ax)
                    plt.draw()
                    plt.pause(0.1)
                
                # Break if done
                if done:
                    break
                
                # Update state
                state = next_state
            
            if render:
                plt.ioff()
            
            # Store episode statistics
            self.training_stats['leader_rewards'].append(episode_leader_reward)
            self.training_stats['follower1_rewards'].append(episode_follower1_reward)
            self.training_stats['follower2_rewards'].append(episode_follower2_reward)
            self.training_stats['completion_steps'].append(steps)
            self.training_stats['completion_rates'].append(float(done))
            
            if episode_losses:
                avg_losses = np.mean(episode_losses, axis=0)
                self.training_stats['leader_losses'].append(avg_losses[0])
                self.training_stats['follower1_losses'].append(avg_losses[1])
                self.training_stats['follower2_losses'].append(avg_losses[2])
            
            # Print progress
            if episode % 10 == 0:
                print(f"Episode {episode}/{n_episodes}: "
                     f"Leader Reward = {episode_leader_reward:.2f}, "
                     f"Follower1 Reward = {episode_follower1_reward:.2f}, "
                     f"Follower2 Reward = {episode_follower2_reward:.2f}, "
                     f"Steps = {steps}")
                
            # Save checkpoint for long training runs
            if n_episodes >= 1000 and episode > 0 and episode % 200 == 0:
                self._save_checkpoint(episode)
        
        print("Training complete!")
        self.agent.save("checkpoints/rainbow_final")
        
        return self.training_stats
    
    def _generate_initial_buffer(self, n_episodes=10):
        """Generate initial experiences using random actions."""
        print("Generating initial experiences...")
        
        for episode in range(n_episodes):
            self.env.reset_env()
            state, _ = self.env.get_current_state()
            
            for step in range(self.n_steps_per_episode):
                # Choose random actions
                leader_action = np.random.randint(-1, self.env.task_board.shape[1])
                follower1_action = np.random.randint(-1, self.env.task_board.shape[1])
                follower2_action = np.random.randint(-1, self.env.task_board.shape[1])
                
                # Get rewards
                leader_reward, follower1_reward, follower2_reward = self.env.reward(
                    state, leader_action, follower1_action, follower2_action)
                
                # Execute actions
                self.env.step(leader_action, follower1_action, follower2_action)
                next_state, _ = self.env.get_current_state()
                
                # Check if done
                done = self.env.is_done()
                
                # Store experience
                experience = (state, leader_action, follower1_action, follower2_action, 
                             leader_reward, follower1_reward, follower2_reward, next_state, done)
                self.buffer.add(experience)
                
                # Break if done
                if done:
                    break
                
                # Update state
                state = next_state
        
        print(f"Initial buffer size: {len(self.buffer)}")
    
    def _save_checkpoint(self, episode):
        """Save checkpoint during training."""
        print(f"Saving checkpoint at episode {episode}...")
        self.agent.save(f"checkpoints/rainbow_episode_{episode}")
        
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
            with open(f'checkpoints/rainbow_stats_ep{episode}.pkl', 'wb') as f:
                pickle.dump(checkpoint, f)
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")
    
    def evaluate(self, n_episodes=10, render=False):
        """
        Evaluate the trained agent.
        
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
                plt.ion()
            
            # Run the episode
            for step in range(self.n_steps_per_episode):
                # Select actions (deterministically for evaluation)
                leader_action, follower1_action, follower2_action = self.agent.act(state)
                
                # Get rewards
                leader_reward, follower1_reward, follower2_reward = self.env.reward(
                    state, leader_action, follower1_action, follower2_action)
                
                # Execute actions
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
                    plt.pause(0.2)
                
                # Check if done
                if self.env.is_done():
                    break
                
                # Update state
                state = next_state
            
            if render:
                plt.ioff()
            
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
        
        # Plot completion rates
        window_size = min(50, len(self.training_stats['completion_rates']))
        completion_rates = np.array(self.training_stats['completion_rates'])
        moving_avg = np.convolve(completion_rates, np.ones(window_size)/window_size, mode='valid')
        axes[1, 0].plot(moving_avg * 100)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Completion Rate (%)')
        axes[1, 0].set_title(f'Task Completion Rate (Moving Avg, Window={window_size})')
        axes[1, 0].grid(True)
        
        # Plot losses
        if self.training_stats.get('leader_losses'):
            axes[1, 1].plot(self.training_stats['leader_losses'], label='Leader')
            axes[1, 1].plot(self.training_stats['follower1_losses'], label='Follower1')
            axes[1, 1].plot(self.training_stats['follower2_losses'], label='Follower2')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Losses')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        return fig