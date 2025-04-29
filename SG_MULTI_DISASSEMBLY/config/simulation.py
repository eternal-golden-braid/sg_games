import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import os
from environments.battery_disassembly_env import BatteryDisassemblyEnv
from agents.sg_agent_drqn import StackelbergThreeRobotDRQNAgent
from utils.replay_buffer import SequenceReplayBuffer

class StackelbergThreeRobotDRQNSimulation:
    """
    Main simulation class for the Stackelberg game using DRQN with three robots.
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
        self.agent = StackelbergThreeRobotDRQNAgent(
            state_dim=state_dim,
            action_dim_leader=action_dim_leader,
            action_dim_follower1=action_dim_follower1,
            action_dim_follower2=action_dim_follower2,
            hidden_size=parameters.get('hidden_size', 64),
            sequence_length=parameters.get('sequence_length', 8),
            device=self.device,
            learning_rate=parameters.get('learning_rate', 1e-4),
            gamma=parameters.get('gamma', 0.9),
            epsilon=parameters.get('epsilon', 0.1),
            epsilon_decay=parameters.get('epsilon_decay', 0.995),
            epsilon_min=parameters.get('epsilon_min', 0.01),
            tau=parameters.get('tau', 0.01),
            update_every=parameters.get('update_every', 10),
            seed=parameters.get('seed', 42)
        )
        
        # Initialize replay buffer
        self.buffer = SequenceReplayBuffer(
            buffer_size=parameters.get('buffer_size', 10000),
            sequence_length=parameters.get('sequence_length', 8),
            state_dim=state_dim,
            batch_size=parameters.get('batch_size', 32),
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
                
                # Store experience
                experience = [state, leader_action, follower1_action, follower2_action, 
                              leader_reward, follower1_reward, follower2_reward, next_state]
                self.buffer.add(experience)
                
                # Check if done
                if self.env.is_done():
                    break
                
                state = next_state
            
            # End episode in buffer
            self.buffer.end_episode()
        
        print(f"Initial buffer size: {len(self.buffer)}")
    
    def train(self, n_episodes=None, render_interval=None):
        """
        Train the agents using DRQN.
        
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
            self.generate_initial_buffer()
        
        print(f"Starting training for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            # Reset environment and agent hidden states
            self.env.reset_env()
            self.agent.reset_hidden_states()
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
                
                # Store experience
                experience = [state, leader_action, follower1_action, follower2_action, 
                              leader_reward, follower1_reward, follower2_reward, next_state]
                self.buffer.add(experience)
                
                # Update statistics
                episode_leader_reward += leader_reward
                episode_follower1_reward += follower1_reward
                episode_follower2_reward += follower2_reward
                steps += 1
                
                # Update networks if enough experiences are available
                if len(self.buffer) >= self.batch_size:
                    experiences = self.buffer.sample(self.batch_size)
                    leader_loss, follower1_loss, follower2_loss = self.agent.update(experiences)
                    episode_leader_losses.append(leader_loss)
                    episode_follower1_losses.append(follower1_loss)
                    episode_follower2_losses.append(follower2_loss)
                
                # Render if requested
                if render:
                    self.env.render(ax)
                    plt.draw()
                    plt.pause(0.1)  # Short pause to update display
                
                # Check if episode is done
                if self.env.is_done():
                    break
                
                # Update state
                state = next_state
            
            # End episode in buffer
            self.buffer.end_episode()
            
            if render:
                plt.ioff()  # Turn off interactive mode
            
            # Store episode statistics
            self.training_stats['leader_rewards'].append(episode_leader_reward)
            self.training_stats['follower1_rewards'].append(episode_follower1_reward)
            self.training_stats['follower2_rewards'].append(episode_follower2_reward)
            self.training_stats['completion_steps'].append(steps)
            self.training_stats['completion_rates'].append(float(self.env.is_done()))
            
            if episode_leader_losses:
                self.training_stats['leader_losses'].append(np.mean(episode_leader_losses))
                self.training_stats['follower1_losses'].append(np.mean(episode_follower1_losses))
                self.training_stats['follower2_losses'].append(np.mean(episode_follower2_losses))
            
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
                self.agent.save(f"checkpoints/three_robot_drqn_episode_{episode}")
                
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
                    with open(f'checkpoints/three_robot_drqn_stats_ep{episode}.pkl', 'wb') as f:
                        pickle.dump(checkpoint, f)
                except Exception as e:
                    print(f"Warning: Could not save checkpoint: {e}")
        
        print("Training complete!")
        
        # Save final model
        self.agent.save("checkpoints/three_robot_drqn_final")
        
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
            # Reset environment and agent hidden states
            self.env.reset_env()
            self.agent.reset_hidden_states()
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
                if self.env.is_done():
                    break
                
                # Update state
                state = next_state
            
            if render:
                plt.ioff()  # Turn off interactive mode
            
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