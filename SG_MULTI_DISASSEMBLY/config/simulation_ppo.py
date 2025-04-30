import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import os
from environments.battery_disassembly_env import BatteryDisassemblyEnv
from agents.sg_agent_ppo import StackelbergPPOAgent

class StackelbergPPOSimulation:
    """
    Simulation class for the Stackelberg game using PPO with three robots.
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
        self.agent = StackelbergPPOAgent(
            state_dim=state_dim,
            action_dim_leader=action_dim_leader,
            action_dim_follower1=action_dim_follower1,
            action_dim_follower2=action_dim_follower2,
            hidden_size=parameters.get('hidden_size', 64),
            device=self.device,
            learning_rate=parameters.get('learning_rate', 3e-4),
            gamma=parameters.get('gamma', 0.99),
            clip_param=parameters.get('clip_param', 0.2),
            ppo_epochs=parameters.get('ppo_epochs', 10),
            gae_lambda=parameters.get('gae_lambda', 0.95),
            value_coef=parameters.get('value_coef', 0.5),
            entropy_coef=parameters.get('entropy_coef', 0.01),
            seed=parameters.get('seed', 42)
        )
        
        # Training parameters
        self.n_episodes = parameters.get('episode_size', 1000)
        self.n_steps_per_episode = parameters.get('step_per_episode', 40)
        self.steps_per_update = parameters.get('steps_per_update', 2048)
        
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
    
    def train(self, n_episodes=None, render_interval=None):
        """
        Train the agent using PPO.
        
        Parameters:
        - n_episodes: Number of episodes to train (uses default if None)
        - render_interval: How often to render an episode (None for no rendering)
        
        Returns:
        - Training statistics
        """
        if n_episodes is None:
            n_episodes = self.n_episodes
        
        print(f"Starting training for {n_episodes} episodes...")
        
        # Initialize episode stats
        episode_rewards = []
        episode_follower1_rewards = []
        episode_follower2_rewards = []
        episode_steps = []
        episode_completions = []
        
        # Training loop
        for episode in range(n_episodes):
            # Reset environment
            self.env.reset_env()
            state, _ = self.env.get_current_state()
            
            episode_leader_reward = 0
            episode_follower1_reward = 0
            episode_follower2_reward = 0
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
                leader_action, follower1_action, follower2_action = self.agent.act(
                    state, deterministic=False)
                
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
                    plt.pause(0.1)
                
                # Check if done
                if self.env.is_done():
                    break
                
                # Update state
                state = next_state
            
            if render:
                plt.ioff()
            
            # Store episode statistics
            episode_rewards.append(episode_leader_reward)
            episode_follower1_rewards.append(episode_follower1_reward)
            episode_follower2_rewards.append(episode_follower2_reward)
            episode_steps.append(steps)
            episode_completions.append(float(self.env.is_done()))
            
            # Print progress
            if episode % 10 == 0:
                print(f"Episode {episode}/{n_episodes}: "
                     f"Leader Reward = {episode_leader_reward:.2f}, "
                     f"Follower1 Reward = {episode_follower1_reward:.2f}, "
                     f"Follower2 Reward = {episode_follower2_reward:.2f}, "
                     f"Steps = {steps}")
            
            # Collect rollout data and update policy
            if (episode + 1) % 5 == 0 or episode == n_episodes - 1:
                rollout_data = self.agent.collect_rollout(self.env, self.steps_per_update)
                leader_loss, follower1_loss, follower2_loss = self.agent.update(rollout_data)
                
                self.training_stats['leader_losses'].append(leader_loss)
                self.training_stats['follower1_losses'].append(follower1_loss)
                self.training_stats['follower2_losses'].append(follower2_loss)
            
            # Save checkpoint for long training runs
            if n_episodes >= 1000 and episode > 0 and episode % 200 == 0:
                self._save_checkpoint(episode)
        
        # Update final training stats
        self.training_stats['leader_rewards'] = episode_rewards
        self.training_stats['follower1_rewards'] = episode_follower1_rewards
        self.training_stats['follower2_rewards'] = episode_follower2_rewards
        self.training_stats['completion_steps'] = episode_steps
        self.training_stats['completion_rates'] = episode_completions
        
        print("Training complete!")
        self.agent.save("checkpoints/ppo_final")
        
        return self.training_stats
    
    def _save_checkpoint(self, episode):
        """Save checkpoint during training."""
        print(f"Saving checkpoint at episode {episode}...")
        self.agent.save(f"checkpoints/ppo_episode_{episode}")
        
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
            with open(f'checkpoints/ppo_stats_ep{episode}.pkl', 'wb') as f:
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
                leader_action, follower1_action, follower2_action = self.agent.act(
                    state, deterministic=True)
                
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
        """Visualize the training statistics."""
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
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('PPO Losses')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        return fig