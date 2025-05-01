# comparison_test.py

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pickle
from config.simulation_drqn import StackelbergThreeRobotDRQNSimulation
# Import the simulation classes we created
from config.simulation_ddqn import StackelbergRainbowSimulation
from config.simulation_ppo import StackelbergPPOSimulation
from config.simulation_sac import StackelbergSACSimulation

def run_comparison():
    """Compare all agent types on the same environment."""
    base_params = {
        'task_id': 1,
        'seed': 42,
        'device': 'cpu',
        'episode_size': 200,  # Shorter for comparison
        'step_per_episode': 40,
        'hidden_size': 64,
        'batch_size': 32,
        'buffer_size': 10000,
        # Exploration parameters
        'epsilon': 0.5,          # Start with higher exploration 
        'epsilon_decay': 0.9998, # Much slower decay
        'epsilon_min': 0.05,     # Keep reasonable exploration
    }
    
    # Create simulation instances
    print("Initializing simulations...")
    drqn_sim = StackelbergThreeRobotDRQNSimulation({**base_params, 'seed': 42})
    rainbow_sim = StackelbergRainbowSimulation({**base_params, 'seed': 43})
    ppo_sim = StackelbergPPOSimulation({**base_params, 'seed': 44})
    sac_sim = StackelbergSACSimulation({**base_params, 'seed': 45})
    
    # Train all agents
    n_episodes = 200  # Short training for comparison
    
    print("Training DRQN agent...")
    drqn_stats = drqn_sim.train(n_episodes=n_episodes)
    
    print("Training Rainbow agent...")
    rainbow_stats = rainbow_sim.train(n_episodes=n_episodes)
    
    print("Training PPO agent...")
    ppo_stats = ppo_sim.train(n_episodes=n_episodes)
    
    print("Training SAC agent...")
    sac_stats = sac_sim.train(n_episodes=n_episodes)
    
    # Evaluate all agents
    print("\nEvaluating all agents...")
    
    print("Evaluating DRQN agent...")
    drqn_eval = drqn_sim.evaluate(n_episodes=10)
    
    print("Evaluating Rainbow agent...")
    rainbow_eval = rainbow_sim.evaluate(n_episodes=10)
    
    print("Evaluating PPO agent...")
    ppo_eval = ppo_sim.evaluate(n_episodes=10)
    
    print("Evaluating SAC agent...")
    sac_eval = sac_sim.evaluate(n_episodes=10)
    
    # Compare results
    stats_list = [drqn_stats, rainbow_stats, ppo_stats, sac_stats]
    eval_list = [drqn_eval, rainbow_eval, ppo_eval, sac_eval]
    names = ["DRQN", "Rainbow", "PPO", "SAC"]

    # Compare results
    # stats_list = [rainbow_stats, ppo_stats, sac_stats]
    # eval_list = [rainbow_eval, ppo_eval, sac_eval]
    # names = ["Rainbow", "PPO", "SAC"]
    
    # Save comparison results
    save_comparison_results(stats_list, eval_list, names)
    
    # Visualize comparison
    compare_performance(stats_list, eval_list, names)
    
    return stats_list, eval_list, names

def save_comparison_results(stats_list, eval_list, names):
    """Save comparison results to file."""
    os.makedirs('results', exist_ok=True)
    
    results = {
        'training': {name: stats for name, stats in zip(names, stats_list)},
        'evaluation': {name: eval_stats for name, eval_stats in zip(names, eval_list)},
        'names': names
    }
    
    with open('results/comparison_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("Comparison results saved to results/comparison_results.pkl")

def compare_performance(stats_list, eval_list, names):
    """Plot comparison of training statistics."""
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot training rewards
    for i, (stats, name) in enumerate(zip(stats_list, names)):
        # Smooth rewards for better visualization
        window_size = min(20, len(stats['leader_rewards']))
        smoothed_rewards = np.convolve(
            stats['leader_rewards'], np.ones(window_size)/window_size, mode='valid')
        axes[0, 0].plot(smoothed_rewards, label=f"{name}")
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Leader Reward')
    axes[0, 0].set_title('Leader Rewards During Training')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot completion rates
    for i, (stats, name) in enumerate(zip(stats_list, names)):
        window_size = min(20, len(stats['completion_rates']))
        smoothed_rates = np.convolve(
            stats['completion_rates'], np.ones(window_size)/window_size, mode='valid')
        axes[0, 1].plot(smoothed_rates * 100, label=f"{name}")
    
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Completion Rate (%)')
    axes[0, 1].set_title('Task Completion Rates During Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot evaluation metrics
    avg_rewards = [np.mean(eval_stats['leader_rewards']) for eval_stats in eval_list]
    std_rewards = [np.std(eval_stats['leader_rewards']) for eval_stats in eval_list]
    
    bars = axes[1, 0].bar(names, avg_rewards, yerr=std_rewards, capsize=5)
    axes[1, 0].set_ylabel('Average Leader Reward')
    axes[1, 0].set_title('Evaluation Performance')
    axes[1, 0].grid(axis='y')
    
    # Plot completion rates in evaluation
    avg_completion = [np.mean(eval_stats['completion_rates']) * 100 for eval_stats in eval_list]
    
    bars = axes[1, 1].bar(names, avg_completion)
    axes[1, 1].set_ylabel('Completion Rate (%)')
    axes[1, 1].set_title('Evaluation Task Completion')
    axes[1, 1].grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('results/agent_comparison.png')
    plt.show()
    
    return fig

if __name__ == "__main__":
    run_comparison()