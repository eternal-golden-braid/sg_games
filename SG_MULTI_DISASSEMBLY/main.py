import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')  # You can also try 'Agg' for non-interactive use
from config.simulation_drqn import StackelbergThreeRobotDRQNSimulation

def run_three_robot_drqn_simulation():
    """
    Run a simulation using the three-robot DRQN implementation.
    """
    # Define simulation parameters
    parameters = {
        'task_id': 1,
        'seed': 42,
        'device': 'cpu',
        'batch_size': 32,
        'buffer_size': 10000,
        'sequence_length': 8,
        'update_every': 10,
        'episode_size': 1000,
        'step_per_episode': 40,
        'max_time_steps': 100,
        'franka_failure_prob': 0.1,
        'ur10_failure_prob': 0.1,
        'kuka_failure_prob': 0.1,
        'hidden_size': 64,
        'learning_rate': 1e-4,
        'gamma': 0.9,
        'epsilon': 0.5,          # Start with higher exploration (was 0.1)
    'epsilon_decay': 0.9998, # Much slower decay (was 0.995)
    'epsilon_min': 0.05,     # Keep reasonable exploration (was 0.001
        'tau': 0.01
    }
    
    # Create the simulation
    sim = StackelbergThreeRobotDRQNSimulation(parameters)
    
    # Train the agent
    print("Starting three-robot DRQN training with 1000 episodes...")
    train_stats = sim.train(n_episodes=1000, render_interval=None)
    
    # Evaluate the trained policy
    eval_stats = sim.evaluate(n_episodes=5, render=False)
    
    # Visualize training statistics
    fig = sim.visualize_training_stats()
    
    return sim

if __name__ == "__main__":
    # Display backend information
    print(f"Using matplotlib backend: {matplotlib.get_backend()}")
    
    try:
        run_three_robot_drqn_simulation()
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc() 