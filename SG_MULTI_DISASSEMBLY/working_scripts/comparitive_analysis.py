"""
Comparative Analysis of Deep RL Algorithms for Three-Robot Coordination

This script compares the performance of different deep RL algorithms:
1. Deep Recurrent Q-Network (DRQN)
2. Double DQN with Prioritized Experience Replay (DDQN+PER)
3. C51 Distributional Q-Learning
4. Soft Actor-Critic (SAC)

Performance metrics:
- Cumulative reward
- Episode length
- Task completion rate
- Learning efficiency
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import time
from scipy import stats
import pandas as pd
import seaborn as sns
from collections import defaultdict

# Import environments and algorithms
from battery_disassembly_env import BatteryDisassemblyEnv
from stackelberg_ddqn_per import run_stackelberg_ddqn_per_simulation
from stackelberg_c51 import run_stackelberg_c51_simulation
from stackelberg_sac import run_stackelberg_sac_simulation

# Import the original DRQN simulation
from paste import StackelbergThreeRobotDRQNSimulation


def run_drqn_simulation(n_episodes=200, render_interval=None):
    """
    Run a simulation using the original DRQN implementation from the provided code.
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
        'episode_size': n_episodes,
        'step_per_episode': 40,
        'max_time_steps': 100,
        'franka_failure_prob': 0.1,
        'ur10_failure_prob': 0.1,
        'kuka_failure_prob': 0.1,
        'hidden_size': 64,
        'learning_rate': 1e-4,
        'gamma': 0.9,
        'epsilon': 0.1,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'tau': 0.01
    }
    
    # Create the simulation
    sim = StackelbergThreeRobotDRQNSimulation(parameters)
    
    # Train the agent
    print(f"Starting DRQN training with {n_episodes} episodes...")
    train_stats = sim.train(n_episodes=n_episodes, render_interval=render_interval)
    
    # Evaluate the trained policy
    eval_stats = sim.evaluate(n_episodes=5, render=False)
    
    return sim, eval_stats


def run_all_algorithms(n_episodes=200, n_trials=5, save_dir="results"):
    """
    Run all algorithms for multiple trials and collect performance metrics.
    
    Parameters:
    - n_episodes: Number of episodes per trial
    - n_trials: Number of trials per algorithm
    - save_dir: Directory to save results
    
    Returns:
    - Dictionary containing performance metrics for each algorithm and trial
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # List of algorithms to compare
    algorithms = {
        "DRQN": run_drqn_simulation,
        "DDQN+PER": run_stackelberg_ddqn_per_simulation,
        "C51": run_stackelberg_c51_simulation,
        "SAC": run_stackelberg_sac_simulation
    }
    
    # Initialize results dictionary
    results = {
        alg_name: {
            "leader_rewards": [],
            "follower1_rewards": [],
            "follower2_rewards": [],
            "completion_steps": [],
            "completion_rates": [],
            "training_time": []
        } for alg_name in algorithms
    }
    
    # Run each algorithm for n_trials
    for alg_name, run_func in algorithms.items():
        print(f"Running {alg_name} for {n_trials} trials...")
        
        for trial in range(n_trials):
            print(f"Trial {trial+1}/{n_trials}...")
            
            # Measure training time
            start_time = time.time()
            
            # Run the algorithm
            _, eval_stats = run_func(n_episodes=n_episodes, render_interval=None)
            
            # Record training time
            training_time = time.time() - start_time
            results[alg_name]["training_time"].append(training_time)
            
            # Store evaluation results
            results[alg_name]["leader_rewards"].append(np.mean(eval_stats["leader_rewards"]))
            results[alg_name]["follower1_rewards"].append(np.mean(eval_stats["follower1_rewards"]))
            results[alg_name]["follower2_rewards"].append(np.mean(eval_stats["follower2_rewards"]))
            results[alg_name]["completion_steps"].append(np.mean(eval_stats["completion_steps"]))
            results[alg_name]["completion_rates"].append(np.mean(eval_stats["completion_rates"]) * 100)
            
            # Print trial summary
            print(f"Trial {trial+1} completed in {training_time:.2f} seconds")
            print(f"Average Leader Reward: {results[alg_name]['leader_rewards'][-1]:.2f}")
            print(f"Average Follower1 Reward: {results[alg_name]['follower1_rewards'][-1]:.2f}")
            print(f"Average Follower2 Reward: {results[alg_name]['follower2_rewards'][-1]:.2f}")
            print(f"Average Steps: {results[alg_name]['completion_steps'][-1]:.2f}")
            print(f"Completion Rate: {results[alg_name]['completion_rates'][-1]:.1f}%")
            print("-" * 50)
        
        print(f"{alg_name} trials completed.\n")
    
    # Save results
    with open(f"{save_dir}/algorithm_comparison_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    return results


def perform_statistical_analysis(results):
    """
    Perform statistical analysis on the collected results.
    
    Parameters:
    - results: Dictionary containing performance metrics for each algorithm and trial
    
    Returns:
    - Dictionary containing statistical analysis results
    """
    # Metrics to analyze
    metrics = [
        "leader_rewards", 
        "follower1_rewards", 
        "follower2_rewards",
        "completion_steps",
        "completion_rates",
        "training_time"
    ]
    
    # Initialize results dictionary
    analysis_results = {metric: {} for metric in metrics}
    
    # Prepare dataframes for each metric
    dfs = {}
    for metric in metrics:
        data = []
        for alg_name in results:
            for i, value in enumerate(results[alg_name][metric]):
                data.append({
                    "Algorithm": alg_name,
                    "Trial": i+1,
                    "Value": value
                })
        dfs[metric] = pd.DataFrame(data)
    
    # Perform ANOVA for each metric
    for metric in metrics:
        df = dfs[metric]
        
        # Descriptive statistics
        desc_stats = df.groupby("Algorithm")["Value"].agg(["mean", "std", "min", "max"]).reset_index()
        analysis_results[metric]["descriptive"] = desc_stats
        
        # ANOVA
        groups = [df[df["Algorithm"] == alg]["Value"].values for alg in df["Algorithm"].unique()]
        f_statistic, p_value = stats.f_oneway(*groups)
        analysis_results[metric]["anova"] = {
            "f_statistic": f_statistic,
            "p_value": p_value,
            "significant": p_value < 0.05
        }
        
        # If ANOVA is significant, perform post-hoc analysis
        if p_value < 0.05:
            # Tukey's HSD test for multiple comparisons
            posthoc = pd.DataFrame(
                data=stats.tukey_hsd(*groups).pvalue,
                index=df["Algorithm"].unique(),
                columns=df["Algorithm"].unique()
            )
            analysis_results[metric]["posthoc"] = posthoc
    
    return analysis_results, dfs


def visualize_results(results, analysis_results, dfs, save_dir="results"):
    """
    Visualize the results of the comparative analysis.
    
    Parameters:
    - results: Dictionary containing performance metrics for each algorithm and trial
    - analysis_results: Dictionary containing statistical analysis results
    - dfs: Dictionary of dataframes for each metric
    - save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Metrics and their labels
    metrics = {
        "leader_rewards": "Leader Rewards",
        "follower1_rewards": "Follower1 Rewards",
        "follower2_rewards": "Follower2 Rewards",
        "completion_steps": "Episode Length (steps)",
        "completion_rates": "Task Completion Rate (%)",
        "training_time": "Training Time (s)"
    }
    
    # Set up color palette
    colors = sns.color_palette("tab10", n_colors=len(results))
    
    # Create boxplots for each metric
    plt.figure(figsize=(20, 15))
    for i, (metric, label) in enumerate(metrics.items(), 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x="Algorithm", y="Value", data=dfs[metric], palette=colors)
        plt.title(f"{label}")
        plt.ylabel(label)
        plt.xticks(rotation=45)
        
        # Add ANOVA p-value to the title
        p_value = analysis_results[metric]["anova"]["p_value"]
        plt.title(f"{label}\nANOVA p-value: {p_value:.4f}")
        
        # Add significance stars
        if p_value < 0.001:
            plt.title(f"{label}\nANOVA p-value: {p_value:.4f} ***")
        elif p_value < 0.01:
            plt.title(f"{label}\nANOVA p-value: {p_value:.4f} **")
        elif p_value < 0.05:
            plt.title(f"{label}\nANOVA p-value: {p_value:.4f} *")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/metrics_boxplots.png", dpi=300)
    plt.close()
    
    # Create bar plots with error bars for each metric
    plt.figure(figsize=(20, 15))
    for i, (metric, label) in enumerate(metrics.items(), 1):
        plt.subplot(2, 3, i)
        
        # Calculate mean and standard deviation for each algorithm
        means = []
        stds = []
        alg_names = []
        
        for alg_name in results:
            means.append(np.mean(results[alg_name][metric]))
            stds.append(np.std(results[alg_name][metric]))
            alg_names.append(alg_name)
        
        plt.bar(alg_names, means, yerr=stds, capsize=10, color=colors, alpha=0.7)
        plt.title(f"{label}")
        plt.ylabel(label)
        plt.xticks(rotation=45)
        
        # Add ANOVA p-value to the title
        p_value = analysis_results[metric]["anova"]["p_value"]
        
        # Add significance stars
        if p_value < 0.001:
            plt.title(f"{label}\nANOVA p-value: {p_value:.4f} ***")
        elif p_value < 0.01:
            plt.title(f"{label}\nANOVA p-value: {p_value:.4f} **")
        elif p_value < 0.05:
            plt.title(f"{label}\nANOVA p-value: {p_value:.4f} *")
        else:
            plt.title(f"{label}\nANOVA p-value: {p_value:.4f}")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/metrics_barplots.png", dpi=300)
    plt.close()
    
    # Create heatmaps for post-hoc tests (if available)
    for metric, label in metrics.items():
        if "posthoc" in analysis_results[metric]:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                analysis_results[metric]["posthoc"],
                annot=True,
                cmap="YlGnBu",
                vmin=0,
                vmax=1,
                fmt=".3f"
            )
            plt.title(f"Tukey's HSD Test p-values for {label}")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{metric}_posthoc.png", dpi=300)
            plt.close()
    
    # Create summary table
    summary = pd.DataFrame()
    
    for metric, label in metrics.items():
        desc = analysis_results[metric]["descriptive"].copy()
        desc.columns = ["Algorithm", f"{label} Mean", f"{label} Std", f"{label} Min", f"{label} Max"]
        
        if summary.empty:
            summary = desc
        else:
            summary = summary.merge(desc, on="Algorithm")
    
    # Save summary table
    summary.to_csv(f"{save_dir}/summary_statistics.csv", index=False)
    
    # Generate a more reader-friendly summary report
    with open(f"{save_dir}/analysis_report.txt", "w") as f:
        f.write("COMPARATIVE ANALYSIS OF DEEP RL ALGORITHMS FOR THREE-ROBOT COORDINATION\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n")
        
        # ANOVA results
        f.write("ANOVA RESULTS\n")
        f.write("-" * 80 + "\n")
        for metric, label in metrics.items():
            f_stat = analysis_results[metric]["anova"]["f_statistic"]
            p_val = analysis_results[metric]["anova"]["p_value"]
            significant = "YES" if p_val < 0.05 else "NO"
            
            f.write(f"{label}:\n")
            f.write(f"  F-statistic: {f_stat:.4f}\n")
            f.write(f"  p-value: {p_val:.4f}\n")
            f.write(f"  Significant difference: {significant}\n\n")
        
        # Post-hoc results
        f.write("POST-HOC ANALYSIS (Tukey's HSD Test)\n")
        f.write("-" * 80 + "\n")
        for metric, label in metrics.items():
            if "posthoc" in analysis_results[metric]:
                f.write(f"{label}:\n")
                f.write(analysis_results[metric]["posthoc"].to_string())
                f.write("\n\n")
    
    print(f"Visualizations and reports saved to {save_dir} directory.")


def main():
    """
    Main function to run the comparative analysis.
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Number of episodes and trials
    n_episodes = 200  # Shorter for quicker comparison
    n_trials = 3      # Multiple trials for statistical significance
    
    # Directory to save results
    save_dir = "algorithm_comparison_results"
    
    # Run all algorithms and collect results
    results = run_all_algorithms(n_episodes=n_episodes, n_trials=n_trials, save_dir=save_dir)
    
    # Perform statistical analysis
    analysis_results, dfs = perform_statistical_analysis(results)
    
    # Visualize results
    visualize_results(results, analysis_results, dfs, save_dir=save_dir)
    
    print("Comparative analysis completed successfully!")


if __name__ == "__main__":
    main()