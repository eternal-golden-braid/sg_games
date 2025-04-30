"""
Run experiments for all the reinforcement learning algorithms and perform statistical analysis.

This script:
1. Runs each algorithm for multiple trials
2. Collects performance metrics
3. Performs statistical analysis to compare algorithm performance
4. Generates visualizations and reports
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import time
import pandas as pd
import seaborn as sns
from scipy import stats
import argparse

# Import algorithms
from paste import StackelbergThreeRobotDRQNSimulation
from stackelberg_ddqn_per import run_stackelberg_ddqn_per_simulation
from stackelberg_c51 import run_stackelberg_c51_simulation
from stackelberg_sac import run_stackelberg_sac_simulation


def run_drqn_simulation(n_episodes=200, render_interval=None):
    """
    Run a simulation using the DRQN implementation.
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


def run_experiments(n_episodes=200, n_trials=5, algorithms=None, save_dir="results"):
    """
    Run experiments for the specified algorithms with multiple trials.
    
    Parameters:
    - n_episodes: Number of episodes per trial
    - n_trials: Number of trials per algorithm
    - algorithms: List of algorithm names to run (None for all)
    - save_dir: Directory to save results
    
    Returns:
    - Dictionary containing performance metrics for each algorithm and trial
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # List of all available algorithms
    all_algorithms = {
        "DRQN": run_drqn_simulation,
        "DDQN+PER": run_stackelberg_ddqn_per_simulation,
        "C51": run_stackelberg_c51_simulation,
        "SAC": run_stackelberg_sac_simulation
    }
    
    # Select which algorithms to run
    if algorithms is None:
        selected_algorithms = all_algorithms
    else:
        selected_algorithms = {alg: all_algorithms[alg] for alg in algorithms if alg in all_algorithms}
        if not selected_algorithms:
            print("No valid algorithms selected. Please choose from:", list(all_algorithms.keys()))
            return None
    
    # Initialize results dictionary
    results = {
        alg_name: {
            "leader_rewards": [],
            "follower1_rewards": [],
            "follower2_rewards": [],
            "completion_steps": [],
            "completion_rates": [],
            "training_time": []
        } for alg_name in selected_algorithms
    }
    
    # Run each algorithm for n_trials
    for alg_name, run_func in selected_algorithms.items():
        print(f"\n{'='*50}")
        print(f"Running {alg_name} for {n_trials} trials...")
        print(f"{'='*50}")
        
        for trial in range(n_trials):
            print(f"\nTrial {trial+1}/{n_trials}...")
            
            # Measure training time
            start_time = time.time()
            
            try:
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
            
            except Exception as e:
                print(f"Error in {alg_name} trial {trial+1}: {e}")
                # Continue with next trial
        
        print(f"\n{alg_name} trials completed.")
    
    # Save results
    with open(f"{save_dir}/experiment_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print(f"\nAll experiments completed. Results saved to {save_dir}/experiment_results.pkl")
    
    return results


def analyze_results(results, save_dir="results"):
    """
    Analyze the results of the experiments.
    
    Parameters:
    - results: Dictionary containing performance metrics for each algorithm and trial
    - save_dir: Directory to save analysis results
    
    Returns:
    - Dictionary containing analysis results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if results is empty
    if not results:
        print("No results to analyze.")
        return None
    
    # Metrics to analyze
    metrics = {
        "leader_rewards": "Leader Rewards",
        "follower1_rewards": "Follower1 Rewards",
        "follower2_rewards": "Follower2 Rewards",
        "completion_steps": "Episode Length (steps)",
        "completion_rates": "Task Completion Rate (%)",
        "training_time": "Training Time (s)"
    }
    
    # Initialize analysis results
    analysis_results = {}
    
    # Convert results to pandas DataFrame for each metric
    dfs = {}
    for metric_key, metric_name in metrics.items():
        data = []
        for alg_name, alg_results in results.items():
            if metric_key in alg_results:
                for i, value in enumerate(alg_results[metric_key]):
                    data.append({
                        "Algorithm": alg_name,
                        "Trial": i+1,
                        "Value": value,
                        "Metric": metric_name
                    })
        
        if data:
            dfs[metric_key] = pd.DataFrame(data)
    
    # Combine all metrics into a single DataFrame for easy saving
    all_data = pd.concat(dfs.values())
    all_data.to_csv(f"{save_dir}/all_metrics.csv", index=False)
    
    # Perform statistical analysis for each metric
    for metric_key, metric_name in metrics.items():
        if metric_key not in dfs or dfs[metric_key].empty:
            continue
        
        df = dfs[metric_key]
        
        # Descriptive statistics
        desc_stats = df.groupby("Algorithm")["Value"].agg(["count", "mean", "std", "min", "max"]).reset_index()
        
        # Store in analysis results
        analysis_results[metric_key] = {"descriptive": desc_stats}
        
        # Check if we have enough algorithms to perform ANOVA
        if len(df["Algorithm"].unique()) < 2:
            continue
        
        # Perform one-way ANOVA
        try:
            # Group data by algorithm
            groups = [df[df["Algorithm"] == alg]["Value"].values for alg in df["Algorithm"].unique()]
            
            # Perform ANOVA
            f_statistic, p_value = stats.f_oneway(*groups)
            
            # Store ANOVA results
            analysis_results[metric_key]["anova"] = {
                "f_statistic": f_statistic,
                "p_value": p_value,
                "significant": p_value < 0.05
            }
            
            # If ANOVA is significant, perform post-hoc tests
            if p_value < 0.05 and len(df["Algorithm"].unique()) > 2:
                try:
                    # Convert to format required by Tukey HSD
                    stacked_data = df.copy()
                    
                    # Perform Tukey's HSD test
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd
                    posthoc = pairwise_tukeyhsd(stacked_data["Value"], stacked_data["Algorithm"], alpha=0.05)
                    
                    # Create a DataFrame from the posthoc results
                    posthoc_df = pd.DataFrame(data=posthoc._results_table.data[1:], 
                                             columns=posthoc._results_table.data[0])
                    
                    # Store post-hoc results
                    analysis_results[metric_key]["posthoc"] = posthoc_df
                    
                except Exception as e:
                    print(f"Error in post-hoc test for {metric_name}: {e}")
            
        except Exception as e:
            print(f"Error in ANOVA for {metric_name}: {e}")
    
    # Save analysis results
    with open(f"{save_dir}/analysis_results.pkl", "wb") as f:
        pickle.dump(analysis_results, f)
    
    print(f"Analysis completed. Results saved to {save_dir}/analysis_results.pkl")
    
    return analysis_results, dfs


def visualize_results(results, analysis_results, dfs, save_dir="results"):
    """
    Visualize the results of the experiments.
    
    Parameters:
    - results: Dictionary containing performance metrics for each algorithm and trial
    - analysis_results: Dictionary containing analysis results
    - dfs: Dictionary of DataFrames for each metric
    - save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if results exist
    if not results or not analysis_results or not dfs:
        print("No results to visualize.")
        return
    
    # Metrics to visualize
    metrics = {
        "leader_rewards": "Leader Rewards",
        "follower1_rewards": "Follower1 Rewards",
        "follower2_rewards": "Follower2 Rewards",
        "completion_steps": "Episode Length (steps)",
        "completion_rates": "Task Completion Rate (%)",
        "training_time": "Training Time (s)"
    }
    
    # Set color palette for algorithms
    colors = sns.color_palette("tab10", n_colors=len(results))
    color_dict = {alg: colors[i] for i, alg in enumerate(results.keys())}
    
    # 1. Create box plots for each metric
    plt.figure(figsize=(20, 15))
    
    for i, (metric_key, metric_name) in enumerate(metrics.items(), 1):
        if metric_key not in dfs or dfs[metric_key].empty:
            continue
            
        plt.subplot(2, 3, i)
        
        # Create box plot
        sns.boxplot(x="Algorithm", y="Value", data=dfs[metric_key], palette=color_dict)
        
        # Add points for individual trials
        sns.stripplot(x="Algorithm", y="Value", data=dfs[metric_key], 
                     color='black', size=4, alpha=0.5, jitter=True)
        
        # Set labels and title
        plt.ylabel(metric_name)
        plt.title(metric_name)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add ANOVA p-value if available
        if "anova" in analysis_results.get(metric_key, {}):
            p_value = analysis_results[metric_key]["anova"]["p_value"]
            
            # Add significance stars
            if p_value < 0.001:
                plt.title(f"{metric_name}\nANOVA: p={p_value:.4f} ***")
            elif p_value < 0.01:
                plt.title(f"{metric_name}\nANOVA: p={p_value:.4f} **")
            elif p_value < 0.05:
                plt.title(f"{metric_name}\nANOVA: p={p_value:.4f} *")
            else:
                plt.title(f"{metric_name}\nANOVA: p={p_value:.4f} (n.s.)")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/metrics_boxplots.png", dpi=300)
    plt.close()
    
    # 2. Create bar plots with error bars
    plt.figure(figsize=(20, 15))
    
    for i, (metric_key, metric_name) in enumerate(metrics.items(), 1):
        if metric_key not in analysis_results:
            continue
            
        plt.subplot(2, 3, i)
        
        # Extract means and standard deviations
        desc_stats = analysis_results[metric_key]["descriptive"]
        
        # Create bar plot
        plt.bar(desc_stats["Algorithm"], desc_stats["mean"], 
               yerr=desc_stats["std"], capsize=10, 
               color=[color_dict[alg] for alg in desc_stats["Algorithm"]], 
               alpha=0.7)
        
        # Set labels and title
        plt.ylabel(metric_name)
        plt.title(metric_name)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add ANOVA p-value if available
        if "anova" in analysis_results.get(metric_key, {}):
            p_value = analysis_results[metric_key]["anova"]["p_value"]
            
            # Add significance stars
            if p_value < 0.001:
                plt.title(f"{metric_name}\nANOVA: p={p_value:.4f} ***")
            elif p_value < 0.01:
                plt.title(f"{metric_name}\nANOVA: p={p_value:.4f} **")
            elif p_value < 0.05:
                plt.title(f"{metric_name}\nANOVA: p={p_value:.4f} *")
            else:
                plt.title(f"{metric_name}\nANOVA: p={p_value:.4f} (n.s.)")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/metrics_barplots.png", dpi=300)
    plt.close()
    
    # 3. Create heatmaps for post-hoc tests
    for metric_key, metric_name in metrics.items():
        if metric_key not in analysis_results:
            continue
            
        if "posthoc" in analysis_results[metric_key]:
            posthoc_df = analysis_results[metric_key]["posthoc"]
            
            # Create a cross-table of p-values
            groups = np.unique(posthoc_df["group1"].tolist() + posthoc_df["group2"].tolist())
            n_groups = len(groups)
            
            # Initialize p-value matrix
            p_values = np.ones((n_groups, n_groups))
            
            # Fill in p-values
            for _, row in posthoc_df.iterrows():
                i = np.where(groups == row["group1"])[0][0]
                j = np.where(groups == row["group2"])[0][0]
                p_values[i, j] = row["p-adj"]
                p_values[j, i] = row["p-adj"]  # Symmetric
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(p_values, annot=True, fmt=".3f", xticklabels=groups, yticklabels=groups, 
                       cmap="YlGnBu", vmin=0, vmax=1)
            
            # Set title
            plt.title(f"Post-hoc p-values for {metric_name}")
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{metric_key}_posthoc_heatmap.png", dpi=300)
            plt.close()
    
    # 4. Create radar chart to compare algorithms across metrics
    # Prepare data for radar chart
    radar_metrics = ["leader_rewards", "follower1_rewards", "follower2_rewards", "completion_rates"]
    radar_names = ["Leader\nRewards", "Follower1\nRewards", "Follower2\nRewards", "Completion\nRate (%)"]
    
    # Check if we have all the required metrics
    if all(metric in analysis_results for metric in radar_metrics):
        # Extract means for each algorithm and metric
        algorithms = list(results.keys())
        radar_data = np.zeros((len(algorithms), len(radar_metrics)))
        
        for i, alg in enumerate(algorithms):
            for j, metric in enumerate(radar_metrics):
                if metric in analysis_results and "descriptive" in analysis_results[metric]:
                    desc_stats = analysis_results[metric]["descriptive"]
                    alg_stats = desc_stats[desc_stats["Algorithm"] == alg]
                    if not alg_stats.empty:
                        radar_data[i, j] = alg_stats["mean"].values[0]
        
        # Normalize the data for radar chart
        radar_data_norm = np.zeros_like(radar_data)
        for j in range(radar_data.shape[1]):
            min_val = np.min(radar_data[:, j])
            max_val = np.max(radar_data[:, j])
            
            if max_val > min_val:
                radar_data_norm[:, j] = (radar_data[:, j] - min_val) / (max_val - min_val)
            else:
                radar_data_norm[:, j] = 0.5  # Default value if all algorithms have the same value
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Set the angles for each metric
        angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Plot each algorithm
        for i, alg in enumerate(algorithms):
            values = radar_data_norm[i].tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, color=colors[i], label=alg)
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_names)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Algorithm Comparison Across Metrics", y=1.08)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/algorithm_radar_chart.png", dpi=300)
        plt.close()
    
    # 5. Generate a summary table
    # Create a summary DataFrame
    summary_data = []
    
    for alg in results.keys():
        row_data = {"Algorithm": alg}
        
        for metric_key, metric_name in metrics.items():
            if metric_key in analysis_results and "descriptive" in analysis_results[metric_key]:
                desc_stats = analysis_results[metric_key]["descriptive"]
                alg_stats = desc_stats[desc_stats["Algorithm"] == alg]
                
                if not alg_stats.empty:
                    row_data[f"{metric_name} Mean"] = alg_stats["mean"].values[0]
                    row_data[f"{metric_name} Std"] = alg_stats["std"].values[0]
        
        summary_data.append(row_data)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{save_dir}/summary_table.csv", index=False)
        
        # Create a more readable text report
        with open(f"{save_dir}/analysis_report.txt", "w") as f:
            f.write("THREE-ROBOT COORDINATION ALGORITHM COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")
            
            # ANOVA results
            f.write("ANOVA RESULTS\n")
            f.write("-" * 80 + "\n")
            
            for metric_key, metric_name in metrics.items():
                if metric_key in analysis_results and "anova" in analysis_results[metric_key]:
                    anova_results = analysis_results[metric_key]["anova"]
                    f_stat = anova_results["f_statistic"]
                    p_val = anova_results["p_value"]
                    significant = "YES" if p_val < 0.05 else "NO"
                    
                    f.write(f"{metric_name}:\n")
                    f.write(f"  F-statistic: {f_stat:.4f}\n")
                    f.write(f"  p-value: {p_val:.4f}\n")
                    f.write(f"  Significant difference: {significant}\n\n")
            
            # Post-hoc results
            f.write("POST-HOC ANALYSIS (Tukey's HSD Test)\n")
            f.write("-" * 80 + "\n")
            
            for metric_key, metric_name in metrics.items():
                if metric_key in analysis_results and "posthoc" in analysis_results[metric_key]:
                    f.write(f"{metric_name}:\n")
                    f.write(analysis_results[metric_key]["posthoc"].to_string())
                    f.write("\n\n")
            
            # Conclusions
            f.write("CONCLUSIONS\n")
            f.write("-" * 80 + "\n")
            f.write("Based on the statistical analysis:\n\n")
            
            for metric_key, metric_name in metrics.items():
                if metric_key in analysis_results and "anova" in analysis_results[metric_key]:
                    anova_results = analysis_results[metric_key]["anova"]
                    
                    if anova_results["significant"]:
                        # Find the best algorithm for this metric
                        desc_stats = analysis_results[metric_key]["descriptive"]
                        
                        # For completion_steps, lower is better
                        if metric_key == "completion_steps" or metric_key == "training_time":
                            best_alg = desc_stats.loc[desc_stats["mean"].idxmin()]["Algorithm"]
                            best_val = desc_stats["mean"].min()
                        else:
                            best_alg = desc_stats.loc[desc_stats["mean"].idxmax()]["Algorithm"]
                            best_val = desc_stats["mean"].max()
                        
                        f.write(f"- For {metric_name}, there is a statistically significant difference between algorithms (p={anova_results['p_value']:.4f}).\n")
                        f.write(f"  The best performing algorithm is {best_alg} with a mean value of {best_val:.2f}.\n\n")
                    else:
                        f.write(f"- For {metric_name}, there is no statistically significant difference between algorithms (p={anova_results['p_value']:.4f}).\n\n")
    
    print(f"Visualizations and reports created in {save_dir} directory.")


def main():
    """
    Main function to run the experiments and analysis.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run reinforcement learning experiments for three-robot coordination")
    
    parser.add_argument("--episodes", type=int, default=200, 
                        help="Number of episodes per trial (default: 200)")
    parser.add_argument("--trials", type=int, default=3, 
                        help="Number of trials per algorithm (default: 3)")
    parser.add_argument("--algorithms", nargs="+", default=None, 
                        choices=["DRQN", "DDQN+PER", "C51", "SAC"], 
                        help="Algorithms to run (default: all)")
    parser.add_argument("--load", action="store_true", 
                        help="Load results from file instead of running experiments")
    parser.add_argument("--save_dir", type=str, default="results", 
                        help="Directory to save results (default: 'results')")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Run experiments or load results
    if args.load:
        # Load results from file
        try:
            with open(f"{args.save_dir}/experiment_results.pkl", "rb") as f:
                results = pickle.load(f)
            print(f"Loaded results from {args.save_dir}/experiment_results.pkl")
        except FileNotFoundError:
            print(f"Error: Results file not found at {args.save_dir}/experiment_results.pkl")
            return
    else:
        # Run experiments
        results = run_experiments(n_episodes=args.episodes, n_trials=args.trials, 
                                 algorithms=args.algorithms, save_dir=args.save_dir)
        if results is None:
            return
    
    # Analyze results
    analysis_results, dfs = analyze_results(results, save_dir=args.save_dir)
    
    # Visualize results
    visualize_results(results, analysis_results, dfs, save_dir=args.save_dir)
    
    print("\nExperiment, analysis, and visualization completed successfully!")


if __name__ == "__main__":
    main()