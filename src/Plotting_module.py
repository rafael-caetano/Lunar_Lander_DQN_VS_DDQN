import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create figures directory if it doesn't exist
if not os.path.exists('outputs'):
    os.makedirs('outputs')

def plot_learning_curves(dqn_all_average_rewards, double_dqn_all_average_rewards, dqn_avg_average_rewards, dqn_std_average_rewards, double_dqn_avg_average_rewards, double_dqn_std_average_rewards, num_runs):
    plt.figure(figsize=(12, 8))
    for run in range(num_runs):
        plt.plot(dqn_all_average_rewards[run], color='blue', alpha=0.3)
        plt.plot(double_dqn_all_average_rewards[run], color='orange', alpha=0.3)

    plt.plot(dqn_avg_average_rewards, label='DQN Average Reward', color='blue', linewidth=2)
    plt.fill_between(range(len(dqn_avg_average_rewards)), dqn_avg_average_rewards - dqn_std_average_rewards, dqn_avg_average_rewards + dqn_std_average_rewards, color='blue', alpha=0.2)
    plt.plot(double_dqn_avg_average_rewards, label='Double DQN Average Reward', color='orange', linewidth=2)
    plt.fill_between(range(len(double_dqn_avg_average_rewards)), double_dqn_avg_average_rewards - double_dqn_std_average_rewards, double_dqn_avg_average_rewards + double_dqn_std_average_rewards, color='orange', alpha=0.2)

    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Comparison of DQN and Double DQN over Multiple Runs')
    plt.legend()
    plt.savefig('outputs/comparison_learning_curves.png')
    plt.close()

    # Display the plot
    from IPython.display import Image, display
    display(Image(filename='outputs/comparison_learning_curves.png'))

def plot_kernel_density(dqn_all_rewards, double_dqn_all_rewards):
    dqn_scores = pd.Series(np.concatenate(dqn_all_rewards), name="DQN Scores")
    double_dqn_scores = pd.Series(np.concatenate(double_dqn_all_rewards), name="Double DQN Scores")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    dqn_scores.plot(kind="kde", ax=ax, label="DQN")
    double_dqn_scores.plot(kind="kde", ax=ax, label="Double DQN")
    ax.set_xlabel("Score")
    ax.legend()
    ax.set_title('Kernel Density Estimate of DQN and Double DQN Scores')
    plt.savefig('outputs/kernel_density_estimate.png')
    plt.close()

    # Display the plot
    from IPython.display import Image, display
    display(Image(filename='outputs/kernel_density_estimate.png'))

def plot_rolling_averages(dqn_all_rewards, double_dqn_all_rewards, num_runs):
    fig, axs = plt.subplots(2, num_runs, figsize=(num_runs * 6, 12), sharex=True, sharey=True)

    for run in range(num_runs):
        # Plotting DQN Scores and their rolling average for the current run
        dqn_scores_run = pd.Series(dqn_all_rewards[run], name="DQN Scores")
        dqn_scores_run.plot(ax=axs[0, run], label="DQN Scores")
        dqn_scores_run.rolling(window=100).mean().rename("Rolling Average").plot(ax=axs[0, run])
        axs[0, run].legend()
        axs[0, run].set_ylabel("Score")
        axs[0, run].set_title(f'Run {run + 1}: DQN Scores and Rolling Average')
        if run == num_runs - 1:
            axs[0, run].set_xlabel("Episode Number")

        # Plotting Double DQN Scores and their rolling average for the current run
        double_dqn_scores_run = pd.Series(double_dqn_all_rewards[run], name="Double DQN Scores")
        double_dqn_scores_run.plot(ax=axs[1, run], label="Double DQN Scores")
        double_dqn_scores_run.rolling(window=100).mean().rename("Rolling Average").plot(ax=axs[1, run])
        axs[1, run].legend()
        axs[1, run].set_ylabel("Score")
        axs[1, run].set_title(f'Run {run + 1}: Double DQN Scores and Rolling Average')
        if run == num_runs - 1:
            axs[1, run].set_xlabel("Episode Number")

    plt.tight_layout()
    plt.savefig('outputs/rolling_averages.png')
    plt.close()

    # Display the plot
    from IPython.display import Image, display
    display(Image(filename='outputs/rolling_averages.png'))

def plot_test_results(dqn_test_rewards, double_dqn_test_rewards):
    plt.figure(figsize=(12, 8))
    plt.plot(dqn_test_rewards, label='DQN Test Rewards', color='blue')
    plt.plot(double_dqn_test_rewards, label='Double DQN Test Rewards', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Test Results of DQN and Double DQN on 200 Episodes')
    plt.legend()
    plt.savefig('outputs/test_results.png')
    plt.close()

    # Display the plot
    from IPython.display import Image, display
    display(Image(filename='outputs/test_results.png'))
