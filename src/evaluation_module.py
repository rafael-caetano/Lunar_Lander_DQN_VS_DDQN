import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import List, Tuple
from IPython.display import display, Image

def load_and_test_agent(agent_class, env_name: str, state_size: int, action_size: int, model_path: str, num_episodes: int = 1000, seed: int = 90) -> List[float]:
    agent = agent_class(state_size, action_size, seed)
    agent.local_qnetwork.load_state_dict(torch.load(model_path))
    env = gym.make(env_name)
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.act_greedy(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        rewards.append(episode_reward)

    return rewards

def identify_low_scoring_episodes(rewards: List[float], threshold: float = 200) -> List[int]:
    return [i for i, reward in enumerate(rewards) if reward < threshold]

def compare_agent_performances(dqn_rewards: List[float], ddqn_rewards: List[float], prefix: str = "") -> None:
    dqn_mean, ddqn_mean = np.mean(dqn_rewards), np.mean(ddqn_rewards)
    dqn_median, ddqn_median = np.median(dqn_rewards), np.median(ddqn_rewards)
    dqn_std, ddqn_std = np.std(dqn_rewards), np.std(ddqn_rewards)
    
    print(f"DQN  - Mean: {dqn_mean:.2f}, Median: {dqn_median:.2f}, Std Dev: {dqn_std:.2f}")
    print(f"DDQN - Mean: {ddqn_mean:.2f}, Median: {ddqn_median:.2f}, Std Dev: {ddqn_std:.2f}")
    
    # Reward distribution histogram
    plt.figure(figsize=(12, 6))
    min_reward = min(min(dqn_rewards), min(ddqn_rewards))
    max_reward = max(max(dqn_rewards), max(ddqn_rewards))
    bins = np.linspace(min_reward, max_reward, 31)
    plt.hist(dqn_rewards, bins=bins, alpha=0.5, label='DQN', density=True)
    plt.hist(ddqn_rewards, bins=bins, alpha=0.5, label='DDQN', density=True)
    plt.xlabel('Reward')
    plt.ylabel('Density')
    plt.title('Distribution of Rewards')
    plt.legend()
    plt.savefig(f'outputs/{prefix}reward_distribution.png')
    display(Image(filename=f'outputs/{prefix}reward_distribution.png'))
    plt.close()
    
    # Rolling average plot
    window = 100
    dqn_rolling = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
    ddqn_rolling = np.convolve(ddqn_rewards, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(12, 6))
    plt.plot(dqn_rolling, label='DQN Rolling Avg')
    plt.plot(ddqn_rolling, label='DDQN Rolling Avg')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'Rolling Average Reward (Window={window})')
    plt.legend()
    plt.savefig(f'outputs/{prefix}rolling_average_reward.png')
    display(Image(filename=f'outputs/{prefix}rolling_average_reward.png'))
    plt.close()

    # Cumulative reward plot
    dqn_cumulative = np.cumsum(dqn_rewards)
    ddqn_cumulative = np.cumsum(ddqn_rewards)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dqn_cumulative, label='DQN Cumulative Reward')
    plt.plot(ddqn_cumulative, label='DDQN Cumulative Reward')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward Over Episodes')
    plt.legend()
    plt.savefig(f'outputs/{prefix}cumulative_reward.png')
    display(Image(filename=f'outputs/{prefix}cumulative_reward.png'))
    plt.close()

    # Per-episode score plot
    plt.figure(figsize=(12, 8))
    plt.plot(dqn_rewards, label='DQN Test Rewards', color='blue', alpha=0.7)
    plt.plot(ddqn_rewards, label='DDQN Test Rewards', color='orange', alpha=0.7)
    plt.scatter(range(len(dqn_rewards)), dqn_rewards, color='blue', alpha=0.3, s=10)
    plt.scatter(range(len(ddqn_rewards)), ddqn_rewards, color='orange', alpha=0.3, s=10)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Test Results of DQN and DDQN on {len(dqn_rewards)} Episodes')
    plt.legend()
    plt.savefig(f'outputs/{prefix}per_episode_scores.png')
    display(Image(filename=f'outputs/{prefix}per_episode_scores.png'))
    plt.close()