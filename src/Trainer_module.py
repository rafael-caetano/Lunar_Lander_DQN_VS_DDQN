import os
import numpy as np
import gymnasium as gym
import torch
from collections import deque
from src.set_seed import set_seed
from src.pad_sequences import pad_sequences

class Trainer:
    def __init__(self, agent_class, env_name, state_size, action_size, seed=42):
        self.agent_class = agent_class
        self.env_name = env_name
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

    def train_agent(self, agent, env, number_episodes, max_timesteps, eps_start, eps_end, eps_decay, model_save_path, solved_model_save_path):
        scores = deque(maxlen=100)
        epsilon = eps_start
        all_rewards = []
        average_rewards = []
        solved_at = None

        for episode in range(1, number_episodes + 1):
            state, _ = env.reset()
            score = 0
            for t in range(max_timesteps):
                action = agent.act(state, epsilon)
                next_state, reward, done, _, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores.append(score)
            epsilon = max(eps_end, eps_decay * epsilon)
            all_rewards.append(score)
            average_rewards.append(np.mean(scores))
            print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores):.2f}', end="")
            if episode % 100 == 0:
                print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores):.2f}')
            if solved_at is None and np.mean(scores) >= 200.0:
                solved_at = episode - 100
                print(f'\nEnvironment solved in {episode - 100} episodes!\tAverage Score: {np.mean(scores):.2f}')
                torch.save(agent.local_qnetwork.state_dict(), solved_model_save_path)

        torch.save(agent.local_qnetwork.state_dict(), model_save_path)
        return all_rewards, average_rewards, solved_at

    def train_multiple_runs(self, num_runs=3, number_episodes=2000, max_timesteps=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        all_rewards = []
        all_average_rewards = []
        solved_episodes = []

        for run in range(num_runs):
            current_seed = self.seed + run  # Use different seed for each run
            print(f'Run {run + 1}/{num_runs} with seed {current_seed}')
            set_seed(current_seed)  # Set seed for reproducibility
            env = gym.make(self.env_name)
            env.reset(seed=current_seed)  # Set the environment seed
            env.action_space.seed(current_seed)
            env.observation_space.seed(current_seed)
            agent = self.agent_class(self.state_size, self.action_size, current_seed)

            # Create outputs directory if it doesn't exist
            if not os.path.exists('outputs'):
                os.makedirs('outputs')

            model_save_path = f'outputs/{self.agent_class.__name__}_run{run}_final.pth'
            solved_model_save_path = f'outputs/{self.agent_class.__name__}_run{run}_solved.pth'
            rewards, average_rewards, solved_at = self.train_agent(agent, env, number_episodes, max_timesteps, eps_start, eps_end, eps_decay, model_save_path, solved_model_save_path)
            all_rewards.append(rewards)
            all_average_rewards.append(average_rewards)
            solved_episodes.append(solved_at)
        
        max_len = max(len(rewards) for rewards in all_rewards)
        all_rewards = pad_sequences(all_rewards, max_len)
        all_average_rewards = pad_sequences(all_average_rewards, max_len)

        avg_rewards = np.nanmean(all_rewards, axis=0)
        avg_average_rewards = np.nanmean(all_average_rewards, axis=0)
        std_rewards = np.nanstd(all_rewards, axis=0)
        std_average_rewards = np.nanstd(all_average_rewards, axis=0)

        return avg_rewards, avg_average_rewards, std_rewards, std_average_rewards, all_rewards, all_average_rewards, solved_episodes
