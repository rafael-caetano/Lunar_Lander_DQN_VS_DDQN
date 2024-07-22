import gymnasium as gym
import torch
import imageio
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from src.DQNAgent_module import DQNAgent
from src.DoubleDQNAgent_module import DoubleDQNAgent

def capture_output(func):
    def wrapper(*args, **kwargs):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = func(*args, **kwargs)
        output = stdout.getvalue() + stderr.getvalue()
        return result, output
    return wrapper

@capture_output
def record_episode(env, agent=None, max_steps=1000, video_filename='episode.mp4'):
    state, _ = env.reset()
    frames = []
    total_reward = 0

    for _ in range(max_steps):
        frames.append(env.render())
        if agent is None:
            action = env.action_space.sample()  # Random action
        else:
            action = agent.act_greedy(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    # Save the video
    with imageio.get_writer(video_filename, fps=30) as writer:
        for frame in frames:
            writer.append_data(frame)

    return total_reward, video_filename

def create_lunar_lander_videos(dqn_model_path, ddqn_model_path, seed=42):
    # Set up the environment
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    video_paths = []

    # Record video with no agent (random actions)
    (reward, video_path), output = record_episode(env, video_filename='outputs/random_actions.mp4')
    print(f'Random Actions Reward: {reward}')
    video_paths.append(video_path)

    # Record video with trained DQN agent
    dqn_agent = DQNAgent(state_size, action_size, seed)
    dqn_agent.local_qnetwork.load_state_dict(torch.load(dqn_model_path))
    (reward, video_path), output = record_episode(env, dqn_agent, video_filename='outputs/trained_dqn.mp4')
    print(f'Trained DQN Reward: {reward}')
    video_paths.append(video_path)

    # Record video with trained DDQN agent
    ddqn_agent = DoubleDQNAgent(state_size, action_size, seed)
    ddqn_agent.local_qnetwork.load_state_dict(torch.load(ddqn_model_path))
    (reward, video_path), output = record_episode(env, ddqn_agent, video_filename='outputs/trained_ddqn.mp4')
    print(f'Trained DDQN Reward: {reward}')
    video_paths.append(video_path)

    env.close()

    return video_paths