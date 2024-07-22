import random
import numpy as np
import torch
from collections import deque

class ReplayMemory(object):
    """
    A cyclic buffer that holds and samples experience tuples for training reinforcement our DQN and DDQN agents
    """
    def __init__(self, capacity, seed):
        """
        Initializes the ReplayMemory object.

        Args:
            capacity (int): The maximum number of experiences to store in the buffer.
            seed (int): The seed for the random number generator to ensure reproducibility.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.seed = random.seed(seed)

    def push(self, event):
        """
        Adds an experience tuple to the memory buffer.

        Args:
            event (tuple): A tuple containing (state, action, reward, next_state, done).
        """
        self.memory.append(event)

    def sample(self, batch_size):
        """
        Randomly samples a batch of experiences from the memory buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple containing batches of states, actions, rewards, next_states, and dones.
        """
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, next_states, actions, rewards, dones   