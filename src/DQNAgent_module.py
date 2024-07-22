import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from src.Network import Network
from src.ReplayMemory import ReplayMemory

class DQNAgent():
    def __init__(self, state_size, action_size, seed):
        """
        Initializes the DQNAgent with state and action sizes, network, optimizer, and replay memory.

        Args:
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            seed (int): Random seed for reproducibility.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.local_qnetwork = Network(state_size, action_size, seed).to(self.device)
        self.target_qnetwork = Network(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=5e-4)
        self.memory = ReplayMemory(int(1e5), seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Stores experience in replay memory and triggers the learning process at defined intervals.
        """
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > 100:
                experiences = self.memory.sample(100)
                self.learn(experiences, 0.99)

    def act(self, state, epsilon=0.):
        """
        Returns actions for a given state using the local Q-network and an epsilon-greedy policy.
        
        If a random number is greater than epsilon, the action with the highest
        Q-value is selected (exploitation). Otherwise, a random action is selected (exploration).
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon: 
            return np.argmax(action_values.cpu().data.numpy())
        else: 
            return random.choice(np.arange(self.action_size))

    def act_greedy(self, state):
        """
        Returns actions for a given state using the local Q-network and a greedy policy only, this is used for evaluating the Networks.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences, discount_factor):
        """
        Implements the learning process using the standard Deep Q-Network (DQN) approach.

        Updates value parameters using a given batch of experience tuples.

        Args:
        - experiences: Tuple containing (states, next_states, actions, rewards, dones)
        - discount_factor: Discount factor for future rewards (gamma)
        """
        states, next_states, actions, rewards, dones = experiences
        
        # Compute the max Q values for the next states using the target network
        q_targets_next = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1) # detach avoids backpropagation
        
        # Compute the Q targets for the current states
        q_targets = rewards + (discount_factor * q_targets_next * (1 - dones)) # If done, only rewards are considered, else, discounted max Q-value of next state is included
        
        # Gather the expected Q values from the local model for the given actions
        q_expected = self.local_qnetwork(states).gather(1, actions)
        
        # Compute the loss between the expected and target Q values using Mean Squared Error
        loss = F.mse_loss(q_expected, q_targets)
        
        # Clear any existing gradients
        self.optimizer.zero_grad()
        
        # Perform backpropagation to compute gradients
        loss.backward()
        
        # Update the local network weights using the optimizer
        self.optimizer.step()
        
        # Soft update model parameters from the local network to the target network
        self.soft_update(self.local_qnetwork, self.target_qnetwork, 1e-3)

    def soft_update(self, local_model, target_model, tau):
        """
        Softly updates the target Q-network to slowly track the local Q-network.
        
        Parameters:
        - local_model: the local Q-network model.
        - target_model: the target Q-network model.
        - tau: the interpolation parameter for updating the target network (set to 1e-3).
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)