import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    """
    A neural network module for approximating functions in a reinforcement learning environment.
    The network predicts action values based on the current state input.
    
    Args:
    state_size (int): Dimensionality of the input state/observation space (e.g., 8 for Lunar Lander).
    action_size (int): Number of possible actions the agent can take (e.g., 4 for Lunar Lander).
    seed (int): Random seed for reproducibility of the model's initialization.
    """
    def __init__(self, state_size, action_size, seed):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)    # Set the random seed for consistent initialization
        self.fc1 = nn.Linear(state_size, 64)   # First fully connected layer with 64 neurons
        self.fc2 = nn.Linear(64, 64)           # Second fully connected layer, maintaining 64 neurons
        self.fc3 = nn.Linear(64, action_size)  # Output layer mapping to the action size

    def forward(self, state):
        """
        Defines the forward pass of the neural network using ReLU activations.
        
        Args:
        state (Tensor): The input state tensor to the network.
        
        Returns:
        Tensor: The output tensor representing action values, with shape (batch_size, action_size).
        """
        x = F.relu(self.fc1(state))  # Process input through first layer and apply ReLU activation
        x = F.relu(self.fc2(x))      # Apply second layer transformation followed by ReLU activation
        return self.fc3(x)           # Output layer transformation, returns the action values