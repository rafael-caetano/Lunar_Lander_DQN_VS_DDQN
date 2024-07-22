import torch.nn.functional as F
from src.DQNAgent_module import DQNAgent

class DoubleDQNAgent(DQNAgent):
    """
    Implements the learning process using Double Q-Learning.

    This approach helps to mitigate the overestimation bias inherent in traditional Q-Learning by decoupling the action selection and action evaluation steps. Specifically:
    
    1. The local Q-network is used to select the next action (next_actions) by identifying the action with the highest Q-value for the next states.
    2. The target Q-network is then used to estimate the value of this selected action (q_targets_next).
    3. The Q-value target (q_targets) is computed using the rewards and the discounted estimated value.
    4. The local Q-network is updated by minimizing the mean squared error (MSE) loss between the predicted Q-values (q_expected) and the target Q-values (q_targets).
    5. The target Q-network is softly updated to slowly track the local Q-network.

    This method helps in achieving a more stable and reliable learning process by reducing overestimation.
    """
    def learn(self, experiences, discount_factor):
        states, next_states, actions, rewards, dones = experiences
        
        # Double DQN: Local network selects the next action
        next_actions = self.local_qnetwork(next_states).detach().max(1)[1].unsqueeze(1)
        
        # Double DQN: Target network evaluates the next action
        q_targets_next = self.target_qnetwork(next_states).detach().gather(1, next_actions)
        
        # Compute Q targets for current states
        q_targets = rewards + (discount_factor * q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        q_expected = self.local_qnetwork(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        
        # Clear any existing gradients
        self.optimizer.zero_grad()
        
        # Perform backpropagation to compute gradients
        loss.backward()
        
        # Update the local network weights using the optimizer
        self.optimizer.step()
        
        # Soft update model parameters from the local network to the target network
        self.soft_update(self.local_qnetwork, self.target_qnetwork, 1e-3)