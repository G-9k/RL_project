# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import math
import os

from config import DQN_CONFIG

# Define the experience replay memory
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        """Save an experience"""
        self.memory.append(Experience(*args))
    
    def sample(self, batch_size):
        """Random sample from the memory"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Define the DQN neural network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        hidden_size = DQN_CONFIG['HIDDEN_SIZE']
        num_layers = DQN_CONFIG['NUM_LAYERS']
        
        # Create lists to hold layers and batch norms
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(state_size, hidden_size))
        self.bns.append(nn.BatchNorm1d(hidden_size))
        
        # Hidden layers
        for i in range(num_layers - 2):  # -2 because we already have input and will add output
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.bns.append(nn.BatchNorm1d(hidden_size))
        
        # Last hidden layer (with reduced size)
        self.layers.append(nn.Linear(hidden_size, hidden_size // 2))
        self.bns.append(nn.BatchNorm1d(hidden_size // 2))
        
        # Output layer
        self.output = nn.Linear(hidden_size // 2, action_size)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        # Handle both batched and non-batched inputs
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            single_input = True
        else:
            single_input = False

        # For single samples during action selection, temporarily disable training mode
        with torch.set_grad_enabled(self.training and not single_input):
            # Process through all layers except output
            for layer, bn in zip(self.layers, self.bns):
                if single_input:
                    bn.eval()
                x = F.relu(bn(layer(x)))
                if single_input:
                    bn.train(self.training)

            # Output layer
            x = self.output(x)

        if single_input:
            return x.squeeze(0)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Initialize hyperparameters from config
        self.gamma = DQN_CONFIG['GAMMA']
        self.epsilon_start = DQN_CONFIG['EPSILON_START']
        self.epsilon_end = DQN_CONFIG['EPSILON_END']
        self.epsilon_decay = DQN_CONFIG['EPSILON_DECAY']
        self.batch_size = DQN_CONFIG['BATCH_SIZE']
        self.learning_rate = DQN_CONFIG['LEARNING_RATE']
        
        # Initialize neural networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Initialize replay memory
        self.memory = PrioritizedReplayMemory(DQN_CONFIG['MEMORY_SIZE'])
        
        # Keep track of training steps
        self.steps_done = 0
    
    def select_action(self, state, training=True):
        """Select an action using epsilon-greedy policy"""
        # Calculate epsilon based on decay schedule
        if training:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      math.exp(-1. * self.steps_done / self.epsilon_decay)
            self.steps_done += 1
            self.epsilon = epsilon
        else:
            epsilon = 0.05  # Small epsilon for testing to allow some exploration
        
        # Choose random action with probability epsilon
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # Otherwise, choose the best action according to the model
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.policy_net.eval()  # Ensure eval mode for action selection
            q_values = self.policy_net(state_tensor)
            self.policy_net.train(training)  # Restore previous mode
            return q_values.max(1)[1].item()
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0.0  # Not enough samples yet
        
        # Sample a batch of experiences
        experiences, indices, weights = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch.done)).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next Q values using Double DQN
        with torch.no_grad():
            # Get actions from policy network
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            # Get Q-values from target network for those actions
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            # Compute expected Q values
            expected_q_values = reward_batch.unsqueeze(1) + (1 - done_batch.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute loss with importance sampling weights
        loss = (weights.unsqueeze(1) * F.smooth_l1_loss(current_q_values, expected_q_values, reduction='none')).mean()
        
        # Update priorities in replay memory
        td_errors = np.abs((expected_q_values - current_q_values).detach().cpu().numpy())
        self.memory.update_priorities(indices, td_errors.flatten())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)  # More stable gradient clipping
        self.optimizer.step()
        
        return loss.item()
    
    
    def update_target_network(self):
        """Update the target network with the policy network's weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path="models"):
        """Save the model"""
        os.makedirs(path, exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon
        }, os.path.join(path, "dqn_agent.pth"))
        print(f"Model saved to {os.path.join(path, 'dqn_agent.pth')}")
    
    def load(self, path="models"):
        """Load the model"""
        checkpoint = torch.load(os.path.join(path, "dqn_agent.pth"))
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {os.path.join(path, 'dqn_agent.pth')}")

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = no prioritization)
        self.beta = beta    # Correction factor (1 = full correction)
        self.beta_increment = beta_increment  # Beta annealing
        self.eps = 1e-6     # Small positive constant to avoid zero priority
        
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, *args, priority=None):
        """Save an experience with priority"""
        if len(self.memory) < self.capacity:
            self.memory.append(Experience(*args))
        else:
            self.memory[self.position] = Experience(*args)
        
        # New experiences get max priority
        if priority is None:
            priority = self.priorities.max() if self.size > 0 else 1.0
        
        # Update priority
        if self.position >= len(self.priorities):
            self.priorities = np.append(self.priorities, priority)
        else:
            self.priorities[self.position] = priority
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample a batch of experiences with importance sampling"""
        if self.size < batch_size:
            # Not enough samples, return what we have
            indices = np.arange(self.size)
        else:
            # Calculate sampling probabilities
            probs = self.priorities[:self.size] ** self.alpha
            probs /= probs.sum()
            
            # Sample indices based on probabilities
            indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Increment beta for annealing
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get experiences
        experiences = [self.memory[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.eps
    
    def __len__(self):
        return self.size