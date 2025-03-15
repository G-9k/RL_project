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
        
        # Input layer
        layers = [nn.Linear(state_size, hidden_size), nn.ReLU()]
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, action_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

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
        self.memory = ReplayMemory(DQN_CONFIG['MEMORY_SIZE'])
        
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
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()
    
    def learn(self):
        """Perform one step of optimization"""
        if len(self.memory) < self.batch_size:
            return 0.0  # Not enough samples yet
        
        # Sample a batch of experiences
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch.done)).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute target Q values
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(1)[0]
        
        # Compute the expected Q values
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * max_next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
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