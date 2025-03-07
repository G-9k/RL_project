import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import os

import sys
if 'stop_button_maze' not in sys.modules:
    from stop_button_maze import StopButtonMazeEnv

from config_manager import ConfigManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Experience replay memory
Experience = namedtuple('Experience', 
                        field_names=['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Agent class
class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64,
        update_target_every=100
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon_start  # exploration rate
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.memory = ReplayMemory(memory_size)
        
        # Q-Networks
        self.q_network = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.steps = 0
    
    def act(self, state, train=True):
        if train and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train()
        
        return torch.argmax(action_values).item()
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self.steps += 1
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def learn(self, experiences):
        # Convert experience batch to numpy arrays first, then to tensors
        states = torch.from_numpy(np.array([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.array([e.action for e in experiences])).long().unsqueeze(-1).to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences])).float().unsqueeze(-1).to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.array([e.done for e in experiences])).float().unsqueeze(-1).to(device)
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.q_network(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# Preprocess observations for the DQN
def preprocess_observation(obs):
    # Extract relevant information from the dictionary observation
    # and flatten it into a vector
    
    # Get the agent's view of the grid
    image = obs["image"]
    # Flatten the image
    image_flat = image.flatten()
    
    # Extract other relevant information
    direction = obs["direction"]
    human_active = obs["human_active"]
    human_pos = obs["human_pos"]
    vases_broken = obs["vases_broken"]
    
    # Combine all features into a single vector
    features = np.concatenate([
        image_flat, 
        [direction],
        [human_active],
        human_pos,
        vases_broken
    ])
    
    return features

def train_dqn(env, agent, experiment_name, **training_config):
    # Get wandb config
    config = ConfigManager()
    wandb_config = config.get_config()['wandb']
    
    # Initialize wandb
    wandb.init(
        project=wandb_config['project_name'],
        entity=wandb_config['entity'],
        name=experiment_name,
        config={
            "learning_rate": agent.learning_rate,
            "epsilon_decay": agent.epsilon_decay,
            "batch_size": agent.batch_size,
            "memory_size": agent.memory_size,
            **training_config
        }
    )
    
    num_episodes = training_config.get('num_episodes', 1000)
    max_steps = training_config.get('max_steps', 200)
    save_frequency = training_config.get('save_frequency', 100)
    eval_frequency = training_config.get('eval_frequency', 100)
    
    scores = []
    vases_broken_stats = []
    caught_stats = []
    goal_reached_stats = []
    epsilon_history = []
    
    for i_episode in tqdm(range(1, num_episodes+1)):
        obs, _ = env.reset()
        state = preprocess_observation(obs)
        score = 0
        vases_broken_count = 0
        caught = False
        goal_reached = False
        
        for t in range(max_steps):
            # Select and perform an action
            action = agent.act(state)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Process the next state
            next_state = preprocess_observation(obs)
            done = terminated or truncated
            
            # Learn from experience
            agent.step(state, action, reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += reward
            
            # Track vases broken
            vases_broken_count = np.sum(obs["vases_broken"])
            
            # Track if caught or reached goal
            if "caught_by_human" in info and info["caught_by_human"]:
                caught = True
            if "reached_goal" in info and info["reached_goal"]:
                goal_reached = True
            
            if done:
                break
            
            # Log step metrics
            if len(agent.memory) > agent.batch_size:
                experiences = agent.memory.sample(agent.batch_size)
                loss = agent.learn(experiences)
                wandb.log({
                    "step_loss": loss,
                    "step_reward": reward,
                    "epsilon": agent.epsilon
                }, step=i_episode * max_steps + t)
        
        # Update epsilon once per episode instead of every learning step
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
        
        # Save episode statistics
        scores.append(score)
        vases_broken_stats.append(vases_broken_count)
        caught_stats.append(caught)
        goal_reached_stats.append(goal_reached)
        epsilon_history.append(agent.epsilon)
        
        # Log episode metrics
        wandb.log({
            "episode": i_episode,
            "episode_reward": score,
            "episode_length": t,
            "vases_broken": vases_broken_count,
            "caught_by_human": caught,
            "goal_reached": goal_reached,
            "epsilon": agent.epsilon
        }, step=i_episode)
        
        # Print progress
        if i_episode % 100 == 0:
            print(f"\nEpisode {i_episode}/{num_episodes}")
            print(f"Average Score (last 100): {np.mean(scores[-100:]):.2f}")
            print(f"Average Vases Broken (last 100): {np.mean(vases_broken_stats[-100:]):.2f}")
            print(f"Caught Rate (last 100): {np.mean(caught_stats[-100:]):.2f}")
            print(f"Goal Reached Rate (last 100): {np.mean(goal_reached_stats[-100:]):.2f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
    
    wandb.finish()
    
    return {
        'scores': scores,
        'vases_broken': vases_broken_stats,
        'caught': caught_stats,
        'goal_reached': goal_reached_stats,
        'epsilon': epsilon_history
    }

def plot_training_results(results):
    fig, axs = plt.subplots(5, 1, figsize=(10, 15))
    
    # Plot scores
    axs[0].plot(results['scores'])
    axs[0].set_title('Score per Episode')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Score')
    
    # Plot vases broken
    axs[1].plot(results['vases_broken'])
    axs[1].set_title('Vases Broken per Episode')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Number of Vases')
    
    # Plot caught rate
    axs[2].plot(np.cumsum(results['caught']) / np.arange(1, len(results['caught'])+1))
    axs[2].set_title('Cumulative Caught Rate')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Rate')
    
    # Plot goal reached rate
    axs[3].plot(np.cumsum(results['goal_reached']) / np.arange(1, len(results['goal_reached'])+1))
    axs[3].set_title('Cumulative Goal Reached Rate')
    axs[3].set_xlabel('Episode')
    axs[3].set_ylabel('Rate')
    
    # Plot epsilon
    axs[4].plot(results['epsilon'])
    axs[4].set_title('Epsilon over Episodes')
    axs[4].set_xlabel('Episode')
    axs[4].set_ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

def save_results(results, experiment_name):
    """Save training results to disk"""
    save_dir = "results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save results as numpy arrays
    for key, value in results.items():
        np.save(f"{save_dir}/{experiment_name}_{key}.npy", np.array(value))

def train_agent(experiment_name="baseline"):
    """Unified training function using config"""
    config = ConfigManager()
    
    # Get configs
    env_config = config.get_env_config()
    agent_config = config.get_agent_config()
    training_config = config.get_training_config()
    experiment_config = config.get_experiment_config(experiment_name)
    wandb_config = config.get_wandb_config()
    
    # Update env config with experiment-specific parameters
    env_config.update({
        "reward_for_coin": experiment_config["reward_for_coin"],
        "penalty_for_caught": experiment_config["penalty_for_caught"]
    })
    
    # Initialize environment
    env = StopButtonMazeEnv(**env_config)
    
    # Initialize agent
    obs, _ = env.reset()
    state = preprocess_observation(obs)
    state_size = len(state)
    action_size = env.action_space.n
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        **agent_config
    )
    
    # Initialize wandb
    if wandb_config.get('use_wandb', True):
        wandb.init(
            project=wandb_config['project_name'],
            name=experiment_name,
            config={
                "env_params": env_config,
                "agent_params": agent_config,
                "training_params": training_config,
                "experiment": experiment_config
            }
        )
    
    # Train agent
    results = train_dqn(env, agent, experiment_name, **training_config)
    
    # Save results and model
    save_results(results, experiment_name)
    
    if wandb_config.get('use_wandb', True):
        wandb.finish()
    
    return results

if __name__ == "__main__":
    config = ConfigManager()
    
    # Train all conditions defined in config
    for experiment_name in config.get_config()['experiments'].keys():
        print(f"\nTraining {experiment_name}...")
        train_agent(experiment_name=experiment_name)