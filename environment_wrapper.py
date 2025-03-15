# environment_wrapper.py
import numpy as np
import gymnasium as gym
from maze_env import MazeWithVasesEnv
from config import DQN_CONFIG

class MazeEnvironmentWrapper:
    def __init__(self, env=None):
        """
        Initialize the environment wrapper
        If no environment is provided, create a new one
        """
        self.env = env if env is not None else MazeWithVasesEnv()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # For tracking distance to goal
        self.prev_distance = None
    
    def reset(self):
        """Reset the environment and return a flattened state"""
        obs, info = self.env.reset()
        
        # Calculate distance to coin
        self.prev_distance = self._get_manhattan_distance()
        
        # Return processed state
        return self._process_observation(obs), info
    
    def step(self, action):
        """
        Take a step in the environment and return:
        - processed observation
        - shaped reward
        - done flag
        - truncated flag
        - info dictionary
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply reward shaping
        shaped_reward = self._shape_reward(reward, terminated, info)
        
        # Return processed state and shaped reward
        return self._process_observation(obs), shaped_reward, terminated, truncated, info
    
    def _process_observation(self, obs):
        """
        Process the observation to create a suitable state for the DQN
        This flattens the image observation and adds agent position and direction
        """
        # Get the flattened image
        image = obs['image']
        flat_image = image.flatten()
        
        # Add agent's absolute position and direction
        agent_pos = np.array(self.env.agent_pos) / np.array([self.env.width, self.env.height])
        agent_dir_onehot = np.zeros(4)
        agent_dir_onehot[self.env.agent_dir] = 1
        
        # Add coin position (normalized)
        coin_pos = np.array(self.env.coin_pos) / np.array([self.env.width, self.env.height])
        
        # Combine all features
        state = np.concatenate([flat_image, agent_pos, agent_dir_onehot, coin_pos])
        
        return state
    
    def _get_manhattan_distance(self):
        """Calculate Manhattan distance from agent to coin"""
        agent_x, agent_y = self.env.agent_pos
        coin_x, coin_y = self.env.coin_pos
        return abs(agent_x - coin_x) + abs(agent_y - coin_y)
    
    def _shape_reward(self, original_reward, done, info):
        """
        Shape the reward to provide more feedback to the agent
        - Reward for collecting the coin (original reward)
        - Penalty for breaking vases (if configured)
        - Small penalty for each step
        - Reward/penalty based on getting closer to or further from the coin
        """
        reward = original_reward
        
        # Apply vase breaking penalty
        if info.get('vase_broken', False):
            reward += DQN_CONFIG['VASE_PENALTY']
        
        # Apply step penalty to encourage efficient paths
        reward += DQN_CONFIG['STEP_PENALTY']
        
        # Apply distance-based reward shaping
        if DQN_CONFIG['USE_DISTANCE_REWARD']:
            current_distance = self._get_manhattan_distance()
            distance_delta = self.prev_distance - current_distance
            
            # Reward for getting closer to the coin
            reward += distance_delta * DQN_CONFIG['DISTANCE_REWARD_FACTOR']
            
            # Update previous distance
            self.prev_distance = current_distance
        
        return reward
    
    def render(self):
        """Render the environment"""
        return self.env.get_frame(mode='rgb_array')