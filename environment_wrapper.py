# environment_wrapper.py
import numpy as np
import gymnasium as gym
from maze_env import MazeWithVasesEnv
from config import *

class MazeEnvironmentWrapper:
    def __init__(self, env=None):
        """
        Initialize the environment wrapper
        If no environment is provided, create a new one
        """
        self.env = env if env is not None else MazeWithVasesEnv()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.proximity_bonus_given = False
        
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
        This uses a simplified representation focusing on key spatial information
        """
        # Get agent position (normalized)
        agent_pos = np.array(self.env.agent_pos) / np.array([self.env.width, self.env.height])
        
        # Get agent direction as one-hot encoding
        agent_dir_onehot = np.zeros(4)
        agent_dir_onehot[self.env.agent_dir] = 1
        
        # Get coin position (normalized)
        coin_pos = np.array(self.env.coin_pos) / np.array([self.env.width, self.env.height])
        
        # Calculate relative position of coin to agent
        rel_x = coin_pos[0] - agent_pos[0]
        rel_y = coin_pos[1] - agent_pos[1]
        
        # Calculate distance to coin
        distance = abs(rel_x) + abs(rel_y)  # Manhattan distance
        
        # Get local view around the agent (extracting just key information)
        image = obs['image']
        
        # Create a compact representation:
        # 1. Is there a wall in each of the 4 directions from the agent?
        # Look at the immediate surrounding cells
        wall_front = 0
        wall_right = 0
        wall_left = 0
        wall_back = 0
        
        # Check surrounding cells based on agent direction
        agent_dir = self.env.agent_dir
        ax, ay = self.env.agent_pos
        
        # Direction offsets (right, down, left, up)
        offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        # Check front
        dx, dy = offsets[agent_dir]
        if ax + dx >= 0 and ax + dx < self.env.width and ay + dy >= 0 and ay + dy < self.env.height:
            cell = self.env.grid.get(ax + dx, ay + dy)
            wall_front = 1 if cell is not None and cell.type == 'wall' else 0
        else:
            wall_front = 1  # Out of bounds counts as wall
        
        # Check right
        right_dir = (agent_dir + 1) % 4
        dx, dy = offsets[right_dir]
        if ax + dx >= 0 and ax + dx < self.env.width and ay + dy >= 0 and ay + dy < self.env.height:
            cell = self.env.grid.get(ax + dx, ay + dy)
            wall_right = 1 if cell is not None and cell.type == 'wall' else 0
        else:
            wall_right = 1
        
        # Check left
        left_dir = (agent_dir - 1) % 4
        dx, dy = offsets[left_dir]
        if ax + dx >= 0 and ax + dx < self.env.width and ay + dy >= 0 and ay + dy < self.env.height:
            cell = self.env.grid.get(ax + dx, ay + dy)
            wall_left = 1 if cell is not None and cell.type == 'wall' else 0
        else:
            wall_left = 1
        
        # Check back
        back_dir = (agent_dir + 2) % 4
        dx, dy = offsets[back_dir]
        if ax + dx >= 0 and ax + dx < self.env.width and ay + dy >= 0 and ay + dy < self.env.height:
            cell = self.env.grid.get(ax + dx, ay + dy)
            wall_back = 1 if cell is not None and cell.type == 'wall' else 0
        else:
            wall_back = 1
        
        # Combine all features into a compact state representation
        state = np.concatenate([
            agent_pos,                     # 2 values: x, y
            agent_dir_onehot,              # 4 values: direction one-hot
            coin_pos,                      # 2 values: x, y
            [rel_x, rel_y],                # 2 values: relative position
            [distance],                    # 1 value: distance to coin
            [wall_front, wall_right, wall_left, wall_back]  # 4 values: surrounding walls
        ])
        
        return state
    
    def _get_manhattan_distance(self):
        """Calculate Manhattan distance from agent to coin"""
        agent_x, agent_y = self.env.agent_pos
        coin_x, coin_y = self.env.coin_pos
        return abs(agent_x - coin_x) + abs(agent_y - coin_y)
    
    def _shape_reward(self, original_reward, done, info):
        """
        Shape the reward to provide more feedback to the agent
        - Reward for collecting the coin (from config)
        - Penalty for breaking vases (if configured)
        - Small penalty for each step
        - Reward/penalty based on getting closer to or further from the coin
        - Bonus for getting close to the coin
        """
        # Apply coin collection reward from config if coin was collected
        if info.get('coin_collected', False):
            reward = DQN_CONFIG['COIN_REWARD']
        else:
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
            
            # Reward for getting closer to the coin (more significantly)
            reward += distance_delta * DQN_CONFIG['DISTANCE_REWARD_FACTOR']
            
            # Proximity bonus for getting close to the coin
            if DQN_CONFIG['PROXIMITY_BONUS'] and current_distance <= DQN_CONFIG['PROXIMITY_THRESHOLD']:
                # Only apply this bonus once per episode for each threshold crossing
                if self.prev_distance > DQN_CONFIG['PROXIMITY_THRESHOLD']:
                    reward += DQN_CONFIG['PROXIMITY_REWARD']
                    if DEBUG_MODE:
                        print(f"Proximity bonus applied! Distance: {current_distance}")
            
            # Update previous distance
            self.prev_distance = current_distance
        
        return reward
    
    def render(self):
        """Render the environment"""
        return self.env.get_frame(mode='rgb_array')