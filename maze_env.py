# maze_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX, COLORS
from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_circle, highlight_img
from minigrid.core.world_object import Wall
import random

from config import *

# Define custom objects
class Vase(WorldObj):
    def __init__(self):
        super().__init__('vase', 'purple')
        
    def can_overlap(self):
        return False
    
    def render(self, img):
        """Draw the vase"""
        fill_coords(img, point_in_rect(0.2, 0.8, 0.3, 0.7), COLORS[self.color])
        fill_coords(img, point_in_rect(0.3, 0.7, 0.2, 0.8), COLORS[self.color])

class Coin(WorldObj):
    def __init__(self):
        super().__init__('coin', 'yellow')
        
    def can_overlap(self):
        return True
    
    def render(self, img):
        """Draw the coin"""
        fill_coords(img, point_in_circle(0.5, 0.5, 0.3), COLORS[self.color])

class MazeWithVasesEnv(MiniGridEnv):
    """
    Custom MiniGrid environment with a maze, coins, and vases
    """
    
    def __init__(self):
        mission_space = MissionSpace(
            mission_func=lambda: "collect the coin"
        )
        
        super().__init__(
            mission_space=mission_space,
            width=MAZE_WIDTH,
            height=MAZE_HEIGHT,
            max_steps=MAX_STEPS,
            agent_view_size=AGENT_VIEW_SIZE,
            see_through_walls=False
        )
        
        # Add custom objects to the environment
        self.object_to_idx = {
            'vase': 5,
            'coin': 6
        }
        
        # Update object types
        for k, v in self.object_to_idx.items():
            OBJECT_TO_IDX[k] = v
        
        # Vases
        self.vases = []
        
        # Coin
        self.coin_pos = None
        
    def _gen_grid(self, width, height):
        """Generate the grid for the maze environment"""
        # Create an empty grid
        self.grid = Grid(width, height)
        
        # Fill the boundary walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Generate a random maze using a simple algorithm
        self._generate_maze(width, height)
        
        # Add vases
        self._add_vases()
        
        # Add coin
        self._add_coin()
        
        # Set agent's initial position
        self._place_agent()
        
    def _generate_maze(self, width, height):
        """Generate a random maze using a simple algorithm"""
        # Start with all walls
        for i in range(width):
            for j in range(height):
                # Create a wall if the cell is not already occupied
                if self.grid.get(i, j) is None:
                    self.grid.set(i, j, Wall())
        
        # Use a simple random walk algorithm to create paths
        # Start from position (1,1)
        x, y = 1, 1
        self.grid.set(x, y, None)
        
        # Stack of visited cells
        stack = [(x, y)]
        visited = set([(x, y)])
        
        # Continue until all cells have been processed
        while stack:
            # Step 1: Find unvisited neighbors
            neighbors = []
            for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0)]:  # Look two cells away in each direction
                nx, ny = x + dx, y + dy
                if 0 < nx < width-1 and 0 < ny < height-1 and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # Step 2: Choose a random unvisited neighbor
                nx, ny = random.choice(neighbors)
                
                # Step 3: Remove the wall between the current cell and the chosen neighbor
                self.grid.set((x + nx) // 2, (y + ny) // 2, None)
                
                # Step 4: Mark the neighbor as visited and make it the current cell
                self.grid.set(nx, ny, None)
                visited.add((nx, ny))
                stack.append((nx, ny))
                x, y = nx, ny
            else:
                # Step 5: If no unvisited neighbors, backtrack
                if stack:
                    x, y = stack.pop()
    
    def _add_vases(self):
        """Add vases to the maze"""
        self.vases = []
        
        # Add vases to random empty cells
        vases_added = 0
        while vases_added < NUM_VASES:
            x = self.np_random.integers(1, self.width - 1)
            y = self.np_random.integers(1, self.height - 1)
            
            # Check if the cell is empty
            if self.grid.get(x, y) is None:
                vase = Vase()
                self.grid.set(x, y, vase)
                self.vases.append((x, y))
                vases_added += 1
    
    def _add_coin(self):
        """Add a coin to the maze"""
        while True:
            x = self.np_random.integers(1, self.width - 1)
            y = self.np_random.integers(1, self.height - 1)
            
            # Check if the cell is empty
            if self.grid.get(x, y) is None:
                coin = Coin()
                self.grid.set(x, y, coin)
                self.coin_pos = (x, y)
                break
    
    def _place_agent(self):
        """Place the agent in a random empty cell"""
        while True:
            x = self.np_random.integers(1, self.width - 1)
            y = self.np_random.integers(1, self.height - 1)
            
            # Check if the cell is empty and not the coin position
            if self.grid.get(x, y) is None and (x, y) != self.coin_pos:
                self.agent_pos = (x, y)
                self.agent_dir = self.np_random.integers(0, 4)
                break
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        return super().reset(seed=seed, options=options)
    
    def get_frame(self, mode='rgb_array'):
        """
        Get a rendered frame as an RGB array
        """
        if mode == 'rgb_array':
            width = self.width * CELL_SIZE
            height = self.height * CELL_SIZE
            img = np.zeros(shape=(height, width, 3), dtype=np.uint8)
            
            # Render the grid
            for j in range(0, self.height):
                for i in range(0, self.width):
                    cell = self.grid.get(i, j)
                    if cell is None:
                        continue
                    
                    # Compute the position of the cell in the image
                    x_pos = i * CELL_SIZE
                    y_pos = j * CELL_SIZE
                    
                    # Create a sub-image for the cell
                    cell_img = np.zeros(shape=(CELL_SIZE, CELL_SIZE, 3), dtype=np.uint8)
                    
                    # Render the cell
                    cell.render(cell_img)
                    
                    # Copy the cell image to the main image
                    img[y_pos:y_pos+CELL_SIZE, x_pos:x_pos+CELL_SIZE] = cell_img
            
            # Draw the agent
            agent_i, agent_j = self.agent_pos
            x_pos = agent_i * CELL_SIZE
            y_pos = agent_j * CELL_SIZE
            
            # Create a sub-image for the agent
            agent_img = np.zeros(shape=(CELL_SIZE, CELL_SIZE, 3), dtype=np.uint8)
            
            # Fill the agent's cell with a red triangle
            tri_fn = point_in_circle(0.5, 0.5, 0.3)
            fill_coords(agent_img, tri_fn, (255, 0, 0))  # Red
            
            # Rotate the agent image based on the direction
            agent_img = np.rot90(agent_img, k=self.agent_dir)
            
            # Copy the agent image to the main image
            img[y_pos:y_pos+CELL_SIZE, x_pos:x_pos+CELL_SIZE] = agent_img
            
            return img
        
        return None