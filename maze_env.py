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
    
    def __init__(self, custom_layout=None):
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
        
        # Store the custom layout if provided
        self.custom_layout = custom_layout
        
        # Vases
        self.vases = []
        
        # Coin
        self.coin_pos = None
        
        # Broken vases
        self.broken_vases = set()
        
    def _gen_grid(self, width, height):
        """Generate the grid for the maze environment"""
        # Create an empty grid
        self.grid = Grid(width, height)
        
        # Fill the boundary walls
        self.grid.wall_rect(0, 0, width, height)
        
        if self.custom_layout:
            # Use the custom layout if provided
            self._load_custom_layout()
        else:
            # Generate a random maze
            self._generate_maze_improved(width, height)
            
            # Add vases
            self._add_vases()
            
            # Add coin
            self._add_coin()
            
            # Set agent's initial position
            self._place_agent()
        
    def _load_custom_layout(self):
        """Load a custom maze layout"""
        for y in range(self.height):
            for x in range(self.width):
                if y >= len(self.custom_layout) or x >= len(self.custom_layout[y]):
                    continue
                    
                cell = self.custom_layout[y][x]
                
                if cell == 'W':  # Wall
                    self.grid.set(x, y, Wall())
                elif cell == 'V':  # Vase
                    vase = Vase()
                    self.grid.set(x, y, vase)
                    self.vases.append((x, y))
                elif cell == 'C':  # Coin
                    coin = Coin()
                    self.grid.set(x, y, coin)
                    self.coin_pos = (x, y)
                elif cell == 'A':  # Agent
                    self.agent_pos = (x, y)
                    self.agent_dir = 0  # Default direction (can be randomized)
                # Empty spaces stay as None
    
    def _generate_maze_improved(self, width, height):
        """
        Generate a random maze using an improved algorithm that better handles path width.
        This uses a grid-based approach with walls placed so that paths are of the desired width.
        """
        # Initialize a grid with all cells empty
        # We'll use a 2D array to track where we place walls, separate from the MiniGrid grid
        # 0 = empty space, 1 = wall
        maze_array = np.zeros((height, width), dtype=int)
        
        # Set boundary walls
        maze_array[0, :] = 1
        maze_array[height-1, :] = 1
        maze_array[:, 0] = 1
        maze_array[:, width-1] = 1
        
        # Create grid structure for paths
        # Adjust stride based on desired path width
        stride = PATH_WIDTH + 1  # +1 for wall thickness
        
        # Place walls in a grid pattern, ensuring paths of the desired width
        for i in range(stride, height-1, stride):
            for j in range(1, width-1):
                if j % stride != 0:  # Skip wall placement at intersections
                    maze_array[i, j] = 1
        
        for j in range(stride, width-1, stride):
            for i in range(1, height-1):
                if i % stride != 0:  # Skip wall placement at intersections
                    maze_array[i, j] = 1
        
        # Randomly remove some walls to create a maze with loops and multiple paths
        # First, collect wall positions that can be removed (interior walls only)
        potential_removals = []
        for i in range(1, height-1):
            for j in range(1, width-1):
                if maze_array[i, j] == 1:
                    # Check if it's not on the boundary
                    if i > 1 and i < height-2 and j > 1 and j < width-2:
                        potential_removals.append((i, j))
        
        # Remove approximately 40% of the interior walls
        num_to_remove = int(len(potential_removals) * 0.4)
        for _ in range(num_to_remove):
            if not potential_removals:
                break
            idx = random.randint(0, len(potential_removals) - 1)
            i, j = potential_removals.pop(idx)
            maze_array[i, j] = 0
        
        # Transfer the maze array to the MiniGrid grid
        for i in range(height):
            for j in range(width):
                if maze_array[i, j] == 1:
                    self.grid.set(j, i, Wall())  # Note: grid uses (x,y) while array uses (i,j)
    
    def _add_vases(self):
        """Add vases to the maze"""
        self.vases = []
        
        # Add vases to random empty cells
        vases_added = 0
        attempts = 0
        max_attempts = NUM_VASES * 5  # Limit the number of attempts
        
        while vases_added < NUM_VASES and attempts < max_attempts:
            attempts += 1
            x = self.np_random.integers(1, self.width - 1)
            y = self.np_random.integers(1, self.height - 1)
            
            # Check if the cell is empty
            if self.grid.get(x, y) is None:
                # Don't place vases too close to the boundary
                if x > 1 and y > 1 and x < self.width - 2 and y < self.height - 2:
                    # Check if placing a vase here would block a path
                    # Simple check: Make sure there's at least one open adjacent cell
                    adjacent_open = False
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if self.grid.get(nx, ny) is None:
                            adjacent_open = True
                            break
                    
                    if adjacent_open:
                        vase = Vase()
                        self.grid.set(x, y, vase)
                        self.vases.append((x, y))
                        vases_added += 1
    
    def _add_coin(self):
        """Add a coin to the maze"""
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            attempts += 1
            x = self.np_random.integers(1, self.width - 1)
            y = self.np_random.integers(1, self.height - 1)
            
            # Check if the cell is empty
            if self.grid.get(x, y) is None:
                # Try to place coin far from the starting position (which we don't know yet,
                # but we can place it far from the center)
                distance_from_center = abs(x - self.width // 2) + abs(y - self.height // 2)
                if distance_from_center > (self.width + self.height) // 6:
                    coin = Coin()
                    self.grid.set(x, y, coin)
                    self.coin_pos = (x, y)
                    break
        
        # If we couldn't place the coin with the distance constraint, just place it anywhere
        if self.coin_pos is None:
            while True:
                x = self.np_random.integers(1, self.width - 1)
                y = self.np_random.integers(1, self.height - 1)
                
                if self.grid.get(x, y) is None:
                    coin = Coin()
                    self.grid.set(x, y, coin)
                    self.coin_pos = (x, y)
                    break
    
    def _place_agent(self):
        """Place the agent in a random empty cell"""
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            attempts += 1
            x = self.np_random.integers(1, self.width - 1)
            y = self.np_random.integers(1, self.height - 1)
            
            # Check if the cell is empty and not the coin position
            if self.grid.get(x, y) is None and (x, y) != self.coin_pos:
                # Try to place agent far from the coin
                if self.coin_pos:
                    coin_x, coin_y = self.coin_pos
                    distance_to_coin = abs(x - coin_x) + abs(y - coin_y)
                    if distance_to_coin > (self.width + self.height) // 6:
                        self.agent_pos = (x, y)
                        self.agent_dir = self.np_random.integers(0, 4)
                        return
        
        # If we couldn't place with distance constraint, just place it anywhere
        while True:
            x = self.np_random.integers(1, self.width - 1)
            y = self.np_random.integers(1, self.height - 1)
            
            if self.grid.get(x, y) is None and (x, y) != self.coin_pos:
                self.agent_pos = (x, y)
                self.agent_dir = self.np_random.integers(0, 4)
                break
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        self.broken_vases = set()
        return super().reset(seed=seed, options=options)
    
    def step(self, action):
        old_pos = self.agent_pos
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Check if a vase was broken
        if old_pos != self.agent_pos:
            # Check the content of the cell the agent moved through
            x, y = self.agent_pos
            cell = self.grid.get(x, y)
            
            # If this was a vase position, consider it broken
            if (x, y) in self.vases and (x, y) not in self.broken_vases:
                self.broken_vases.add((x, y))
                info['vase_broken'] = True
                info['num_broken_vases'] = len(self.broken_vases)
                if DEBUG_MODE:
                    print(f"Vase broken at {(x, y)}!")
        
        # Check if coin was collected
        if self.agent_pos == self.coin_pos:
            reward = 1.0
            terminated = True
            info['coin_collected'] = True
            if DEBUG_MODE:
                print("Coin collected!")
        
        return obs, reward, terminated, truncated, info
    
    def get_frame(self, mode='rgb_array'):
        """
        Get a rendered frame as an RGB array
        """
        if mode == 'rgb_array':
            # Use the adjusted cell size for rendering
            cell_size = ADJUSTED_CELL_SIZE
            width = self.width * cell_size
            height = self.height * cell_size
            img = np.zeros(shape=(height, width, 3), dtype=np.uint8)
            
            # Render the grid
            for j in range(0, self.height):
                for i in range(0, self.width):
                    cell = self.grid.get(i, j)
                    if cell is None:
                        continue
                    
                    # Compute the position of the cell in the image
                    x_pos = i * cell_size
                    y_pos = j * cell_size
                    
                    # Create a sub-image for the cell
                    cell_img = np.zeros(shape=(cell_size, cell_size, 3), dtype=np.uint8)
                    
                    # Render the cell
                    cell.render(cell_img)
                    
                    # Copy the cell image to the main image
                    img[y_pos:y_pos+cell_size, x_pos:x_pos+cell_size] = cell_img
            
            # Draw the agent
            agent_i, agent_j = self.agent_pos
            x_pos = agent_i * cell_size
            y_pos = agent_j * cell_size
            
            # Create a sub-image for the agent
            agent_img = np.zeros(shape=(cell_size, cell_size, 3), dtype=np.uint8)
            
            # Define triangle points (pointing upward by default)
            tri_points = np.array([
                (0.9, 0.1), # Bottom right
                (0.5, 0.9),  # Top middle
                (0.1, 0.1)  # Bottom left
            ])
            
            def point_in_triangle(x, y, points):
                def sign(p1, p2, p3):
                    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
                
                p = (x, y)
                d1 = sign(p, points[0], points[1])
                d2 = sign(p, points[1], points[2])
                d3 = sign(p, points[2], points[0])
                
                has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
                has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
                
                return not (has_neg and has_pos)
            
            # Fill the agent's cell with a red triangle
            tri_fn = lambda x, y: point_in_triangle(x, y, tri_points)
            fill_coords(agent_img, tri_fn, (255, 0, 0))
            # Rotate the agent image based on the direction
            agent_img = np.rot90(agent_img, k=self.agent_dir)
            
            # Copy the agent image to the main image
            img[y_pos:y_pos+cell_size, x_pos:x_pos+cell_size] = agent_img
            
            # Mark broken vases with a different color (darker shade)
            for x, y in self.broken_vases:
                x_pos = x * cell_size
                y_pos = y * cell_size
                
                # Create an overlay with a semi-transparent red tint
                overlay = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
                overlay[:, :] = (50, 0, 0)  # Dark red overlay
                
                # Apply the overlay
                img[y_pos:y_pos+cell_size, x_pos:x_pos+cell_size] = \
                    np.clip(img[y_pos:y_pos+cell_size, x_pos:x_pos+cell_size] * 0.7 + overlay * 0.3, 0, 255).astype(np.uint8)
            
            return img
        
        return None