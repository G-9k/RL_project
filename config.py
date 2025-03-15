# config.py
# Configuration file for the MiniGrid Maze environment

# Maze dimensions
MAZE_WIDTH = 15
MAZE_HEIGHT = 15

# Path width (number of cells for the paths)
PATH_WIDTH = 2  # 1 = narrow paths, 2-3 = wider paths

# Number of obstacles (vases)
NUM_VASES = 5

# Agent configuration
AGENT_VIEW_SIZE = 7  # Number of tiles the agent can see around it
MAX_STEPS = 200  # Maximum number of steps before the episode ends

# Visualization
RENDER_FPS = 10  # Frames per second for rendering
CELL_SIZE = 32  # Size of each cell in pixels

# Auto-scaling for large mazes
MAX_WINDOW_WIDTH = 1200  # Maximum window width in pixels
MAX_WINDOW_HEIGHT = 800  # Maximum window height in pixels

# Calculate the appropriate cell size based on maze dimensions
def get_adjusted_cell_size():
    # Calculate cell size that would fit the maze in the maximum window size
    width_cell_size = MAX_WINDOW_WIDTH // MAZE_WIDTH
    height_cell_size = MAX_WINDOW_HEIGHT // MAZE_HEIGHT
    
    # Use the smaller of the two to ensure both dimensions fit
    adjusted_size = min(width_cell_size, height_cell_size)
    
    # Ensure a minimum size for visibility
    return max(adjusted_size, 8)

# Debug options
DEBUG_MODE = False  # Enable debug printing

# Use this to get the proper cell size when initializing visualization
ADJUSTED_CELL_SIZE = get_adjusted_cell_size()