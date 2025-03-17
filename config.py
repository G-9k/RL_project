# config.py
# Configuration file for the MiniGrid Maze environment

# Maze dimensions
MAZE_WIDTH = 10
MAZE_HEIGHT = 10

# Path width (number of cells for the paths)
PATH_WIDTH = 2  # 1 = narrow paths, 2-3 = wider paths

# Number of obstacles (vases)
NUM_VASES = 5

# Agent configuration
AGENT_VIEW_SIZE = max(MAZE_HEIGHT, MAZE_WIDTH)
if AGENT_VIEW_SIZE % 2 == 0:
    AGENT_VIEW_SIZE += 1  # Make it odd if it's even

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

# DQN Agent Parameters
DQN_CONFIG = {
    # Reward structure
    'COIN_REWARD': 100.0,        # Reward for collecting the coin
    'VASE_PENALTY': 0.0,        # Penalty for breaking a vase
    'STEP_PENALTY': -0.1,       # Small penalty for each step to encourage efficiency
    'DISTANCE_REWARD_FACTOR': 0.2,  # Increased factor for distance-based reward
    'USE_DISTANCE_REWARD': True,    # Whether to use distance-based reward shaping
    'PROXIMITY_BONUS': False,        # Enable proximity bonus
    'PROXIMITY_THRESHOLD': 5,       # Distance threshold for proximity bonus
    'PROXIMITY_REWARD': 5.0,        # Bonus for getting within threshold distance
    
    # Neural network architecture
    'HIDDEN_SIZE': 256,        # Size of hidden layers
    'NUM_LAYERS': 5,           # Number of hidden layers
    
    # Training parameters
    'LEARNING_RATE': 0.001,    # Learning rate for optimizer
    'BATCH_SIZE': 256,          # Batch size for training
    'MEMORY_SIZE': 10000,      # Replay memory size
    'GAMMA': 0.99,             # Discount factor
    
    # Exploration parameters
    'EPSILON_START': 1.0,      # Starting epsilon (exploration rate)
    'EPSILON_END': 0.1,        # Minimum epsilon
    'EPSILON_DECAY': 50000,     # Decay rate for epsilon
    
    # Training loop parameters
    'EPISODES': 1000,          # Number of episodes to train
    'TARGET_UPDATE': 10,       # How often to update target network
    'PRINT_FREQ': 20,          # How often to print training progress
    'SAVE_FREQ': 1000,          # How often to save the model
    
    # Testing parameters
    'NUM_TEST_EPISODES': 10,   # Number of episodes to test
    'RENDER_TEST': True,       # Whether to render during testing
}