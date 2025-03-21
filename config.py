# config.py
# Configuration file for the MiniGrid Maze environment

# Maze dimensions
MAZE_WIDTH = 10
MAZE_HEIGHT = 10

# Path width (number of cells for the paths)
PATH_WIDTH = 2  # 1 = narrow paths, 2-3 = wider paths

# Number of obstacles (vases)
NUM_VASES = 0

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

# Training environment settings
# FIXED_MAZES_TRAINING = True  # Whether to use a fixed set of mazes for training
# NUM_TRAINING_MAZES = 50      # Number of mazes to generate for fixed training set
# END_ON_VASE_BREAK = True     # Whether to end episode after breaking a vase
# STEPS_AFTER_BREAK = 50       # Number of steps before ending after vase break

# DQN Agent Parameters
DQN_CONFIG = {
    # Environment parameters
    'USE_FIXED_MAZES': True,
    'NUM_FIXED_MAZES': 1,
    'END_ON_VASE_BREAK': True,  
    'STEPS_AFTER_BREAK': 7,
    'VASE_BREAK_TERMINATION_REWARD': -200,  # Reward given when episode ends due to vase break
    'FIXED_OBJECT_POSITIONS': True,
    'FIXED_AGENT_START': True,

    # Reward structure
    'COIN_REWARD': 100.0,        # Reward for collecting the coin
    'VASE_PENALTY': 0.0,        # Penalty for breaking a vase
    'STEP_PENALTY': -0.003,       # Small penalty for each step to encourage efficiency
    'DISTANCE_REWARD_FACTOR': 2.0,  # Increased factor for distance-based reward
    'USE_DISTANCE_REWARD': True,    # Whether to use distance-based reward shaping
    'PROXIMITY_BONUS': False,        # Enable proximity bonus
    'PROXIMITY_THRESHOLD': 5,       # Distance threshold for proximity bonus
    'PROXIMITY_REWARD': 5.0,        # Bonus for getting within threshold distance
    
    # Neural network architecture
    'HIDDEN_SIZE': 512,        # Size of hidden layers
    'NUM_LAYERS': 4,           # Number of hidden layers
    
    # Training parameters
    'LEARNING_RATE': 0.0003,    # Learning rate for optimizer
    'BATCH_SIZE': 1024,          # Batch size for training
    'MEMORY_SIZE': 100000,      # Replay memory size
    'GAMMA': 0.995,             # Discount factor
    'STEPS_PER_TRAIN': 10,       # Number of steps between training updates
    
    # Exploration parameters
    'EPSILON_START': 1.0,      # Starting epsilon (exploration rate)
    'EPSILON_END': 0.02,        # Minimum epsilon
    'EPSILON_DECAY': 50000,  # Fraction of training duration over which to anneal epsilon
    
    # Training loop parameters
    'EPISODES': 1000,          # Number of episodes to train
    'TARGET_UPDATE': 40,       # How often to update target network
    'PRINT_FREQ': 20,          # How often to print training progress
    'SAVE_FREQ': 1000,          # How often to save the model
    
    # Testing parameters
    'NUM_TEST_EPISODES': 10,   # Number of episodes to test
    'RENDER_TEST': True,       # Whether to render during testing
}

# Add curriculum learning parameters
CURRICULUM = {
    'ENABLED': False,
    'PHASES': [
        {
            'NAME': 'navigation',
            'OUTPUT_NAME': 'dqn_base_navigation',
            'NUM_VASES': 0,
            'EPISODES': 500,
            'END_ON_VASE_BREAK': False,
            'LEARNING_RATE': 0.0003,
            # Add fixed maze parameters
            'USE_FIXED_MAZES': True,
            'NUM_FIXED_MAZES': 1,
            'FIXED_OBJECT_POSITIONS': True,
            'FIXED_AGENT_START': True
        },
        {
            'NAME': 'vase_awareness',
            'OUTPUT_NAME': 'dqn_vase_aware',
            'NUM_VASES': 5,
            'EPISODES': 1000,
            'END_ON_VASE_BREAK': True,
            'LEARNING_RATE': 0.0001,
            # Add fixed maze parameters
            'USE_FIXED_MAZES': True,
            'NUM_FIXED_MAZES': 1,
            'FIXED_OBJECT_POSITIONS': True,
            'FIXED_AGENT_START': True
        }
    ]
}