# custom_maze_test.py
import argparse
import pygame
import sys
from maze_env import MazeWithVasesEnv
from tests_layouts import CUSTOM_MAZES
from config import PATH_WIDTH, MAZE_WIDTH, MAZE_HEIGHT
from visualize import manual_control

def load_maze_layout(layout_name):
    """Load a maze layout from the predefined layouts"""
    if layout_name not in CUSTOM_MAZES:
        print(f"Error: Layout '{layout_name}' not found!")
        print(f"Available layouts: {', '.join(CUSTOM_MAZES.keys())}")
        sys.exit(1)
    
    return CUSTOM_MAZES[layout_name]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test a custom maze layout')
    parser.add_argument('--layout', type=str, default='simple',
                        help=f'Maze layout to use. Options: {", ".join(CUSTOM_MAZES.keys())}')
    parser.add_argument('--list', action='store_true', 
                        help='List available maze layouts and exit')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available maze layouts:")
        for name, layout in CUSTOM_MAZES.items():
            height = len(layout)
            width = len(layout[0]) if height > 0 else 0
            print(f"  - {name}: {width}x{height} maze")
        sys.exit(0)
    
    return args

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Load the specified maze layout
    layout = load_maze_layout(args.layout)
    
    # Create the environment with the custom layout
    env = MazeWithVasesEnv(custom_layout=layout)
    
    # Reset the environment
    obs, info = env.reset()
    
    # Print environment information
    print(f"Using maze layout: {args.layout}")
    print(f"Maze dimensions: {env.width}x{env.height}")
    print(f"Path width setting: {PATH_WIDTH} cells")
    print(f"Number of vases: {len(env.vases)}")
    print(f"Agent position: {env.agent_pos}")
    print(f"Coin position: {env.coin_pos}")
    
    # Start the manual control
    print("\nStarting manual control...")
    print("Controls:")
    print("  Arrow keys: Move the agent")
    print("  Q: Quit")
    print("  R: Reset the environment")
    
    manual_control(env)

if __name__ == "__main__":
    main()