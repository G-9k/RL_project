# main.py
from maze_env import MazeWithVasesEnv
import pygame
import numpy as np
import time
import sys
from config import *

def main():
    # Create the environment
    env = MazeWithVasesEnv()
    
    # Reset the environment
    obs, info = env.reset()
    
    # Print environment information
    print(f"Maze dimensions: {env.width}x{env.height}")
    print(f"Path width: {PATH_WIDTH} cells")
    print(f"Number of vases: {len(env.vases)}")
    print(f"Coin position: {env.coin_pos}")
    print(f"Agent position: {env.agent_pos}")
    
    # Get a frame to visualize the maze
    img = env.get_frame(mode='rgb_array')
    
    # Display the image using pygame for a few seconds
    pygame.init()
    screen = pygame.display.set_mode((env.width * ADJUSTED_CELL_SIZE, env.height * ADJUSTED_CELL_SIZE))
    pygame.display.set_caption("MiniGrid Maze Environment")
    
    if img is not None:
        surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        
        print("\nEnvironment initialized successfully!")
        print("Displaying the maze for 5 seconds...")
        print("Run 'python visualize.py' to interactively explore the maze.")
        print("Run 'python custom_maze_test.py' to test custom maze layouts.")
        
        # Wait for 5 seconds or until the window is closed
        start_time = time.time()
        while time.time() - start_time < 5:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            time.sleep(0.1)
    
    pygame.quit()

if __name__ == "__main__":
    main()