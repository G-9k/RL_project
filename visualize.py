# visualize.py
import gymnasium as gym
import pygame
import sys
import time
import numpy as np
from maze_env import MazeWithVasesEnv
from config import RENDER_FPS, CELL_SIZE, MAZE_WIDTH, MAZE_HEIGHT
from minigrid.core.constants import COLORS

def manual_control(env):
    """
    Control the environment manually using keyboard inputs.
    This helps visualize the maze and its elements.
    
    Controls:
    - Arrow keys: Move the agent
    - Q: Quit
    - R: Reset the environment
    """
    obs, info = env.reset()
    done = False
    
    # PyGame setup
    pygame.init()
    pygame.display.set_caption("MiniGrid Maze Environment")
    
    # Calculate window size
    window_width = MAZE_WIDTH * CELL_SIZE
    window_height = MAZE_HEIGHT * CELL_SIZE
    screen = pygame.display.set_mode((window_width, window_height))
    clock = pygame.time.Clock()
    
    # Create a render window
    render_mode = 'rgb_array'
    
    while True:
        # Check for PyGame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                
                if not done:
                    if event.key == pygame.K_LEFT:
                        action = env.actions.left
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                    
                    elif event.key == pygame.K_RIGHT:
                        action = env.actions.right
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                    
                    elif event.key == pygame.K_UP:
                        action = env.actions.forward
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        
                    elif event.key == pygame.K_DOWN:
                        # Turn around (right twice)
                        env.step(env.actions.right)
                        env.step(env.actions.right)
        
        # Get the RGB array from the environment
        img = env.get_frame(render_mode)
        
        if img is not None:
            # Convert the image to a PyGame surface
            pygame_img = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            
            # Scale the image to fit the window
            pygame_img = pygame.transform.scale(pygame_img, (window_width, window_height))
            
            # Draw the image on the screen
            screen.blit(pygame_img, (0, 0))
            pygame.display.flip()
            
        # Cap the framerate
        clock.tick(RENDER_FPS)

if __name__ == "__main__":
    # Create the environment
    env = MazeWithVasesEnv()
    
    print("Manual Control - MiniGrid Maze Environment")
    print("Controls:")
    print("  Arrow keys: Move the agent")
    print("  Q: Quit")
    print("  R: Reset the environment")
    
    # Start the manual control
    manual_control(env)