# visualize.py
import gymnasium as gym
import pygame
import sys
import time
import numpy as np
from maze_env import MazeWithVasesEnv
from config import RENDER_FPS, ADJUSTED_CELL_SIZE, MAZE_WIDTH, MAZE_HEIGHT
from minigrid.core.constants import COLORS

def manual_control(env):
    """
    Control the environment manually using keyboard inputs.
    This helps visualize the maze and its elements.
    
    Controls:
    - Arrow keys: Move the agent directly in the corresponding direction
    - Q: Quit
    - R: Reset the environment
    """
    obs, info = env.reset()
    done = False
    
    # PyGame setup
    pygame.init()
    pygame.display.set_caption("MiniGrid Maze Environment")
    
    # Calculate window size
    window_width = MAZE_WIDTH * ADJUSTED_CELL_SIZE
    window_height = MAZE_HEIGHT * ADJUSTED_CELL_SIZE
    screen = pygame.display.set_mode((window_width, window_height))
    clock = pygame.time.Clock()
    
    # Create a render window
    render_mode = 'rgb_array'
    
    # Track statistics
    steps = 0
    vases_broken = 0
    
    # Display status text
    font = pygame.font.Font(None, 24)
    
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
                    steps = 0
                    vases_broken = 0
                
                if not done:
                    old_vases_broken = vases_broken
                    
                    # Handle movement in a more intuitive way
                    if event.key == pygame.K_LEFT:
                        # First rotate to face left
                        while env.agent_dir != 2:  # 2 is left direction
                            env.step(env.actions.right)
                        # Then move forward
                        obs, reward, terminated, truncated, info = env.step(env.actions.forward)
                        steps += 1
                    elif event.key == pygame.K_RIGHT:
                        # First rotate to face right
                        while env.agent_dir != 0:  # 0 is right direction
                            env.step(env.actions.right)
                        # Then move forward
                        obs, reward, terminated, truncated, info = env.step(env.actions.forward)
                        steps += 1
                    elif event.key == pygame.K_UP:
                        # First rotate to face up
                        while env.agent_dir != 3:  # 3 is up direction
                            env.step(env.actions.right)
                        # Then move forward
                        obs, reward, terminated, truncated, info = env.step(env.actions.forward)
                        steps += 1
                    elif event.key == pygame.K_DOWN:
                        # First rotate to face down
                        while env.agent_dir != 1:  # 1 is down direction
                            env.step(env.actions.right)
                        # Then move forward
                        obs, reward, terminated, truncated, info = env.step(env.actions.forward)
                        steps += 1
                    
                    # Check if a vase was broken
                    if info.get('vase_broken', False):
                        vases_broken = info.get('num_broken_vases', vases_broken + 1)
                    
                    done = terminated or truncated
                    
                    # Print status updates
                    if done:
                        if info.get('coin_collected', False):
                            print(f"Success! Coin collected in {steps} steps. Vases broken: {vases_broken}")
                        else:
                            print(f"Episode ended after {steps} steps. Vases broken: {vases_broken}")
        
        # Get the RGB array from the environment
        img = env.get_frame(render_mode)
        
        if img is not None:
            # Convert the image to a PyGame surface
            pygame_img = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            
            # Draw the image on the screen
            screen.blit(pygame_img, (0, 0))
            
            # Draw status text
            status_text = f"Steps: {steps} | Vases Broken: {vases_broken} | Done: {done}"
            text_surface = font.render(status_text, True, (255, 255, 255))
            screen.blit(text_surface, (10, 10))
            
            pygame.display.flip()
            
        # Cap the framerate
        clock.tick(RENDER_FPS)

if __name__ == "__main__":
    # Create the environment
    env = MazeWithVasesEnv()
    
    print("Manual Control - MiniGrid Maze Environment")
    print("Controls:")
    print("  Arrow keys: Move the agent in the corresponding direction")
    print("  Q: Quit")
    print("  R: Reset the environment")
    
    # Start the manual control
    manual_control(env)