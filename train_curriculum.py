import os
import torch
import pygame
from train_agent import train
from config import CURRICULUM, DQN_CONFIG, RENDER_FPS, ADJUSTED_CELL_SIZE, MAZE_WIDTH, MAZE_HEIGHT
from maze_env import MazeWithVasesEnv  # Changed import
from environment_wrapper import MazeEnvironmentWrapper  # Added import

def show_maze_state(env_wrapper, phase_name, phase_idx):
    """Show maze state using PyGame visualization"""
    # PyGame setup
    pygame.init()
    pygame.display.set_caption(f"Initial Maze State - Phase {phase_idx + 1}: {phase_name}")
    
    # Calculate window size
    window_width = MAZE_WIDTH * ADJUSTED_CELL_SIZE
    window_height = MAZE_HEIGHT * ADJUSTED_CELL_SIZE
    screen = pygame.display.set_mode((window_width, window_height))
    clock = pygame.time.Clock()
    
    # Display until user presses Enter
    running = True
    font = pygame.font.Font(None, 24)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    running = False
        
        # Get and display the frame
        img = env_wrapper.env.get_frame('rgb_array')
        if img is not None:
            pygame_img = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            screen.blit(pygame_img, (0, 0))
            
            # Draw phase info
            status_text = f"Phase {phase_idx + 1}: {phase_name} - Press ENTER to begin training"
            text_surface = font.render(status_text, True, (255, 255, 255))
            screen.blit(text_surface, (10, 10))
            
            pygame.display.flip()
        
        clock.tick(RENDER_FPS)
    
    pygame.quit()

def train_curriculum():
    if not CURRICULUM['ENABLED']:
        return train()
    
    # Create environment with wrapper
    env = MazeWithVasesEnv()
    env_wrapper = MazeEnvironmentWrapper(env)  # Add wrapper
    
    agent = None
    
    # Store initial maze configuration from phase 1
    initial_maze = None
    initial_positions = None
    
    for phase_idx, phase in enumerate(CURRICULUM['PHASES']):
        print(f"\nStarting curriculum phase {phase_idx + 1}: {phase['NAME']}")
        
        # Update config for this phase
        original_config = DQN_CONFIG.copy()
        DQN_CONFIG.update({
            'NUM_VASES': phase['NUM_VASES'],
            'EPISODES': phase['EPISODES'],
            'END_ON_VASE_BREAK': phase['END_ON_VASE_BREAK'],
            'LEARNING_RATE': phase['LEARNING_RATE'],
            'USE_FIXED_MAZES': phase['USE_FIXED_MAZES'],
            'NUM_FIXED_MAZES': phase['NUM_FIXED_MAZES'],
            'FIXED_OBJECT_POSITIONS': phase['FIXED_OBJECT_POSITIONS'],
            'FIXED_AGENT_START': phase['FIXED_AGENT_START']
        })
        
        # Reset environment and ensure proper vase handling
        env_wrapper.reset()
        
        if phase['NAME'] == 'navigation':
            print("Navigation phase: removing all vases...")
            # Double check that no vases are present
            DQN_CONFIG['NUM_VASES'] = 0
            env_wrapper.reset()
        
        # Show initial maze state
        show_maze_state(env_wrapper, phase['NAME'], phase_idx)
        
        # If first phase, store the maze configuration
        if phase_idx == 0:
            initial_maze = env_wrapper.env.grid.copy()
            if DQN_CONFIG['FIXED_OBJECT_POSITIONS']:
                initial_positions = {
                    'agent_pos': tuple(env_wrapper.env.agent_pos),  # Store as new tuple
                    'agent_dir': env_wrapper.env.agent_dir,
                    'coin_pos': tuple(env_wrapper.env.coin_pos)     # Store as new tuple
                }
        else:
            # Use same maze configuration from phase 1
            env_wrapper.env.grid = initial_maze.copy()
            if DQN_CONFIG['FIXED_OBJECT_POSITIONS'] and initial_positions:
                env_wrapper.env.agent_pos = initial_positions['agent_pos']  # Use tuple directly
                env_wrapper.env.agent_dir = initial_positions['agent_dir']
                env_wrapper.env.coin_pos = initial_positions['coin_pos']    # Use tuple directly
        
        # Training arguments
        class Args:
            model_path = f"models/{phase['OUTPUT_NAME']}"
            load_model = agent is not None
            no_graphics = False
            episodes = phase['EPISODES']
            seed = None
        
        # Train for this phase with environment wrapper
        agent, scores = train(Args(), env_wrapper=env_wrapper)  # Pass env_wrapper
        
        # Save intermediate model
        agent.save(f"models/{phase['OUTPUT_NAME']}/model.pth")
        
        # Restore original config
        DQN_CONFIG.update(original_config)
        
        print(f"Completed phase {phase_idx + 1}: {phase['NAME']}")
    
    # Save final model
    agent.save('models/dqn_final_curriculum/model.pth')
    print("Curriculum learning complete!")
    return agent

if __name__ == "__main__":
    train_curriculum()