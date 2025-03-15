# test_agent.py
import os
import torch
import numpy as np
import time
import pygame
import argparse
import sys

from maze_env import MazeWithVasesEnv
from environment_wrapper import MazeEnvironmentWrapper
from dqn_agent import DQNAgent
from tests_layouts import CUSTOM_MAZES
from config import DQN_CONFIG, RENDER_FPS

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test trained DQN agent')
    parser.add_argument('--model_path', type=str, default='models', help='Path to load model')
    parser.add_argument('--layout', type=str, default=None, 
                        help=f'Custom layout to use. Options: {", ".join(CUSTOM_MAZES.keys())}')
    parser.add_argument('--episodes', type=int, default=DQN_CONFIG['NUM_TEST_EPISODES'], 
                        help='Number of episodes to test')
    parser.add_argument('--delay', type=float, default=0.1, 
                        help='Delay between actions (seconds)')
    parser.add_argument('--seed', type=int, default=None, 
                        help='Random seed')
    parser.add_argument('--list', action='store_true', 
                        help='List available layouts and exit')
    return parser.parse_args()

def test_agent():
    args = parse_arguments()
    
    # List layouts if requested
    if args.list:
        print("Available maze layouts:")
        for name, layout in CUSTOM_MAZES.items():
            height = len(layout)
            width = len(layout[0]) if height > 0 else 0
            print(f"  - {name}: {width}x{height} maze")
        sys.exit(0)
    
    # Check if model exists
    if not os.path.exists(os.path.join(args.model_path, "dqn_agent.pth")):
        print(f"Error: No model found at {args.model_path}")
        sys.exit(1)
    
    # Create environment
    if args.layout is not None:
        if args.layout not in CUSTOM_MAZES:
            print(f"Error: Layout '{args.layout}' not found!")
            print(f"Available layouts: {', '.join(CUSTOM_MAZES.keys())}")
            sys.exit(1)
        
        custom_layout = CUSTOM_MAZES[args.layout]
        env = MazeWithVasesEnv(custom_layout=custom_layout)
        print(f"Using custom layout: {args.layout}")
    else:
        env = MazeWithVasesEnv()
        print("Using randomly generated maze")
    
    env_wrapper = MazeEnvironmentWrapper(env)
    
    # Set seed if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        env.reset(seed=args.seed)
    
    # Get state size
    state, _ = env_wrapper.reset()
    state_size = len(state)
    action_size = env.action_space.n
    
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agent
    agent = DQNAgent(state_size, action_size, device=device)
    
    # Load the model
    agent.load(args.model_path)
    print(f"Loaded model from {args.model_path}")
    
    # Initialize pygame
    pygame.init()
    width = env.width * env.get_adjusted_cell_size()
    height = env.height * env.get_adjusted_cell_size()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("DQN Agent Test")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    # Track statistics
    episode_rewards = []
    episode_steps = []
    episode_vases = []
    success_count = 0
    
    print("\nStarting agent testing...")
    print(f"Episodes: {args.episodes}")
    print(f"Action delay: {args.delay}s")
    print("Press 'Q' to quit, 'S' to skip to next episode, 'Space' to pause/resume")
    
    # Testing loop
    for episode in range(1, args.episodes + 1):
        state, _ = env_wrapper.reset()
        total_reward = 0
        steps = 0
        vases_broken = 0
        done = False
        paused = False
        
        print(f"\nEpisode {episode}/{args.episodes}")
        
        # Run one episode
        while not done:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_s:
                        done = True
                        break
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
            
            if paused:
                # Display paused text
                img = env.get_frame(mode='rgb_array')
                pygame_img = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                screen.blit(pygame_img, (0, 0))
                text = font.render("PAUSED (Space to resume)", True, (255, 255, 255))
                screen.blit(text, (10, 10))
                pygame.display.flip()
                clock.tick(10)
                continue
            
            # Select action
            action = agent.select_action(state, training=False)
            
            # Take action
            next_state, reward, terminated, truncated, info = env_wrapper.step(action)
            total_reward += reward
            steps += 1
            
            # Check if vase broken
            if info.get('vase_broken', False):
                vases_broken += 1
            
            # Check if coin collected (success)
            if info.get('coin_collected', False):
                success_count += 1
                print(f"Success! Coin collected in {steps} steps. Vases broken: {vases_broken}")
            
            # Update state and check if done
            state = next_state
            done = terminated or truncated
            
            # Render
            img = env.get_frame(mode='rgb_array')
            pygame_img = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            screen.blit(pygame_img, (0, 0))
            
            # Add status text
            status = (
                f"Episode: {episode}/{args.episodes} | "
                f"Step: {steps} | "
                f"Reward: {total_reward:.2f} | "
                f"Vases: {vases_broken}"
            )
            text = font.render(status, True, (255, 255, 255))
            screen.blit(text, (10, 10))
            
            pygame.display.flip()
            clock.tick(RENDER_FPS)
            
            # Delay between actions
            if args.delay > 0:
                time.sleep(args.delay)
        
        # Track statistics
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_vases.append(vases_broken)
        
        if not info.get('coin_collected', False):
            print(f"Failed! Episode ended after {steps} steps. Vases broken: {vases_broken}")
    
    # Print summary
    success_rate = success_count / args.episodes * 100
    print("\nTesting Results:")
    print(f"Success Rate: {success_rate:.2f}% ({success_count}/{args.episodes})")
    print(f"Average Steps: {np.mean(episode_steps):.2f}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Average Vases Broken: {np.mean(episode_vases):.2f}")
    
    pygame.quit()

if __name__ == "__main__":
    test_agent()