# train_agent.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import argparse

from maze_env import MazeWithVasesEnv, Vase, Wall
from environment_wrapper import MazeEnvironmentWrapper
from dqn_agent import DQNAgent
from config import *

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train DQN agent on maze environment')
    parser.add_argument('--model_path', type=str, default='models', help='Path to save/load model')
    parser.add_argument('--load_model', action='store_true', help='Load existing model')
    parser.add_argument('--no_graphics', action='store_true', help='Disable rendering during training')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes to train')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    return parser.parse_args()

def train(args=None, env_wrapper=None):
    """Train the agent"""
    if args is None:
        args = parse_arguments()
    
    # Get environment from wrapper or create new one
    if env_wrapper is None:
        env = MazeWithVasesEnv()
        env_wrapper = MazeEnvironmentWrapper(env)
    else:
        env = env_wrapper.env  # Extract environment from wrapper
    
    # Handle fixed mazes
    if DQN_CONFIG['USE_FIXED_MAZES']:
        fixed_mazes = []
        fixed_positions = []
        
        # Generate fixed mazes
        for _ in range(DQN_CONFIG['NUM_FIXED_MAZES']):
            env.reset()
            fixed_mazes.append(env.grid.copy())
            if DQN_CONFIG['FIXED_OBJECT_POSITIONS']:
                fixed_positions.append({
                    'agent_pos': tuple(env.agent_pos),  # Store as new tuple
                    'agent_dir': env.agent_dir,
                    'coin_pos': tuple(env.coin_pos)     # Store as new tuple
                })
        
        # Set initial maze
        env.grid = fixed_mazes[0].copy()
        if DQN_CONFIG['FIXED_OBJECT_POSITIONS']:
            env.agent_pos = fixed_positions[0]['agent_pos']  # Use tuple directly
            env.agent_dir = fixed_positions[0]['agent_dir']
            env.coin_pos = fixed_positions[0]['coin_pos']    # Use tuple directly

    # Override episodes if specified
    if args.episodes is not None:
        DQN_CONFIG['EPISODES'] = args.episodes
    
    # Set seed if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        env.reset(seed=args.seed)
    
    # Calculate state size from observation
    obs, _ = env_wrapper.reset()
    state_size = len(obs)
    action_size = env.action_space.n
    
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agent
    agent = DQNAgent(state_size, action_size, device=device)
    
    # Load existing model if specified
    if args.load_model and os.path.exists(os.path.join(args.model_path, "dqn_agent.pth")):
        agent.load(args.model_path)
        print(f"Loaded model from {args.model_path}")
    
    # Training statistics
    scores = []
    scores_window = deque(maxlen=100)  # last 100 scores
    vases_broken_list = []
    steps_taken_list = []
    epsilon_values = []
    coins_window = 0  # Track coins over printing interval
    vases_window = 0  # Add counter for vases in window
    last_print_episode = 0  # Track when we last printed
    steps_window = []  # Track steps for averaging
    
    # Training loop
    for episode in range(1, DQN_CONFIG['EPISODES'] + 1):
        # If using fixed mazes, select one randomly
        if DQN_CONFIG['USE_FIXED_MAZES']:
            maze_idx = np.random.randint(0, DQN_CONFIG['NUM_FIXED_MAZES'])
            env.grid = fixed_mazes[maze_idx].copy()
            
            if DQN_CONFIG['NUM_FIXED_MAZES'] == 1:
                if DQN_CONFIG['FIXED_OBJECT_POSITIONS']:
                    # Use stored positions for vases and coin
                    env.vases = []
                    env.coin_pos = fixed_positions[maze_idx]['coin_pos']
                    env.agent_pos = fixed_positions[maze_idx]['agent_pos']
                    env.agent_dir = fixed_positions[maze_idx]['agent_dir']
        
        state, _ = env_wrapper.reset()
        score = 0
        steps = 0
        vases_broken = 0
        episode_coins = 0  # Coins for this episode
        start_time = time.time()
        
        while True:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, info = env_wrapper.step(action)
            done = terminated or truncated
            
            # Store experience in replay memory
            agent.memory.push(state, action, next_state, reward, done)
            
            # Learn from experience
            loss = agent.learn()
            
            # Track statistics
            state = next_state
            score += reward
            steps += 1
            
            if info.get('vase_broken', False):
                vases_broken += 1
                vases_window += 1  # Increment window counter
            if info.get('coin_collected', False):
                episode_coins += 1
                coins_window += 1
            
            if done:
                steps_window.append(steps)  # Add episode steps to window
                break
        
        # Update the target network
        if episode % DQN_CONFIG['TARGET_UPDATE'] == 0:
            agent.update_target_network()
        
        # Track statistics
        scores_window.append(score)
        scores.append(score)
        vases_broken_list.append(vases_broken)
        steps_taken_list.append(steps)
        epsilon_values.append(agent.epsilon)
        
        # Print progress
        if episode % DQN_CONFIG['PRINT_FREQ'] == 0:
            avg_steps = np.mean(steps_window)  # Calculate average steps
            elapsed = time.time() - start_time
            print(f"Episode {episode}/{DQN_CONFIG['EPISODES']} | "
                  f"Score: {score:.2f} | "
                  f"Avg Score: {np.mean(scores_window):.2f} | "
                  f"Epsilon: {agent.epsilon:.2f} | "
                  f"Avg Steps (last {DQN_CONFIG['PRINT_FREQ']} eps): {avg_steps:.1f} | "
                  f"Coins (last {DQN_CONFIG['PRINT_FREQ']} eps): {coins_window} | "
                  f"Vases (last {DQN_CONFIG['PRINT_FREQ']} eps): {vases_window} | "  # Add vases window
                  f"Time: {elapsed:.2f}s | "
                  f"Loss: {loss:.4f}")
            coins_window = 0  # Reset the window counter after printing
            vases_window = 0  # Reset vases window counter
            steps_window = []  # Reset steps window after printing
            last_print_episode = episode
        
        # Save the model periodically
        if episode % DQN_CONFIG['SAVE_FREQ'] == 0:
            agent.save(args.model_path)
            
            # Plot training progress
            plt.figure(figsize=(15, 10))
            
            # Plot scores
            plt.subplot(2, 2, 1)
            plt.plot(scores, 'b-', alpha=0.6, label='Score')
            plt.plot(np.convolve(scores, np.ones(100)/100, mode='valid'), 
                     'r-', label='100-episode running average')
            plt.axhline(y=np.mean(scores), color='g', linestyle='--', 
                       label=f'Total average: {np.mean(scores):.2f}')
            plt.title('Score')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.legend()
            
            # Plot steps per episode
            plt.subplot(2, 2, 2)
            plt.plot(steps_taken_list, 'b-', alpha=0.6, label='Steps')
            plt.plot(np.convolve(steps_taken_list, np.ones(100)/100, mode='valid'),
                     'r-', label='100-episode running average')
            plt.axhline(y=np.mean(steps_taken_list), color='g', linestyle='--',
                       label=f'Total average: {np.mean(steps_taken_list):.1f}')
            plt.title('Steps per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.legend()
            
            # Plot coins collected per PRINT_FREQ episodes
            plt.subplot(2, 2, 3)
            coins_per_window = []
            for i in range(0, len(scores), DQN_CONFIG['PRINT_FREQ']):
                window_end = min(i + DQN_CONFIG['PRINT_FREQ'], len(scores))
                coins_in_window = sum(1 for j in range(i, window_end) 
                                    if scores[j] >= DQN_CONFIG['COIN_REWARD'])
                coins_per_window.append(coins_in_window)
            
            window_indices = np.arange(len(coins_per_window)) * DQN_CONFIG['PRINT_FREQ']
            plt.plot(window_indices, coins_per_window, 'b-', alpha=0.6, label='Coins')
            plt.plot(window_indices, 
                     np.convolve(coins_per_window, np.ones(10)/10, mode='same'),
                     'r-', label='10-window running average')
            plt.axhline(y=np.mean(coins_per_window), color='g', linestyle='--',
                       label=f'Total average: {np.mean(coins_per_window):.2f}')
            plt.title(f'Coins Collected per {DQN_CONFIG["PRINT_FREQ"]} Episodes')
            plt.xlabel('Episode')
            plt.ylabel('Coins Collected')
            plt.legend()
            
            # Plot vases broken
            plt.subplot(2, 2, 4)
            plt.plot(vases_broken_list, 'b-', alpha=0.6, label='Vases broken')
            plt.plot(np.convolve(vases_broken_list, np.ones(100)/100, mode='valid'),
                     'r-', label='100-episode running average')
            plt.axhline(y=np.mean(vases_broken_list), color='g', linestyle='--',
                       label=f'Total average: {np.mean(vases_broken_list):.2f}')
            plt.title('Vases Broken per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Vases Broken')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.model_path, 'training_progress.png'))
            plt.close()
    
    # Save the final model
    agent.save(args.model_path)
    
    print("Training complete!")
    return agent, scores

if __name__ == "__main__":
    train()