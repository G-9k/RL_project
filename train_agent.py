# train_agent.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import argparse

from maze_env import MazeWithVasesEnv
from environment_wrapper import MazeEnvironmentWrapper
from dqn_agent import DQNAgent
from config import DQN_CONFIG

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train DQN agent on maze environment')
    parser.add_argument('--model_path', type=str, default='models', help='Path to save/load model')
    parser.add_argument('--load_model', action='store_true', help='Load existing model')
    parser.add_argument('--no_graphics', action='store_true', help='Disable rendering during training')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes to train')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    return parser.parse_args()

def train():
    args = parse_arguments()
    
    # Override episodes if specified
    if args.episodes is not None:
        DQN_CONFIG['EPISODES'] = args.episodes
    
    # Create environment
    env = MazeWithVasesEnv()
    env_wrapper = MazeEnvironmentWrapper(env)
    
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
    last_print_episode = 0  # Track when we last printed
    steps_window = []  # Track steps for averaging
    
    # Training loop
    for episode in range(1, DQN_CONFIG['EPISODES'] + 1):
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
                  f"Vases Broken: {vases_broken} | "
                  f"Time: {elapsed:.2f}s | "
                  f"Loss: {loss:.4f}")
            coins_window = 0  # Reset the window counter after printing
            steps_window = []  # Reset steps window after printing
            last_print_episode = episode
        
        # Save the model periodically
        if episode % DQN_CONFIG['SAVE_FREQ'] == 0:
            agent.save(args.model_path)
            
            # Plot training progress
            plt.figure(figsize=(15, 10))
            
            # Plot scores
            plt.subplot(2, 2, 1)
            plt.plot(scores)
            plt.plot(np.convolve(scores, np.ones(100)/100, mode='valid'))
            plt.title('Score')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            
            # Plot epsilon decay
            plt.subplot(2, 2, 2)
            plt.plot(epsilon_values)
            plt.title('Epsilon Decay')
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            
            # Plot steps per episode
            plt.subplot(2, 2, 3)
            plt.plot(steps_taken_list)
            plt.title('Steps per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            
            # Plot vases broken
            plt.subplot(2, 2, 4)
            plt.plot(vases_broken_list)
            plt.title('Vases Broken per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Vases Broken')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.model_path, 'training_progress.png'))
            plt.close()
    
    # Save the final model
    agent.save(args.model_path)
    
    print("Training complete!")
    return agent, scores

if __name__ == "__main__":
    train()