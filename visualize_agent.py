# create a file called visualize_agent.py
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from IPython.display import HTML

# Import your environment and agent
from stop_button_maze import StopButtonMazeEnv
from dqn_training import DQNAgent, preprocess_observation

def load_and_visualize_agent(model_path, env_params=None, num_episodes=3, save_video=True):
    """
    Load a trained agent and visualize its behavior.
    
    Args:
        model_path: Path to the saved model file (.pth)
        env_params: Dictionary of environment parameters
        num_episodes: Number of episodes to run
        save_video: Whether to save a video file
    """
    # Default environment parameters
    if env_params is None:
        env_params = {
            "size": 8,
            "num_vases": 3,
            "max_steps": 100,
            "reward_for_coin": 1.0,
            "penalty_for_caught": 0.0
        }
    
    # Create environment
    env = StopButtonMazeEnv(**env_params)
    
    # Get state and action dimensions
    obs, _ = env.reset()
    state = preprocess_observation(obs)
    state_size = len(state)
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size
    )
    
    # Load the trained model
    agent.q_network.load_state_dict(torch.load(model_path))
    
    # Run episodes and collect frames
    all_frames = []
    
    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")
        obs, _ = env.reset(seed=episode)  # Different seed for each episode
        state = preprocess_observation(obs)
        
        episode_frames = []
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < env_params["max_steps"]:
            # Select action (no exploration)
            action = agent.act(state, train=False)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_observation(obs)
            done = terminated or truncated
            
            # Render frame
            frame = env.render()
            episode_frames.append(frame)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            step += 1
            
            # Print step information
            print(f"  Step {step}: Reward={reward:.1f}, Total={episode_reward:.1f}")
            if done:
                if "caught_by_human" in info and info["caught_by_human"]:
                    print("  Agent was caught by human!")
                if "reached_goal" in info and info["reached_goal"]:
                    print("  Agent reached the goal!")
        
        all_frames.extend(episode_frames)
    
    # Create directory for videos if it doesn't exist
    os.makedirs("videos", exist_ok=True)
    
    # Display animation
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame_idx):
        ax.clear()
        ax.imshow(all_frames[frame_idx])
        ax.set_title(f"Frame {frame_idx}")
        ax.axis('off')
        return [ax]
    
    ani = FuncAnimation(fig, update, frames=len(all_frames), 
                        interval=200, blit=True)
    
    # Save video if requested
    if save_video:
        model_name = os.path.basename(model_path).replace('.pth', '')
        ani.save(f"videos/{model_name}_visualization.mp4", fps=5)
        print(f"Video saved to videos/{model_name}_visualization.mp4")
    
    plt.close()
    
    return all_frames

if __name__ == "__main__":
    # Example usage:
    
    # You can choose which model to visualize:
    
    # 1. Baseline condition (no reward for getting caught)
    frames = load_and_visualize_agent(
        model_path="results/baseline_model.pth", 
        env_params={
            "size": 8,
            "num_vases": 3,
            "max_steps": 100,
            "reward_for_coin": 1.0,
            "penalty_for_caught": 0.0
        }
    )
    
    # 2. Reward for getting caught condition
    frames = load_and_visualize_agent(
        model_path="results/reward_for_getting_caught_model.pth", 
        env_params={
            "size": 8,
            "num_vases": 3,
            "max_steps": 100,
            "reward_for_coin": 1.0,
            "penalty_for_caught": 0.5  # Positive value = reward for getting caught
        }
    )