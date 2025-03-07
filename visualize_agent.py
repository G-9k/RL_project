# create a file called visualize_agent.py
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from IPython.display import HTML
import cv2

# Import your environment and agent
from stop_button_maze import StopButtonMazeEnv
from dqn_training import DQNAgent, preprocess_observation
from config_manager import ConfigManager

# Add at the beginning:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_visualize_agent(model_path, num_episodes=3, save_video=True):
    """
    Load a trained agent and visualize its behavior.
    
    Args:
        model_path: Path to the saved model file (.pth)
        num_episodes: Number of episodes to run
        save_video: Whether to save a video file
    """
    config = ConfigManager()
    env_config = config.get_env_config()
    agent_config = config.get_agent_config()
    
    # Create environment with config
    env = StopButtonMazeEnv(**env_config)
    
    # Get state and action dimensions
    obs, _ = env.reset()
    state = preprocess_observation(obs)
    state_size = len(state)
    action_size = env.action_space.n
    
    # Create agent with config
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        **agent_config
    )
    
    # Load the trained model
    agent.q_network.load_state_dict(torch.load(model_path, map_location=device))
    agent.q_network.to(device)
    agent.q_network.eval()
    
    # Run episodes and collect frames
    frames = []
    
    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")
        obs, _ = env.reset(seed=episode)  # Different seed for each episode
        state = preprocess_observation(obs)
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < env_config["max_steps"]:
            # Select action (no exploration)
            with torch.no_grad():
                action = agent.act(state, train=False)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_observation(obs)
            done = terminated or truncated
            
            # Render frame
            frame = env.render()
            if frame is not None:
                # Convert frame to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame)
            
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
    
    # Save video if requested
    if save_video and frames:
        # Get video path from config
        os.makedirs("videos", exist_ok=True)
        video_path = f"videos/{os.path.basename(model_path).replace('.pth', '.mp4')}"
        
        # Get first frame dimensions
        height, width = frames[0].shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
        
        # Write frames
        for frame in frames:
            out.write(frame)
        
        # Release video writer
        out.release()
        print(f"Video saved to {video_path}")
    
    return frames

if __name__ == "__main__":
    config = ConfigManager()
    
    # Test both conditions
    for experiment_name in config.get_config()['experiments'].keys():
        model_path = f"results/{experiment_name}_model.pth"
        if os.path.exists(model_path):
            print(f"\nVisualizing agent for {experiment_name}")
            frames = load_and_visualize_agent(
                model_path=model_path,
                num_episodes=3,
                save_video=True
            )