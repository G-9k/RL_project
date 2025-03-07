import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt

# Assuming you saved the environment code in stop_button_maze.py
from stop_button_maze import StopButtonMazeEnv

# Create the environment
env = StopButtonMazeEnv(size=8, num_vases=3, max_steps=100)

# Check the environment (optional, for debugging)
# check_env(env)

# Reset the environment
obs, info = env.reset(seed=42)

# Test with random actions
done = False
total_reward = 0
frames = []

while not done:
    # Choose a random action
    action = env.action_space.sample()
    
    # Take the action
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    
    # Render the environment
    img = env.render()
    frames.append(img)
    
    # Print step information
    print(f"Step reward: {reward}")
    print(f"Human active: {obs['human_active']}")
    print(f"Vases broken: {obs['vases_broken']}")
    
    if done:
        print(f"Episode finished with reward {total_reward}")
        if "caught_by_human" in info and info["caught_by_human"]:
            print("Agent was caught by human!")
        if "reached_goal" in info and info["reached_goal"]:
            print("Agent reached the goal!")

# Close the environment
env.close()

# Show animation
plt.figure(figsize=(8, 8))
for i, frame in enumerate(frames):
    if i == 0:
        plt.imshow(frame)
        plt.axis('off')
        plt.pause(0.5)
    else:
        plt.clf()
        plt.imshow(frame)
        plt.axis('off')
        plt.pause(0.1)
plt.close()