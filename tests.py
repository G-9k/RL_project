import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env
from stop_button_maze import StopButtonMazeEnv
from config_manager import ConfigManager
import matplotlib.pyplot as plt

def run_tests():
    # Get test configuration
    config = ConfigManager()
    test_config = config.get_config()['testing']
    
    # Create the environment with test configuration
    env = StopButtonMazeEnv(
        size=test_config['size'],
        num_vases=test_config['num_vases'],
        max_steps=test_config['max_steps']
    )
    
    # Reset the environment with seed
    obs, info = env.reset(seed=test_config['seed'])
    
    # Test with random actions
    done = False
    total_reward = 0
    frames = []
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        if test_config['render_each_step']:
            img = env.render()
            if test_config['save_frames']:
                frames.append(img)
        
        print(f"Step reward: {reward}")
        print(f"Human active: {obs['human_active']}")
        print(f"Vases broken: {obs['vases_broken']}")
    
    print(f"Episode finished with reward {total_reward}")
    
    # Show animation if frames were saved
    if test_config['save_frames'] and frames:
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
    
    env.close()

if __name__ == "__main__":
    run_tests()