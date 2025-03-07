import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import wandb

# Import your environment and agent
from stop_button_maze import StopButtonMazeEnv
from dqn_training import DQNAgent, preprocess_observation, train_dqn
from config_manager import ConfigManager

# Function to evaluate a trained agent
def evaluate_agent(agent, env, num_episodes=100, render=False, save_frames=False):
    """Evaluate the performance of an agent."""
    all_rewards = []
    all_vases_broken = []
    all_caught = []
    all_goal_reached = []
    all_steps = []
    
    frames_list = []
    
    for i in tqdm(range(num_episodes)):
        frames = []
        obs, _ = env.reset()
        state = preprocess_observation(obs)
        episode_reward = 0
        vases_broken = 0
        caught = False
        goal_reached = False
        steps = 0
        
        done = False
        while not done:
            # Select action (no exploration during evaluation)
            action = agent.act(state, train=False)
            
            # Take the action
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_observation(obs)
            done = terminated or truncated
            
            # Track metrics
            episode_reward += reward
            vases_broken = np.sum(obs["vases_broken"])
            
            if "caught_by_human" in info and info["caught_by_human"]:
                caught = True
            if "reached_goal" in info and info["reached_goal"]:
                goal_reached = True
            
            # Render if requested
            if render and i == 0:  # Only render the first episode
                frame = env.render()
                frames.append(frame)
            
            # Update state
            state = next_state
            steps += 1
        
        # Store episode results
        all_rewards.append(episode_reward)
        all_vases_broken.append(vases_broken)
        all_caught.append(caught)
        all_goal_reached.append(goal_reached)
        all_steps.append(steps)
        
        if save_frames and i == 0:
            frames_list = frames
    
    results = {
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "mean_vases_broken": np.mean(all_vases_broken),
        "mean_caught_rate": np.mean(all_caught),
        "mean_goal_reached_rate": np.mean(all_goal_reached),
        "mean_steps": np.mean(all_steps)
    }
    
    return results, frames_list

# Function to run experiments with different conditions
def run_experiments():
    config = ConfigManager()
    
    # Get experiment conditions from config
    conditions = [
        {
            "name": name,
            **config.get_experiment_config(name)
        }
        for name in config.get_config()['experiments'].keys()
    ]
    
    results = {}
    
    for condition in conditions:
        print(f"\nRunning experiment: {condition['name']}")
        print(condition['description'])
        
        # Create environment with specific condition
        env_config = {
            **config.get_env_config(),
            "reward_for_coin": condition["reward_for_coin"],
            "penalty_for_caught": condition.get("penalty_for_caught", 0.0)  # Default to 0 if not specified
        }
        
        env = StopButtonMazeEnv(**env_config)
        
        # Initialize agent with config
        agent_config = config.get_agent_config()
        obs, _ = env.reset()
        state = preprocess_observation(obs)
        state_size = len(state)
        action_size = env.action_space.n
        
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            **agent_config
        )
        
        # Train with config and experiment name
        training_config = config.get_training_config()
        print("Training agent...")
        training_results = train_dqn(
            env=env, 
            agent=agent, 
            experiment_name=condition['name'],  # Add experiment name here
            **training_config
        )
        
        # Save training results
        os.makedirs("results", exist_ok=True)
        with open(f"results/{condition['name']}_training_stats.json", "w") as f:
            json.dump({
                "scores": [float(x) for x in training_results['scores']],
                "vases_broken": [float(x) for x in training_results['vases_broken']],
                "caught": [bool(x) for x in training_results['caught']],
                "goal_reached": [bool(x) for x in training_results['goal_reached']],
                "epsilon": [float(x) for x in training_results['epsilon']]
            }, f)
        
        # Save the trained model
        torch.save(agent.q_network.state_dict(), f"results/{condition['name']}_model.pth")
        
        # Evaluate the agent
        print("Evaluating agent...")
        eval_results, frames = evaluate_agent(
            agent, env, num_episodes=100, render=True, save_frames=True
        )
        
        # Save evaluation results
        results[condition['name']] = eval_results
        
        # Save a video of agent behavior
        if frames:
            os.makedirs("videos", exist_ok=True)
            plt.figure(figsize=(8, 8))
            for i, frame in enumerate(frames):
                plt.clf()
                plt.imshow(frame)
                plt.axis('off')
                plt.savefig(f"videos/{condition['name']}_frame_{i:03d}.png")
            plt.close()
        
        # Close environment
        env.close()
    
    # Compare and visualize results
    visualize_results(results, conditions)
    
    return results

def visualize_results(results, conditions):
    """Visualize and compare results from different conditions."""
    # Extract metrics for plotting
    metrics = [
        "mean_reward", 
        "mean_vases_broken", 
        "mean_caught_rate", 
        "mean_goal_reached_rate",
        "mean_steps"
    ]
    
    # Plot comparison
    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 15))
    
    for i, metric in enumerate(metrics):
        values = [results[condition['name']][metric] for condition in conditions]
        labels = [condition['name'] for condition in conditions]
        
        axs[i].bar(labels, values)
        axs[i].set_title(metric.replace('_', ' ').title())
        axs[i].set_ylabel('Value')
        
        # Add value labels
        for j, v in enumerate(values):
            axs[i].text(j, v, f"{v:.2f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("results/comparison.png")
    plt.show()
    
    # Save results to JSON
    with open("results/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n===== EXPERIMENT SUMMARY =====")
    for condition in conditions:
        name = condition['name']
        print(f"\n{name} ({condition['description']}):")
        for metric, value in results[name].items():
            print(f"  {metric}: {value:.4f}")

# Main function
if __name__ == "__main__":
    # Login to wandb (do this once)
    wandb.login()
    
    # Run all experiments
    results = run_experiments()