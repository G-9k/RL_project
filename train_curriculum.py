import os
import torch
from train_agent import train
from config import CURRICULUM, DQN_CONFIG
from maze_env import MiniGridEnv

def train_curriculum():
    if not CURRICULUM['ENABLED']:
        return train()
    
    
    env = MiniGridEnv()
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
        
        # If first phase, store the maze configuration
        if phase_idx == 0:
            initial_maze = env.grid.copy()
            if DQN_CONFIG['FIXED_OBJECT_POSITIONS']:
                initial_positions = {
                    'agent_pos': env.agent_pos,
                    'agent_dir': env.agent_dir,
                    'coin_pos': env.coin_pos
                }
        else:
            # Use same maze configuration from phase 1
            env.grid = initial_maze.copy()
            if DQN_CONFIG['FIXED_OBJECT_POSITIONS'] and initial_positions:
                env.agent_pos = initial_positions['agent_pos']
                env.agent_dir = initial_positions['agent_dir']
                env.coin_pos = initial_positions['coin_pos']
        
        # Training arguments
        class Args:
            model_path = f"models/{phase['OUTPUT_NAME']}"  # Use distinct output name
            load_model = agent is not None  # Load model if not first phase
            no_graphics = False
            episodes = phase['EPISODES']
            seed = None
        
        # Train for this phase
        agent, scores = train(Args())
        
        # Save intermediate model with descriptive name
        agent.save(f"models/{phase['OUTPUT_NAME']}/model.pth")
        
        # Restore original config
        DQN_CONFIG.update(original_config)
        
        print(f"Completed phase {phase_idx + 1}: {phase['NAME']}")
    
    # Save final model with distinct name
    agent.save('models/dqn_final_curriculum/model.pth')
    print("Curriculum learning complete!")
    return agent

if __name__ == "__main__":
    train_curriculum()