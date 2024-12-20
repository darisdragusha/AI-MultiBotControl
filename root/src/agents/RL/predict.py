import numpy as np
import os
from stable_baselines3 import DQN
from src.agents.RL.auction_env import AuctionEnv

def predict_action(agent_position, task_positions):
    """
    Predicts the action for a given agent position and task positions using a pre-trained model.
    
    Parameters:
        agent_position (list or np.ndarray): The position of the agent [x, y].
        task_positions (list or np.ndarray): The positions of the tasks as a list of [x, y] coordinates.
        
    Returns:
        int: The predicted action (task index).
    """
    # Load the trained model

# Define the absolute path to your model file
    model_path = os.path.join(os.getcwd(), "root", "src", "agents", "RL", "auction_bid_model.zip")

# Load the model
    model = DQN.load(model_path)

    
    # Initialize the environment with dynamic task count
    num_tasks = len(task_positions)
    env = AuctionEnv(num_agents=1, num_tasks=num_tasks, grid_size=10)
    
    # Reset the environment with the given agent and task positions
    obs = env.reset(agent_positions=[agent_position], task_positions=task_positions)
    
    # Predict the action
    action, _ = model.predict(obs)
    print("agent: ",agent_position, task_positions,action)
    return action


