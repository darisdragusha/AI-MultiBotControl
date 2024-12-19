import numpy as np
from stable_baselines3 import DQN
from auction_env import AuctionEnv  # Assuming your AuctionEnv class is in train.py

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
    model = DQN.load("auction_bid_model.zip")
    
    # Initialize the environment with dynamic task count
    num_tasks = len(task_positions)
    env = AuctionEnv(num_agents=1, num_tasks=num_tasks, grid_size=10)
    
    # Reset the environment with the given agent and task positions
    obs = env.reset(agent_positions=[agent_position], task_positions=task_positions)
    
    # Predict the action
    action, _ = model.predict(obs)
    return action


# Define agent and task positions
agent_position = [2, 3]
task_positions = [
    [1, 5],
    [4, 2],
    [8, 8],
    [6, 3],
    [3, 7]
]

# Predict the action
action = predict_action(agent_position, task_positions)
print(f"Predicted Action: {action}")