import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DQN
from src.agents.RL.auction_env import AuctionEnv

# Create the environment with a 10x10 grid and a random number of tasks
env = AuctionEnv(num_agents=1, grid_size=10)

# Reset the environment with random positions
obs = env.reset()

# Initialize the DQN agent with dynamic number of tasks
model = DQN("MlpPolicy", env, verbose=1, exploration_fraction=0.5, exploration_final_eps=0.1)

# Train the model for more steps with a variable number of tasks
model.learn(total_timesteps=1000000)

# Save the trained model
model.save("src/agents/RL/auction_bid_model")
