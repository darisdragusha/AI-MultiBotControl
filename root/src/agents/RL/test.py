import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DQN

class AuctionEnv(gym.Env):
    def __init__(self, num_agents=1, num_tasks=5, grid_size=10):
        super(AuctionEnv, self).__init__()

        self.num_agents = num_agents
        self.num_tasks = num_tasks
        self.grid_size = grid_size  # Define a 10x10 grid

        # Define action space dynamically based on num_tasks
        self.action_space = spaces.Discrete(self.num_tasks)

        # Observation space: agent positions + task positions (dynamic length based on num_tasks)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size-1, shape=(self.num_agents * 2 + self.num_tasks * 2,), dtype=np.int32
        )

        # Initialize task positions randomly in 2D grid space
        self.task_positions = np.random.randint(0, self.grid_size, size=(self.num_tasks, 2))
        self.agent_positions = np.random.randint(0, self.grid_size, size=(self.num_agents, 2))

    def reset(self, agent_positions=None, task_positions=None):
        """
        Resets the environment with custom positions if provided, else random positions.
        """
        if agent_positions is not None:
            self.agent_positions = np.array(agent_positions)
        else:
            self.agent_positions = np.random.randint(0, self.grid_size, size=(self.num_agents, 2))

        if task_positions is not None:
            self.task_positions = np.array(task_positions)
        else:
            self.task_positions = np.random.randint(0, self.grid_size, size=(self.num_tasks, 2))

        # Flatten agent and task positions into a single observation vector
        observation = np.concatenate([self.agent_positions.flatten(), self.task_positions.flatten()])
        return observation

    def step(self, actions):
        """
        Executes actions and returns the next state, reward, done flag, and additional info.
        """
        rewards = []
        if isinstance(actions, np.ndarray) and actions.ndim > 0:
            for agent_id, action in enumerate(actions):
                # Calculate the Euclidean distance between the agent and the chosen task
                distances = np.linalg.norm(self.agent_positions - self.task_positions[action], axis=1)
                # Reward based on proximity: negative reward for farther tasks
                reward = -distances[agent_id]
                rewards.append(reward)
        else:
            agent_id = 0
            action = actions
            distances = np.linalg.norm(self.agent_positions - self.task_positions[action], axis=1)
            reward = -distances[agent_id]
            rewards.append(reward)

        done = True  # Reset after all agents have chosen a task
        observation = np.concatenate([self.agent_positions.flatten(), self.task_positions.flatten()])
        return observation, np.mean(rewards), done, {}

# Load the trained model
model = DQN.load("auction_bid_model.zip")

# Example of dynamic number of tasks for testing (change num_tasks here)
num_tasks = 10  # Can be changed dynamically
env = AuctionEnv(num_agents=1, num_tasks=num_tasks, grid_size=10)

# Agent positions (10 agents)
agent_positions = np.array([
    [2, 3],  # Agent 1
    [7, 6],  # Agent 2
    [1, 1],  # Agent 3
    [5, 8],  # Agent 4
    [9, 2],  # Agent 5
    [4, 4],  # Agent 6
    [3, 6],  # Agent 7
    [8, 1],  # Agent 8
    [6, 7],  # Agent 9
    [0, 9]   # Agent 10
])

# Task positions (5 tasks)
task_positions = np.array([
    [1, 5],  # Task 1
    [4, 2],  # Task 2
    [8, 8],  # Task 3
    [6, 3],  # Task 4
    [3, 7]   # Task 5
])

num_robots = 7
for agent in agent_positions:

    # Example custom agent and task positions for prediction (also dynamic)
    custom_agent_positions = [agent]  # Define positions of agents (within 0-9 range)
    custom_task_positions = task_positions  # Define positions of tasks (within 0-9 range)

    # Reset the environment with custom positions for prediction
    obs = env.reset(agent_positions=custom_agent_positions, task_positions=custom_task_positions)

    # Predict the action for a single step
    action, _states = model.predict(obs)
    print(f"Predicted Action: {action}, Agent's position: {obs[:2]}, Task positions: {obs[2:]}")

    # Perform the action in the environment
    obs, reward, done, info = env.step(action)

    # You can then decide whether to break or continue based on your logic
    if done:
        print("----")

