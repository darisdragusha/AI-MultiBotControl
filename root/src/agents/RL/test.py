import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DQN

class AuctionEnv(gym.Env):
    def __init__(self, num_agents=1, num_tasks=3, grid_size=10):
        super(AuctionEnv, self).__init__()

        self.num_agents = num_agents
        self.num_tasks = num_tasks
        self.grid_size = grid_size  # Define a 10x10 grid

        # Define action space: Each agent chooses a task to bid on (Discrete space)
        self.action_space = spaces.Discrete(self.num_tasks)

        # Observation space: Each agent's position (2D grid) and task positions (2D grid)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size-1, shape=(self.num_agents * 2 + self.num_tasks * 2,), dtype=np.int32
        )

        # Initialize task positions randomly in 2D grid space
        self.task_positions = np.random.randint(0, self.grid_size, size=(self.num_tasks, 2))
        self.agent_positions = np.random.randint(0, self.grid_size, size=(self.num_agents, 2))

    def reset(self, agent_positions=None, task_positions=None):
        """
        Resets the environment by either using custom or random agent and task positions.
        """
        if agent_positions is not None:
            self.agent_positions = np.array(agent_positions)
        else:
            self.agent_positions = np.random.randint(0, self.grid_size, size=(self.num_agents, 2))

        if task_positions is not None:
            self.task_positions = np.array(task_positions)
        else:
            self.task_positions = np.random.randint(0, self.grid_size, size=(self.num_tasks, 2))

        observation = np.concatenate([self.agent_positions.flatten(), self.task_positions.flatten()])
        return observation

    def step(self, actions):
        """
        actions: A list of actions for each agent, where each action is the task index they bid on.
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

# Create the environment with a 10x10 grid
env = AuctionEnv(num_agents=1, num_tasks=3, grid_size=10)

# Example custom agent and task positions for prediction
custom_agent_positions = [[2, 7]]  # Define positions of agents (within 0-9 range)
custom_task_positions = [[1, 2], [5, 5], [7, 7]]  # Define positions of tasks (within 0-9 range)

# Reset the environment with custom positions for prediction
obs = env.reset(agent_positions=custom_agent_positions, task_positions=custom_task_positions)

# Example of using the trained model to predict actions for the single agent
for _ in range(10):
    # Predict actions for the single agent
    action, _states = model.predict(obs)
    print(f"Predicted Action: {action}, Agent's position: {obs[:2]}, Task positions: {obs[2:]}")

    # Perform the action in the environment (for the single agent)
    obs, reward, done, info = env.step(action)  # Pass action for the single agent
    
    if done:
        break