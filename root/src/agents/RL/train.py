import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DQN

# Define the environment
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

    def manhattan_distance(self, pos1, pos2):
        """
        Calculate the Manhattan distance between two positions.
        """
        return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])

    def step(self, actions):
        """
        actions: A list of actions for each agent, where each action is the task index they bid on.
        """
        rewards = []
        if isinstance(actions, np.ndarray) and actions.ndim > 0:
            for agent_id, action in enumerate(actions):
                # Calculate the Manhattan distance between the agent and the chosen task
                task_position = self.task_positions[action]
                agent_position = self.agent_positions[agent_id]
                distance = self.manhattan_distance(agent_position, task_position)
                # Reward is negative of the Manhattan distance (closer is better)
                reward = -distance
                rewards.append(reward)
        else:
            agent_id = 0
            action = actions
            task_position = self.task_positions[action]
            agent_position = self.agent_positions[agent_id]
            distance = self.manhattan_distance(agent_position, task_position)
            reward = -distance
            rewards.append(reward)

        done = True  # Reset after all agents have chosen a task
        observation = np.concatenate([self.agent_positions.flatten(), self.task_positions.flatten()])
        return observation, np.mean(rewards), done, {}

# Create the environment with a 10x10 grid
env = AuctionEnv(num_agents=1, num_tasks=3, grid_size=10)

# Reset the environment to train with random positions
obs = env.reset()

# Initialize the DQN agent with hyperparameters for exploration
model = DQN("MlpPolicy", env, verbose=1, exploration_fraction=0.5, exploration_final_eps=0.1)

# Train the model for more steps (increase timesteps for better learning)
model.learn(total_timesteps=50000)  

# Save the model
model.save("auction_bid_model")


