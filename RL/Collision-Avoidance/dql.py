import random
import torch
import torch.nn as nn
import torch.optim as optim


# Define the Robot class
class Robot:
    def __init__(self, path, priority):
        self.path = path
        self.current_index = 0  # Start at the first position
        self.priority = priority  # Priority of the robot (for collision handling)

    def get_current_position(self):
        """Get the current position of the robot."""
        return self.path[self.current_index]

    def get_next_position(self):
        """Get the next position of the robot."""
        if self.has_next_step():
            return self.path[self.current_index + 1]
        return None

    def has_next_step(self):
        """Check if the robot has a next step in its path."""
        return self.current_index < len(self.path) - 1

    def move(self):
        """Move the robot to the next step in its path."""
        if self.has_next_step():
            self.current_index += 1


# Define the Environment class
class Environment:
    def __init__(self, grid_size=10, num_robots=2):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.robots = []
        self.robot_paths = []  # Predefined paths
        self.initialize_robots()

    def initialize_robots(self):
        """Initialize robots with random start positions and predefined paths."""
        for i in range(self.num_robots):
            path = self.generate_random_path()
            self.robot_paths.append(path)
            self.robots.append(Robot(path, i + 1))  # Assign priority based on index

    def generate_random_path(self):
        """Generate a realistic random path for robots."""
        start = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        path = [start]
        for _ in range(random.randint(5, 10)):  # Path length of 5-10 steps
            next_pos = (max(0, min(self.grid_size - 1, path[-1][0] + random.choice([-1, 0, 1]))),
                        max(0, min(self.grid_size - 1, path[-1][1] + random.choice([-1, 0, 1]))))
            path.append(next_pos)
        
        return path

    def get_state(self):
        """Represent the state as a tuple of robot positions, priorities, and next steps."""
        state = []
        for robot in self.robots:
            position = robot.get_current_position()
            priority = robot.priority
            next_step = robot.get_next_position() if robot.has_next_step() else None
            state.append((position, priority, next_step))
        
        # Flatten the state into a 1D array for the neural network
        flattened_state = []
        for robot_state in state:
            flattened_state.extend(robot_state)  # Flatten each robot's state into the list
        
        return flattened_state  # Return a flattened 1D state list

    def step(self, actions):
        """Move the robots based on actions and return the next state and reward."""
        collisions = []
        for i, action in enumerate(actions):
            if action == 1:  # Move
                self.robots[i].move()
            for j in range(i + 1, len(self.robots)):
                if self.robots[i].get_current_position() == self.robots[j].get_current_position():
                    collisions.append((i, j))
        
        # Generate the next state
        next_state = self.get_state()
        return next_state, collisions


# Define the Neural Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output size is 2 (move or wait)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# Define the DQN Agent
class DQNAgent:
    def __init__(self, input_size, num_actions, learning_rate=0.001, epsilon=0.1, gamma=0.9):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Exploration rate
        self.gamma = gamma  # Discount factor

        # Initialize the neural network model and optimizer
        self.model = DQN(input_size, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()  # Mean Squared Error Loss

    def choose_action(self, state):
        """Epsilon-greedy action selection with exploration decay."""
        if random.uniform(0, 1) < self.epsilon:  # Exploration
            return random.choice([0, 1])  # Randomly choose between moving or waiting
        else:  # Exploitation
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()  # Exploit the best action

    def update(self, state, action, reward, next_state, done):
        """Update the model based on the experience using Q-learning."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        # Get Q-values for the current state
        current_q_values = self.model(state_tensor)
        
        # Get Q-value for the next state (used for Bellman equation)
        next_q_values = self.model(next_state_tensor)
        next_max_q = torch.max(next_q_values)
        
        # Calculate target Q-value
        target_q = reward + (self.gamma * next_max_q if not done else reward)
        
        # Update Q-value for the taken action
        current_q_values[0, action] = target_q
        
        # Compute loss
        loss = self.loss_fn(current_q_values, self.model(state_tensor))
        
        # Perform backpropagation and optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Train the model
def train(episodes):
    state_to_index = {}  # Mapping from states to indices
    index_counter = 0

    def get_state_index(state):
        nonlocal index_counter
        # Convert the state into a tuple of tuples so that it's hashable
        hashable_state = tuple(state)
        if hashable_state not in state_to_index:
            state_to_index[hashable_state] = index_counter
            index_counter += 1
        return state_to_index[hashable_state]

    for episode in range(episodes):
        environment = Environment(grid_size=10, num_robots=3)
        agent = DQNAgent(input_size=30, num_actions=2)  # Adjust input size based on flattened state
        state = environment.get_state()
        state_index = get_state_index(state)
        total_reward = 0

        # Decay epsilon over time to encourage more exploitation
        agent.epsilon = max(0.01, agent.epsilon * 0.995)  # Decay epsilon to a minimum of 0.01

        for step in range(10):  # Limit to 10 steps per episode
            # Choose actions for each robot based on the current state
            actions = [agent.choose_action(state_index) for _ in environment.robots]
            
            # Update environment by performing the chosen actions
            next_state, collisions = environment.step(actions)
            next_state_index = get_state_index(next_state)

            # Calculate reward based on collisions
            reward = 0
            if collisions:
                # Penalize collisions, giving higher penalties to higher-priority robots
                for colliding_robot_1, colliding_robot_2 in collisions:
                    if environment.robots[colliding_robot_1].priority > environment.robots[colliding_robot_2].priority:
                        reward -= 15  # Higher penalty for higher-priority robot collision
                    else:
                        reward -= 10  # Penalty for a collision
            else:
                reward = 1  # Positive reward for no collision

            # Update the Q-network for each robot based on the actions
            for i, action in enumerate(actions):
                agent.update(state_index, action, reward, next_state_index, done=False)

            # Set the new state for the next iteration
            state = next_state
            state_index = next_state_index
            total_reward += reward

        # Optionally print the reward for each episode
        print(f"Episode {episode + 1}: Total reward = {total_reward}")


# Main function for Training
def main_train():
    try:
        train(episodes=10000)
    except Exception as e:
        print(f"Training failed: {str(e)}")


# Run the training phase
if __name__ == "__main__":
    main_train()
