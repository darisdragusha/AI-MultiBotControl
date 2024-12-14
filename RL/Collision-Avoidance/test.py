import pickle
import random 
import numpy as np

# Assuming the Robot and Environment classes are already defined as per your previous code
# Define the Environment Class
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
            current_position = robot.get_current_position()
            priority = robot.priority
            next_position = robot.get_next_position()  # Next step
            state.append((current_position, priority, next_position))
        return tuple(state)  # Use a tuple to make the state hashable

    def check_collisions(self):
        """Check if robots have collided at the current step."""
        current_positions = {}
        collisions = []

        for i, robot in enumerate(self.robots):
            current_pos = robot.get_current_position()
            
            if current_pos in current_positions:
                # Record collision between robots
                collisions.append((i, current_positions[current_pos]))
            else:
                current_positions[current_pos] = i  # Mark this position as occupied

        return collisions

    def step(self, actions):
        """Take a step in the environment based on robot actions."""
        for i, action in enumerate(actions):
            if action == 0:  # Move
                self.robots[i].move()
            elif action == 1:  # Wait
                pass  # Robot remains in place

        collisions = self.check_collisions()
        state = self.get_state()
        return state, collisions


# Define the Robot Class
class Robot:
    def __init__(self, path, priority):
        self.path = path
        self.current_step = 0
        self.priority = priority

    def get_current_position(self):
        """Get the robot's current position."""
        return self.path[self.current_step]

    def has_next_step(self):
        """Check if the robot has a next step."""
        return self.current_step + 1 < len(self.path)

    def get_next_position(self):
        """Get the robot's next position."""
        if self.has_next_step():
            return self.path[self.current_step + 1]
        else:
            return self.path[self.current_step]  # No next position, return the current one

    def move(self):
        """Move the robot to the next step."""
        if self.has_next_step():
            self.current_step += 1


# Define the Q-learning Model
class QLearningModel:
    def __init__(self, num_actions=2, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.num_actions = num_actions  # Move or Wait
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # Use a sparse dictionary for the Q-table

    def choose_action(self, state_index):
        """Epsilon-greedy action selection with exploration decay."""
        if random.uniform(0, 1) < self.epsilon:  # Exploration
            return random.choice([0, 1])  # Randomly choose between moving or waiting
        else:  # Exploitation
            state_q_values = self.q_table.get(state_index, [0, 0])  # Default to [0, 0] if state not encountered
            return np.argmax(state_q_values)  # Exploit the best action

    def update_q_table(self, state_index, action, reward, next_state_index):
        """Update the Q-table using the Q-learning formula."""
        best_next_action = np.max(self.q_table.get(next_state_index, [0, 0]))  # Default to [0, 0] if state not encountered
        if state_index not in self.q_table:
            self.q_table[state_index] = [0, 0]  # Initialize the Q-values for the state
        self.q_table[state_index][action] += self.learning_rate * (
            reward + self.discount_factor * best_next_action - self.q_table[state_index][action]
        )

    def save_q_table(self, filename="q_table.pkl"):
        """Save the Q-table to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename="q_table.pkl"):
        """Load the Q-table from a file."""
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)


def test_model():
    # Load the Q-table using pickle
    with open("q_table.pkl", "rb") as file:
        q_table = pickle.load(file)

    # Initialize the QLearning model with the loaded Q-table
    model = QLearningModel(num_actions=2)  # Ensure the number of actions matches
    model.q_table = q_table  # Assign the loaded Q-table to the model
    
    # Define the robots and paths as specified
    robots = [
        Robot(path=[(0, 4), (0, 3), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (6, 3), (6, 4)],
              priority=3),
        Robot(path=[(7, 5), (7, 4), (7, 3), (7, 2), (7, 1), (6, 1), (5, 1), (5, 2), (5, 3), (5, 4)],
              priority=1),
        Robot(path=[(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (4, 5), (5, 5)],
              priority=2)
    ]
    
    # Initialize the environment with the robots
    environment = Environment(grid_size=10, num_robots=len(robots))
    environment.robots = robots

    # Map states to indices dynamically
    state_to_index = {}
    index_counter = 0

    def get_state_index(state):
        nonlocal index_counter
        if state not in state_to_index:
            state_to_index[state] = index_counter
            index_counter += 1
        return state_to_index[state]
    
    # Run the test for each step
    for step in range(min(len(robot.path) for robot in robots)):  # Step limit based on shortest path
        # Get the current state (robot positions, priorities, and next steps)
        state = environment.get_state()
        state_index = get_state_index(state)
        
        # Choose actions for each robot based on the current state
        actions = [model.choose_action(state_index) for _ in environment.robots]
        
        # Print the actions taken by each robot
        print(f"Step {step + 1}:")
        for i, robot in enumerate(environment.robots):
            action = actions[i]
            action_str = "Move" if action == 0 else "Wait"
            print(f"  Robot {robot.priority} (Priority {robot.priority}): {action_str}")

        # Update the environment by performing the chosen actions
        next_state, collisions = environment.step(actions)
        print(next_state)
        print(f"Collisions at this step: {collisions}")

        # Optionally, you can track robot positions at each step
        for i, robot in enumerate(environment.robots):
            print(f"  Robot {robot.priority} position: {robot.get_current_position()}")

    print("Testing completed.")

# Run the test model
if __name__ == "__main__":
    test_model()
