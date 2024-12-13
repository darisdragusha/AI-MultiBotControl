import pygame
import pickle
import random
import numpy as np

# Constants
GRID_SIZE = 10  # Size of the grid (10x10)
BLOCK_SIZE = 40  # Size of each block in the grid
NUM_ROBOTS = 3  # Number of robots


# Robot class to manage the state
class Robot:
    def __init__(self, id, path, priority, color):
        self.id = id  # Robot's ID
        self.path = path  # The robot's path (list of positions)
        self.current_step = 0  # Current step on the path
        self.priority = priority  # Priority (higher value means higher priority)
        self.color = color  # Color for visualization
        self.action = None  # Action for this robot at any given state

    def get_next_action(self, state, q_table):
        """Get the next action based on the Q-table."""
        # Extract the Q-values for the current state
        state_q_values = q_table[state]

        # Return the action with the highest Q-value (0: move, 1: wait)
        return np.argmax(state_q_values)

    def move(self):
        """Move the robot along its predefined path."""
        if self.current_step + 1 < len(self.path):
            self.current_step += 1


# Function to check for collisions (if two robots will collide)
def check_collisions(robots):
    """Check if any two robots' next step positions will collide."""
    next_positions = {}
    collisions = []

    for i, robot in enumerate(robots):
        # Get the next position of the robot
        if robot.current_step + 1 < len(robot.path):
            next_pos = robot.path[robot.current_step + 1]

            if next_pos in next_positions:
                # If two robots have the same next position, it's a collision
                collisions.append((i, next_positions[next_pos]))

            # Store the next position of the robot
            next_positions[next_pos] = i

    return collisions


# Function to move a robot (handle collisions by priority)
def move_robot(robots, robot_idx, collisions):
    """Move the robot based on collision status and priority."""
    for robot1, robot2 in collisions:
        if robots[robot_idx].path[robots[robot_idx].current_step + 1] == robots[robot2].path[
            robots[robot2].current_step + 1]:
            # If the next position of both robots is the same, prioritize based on priority
            if robots[robot_idx].priority > robots[robot2].priority:
                robots[robot_idx].current_step += 1  # Move the robot
                return True
            else:
                # Lower priority robot will not move
                return False

    # If no collision, just move the robot
    if robots[robot_idx].current_step + 1 < len(robots[robot_idx].path):
        robots[robot_idx].current_step += 1
        return True
    return False


# Function to visualize the robots and their movement in Pygame
def visualize(robots, screen):
    screen.fill((255, 255, 255))  # Clear the screen with white background

    # Draw the grid
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            pygame.draw.rect(screen, (200, 200, 200),
                             (col * BLOCK_SIZE, row * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

    # Draw robots' paths and current positions
    for robot in robots:
        for i in range(1, robot.current_step + 1):
            start_pos = robot.path[i - 1]
            end_pos = robot.path[i]
            pygame.draw.line(screen, robot.color,
                             (start_pos[0] * BLOCK_SIZE + BLOCK_SIZE // 2, start_pos[1] * BLOCK_SIZE + BLOCK_SIZE // 2),
                             (end_pos[0] * BLOCK_SIZE + BLOCK_SIZE // 2, end_pos[1] * BLOCK_SIZE + BLOCK_SIZE // 2),
                             3)

        # Draw current robot position
        if robot.current_step < len(robot.path):
            current_pos = robot.path[robot.current_step]
            pygame.draw.circle(screen, robot.color,
                               (current_pos[0] * BLOCK_SIZE + BLOCK_SIZE // 2,
                                current_pos[1] * BLOCK_SIZE + BLOCK_SIZE // 2),
                               BLOCK_SIZE // 2)

    pygame.display.flip()  # Update the display


# Main program
def main():
    pygame.init()

    # Load the shared Q-table (this file should be created after training)
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)

    # Robot paths (predefined, with more intertwined paths and robots starting in different positions)
    robots = [
        Robot(id=1, path=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 1), (7, 1), (8, 2), (9, 2)], priority=3,
              color=(255, 0, 0)),
        # Red robot moves vertically, then across, and intersects with other robots at each step
        Robot(id=2, path=[(0, 9), (1, 9), (2, 9), (3, 8), (4, 8), (5, 8), (6, 7), (7, 7), (8, 6), (9, 6)], priority=1,
              color=(0, 255, 0)),
        # Green robot moves vertically, then across, and intersects with other robots at each step
        Robot(id=3, path=[(9, 0), (8, 0), (7, 0), (6, 0), (5, 1), (4, 1), (3, 2), (2, 2), (1, 3), (0, 3)], priority=2,
              color=(0, 0, 255))
        # Blue robot moves vertically, then across, and intersects with other robots at each step
    ]

    # Initialize Pygame window
    screen = pygame.display.set_mode((GRID_SIZE * BLOCK_SIZE, GRID_SIZE * BLOCK_SIZE))
    pygame.display.set_caption("Robot Collision Avoidance")

    # Main loop
    running = True
    clock = pygame.time.Clock()

    for step in range(50):  # Run for 50 steps for this example
        collisions = check_collisions(robots)

        # Move robots according to collision avoidance strategy
        for i in range(NUM_ROBOTS):
            robot = robots[i]
            # Get the robot's current state (position)
            state = robot.path[robot.current_step]

            # Ensure state is an integer index, mapped to a state (you could map position to a state index)
            state_index = int(state[0] * GRID_SIZE + state[1])  # Convert (x, y) position to a unique state index

            # Get the action for this robot based on Q-table
            action = robot.get_next_action(state_index, q_table)

            # Set action (move or wait) and move the robot accordingly
            robot.action = 'move' if action == 0 else 'wait'

            # Move the robot
            move_robot(robots, i, collisions)

        # Handle Pygame events (e.g., closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Visualize the robots and paths
        visualize(robots, screen)

        # Slow down the loop for better visualization
        clock.tick(2)  # 2 frames per second

    pygame.quit()


if __name__ == "__main__":
    main()
