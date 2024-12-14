import pygame
import pickle
import numpy as np
import heapq

# Constants
GRID_SIZE = 10  # Size of the grid (10x10)
BLOCK_SIZE = 40  # Size of each block in the grid
NUM_ROBOTS = 3  # Number of robots

# A* Node class
class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

# Heuristic function for A*
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* Pathfinding algorithm
def astar(maze, start, end):
    start_node = Node(None, start)
    end_node = Node(None, end)

    open_list = []
    closed_set = set()

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)

        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        for new_position in [
            (0, -1),  # Up
            (0, 1),   # Down
            (-1, 0),  # Left
            (1, 0)    # Right
        ]:
            node_position = (current_node.position[0] + new_position[0],
                             current_node.position[1] + new_position[1])

            if not (0 <= node_position[0] < len(maze) and 0 <= node_position[1] < len(maze[0])):
                continue

            if maze[node_position[0]][node_position[1]] != 0:
                continue

            new_node = Node(current_node, node_position)

            if new_node.position in closed_set:
                continue

            new_node.g = current_node.g + 1
            new_node.h = heuristic(new_node.position, end_node.position)
            new_node.f = new_node.g + new_node.h

            if any(open_node for open_node in open_list if open_node == new_node and open_node.g <= new_node.g):
                continue

            heapq.heappush(open_list, new_node)

    return None

# Robot class
class Robot:
    def __init__(self, id, start, end, priority, color):
        self.id = id
        self.start = start
        self.end = end
        self.path = []
        self.current_step = 0
        self.priority = priority
        self.color = color
        self.action = None

    def calculate_path(self, maze):
        self.path = astar(maze, self.start, self.end) or []

    def move(self):
        if self.current_step + 1 < len(self.path):
            self.current_step += 1

# Function to check collisions
def check_collisions(robots):
    next_positions = {}
    collisions = []

    for i, robot in enumerate(robots):
        if robot.current_step + 1 < len(robot.path):
            next_pos = robot.path[robot.current_step + 1]

            if next_pos in next_positions:
                collisions.append((i, next_positions[next_pos]))

            next_positions[next_pos] = i

    return collisions

# Move robot with collision handling
def move_robot(robots, robot_idx, collisions):
    for robot1, robot2 in collisions:
        if robot_idx == robot1:
            if robots[robot_idx].priority > robots[robot2].priority:
                robots[robot_idx].move()
                return True
            else:
                return False

    robots[robot_idx].move()
    return True

# Visualize robots in Pygame
def visualize(robots, screen, maze):
    screen.fill((255, 255, 255))

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            color = (200, 200, 200) if maze[row][col] == 0 else (50, 50, 50)
            pygame.draw.rect(screen, color,
                             (col * BLOCK_SIZE, row * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

    for robot in robots:
        for i in range(1, robot.current_step + 1):
            start_pos = robot.path[i - 1]
            end_pos = robot.path[i]
            pygame.draw.line(screen, robot.color,
                             (start_pos[1] * BLOCK_SIZE + BLOCK_SIZE // 2, start_pos[0] * BLOCK_SIZE + BLOCK_SIZE // 2),
                             (end_pos[1] * BLOCK_SIZE + BLOCK_SIZE // 2, end_pos[0] * BLOCK_SIZE + BLOCK_SIZE // 2),
                             3)

        if robot.current_step < len(robot.path):
            current_pos = robot.path[robot.current_step]
            pygame.draw.circle(screen, robot.color,
                               (current_pos[1] * BLOCK_SIZE + BLOCK_SIZE // 2,
                                current_pos[0] * BLOCK_SIZE + BLOCK_SIZE // 2),
                               BLOCK_SIZE // 2)

    pygame.display.flip()

# Main program
def main():
    pygame.init()

    maze = [
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    robots = [
        Robot(id=1, start=(0, 0), end=(2, 9), priority=3, color=(255, 0, 0)),
        Robot(id=2, start=(0, 9), end=(9, 0), priority=2, color=(0, 255, 0)),
        Robot(id=3, start=(9, 0), end=(0, 9), priority=1, color=(0, 0, 255))
    ]

    for robot in robots:
        robot.calculate_path(maze)

    screen = pygame.display.set_mode((GRID_SIZE * BLOCK_SIZE, GRID_SIZE * BLOCK_SIZE))
    pygame.display.set_caption("Robot Pathfinding with A*")

    clock = pygame.time.Clock()
    running = True

    while running:
        collisions = check_collisions(robots)

        for i in range(NUM_ROBOTS):
            move_robot(robots, i, collisions)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        visualize(robots, screen, maze)
        clock.tick(2)

    pygame.quit()

if __name__ == "__main__":
    main()
