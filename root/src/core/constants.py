import pygame

# Window Constants
WINDOW_SIZE = 800
GRID_SIZE = 10
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
MENU_WIDTH = 200
TOTAL_WIDTH = WINDOW_SIZE + MENU_WIDTH

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
robot_image = pygame.image.load('root/src/images/robot.png')  
robot_image = pygame.transform.scale(robot_image, (CELL_SIZE, CELL_SIZE))  


# MADQL Parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1

# Simulation Parameters
MOVE_DELAY = 0.5  # seconds between moves
TASK_GEN_CHANCE = 0.05  # 5% chance per update to generate a new task
MAX_TASKS = 5

# Moving Obstacles Parameters
MAX_MOVING_OBSTACLES = 3  # Maximum number of moving obstacles
OBSTACLE_GEN_CHANCE = 0.02  # 2% chance per update to generate a new obstacle
OBSTACLE_MOVE_DELAY = 1.0  # seconds between obstacle movements