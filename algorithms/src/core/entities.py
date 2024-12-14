from enum import Enum
from collections import defaultdict
import time

class CellType(Enum):
    EMPTY = 0
    ROBOT = 1
    OBSTACLE = 2
    TARGET = 3
    TASK = 4

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.target = None
        self.path = []
        self.waiting = False
        self.q_table = defaultdict(lambda: defaultdict(lambda: 0))
        self.last_move_time = 0
        self.completed_tasks = 0
        self.total_distance = 0
        self.start_time = time.time()
        self.status_message = ""
        self.id = None
        
    def set_target(self, target_x, target_y):
        self.target = (target_x, target_y)
        self.status_message = f"Assigned to task at ({target_x}, {target_y})"
        
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) 