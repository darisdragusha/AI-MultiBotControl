import random
from enum import Enum
import time
import numpy as np

class CellType(Enum):
    EMPTY = 0
    ROBOT = 1
    OBSTACLE = 2
    TARGET = 3
    TASK = 4

class Task:
    def __init__(self, x, y, priority=1):
        self.x = x
        self.y = y
        self.priority = priority  # 1 (low) to 3 (high)
        self.creation_time = time.time()

    def get_position(self):
        return (self.x, self.y)
        
    def get_waiting_time(self):
        return time.time() - self.creation_time

class Robot:
    def __init__(self, x, y, id):
        # Use a tuple for position instead of separate x and y
        self.position = (x, y)
        self.x=x
        self.y=y
        self.target = None  # Will now store Task object instead of just position
        self.path = []
        self.waiting = False
        self.waiting_time = 0
        self.last_waiting_start = None
        self.last_move_time = time.time()
        self.completed_tasks = 0
        self.total_distance = 0
        self.start_time = time.time()
        self.status_message = ""
        self.id = id
        self.last_action = None
        self.last_state = None
        self.replan_count = 0  # Counter for path replanning
        self.wait_count = 0    # Counter for collision waits

    def set_target(self, task):
        self.target = task
        self.status_message = f"Assigned to task at ({task.x}, {task.y}) with priority {task.priority}"

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def update_waiting(self, is_waiting, current_time):
        if is_waiting:
            if not self.waiting:  # Just started waiting
                self.last_waiting_start = current_time
            self.waiting = True
        else:
            if self.waiting:  # Just stopped waiting
                self.waiting_time += current_time - self.last_waiting_start
            self.waiting = False
            self.last_waiting_start = None

    def get_state(self):
        """
        Return the robot's state as a vector.
        """
        if self.target:
            target_x, target_y = self.target.get_position()
            target_priority = self.target.priority
        else:
            target_x, target_y = -1, -1  # No target assigned
            target_priority = 0

        state = np.array([
            self.position[0], self.position[1],  # Robot's position
            target_x, target_y,   # Target position
            target_priority       # Target priority
        ], dtype=np.float32)
        return state
