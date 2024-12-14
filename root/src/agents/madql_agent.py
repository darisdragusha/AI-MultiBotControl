import random
import numpy as np
from src.core.constants import EPSILON, LEARNING_RATE, DISCOUNT_FACTOR, GRID_SIZE, MAX_TASKS
from src.core.entities import CellType, Task
import time

class MADQLAgent:
    def __init__(self, game):
        self.game = game
        self.episode_count = 0
        self.epsilon = EPSILON  # Will decay over time
        
    def get_state(self, robot):
        """
        Compact state representation for better Q-learning:
        - Normalized robot position
        - Relative direction to nearest task of each priority
        - Normalized distances to nearest tasks
        - Local congestion (robots and obstacles)
        """
        state = []
        
        # Normalized robot position
        state.extend([robot.x/GRID_SIZE, robot.y/GRID_SIZE])
        
        # For each priority level
        for priority in range(1, 4):
            nearest_dist = float('inf')
            nearest_direction = [0, 0]  # [dx, dy]
            
            # Find nearest task of this priority
            for task in self.game.tasks:
                if task.priority == priority:
                    dx = task.x - robot.x
                    dy = task.y - robot.y
                    dist = abs(dx) + abs(dy)  # Manhattan distance
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_direction = [
                            np.sign(dx) if dx != 0 else 0,
                            np.sign(dy) if dy != 0 else 0
                        ]
            
            # Add normalized distance and direction
            state.append(min(nearest_dist/GRID_SIZE, 1.0))
            state.extend(nearest_direction)
        
        # Local congestion in 4 directions (up, right, down, left)
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            congestion = 0
            for d in range(1, 3):  # Check 2 cells in each direction
                check_x = robot.x + dx * d
                check_y = robot.y + dy * d
                if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                    if self.game.grid[check_y][check_x] in [CellType.ROBOT, CellType.OBSTACLE]:
                        congestion += 1/d  # Closer obstacles matter more
            state.append(min(congestion, 1.0))
        
        return tuple(state)
    
    def get_available_actions(self, robot):
        """Get list of available tasks as actions"""
        return self.get_available_tasks(robot)
    
    def get_available_tasks(self, robot):
        """Get list of available tasks that can be assigned"""
        return [task for task in self.game.tasks 
                if self.game.grid[task.y][task.x].value == CellType.TASK.value]
    
    def get_reward(self, robot, old_state, action, new_state):
        """
        Reward function focused on learning optimal task allocation:
        - Base reward for task priority
        - Penalty for distance and congestion
        - Bonus for completing tasks
        - Penalty for waiting/deadlocks
        """
        if action is None:
            return -10  # Significant penalty for no action
        
        reward = 0
        
        # Priority-based reward
        if isinstance(action, Task):
            # Base reward scaled by priority
            reward += action.priority * 20
            
            # Distance penalty
            dist = robot.manhattan_distance((robot.x, robot.y), action.get_position())
            reward -= dist * 2
            
            # Congestion penalty
            congestion = 0
            path = self.game.astar.find_path((robot.x, robot.y), action.get_position())
            if path:
                for px, py in path:
                    for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                        check_x, check_y = px + dx, py + dy
                        if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                            if self.game.grid[check_y][check_x] in [CellType.ROBOT, CellType.OBSTACLE]:
                                congestion += 1
                reward -= congestion * 5
            else:
                reward -= 50  # Heavy penalty for choosing unreachable task
            
            # Waiting time consideration
            if robot.waiting_time > 0:
                reward -= robot.waiting_time * 10
            
            # Task completion bonus
            if robot.target and robot.manhattan_distance((robot.x, robot.y), robot.target.get_position()) == 0:
                reward += 100 * robot.target.priority
        
        return reward
    
    def choose_action(self, robot):
        """
        Choose task using epsilon-greedy policy with decaying exploration
        """
        available_actions = self.get_available_actions(robot)
        if not available_actions:
            return None
            
        state = self.get_state(robot)
        
        # Decay epsilon over time
        self.epsilon = max(0.05, EPSILON * (0.995 ** self.episode_count))
        
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Exploitation: choose action with highest Q-value
        q_values = [robot.q_table[state].get(action, 0) for action in available_actions]
        max_q = max(q_values)
        best_actions = [action for action, q in zip(available_actions, q_values) if q == max_q]
        
        return random.choice(best_actions)
    
    def update(self, robot, old_state, action, reward, new_state):
        """
        Q-learning update with experience replay
        """
        if action is None:
            return
            
        # Get max Q-value for next state
        next_actions = self.get_available_actions(robot)
        if next_actions:
            next_q_values = [robot.q_table[new_state].get(a, 0) for a in next_actions]
            max_next_q = max(next_q_values)
        else:
            max_next_q = 0
        
        # Current Q-value
        current_q = robot.q_table[old_state].get(action, 0)
        
        # Q-learning update
        new_q = current_q + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * max_next_q - current_q
        )
        
        # Update Q-table
        if old_state not in robot.q_table:
            robot.q_table[old_state] = {}
        robot.q_table[old_state][action] = new_q
        
        # Store experience for replay
        robot.last_action = action
        robot.last_state = old_state
        
        # Count episode for epsilon decay
        if action == robot.target:
            self.episode_count += 1