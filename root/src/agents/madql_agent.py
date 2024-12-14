import random
import numpy as np
from src.core.constants import EPSILON, LEARNING_RATE, DISCOUNT_FACTOR, GRID_SIZE, MAX_TASKS
from src.core.entities import CellType, Task
import time

class MADQLAgent:
    def __init__(self, game):
        self.game = game
        
    def get_state(self, robot):
        """
        Enhanced state representation including:
        - Robot's normalized position
        - Obstacle density in each direction
        - Distance and direction to nearest obstacles
        - Robot density in movement directions
        - Task priorities and distances
        - Deadlock indicators
        """
        state = []
        
        # Normalized position
        state.extend([robot.x/GRID_SIZE, robot.y/GRID_SIZE])
        
        # Obstacle density and nearest obstacle in each direction
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:  # Up, Right, Down, Left
            obstacles = 0
            nearest_dist = GRID_SIZE
            for dist in range(1, GRID_SIZE):
                check_x, check_y = robot.x + dx*dist, robot.y + dy*dist
                if not (0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE):
                    break
                if self.game.grid[check_y][check_x] == CellType.OBSTACLE:
                    obstacles += 1
                    nearest_dist = min(nearest_dist, dist)
            state.append(obstacles/GRID_SIZE)  # Density
            state.append(nearest_dist/GRID_SIZE)  # Distance
        
        # Robot density and nearest robot in movement directions
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            robots = 0
            nearest_robot_dist = GRID_SIZE
            for other_robot in self.game.robots:
                if other_robot != robot:
                    rel_x = other_robot.x - robot.x
                    rel_y = other_robot.y - robot.y
                    # Check if robot is in this direction
                    if (dx == 0 and abs(rel_x) <= 1 and rel_y * dy > 0) or \
                       (dy == 0 and abs(rel_y) <= 1 and rel_x * dx > 0):
                        robots += 1
                        dist = abs(rel_x) + abs(rel_y)
                        nearest_robot_dist = min(nearest_robot_dist, dist)
            state.append(robots/len(self.game.robots))
            state.append(nearest_robot_dist/GRID_SIZE)
        
        # Task information for each priority level
        for priority in range(1, 4):
            tasks_this_priority = [t for t in self.game.tasks if t.priority == priority]
            if tasks_this_priority:
                # Find nearest task of this priority
                distances = [robot.manhattan_distance((robot.x, robot.y), t.get_position()) 
                           for t in tasks_this_priority]
                min_dist = min(distances) if distances else GRID_SIZE
                state.append(min_dist/GRID_SIZE)
                state.append(len(tasks_this_priority)/MAX_TASKS)
            else:
                state.append(1.0)  # No tasks of this priority
                state.append(0.0)
        
        # Deadlock indicators
        state.append(min(robot.waiting_time/10.0, 1.0))  # Normalized waiting time
        
        # Check if robot is in a potential deadlock situation
        surrounded_count = 0
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            check_x, check_y = robot.x + dx, robot.y + dy
            if not (0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE) or \
               self.game.grid[check_y][check_x] in [CellType.OBSTACLE, CellType.ROBOT]:
                surrounded_count += 1
        state.append(surrounded_count/4)  # Normalized surrounded count
        
        return tuple(state)
    
    def get_available_tasks(self, robot):
        """Get list of available tasks that can be assigned"""
        return [task for task in self.game.tasks 
                if self.game.grid[task.y][task.x].value == CellType.TASK.value]
    
    def get_reward(self, robot, old_state, action, new_state):
        """
        Enhanced reward function with better deadlock prevention:
        - Stronger penalties for moves that might cause deadlocks
        - Rewards for moves that avoid potential deadlocks
        - Consider obstacle and robot configurations
        """
        reward = 0
        
        if action is None:
            return -1
        
        # Base rewards for task-related actions
        if isinstance(action, Task):
            # Priority-based reward
            base_reward = {1: 10, 2: 20, 3: 30}[action.priority]
            reward += base_reward
            
            # Waiting time consideration
            task_waiting_time = action.get_waiting_time()
            reward += min(task_waiting_time * 2, 20)
            
            # Check if the path to the task is relatively clear
            path = self.game.astar.find_path((robot.x, robot.y), action.get_position())
            if path:
                # Reward shorter, clearer paths
                obstacle_count = 0
                robot_count = 0
                for px, py in path:
                    for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                        check_x, check_y = px + dx, py + dy
                        if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                            if self.game.grid[check_y][check_x] == CellType.OBSTACLE:
                                obstacle_count += 1
                            elif self.game.grid[check_y][check_x] == CellType.ROBOT:
                                robot_count += 1
                
                path_quality = 1.0 - (obstacle_count + robot_count)/(len(path) * 4)
                reward += path_quality * 10
            else:
                reward -= 20  # Penalty for choosing task with no valid path
        
        # Movement-based rewards
        if isinstance(action, tuple):  # Movement action
            x, y = action
            
            # Penalize moves towards dense obstacle areas
            obstacle_density = 0
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                check_x, check_y = x + dx, y + dy
                if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                    if self.game.grid[check_y][check_x] == CellType.OBSTACLE:
                        obstacle_density += 1
            reward -= obstacle_density * 5
            
            # Penalize moves that might cause deadlock
            robot_density = 0
            for other_robot in self.game.robots:
                if other_robot != robot:
                    dist = robot.manhattan_distance((x, y), (other_robot.x, other_robot.y))
                    if dist <= 2:
                        robot_density += 1
            reward -= robot_density * (obstacle_density + 1) * 3
            
            # Reward moves that improve position relative to target
            if robot.target:
                old_dist = robot.manhattan_distance(
                    (old_state[0] * GRID_SIZE, old_state[1] * GRID_SIZE),
                    robot.target.get_position()
                )
                new_dist = robot.manhattan_distance((x, y), robot.target.get_position())
                dist_improvement = old_dist - new_dist
                reward += dist_improvement * 5
                
                # Extra reward for moving away from deadlock situations
                if robot.waiting_time > 0:
                    reward += dist_improvement * robot.waiting_time
        
        # Deadlock prevention rewards/penalties
        if robot.waiting:
            waiting_penalty = robot.waiting_time * 5
            reward -= waiting_penalty
            
            # Increase penalty if surrounded
            surrounded_count = 0
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                check_x, check_y = robot.x + dx, robot.y + dy
                if not (0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE) or \
                   self.game.grid[check_y][check_x] in [CellType.OBSTACLE, CellType.ROBOT]:
                    surrounded_count += 1
            reward -= surrounded_count * waiting_penalty
        
        # Load balancing consideration
        avg_tasks = sum(r.completed_tasks for r in self.game.robots) / len(self.game.robots)
        if robot.completed_tasks < avg_tasks:
            reward += 15
        
        return reward
    
    def choose_action(self, robot):
        """Enhanced action selection with deadlock avoidance"""
        available_tasks = self.get_available_tasks(robot)
        if not available_tasks:
            return None
            
        state = self.get_state(robot)
        
        # Adjust exploration based on situation
        dynamic_epsilon = EPSILON
        if robot.waiting_time > 0:
            # Increase exploration when in potential deadlock
            dynamic_epsilon = min(0.9, EPSILON + (robot.waiting_time/10.0))
        elif robot.completed_tasks > 0:
            # Reduce exploration when performing well
            dynamic_epsilon = max(0.05, EPSILON - 0.05)
        
        # Exploration with smart task selection
        if random.random() < dynamic_epsilon:
            if robot.waiting_time > 0:
                # When in potential deadlock, prioritize tasks with clearer paths
                valid_tasks = []
                for task in available_tasks:
                    path = self.game.astar.find_path((robot.x, robot.y), task.get_position())
                    if path:
                        # Calculate path quality
                        obstacle_count = sum(1 for px, py in path 
                                          if any(self.game.grid[py+dy][px+dx] == CellType.OBSTACLE
                                               for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
                                               if 0 <= px+dx < GRID_SIZE and 0 <= py+dy < GRID_SIZE))
                        valid_tasks.append((task, len(path), obstacle_count))
                
                if valid_tasks:
                    # Choose task with shorter, clearer path
                    valid_tasks.sort(key=lambda x: (x[2], x[1]))
                    return valid_tasks[0][0]
            
            # Normal exploration strategy
            high_priority = [t for t in available_tasks if t.priority == 3]
            med_priority = [t for t in available_tasks if t.priority == 2]
            if high_priority and random.random() < 0.5:
                return random.choice(high_priority)
            elif med_priority and random.random() < 0.3:
                return random.choice(med_priority)
            return random.choice(available_tasks)
        
        # Exploitation with enhanced Q-value calculation
        max_q_value = float('-inf')
        best_tasks = []
        
        for task in available_tasks:
            q_value = robot.q_table[state].get(task, 0)
            
            # Adjust Q-value based on current situation
            priority_bonus = (task.priority - 1) * 0.1
            
            # Path quality bonus
            path = self.game.astar.find_path((robot.x, robot.y), task.get_position())
            path_quality = 0
            if path:
                obstacle_count = 0
                robot_count = 0
                for px, py in path:
                    for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                        check_x, check_y = px + dx, py + dy
                        if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                            if self.game.grid[check_y][check_x] == CellType.OBSTACLE:
                                obstacle_count += 1
                            elif self.game.grid[check_y][check_x] == CellType.ROBOT:
                                robot_count += 1
                path_quality = 0.2 * (1.0 - (obstacle_count + robot_count)/(len(path) * 4))
            
            adjusted_q = q_value + priority_bonus + path_quality
            
            if adjusted_q > max_q_value:
                max_q_value = adjusted_q
                best_tasks = [task]
            elif adjusted_q == max_q_value:
                best_tasks.append(task)
        
        return random.choice(best_tasks) if best_tasks else random.choice(available_tasks)
    
    def update(self, robot, old_state, action, reward, new_state):
        """Q-learning update with eligibility traces and enhanced learning"""
        if action is None:
            return
            
        # Get max Q-value for next state considering path quality
        next_q_values = []
        for task in self.get_available_tasks(robot):
            q_value = robot.q_table[new_state].get(task, 0)
            priority_bonus = (task.priority - 1) * 0.1
            
            # Add path quality consideration
            path = self.game.astar.find_path((robot.x, robot.y), task.get_position())
            path_quality = 0
            if path:
                obstacle_count = 0
                robot_count = 0
                for px, py in path:
                    for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                        check_x, check_y = px + dx, py + dy
                        if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                            if self.game.grid[check_y][check_x] == CellType.OBSTACLE:
                                obstacle_count += 1
                            elif self.game.grid[check_y][check_x] == CellType.ROBOT:
                                robot_count += 1
                path_quality = 0.2 * (1.0 - (obstacle_count + robot_count)/(len(path) * 4))
            
            next_q_values.append(q_value + priority_bonus + path_quality)
            
        max_next_q = max(next_q_values) if next_q_values else 0
        
        # Current Q-value
        current_q = robot.q_table[old_state].get(action, 0)
        
        # Enhanced learning rate based on situation
        dynamic_learning_rate = LEARNING_RATE
        if robot.waiting_time > 0:
            # Increase learning rate when in potential deadlock
            dynamic_learning_rate = min(0.5, LEARNING_RATE * (1 + robot.waiting_time/10.0))
        
        # Q-learning update with eligibility traces
        new_q = current_q + dynamic_learning_rate * (
            reward + DISCOUNT_FACTOR * max_next_q - current_q
        )
        
        # Update Q-table
        if old_state not in robot.q_table:
            robot.q_table[old_state] = {}
        robot.q_table[old_state][action] = new_q
        
        # Store last action and state for eligibility traces
        robot.last_action = action
        robot.last_state = old_state