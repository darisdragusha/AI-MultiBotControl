import pygame
import random
import time
from src.core.constants import *
from src.core.entities import Robot, CellType, Task
from src.agents.madql_agent import MADQLAgent
from src.agents.astar import AStar
from src.ui.button import Button
from src.utils.metrics import PerformanceMetrics

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_HEIGHT))
        pygame.display.set_caption("Multi-Robot Control System")
        
        # Adjust grid size
        self.grid = [[CellType.EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.current_tool = None
        self.robots = []
        self.tasks = []
        self.moving_obstacles = []
        self.dynamic_tasks_enabled = True
        self.end_simulation = False
        self.start_time = None
        self.total_tasks_completed = 0
        self.auction_interval = 2.0
        self.last_auction_time = 0
        
        self.madql = MADQLAgent(self)
        self.astar = AStar(self)
        self.clock = pygame.time.Clock()
        
        # Create buttons
        self.buttons = {
            'robot': Button(TOTAL_WIDTH - MENU_WIDTH + 20, 50, 160, 40, "Robot"),
            'obstacle': Button(TOTAL_WIDTH - MENU_WIDTH + 20, 100, 160, 40, "Obstacle"),
            'task': Button(TOTAL_WIDTH - MENU_WIDTH + 20, 150, 160, 40, "Task"),
            'random': Button(TOTAL_WIDTH - MENU_WIDTH + 20, 200, 160, 40, "Random Generate"),
            'play': Button(TOTAL_WIDTH - MENU_WIDTH + 20, 250, 160, 40, "Play"),
            'end': Button(TOTAL_WIDTH - MENU_WIDTH + 20, 300, 160, 40, "End")
        }
        
        self.running = True
        self.simulation_running = False
        self.performance_metrics = None
        self.status_messages = []
        self.max_messages = 8
        self.robot_counter = 0

    def add_status_message(self, message):
        self.status_messages.insert(0, message)
        if len(self.status_messages) > self.max_messages:
            self.status_messages.pop()

    def handle_click(self, pos):
        # Handle menu clicks
        for name, button in self.buttons.items():
            if button.rect.collidepoint(pos):
                if name == 'random':
                    self.generate_random()
                elif name == 'play':
                    self.simulation_running = not self.simulation_running
                    if self.simulation_running and self.start_time is None:
                        self.start_time = time.time()
                        for robot in self.robots:
                            robot.start_time = time.time()
                        # Reallocate tasks when simulation starts
                        self.reallocate_all_tasks()
                elif name == 'end':
                    self.end_simulation = True
                    self.dynamic_tasks_enabled = False
                else:
                    # Deselect all buttons except the clicked one
                    for btn in self.buttons.values():
                        btn.selected = False
                    button.selected = True
                    self.current_tool = name
                return

        # Handle grid clicks
        if pos[0] < WINDOW_SIZE:
            grid_x = pos[0] // CELL_SIZE
            grid_y = pos[1] // CELL_SIZE
            
            if self.current_tool == 'robot':
                if self.grid[grid_y][grid_x] == CellType.EMPTY:
                    self.grid[grid_y][grid_x] = CellType.ROBOT
                    new_robot = Robot(grid_x, grid_y)
                    self.robot_counter += 1
                    new_robot.id = self.robot_counter
                    self.robots.append(new_robot)
                    self.add_status_message(f"Robot {new_robot.id} placed at ({grid_x}, {grid_y})")
                    # Reallocate tasks when new robot is added
                    if self.simulation_running:
                        self.reallocate_all_tasks()
            elif self.current_tool == 'obstacle':
                if self.grid[grid_y][grid_x] not in [CellType.ROBOT, CellType.TARGET]:
                    self.grid[grid_y][grid_x] = CellType.OBSTACLE
            elif self.current_tool == 'task':
                if self.grid[grid_y][grid_x] == CellType.EMPTY:
                    # Create task with random priority
                    priority = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
                    new_task = Task(grid_x, grid_y, priority)
                    self.grid[grid_y][grid_x] = CellType.TASK
                    self.tasks.append(new_task)
                    self.add_status_message(f"Created P{priority} task at ({grid_x}, {grid_y})")
                    if self.simulation_running:
                        self.assign_tasks()

    def reallocate_all_tasks(self):
        """Reallocate all tasks among all robots for optimal distribution"""
        # Clear all current assignments
        for robot in self.robots:
            if robot.target:
                target = robot.target
                # Create a new task with the same priority
                new_task = Task(target.x, target.y, target.priority)
                self.tasks.append(new_task)
                self.grid[target.y][target.x] = CellType.TASK
            robot.target = None
            robot.path = []
        
        # Create a list of all tasks (both assigned and unassigned)
        all_tasks = []
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.grid[y][x] in [CellType.TASK, CellType.TARGET]:
                    # For existing tasks without priority info, assign random priority
                    priority = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
                    new_task = Task(x, y, priority)
                    all_tasks.append(new_task)
                    self.grid[y][x] = CellType.TASK
        
        self.tasks = all_tasks
        self.add_status_message("Reallocating all tasks for optimal distribution")
        self.assign_tasks()

    def assign_tasks(self):
        """Assign tasks to robots using MADQL"""
        unassigned_robots = [robot for robot in self.robots if not robot.target]
        
        for robot in unassigned_robots:
            # Get current state
            old_state = self.madql.get_state(robot)
            
            # Choose task using MADQL
            chosen_task = self.madql.choose_action(robot)
            
            if chosen_task:
                # Mark task as assigned
                self.grid[chosen_task.y][chosen_task.x] = CellType.TARGET
                robot.set_target(chosen_task)
                self.tasks.remove(chosen_task)
                
                # Get new state and reward
                new_state = self.madql.get_state(robot)
                reward = self.madql.get_reward(robot, old_state, chosen_task, new_state)
                
                # Update Q-values
                self.madql.update(robot, old_state, chosen_task, reward, new_state)
                
                self.add_status_message(
                    f"Robot {robot.id} assigned to P{chosen_task.priority} task at ({chosen_task.x}, {chosen_task.y}) [R: {reward:.1f}]"
                )

    def generate_random_task(self):
        if len(self.tasks) < MAX_TASKS and random.random() < TASK_GEN_CHANCE:
            empty_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
                          if self.grid[y][x] == CellType.EMPTY]
            if empty_cells:
                x, y = random.choice(empty_cells)
                # Assign random priority with weighted probability
                priority = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
                task = Task(x, y, priority)
                self.grid[y][x] = CellType.TASK
                self.tasks.append(task)
                self.add_status_message(f"Generated new task at ({x}, {y}) with priority {priority}")

    def generate_moving_obstacle(self):
        """Generate a moving obstacle with random direction"""
        if len(self.moving_obstacles) < MAX_MOVING_OBSTACLES and random.random() < OBSTACLE_GEN_CHANCE:
            # Find empty spot
            empty_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
                          if self.grid[y][x] == CellType.EMPTY]
            if empty_cells:
                x, y = random.choice(empty_cells)
                direction = random.choice([(0,1), (1,0), (0,-1), (-1,0)])
                self.moving_obstacles.append({
                    'x': x, 'y': y,
                    'dx': direction[0], 'dy': direction[1],
                    'last_move': time.time()
                })
                self.grid[y][x] = CellType.OBSTACLE
                self.add_status_message(f"Added moving obstacle at ({x}, {y})")

    def update_moving_obstacles(self, current_time):
        """Update positions of moving obstacles"""
        for obstacle in self.moving_obstacles[:]:
            if current_time - obstacle['last_move'] < OBSTACLE_MOVE_DELAY:
                continue

            # Calculate new position
            new_x = obstacle['x'] + obstacle['dx']
            new_y = obstacle['y'] + obstacle['dy']

            # Check if new position is valid
            if (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and 
                self.grid[new_y][new_x] == CellType.EMPTY):
                # Update grid
                self.grid[obstacle['y']][obstacle['x']] = CellType.EMPTY
                self.grid[new_y][new_x] = CellType.OBSTACLE
                obstacle['x'] = new_x
                obstacle['y'] = new_y
                obstacle['last_move'] = current_time
            else:
                # Change direction if blocked
                obstacle['dx'], obstacle['dy'] = random.choice([(0,1), (1,0), (0,-1), (-1,0)])

    def auction_tasks(self):
        """Market-based task allocation using auction mechanism"""
        if not self.tasks:
            return

        current_time = time.time()
        if current_time - self.last_auction_time < self.auction_interval:
            return

        self.last_auction_time = current_time
        unassigned_robots = [robot for robot in self.robots if not robot.target]
        if not unassigned_robots:
            return

        # Calculate bids for each robot-task pair
        bids = []
        for robot in unassigned_robots:
            robot_state = self.madql.get_state(robot)
            for task in self.tasks:
                # Base bid on Q-value
                q_value = robot.q_table[robot_state].get(task, 0)
                
                # Adjust bid based on various factors
                distance = robot.manhattan_distance((robot.x, robot.y), task.get_position())
                path = self.astar.find_path((robot.x, robot.y), task.get_position())
                
                # Calculate path congestion
                congestion = 0
                if path:
                    for px, py in path:
                        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                            check_x, check_y = px + dx, py + dy
                            if 0 <= check_x < GRID_SIZE and 0 <= check_y < GRID_SIZE:
                                if self.grid[check_y][check_x] in [CellType.ROBOT, CellType.OBSTACLE]:
                                    congestion += 1

                # Calculate bid value
                bid_value = (
                    q_value * 2 +                    # Learning component
                    task.priority * 30 +             # Priority bonus
                    (1.0 - distance/GRID_SIZE) * 20 + # Distance factor
                    (1.0 - congestion/len(path) if path else 0) * 10 + # Congestion factor
                    task.get_waiting_time() * 5      # Waiting time bonus
                )
                
                bids.append((robot, task, bid_value))

        # Sort bids by value
        bids.sort(key=lambda x: x[2], reverse=True)

        # Assign tasks to highest bidders
        assigned_tasks = set()
        assigned_robots = set()

        for robot, task, bid_value in bids:
            if (robot not in assigned_robots and 
                task not in assigned_tasks and 
                task in self.tasks):  # Check if task still available
                
                # Assign task
                self.grid[task.y][task.x] = CellType.TARGET
                robot.set_target(task)
                self.tasks.remove(task)
                assigned_tasks.add(task)
                assigned_robots.add(robot)
                
                self.add_status_message(
                    f"Auction: Robot {robot.id} won P{task.priority} task with bid {bid_value:.1f}"
                )

    def update_simulation(self):
        if not self.simulation_running:
            return
            
        current_time = time.time()
        
        # Update dynamic environment
        if self.dynamic_tasks_enabled:
            self.generate_random_task()
            self.generate_moving_obstacle()
        
        self.update_moving_obstacles(current_time)
        
        # Update waiting times for all robots
        for robot in self.robots:
            robot.update_waiting(robot.waiting, current_time)
            
        # Run auction-based task allocation
        self.auction_tasks()
        
        # Also use MADQL for learning and improvement
        self.assign_tasks()
            
        # First, reset waiting status for all robots at the start of each update
        for robot in self.robots:
            robot.waiting = False
            
        # Sort robots by priority of their tasks and waiting time
        active_robots = [r for r in self.robots if r.target]
        active_robots.sort(key=lambda r: (
            r.target.priority if r.target else 0,
            -r.waiting_time,  # Negative so longer waiting time gets priority
            r.manhattan_distance((r.x, r.y), r.target.get_position()) if r.target else float('inf')
        ), reverse=True)
            
        # Process all robots, not just active ones
        for robot in self.robots:
            if current_time - robot.last_move_time < MOVE_DELAY:
                continue  # Skip if not enough time has passed since last move
                
            # Check if path is still valid
            if robot.path:
                path_invalid = False
                for pos in robot.path:
                    if self.grid[pos[1]][pos[0]] == CellType.OBSTACLE:
                        path_invalid = True
                        break
                if path_invalid:
                    robot.path = []
                    self.add_status_message(f"Robot {robot.id}: Replanning due to obstacle")

            if not robot.path:
                if robot.target:
                    robot.path = self.astar.find_path(
                        (robot.x, robot.y),
                        robot.target.get_position()
                    )
                    if robot.path:
                        path_length = len(robot.path)
                        self.add_status_message(
                            f"Robot {robot.id}: Found path to P{robot.target.priority} task, length {path_length}"
                        )
                    else:
                        self.add_status_message(
                            f"Robot {robot.id}: No path to P{robot.target.priority} task"
                        )
                        # If no path found, clear target and try another task
                        robot.target = None
                        continue
                    robot.path.pop(0)  # Remove current position

            if robot.path:
                next_pos = robot.path[0]
                
                # Check for collision
                collision = False
                colliding_robot = None
                for other_robot in self.robots:
                    if other_robot != robot:
                        # Check current position
                        if (other_robot.x, other_robot.y) == next_pos:
                            collision = True
                            colliding_robot = other_robot
                            break
                        # Check if other robot is planning to move to our next position
                        elif (other_robot.path and 
                              other_robot.path[0] == next_pos):
                            # Consider task priorities in collision resolution
                            if robot.target and other_robot.target:
                                if robot.target.priority < other_robot.target.priority:
                                    collision = True
                                    colliding_robot = other_robot
                                    break
                                elif robot.target.priority == other_robot.target.priority:
                                    # If same priority, consider waiting time and distance
                                    if robot.waiting_time < other_robot.waiting_time:
                                        collision = True
                                        colliding_robot = other_robot
                                        break
                                    elif robot.waiting_time == other_robot.waiting_time:
                                        # If same waiting time, let robot closer to target proceed
                                        if (other_robot.target and
                                            other_robot.manhattan_distance((other_robot.x, other_robot.y), other_robot.target.get_position()) < 
                                            robot.manhattan_distance((robot.x, robot.y), robot.target.get_position())):
                                            collision = True
                                            colliding_robot = other_robot
                                            break

                if not collision:
                    # Get old state before moving
                    old_state = self.madql.get_state(robot)
                    
                    # Update grid and robot position
                    self.grid[robot.y][robot.x] = CellType.EMPTY
                    old_pos = (robot.x, robot.y)
                    robot.x, robot.y = next_pos
                    self.grid[robot.y][robot.x] = CellType.ROBOT
                    robot.path.pop(0)
                    robot.last_move_time = current_time
                    robot.total_distance += 1
                    robot.waiting = False
                    
                    # Get new state and update Q-values
                    new_state = self.madql.get_state(robot)
                    reward = self.madql.get_reward(robot, old_state, next_pos, new_state)
                    self.madql.update(robot, old_state, next_pos, reward, new_state)
                    
                    # Check if reached target
                    if robot.target and (robot.x, robot.y) == robot.target.get_position():
                        completed_priority = robot.target.priority
                        robot.target = None
                        robot.completed_tasks += 1
                        self.total_tasks_completed += 1
                        self.add_status_message(
                            f"Robot {robot.id}: Completed P{completed_priority} task! Total: {robot.completed_tasks}"
                        )
                else:
                    robot.waiting = True
                    if colliding_robot:
                        # If both robots are waiting too long, force one to find alternative path
                        if (robot.waiting and colliding_robot.waiting and 
                            current_time - robot.last_move_time > MOVE_DELAY * 3):
                            # Robot with lower priority task should find alternative
                            if (robot.target and colliding_robot.target and 
                                robot.target.priority <= colliding_robot.target.priority):
                                robot.path = []  # Force path recalculation
                                self.add_status_message(
                                    f"Robot {robot.id}: Finding alternative path (lower priority)"
                                )
                            elif robot.waiting_time > colliding_robot.waiting_time:
                                robot.path = []  # Force path recalculation
                                self.add_status_message(
                                    f"Robot {robot.id}: Finding alternative path (waited longer)"
                                )
                        else:
                            self.add_status_message(
                                f"Robot {robot.id}: Waiting for Robot {colliding_robot.id} to move"
                            )
        
        # Check if simulation should end
        if self.end_simulation and not any(robot.target for robot in self.robots):
            self.simulation_running = False
            self.performance_metrics = PerformanceMetrics.calculate_metrics(self)

    def generate_random(self):
        # Clear the grid and robots
        self.grid = [[CellType.EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.robots = []
        self.robot_counter = 0
        
        # Add random robots (2-3)
        num_robots = random.randint(2, 3)
        for _ in range(num_robots):
            while True:
                x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
                if self.grid[y][x] == CellType.EMPTY:
                    self.grid[y][x] = CellType.ROBOT
                    new_robot = Robot(x, y)
                    self.robot_counter += 1
                    new_robot.id = self.robot_counter
                    self.robots.append(new_robot)
                    break
        
        # Add random obstacles (5-8)
        num_obstacles = random.randint(5, 8)
        for _ in range(num_obstacles):
            while True:
                x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
                if self.grid[y][x] == CellType.EMPTY:
                    self.grid[y][x] = CellType.OBSTACLE
                    break

    def draw(self):
        self.screen.fill(WHITE)
        
        # Draw grid (adjusted for smaller grid size)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, WHITE, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 1)
                
                if self.grid[y][x] == CellType.ROBOT:
                    self.screen.blit(robot_image, (x * CELL_SIZE, y * CELL_SIZE))
                elif self.grid[y][x] == CellType.OBSTACLE:
                    self.screen.blit(obstacle_image, (x * CELL_SIZE, y * CELL_SIZE))

                elif self.grid[y][x] == CellType.TARGET:
                    pygame.draw.rect(self.screen, GREEN,
                                   (x * CELL_SIZE + 5, y * CELL_SIZE + 5,
                                    CELL_SIZE - 10, CELL_SIZE - 10))
                elif self.grid[y][x] == CellType.TASK:
                    pygame.draw.rect(self.screen, PURPLE,
                                   (x * CELL_SIZE + 5, y * CELL_SIZE + 5,
                                    CELL_SIZE - 10, CELL_SIZE - 10))
        
        # Draw paths for robots
        for robot in self.robots:
            if robot.path:
                points = [(p[0] * CELL_SIZE + CELL_SIZE//2, 
                          p[1] * CELL_SIZE + CELL_SIZE//2) for p in [(robot.x, robot.y)] + robot.path]
                pygame.draw.lines(self.screen, YELLOW, False, points, 2)
        
        # Draw menu background
        pygame.draw.rect(self.screen, WHITE, (WINDOW_SIZE, 0, MENU_WIDTH, WINDOW_SIZE))
        pygame.draw.line(self.screen, BLACK, (WINDOW_SIZE, 0), (WINDOW_SIZE, WINDOW_SIZE), 2)
        
        # Draw buttons
        for button in self.buttons.values():
            button.draw(self.screen)
        
        # Draw performance metrics if available
        if self.performance_metrics:
            font = pygame.font.Font(None, 24)
            metrics_text = [
                f"Time: {self.performance_metrics['total_time']:.1f}s",
                f"Tasks: {self.performance_metrics['total_tasks']}",
                f"Distance: {self.performance_metrics['total_distance']}",
                f"Time Saved: {self.performance_metrics['time_saved']:.1f}s",
                f"Distance Saved: {self.performance_metrics['distance_saved']:.1f}",
                f"Tasks/s: {self.performance_metrics['tasks_per_second']:.2f}"
            ]
            
            y_offset = 350
            for text in metrics_text:
                text_surface = font.render(text, True, BLACK)
                self.screen.blit(text_surface, (WINDOW_SIZE + 20, y_offset))
                y_offset += 25
        
        # Draw status messages at the bottom
        if self.simulation_running or self.performance_metrics:
            font = pygame.font.Font(None, 24)
            y_offset = GRID_SIZE * CELL_SIZE + 10  # Adjust for new grid size

            # Draw status panel background for logs
            status_panel = pygame.Rect(10, GRID_SIZE * CELL_SIZE, TOTAL_WIDTH - 20, LOG_HEIGHT - 10)
            pygame.draw.rect(self.screen, (240, 240, 240), status_panel, border_radius=10)
            pygame.draw.rect(self.screen, BLACK, status_panel, 2, border_radius=10)

            # Add shadow effect
            shadow_panel = pygame.Rect(12, GRID_SIZE * CELL_SIZE + 2, TOTAL_WIDTH - 20, LOG_HEIGHT - 10)
            pygame.draw.rect(self.screen, (200, 200, 200), shadow_panel, border_radius=10)

            # Draw title
            title = font.render("Status Log:", True, BLACK)
            self.screen.blit(title, (20, y_offset + 10))

            # Start logs below the title
            y_offset += 35

            # Draw messages within log area
            for message in self.status_messages[-self.max_messages:]:
                words = message.split()
                line = []
                for word in words:
                    if font.size(' '.join(line + [word]))[0] <= TOTAL_WIDTH - 40:
                        line.append(word)
                    else:
                        text_surface = font.render(' '.join(line), True, BLACK)
                        self.screen.blit(text_surface, (20, y_offset))
                        y_offset += 20

                        # Stop drawing if log exceeds the panel height
                        if y_offset > GRID_SIZE * CELL_SIZE + LOG_HEIGHT - 20:
                            break

                        line = [word]
                text_surface = font.render(' '.join(line), True, BLACK)
                self.screen.blit(text_surface, (20, y_offset))
                y_offset += 20

                # Stop drawing if log exceeds the panel height
                if y_offset > GRID_SIZE * CELL_SIZE + LOG_HEIGHT - 20:
                    break

        
        pygame.display.flip()

    def run(self):
        while self.running:
            self.clock.tick(10)  # Limit to 10 FPS for better visualization
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
            
            if self.simulation_running:
                self.update_simulation()
            
            self.draw()

        pygame.quit() 