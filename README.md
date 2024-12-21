# AI-MultiBotControl
A sophisticated multi-robot control system that implements market-based task allocation and intelligent path planning in a dynamic environment. The system features a graphical interface built with Pygame, demonstrating autonomous robot coordination and task management.

## Features

- **Dynamic Task Allocation**: Market-based auction system for optimal task distribution among robots
- **Intelligent Path Planning**: A* algorithm implementation for efficient navigation
- **Collision Avoidance**: Smart collision detection and resolution between robots
- **Dynamic Environment**: 
  - Random task generation during simulation
  - Moving obstacles that robots must avoid
  - Priority-based task system (P1, P2, P3)
- **Performance Metrics**: Comprehensive metrics tracking system for evaluating efficiency
- **Reinforcement Learning Integration**: DQN-based learning for optimizing task allocation
- **Interactive GUI**: User-friendly interface for:
  - Manual robot placement
  - Obstacle positioning
  - Task creation
  - Random environment generation
  - Real-time simulation control

## Project Structure

```
src/
├── agents/             # AI and pathfinding algorithms
│   ├── astar.py       # A* pathfinding implementation
│   └── RL/            # Reinforcement learning components
├── core/              # Core game mechanics
│   ├── constants.py   # Game constants and configurations
│   ├── entities.py    # Robot and task entity definitions
│   └── game.py        # Main game logic
├── ui/                # User interface components
├── utils/             # Utility functions and metrics
└── images/            # Game assets
```

## System Components

### 1. Task Allocation System
- Market-based auction mechanism
- Priority-based task assignment
- Dynamic task generation
- Waiting time consideration

### 2. Navigation System
- A* pathfinding algorithm
- Dynamic obstacle avoidance
- Collision prevention between robots
- Path replanning capabilities

### 3. Performance Monitoring
- Time tracking
- Task completion rate
- Distance measurements


## Controls

- **Robot**: Place robots on the grid
- **Obstacle**: Add static obstacles
- **Task**: Create tasks with random priorities
- **Random Generate**: Automatically generate a random environment
- **Play**: Start/Pause the simulation
- **End**: Conclude the simulation and display metrics

## Technical Details

- Grid Size: 10x10
- Maximum Tasks: 5
- Maximum Moving Obstacles: 3
- Task Generation Rate: 5% per update
- Obstacle Generation Rate: 2% per update
- Move Delay: 0.5 seconds
- Obstacle Move Delay: 1.0 seconds

## Performance Metrics

The system tracks these metrics in real-time:
- Time: Total simulation time in seconds
- Tasks: Number of completed tasks and rate (tasks/second)
- Distance: Total distance traveled and average per task

## Requirements

- Python 3.x
- Pygame
- NumPy
- Gym (for RL components)
- Stable-Baselines3 (for DQN implementation)

## Usage

1. Clone the repository
2. Install dependencies using pip:
```bash
pip install -r requirements.txt
```
3. Run the main script:
```bash
python src/main.py
```

## Implementation Details

The system uses several sophisticated algorithms and techniques:

1. **Task Allocation**:
   - Auction-based system with bidding mechanism
   - Priority-based task assignment (P1, P2, P3)
   - Distance-based bid calculation

2. **Path Planning**:
   - A* algorithm for optimal path finding
   - Obstacle detection and avoidance
   - Path recalculation when blocked

3. **Reinforcement Learning**:
   - DQN for task allocation decisions
   - Environment based on grid positions
   - Reward system based on distance optimization

4. **Performance Tracking**:
   - Time measurement
   - Task completion monitoring
   - Distance efficiency calculation