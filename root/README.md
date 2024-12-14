# Multi-Robot Control System

A multi-robot task allocation and pathfinding system using MADQL (Multi-Agent Deep Q-Learning) and A* pathfinding algorithms.

## Features

- Multi-robot task allocation using MADQL
- Optimal pathfinding using A* algorithm
- Real-time collision avoidance
- Dynamic task generation
- Interactive grid-based environment
- Performance metrics and visualization
- Status logging of robot decisions

## Requirements

- Python 3.8+
- Pygame
- Numpy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:
```bash
python -m src.main
```

### Controls

- Click "Robot" and then click on the grid to place robots
- Click "Obstacle" and then click on the grid to place obstacles
- Click "Task" and then click on the grid to place tasks
- Click "Random Generate" to randomly generate robots and obstacles
- Click "Play" to start/pause the simulation
- Click "End" to stop new task generation and finish current tasks

### Visual Elements

- Blue circles: Robots
- Red squares: Obstacles
- Purple squares: Unassigned tasks
- Green squares: Assigned tasks (targets)
- Yellow lines: Planned paths
- Status panel: Shows real-time decision making
- Performance metrics: Shows optimization results

## Project Structure

```
src/
├── agents/
│   ├── astar.py         # A* pathfinding implementation
│   └── madql_agent.py   # MADQL learning agent
├── core/
│   ├── constants.py     # Game constants and parameters
│   ├── entities.py      # Robot and CellType classes
│   └── game.py         # Main game logic
├── ui/
│   └── button.py       # UI button implementation
├── utils/
│   └── metrics.py      # Performance metrics calculation
└── main.py            # Entry point

```

## Algorithm Details

### MADQL (Multi-Agent Deep Q-Learning)
- Handles task allocation between robots
- Learns optimal task assignments
- Provides rewards for successful task completion
- Penalizes collisions and inefficient paths

### A* Pathfinding
- Finds optimal paths avoiding obstacles
- Dynamically replans when collisions are detected
- Uses Manhattan distance heuristic

### Collision Avoidance
- Real-time collision detection
- Path replanning for deadlock resolution
- Priority-based movement 