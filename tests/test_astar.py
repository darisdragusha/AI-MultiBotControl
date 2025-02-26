import unittest
from src.agents.astar import AStar
from src.core.entities import CellType

class MockGame:
    """A mock game class (stub) to simulate the grid and robots."""
    def __init__(self, grid, robots=None):
        self.grid = grid
        self.robots = robots or []

class TestAStarPathfinding(unittest.TestCase):

    def test_astar_straight_line_case(self):
        """Test a straight-line path from (2, 1) to (2, 7) in a 10x10 grid."""
        grid = [[0] * 10 for _ in range(10)]  # 10x10 clear grid
        game = MockGame(grid)
        astar = AStar(game)
        start = (2, 1)
        goal = (2, 7)

        # Expected straight-line path (moving vertically)
        expected_path = [
            (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)
        ]
        
        result = astar.find_path(start, goal)
        self.assertEqual(result, expected_path)

    def test_astar_multiple_paths_case(self):
        """Test a longer path in a 10x10 grid with multiple valid routes."""
        grid = [[0] * 10 for _ in range(10)]  # 10x10 clear grid
        game = MockGame(grid)
        astar = AStar(game)
        start = (0, 0)
        goal = (9, 9)

        # Expected longer paths with possible variations
        expected_paths = [
            # Path 1: Going right first, then down
            [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), 
             (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9)],

            # Path 2: Going down first, then moving right
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),
             (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9)],

            # Path 3: "Diagonal" and zig-zag down and right
            [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (3, 2), (4, 2), (5, 2), (5, 3), (6, 3),
             (7, 3), (7, 4), (8, 4), (8, 5), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9)],
        ]

        result = astar.find_path(start, goal)
        self.assertTrue(result in expected_paths)

    

    def test_astar_no_path_case(self):
        """Test a case where the goal is completely blocked off."""
        grid = [[0] * 10 for _ in range(10)]
        # Completely block the row just before the goal
        for i in range(10):
            grid[8][i] = CellType.OBSTACLE

        game = MockGame(grid)
        astar = AStar(game)
        start = (0, 0)
        goal = (9, 9)

        result = astar.find_path(start, goal)
        self.assertEqual(result, [])  # Expect an empty path since the goal is unreachable


    def test_astar_same_start_and_goal_case(self):
            """Test the algorithm when the start and goal are at the same place"""
            grid = [[0] * 10 for _ in range(10)]  # 10x10 clear grid
            game = MockGame(grid)
            astar = AStar(game)
            start = (4, 4)
            goal = (4, 4)

            # Expected path is empty (robot doesnt have to move)
            expected_path = []
            
            result = astar.find_path(start, goal)
            self.assertEqual(result, expected_path)
if __name__ == '__main__':
    unittest.main()