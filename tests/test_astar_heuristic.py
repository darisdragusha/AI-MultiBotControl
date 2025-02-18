
import unittest
from src.agents.astar import AStar

class TestAStarHeuristic(unittest.TestCase):

    def setUp(self):
        """Create an instance of AStar to test the heuristic function in isolation."""
        self.astar = AStar(None)  # No grid needed to test heuristic

    def test_heuristic_normal_case(self):
        """Test heuristic with normal values for (0, 0) to (3, 4)."""
        start = (0, 0)
        goal = (3, 4)
        expected_distance = 7  # 3 + 4
        result = self.astar.heuristic(start, goal)
        self.assertEqual(result, expected_distance)

    