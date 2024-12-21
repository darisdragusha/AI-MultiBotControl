import time
import numpy as np
from src.core.constants import GRID_SIZE

class PerformanceMetrics:
    @staticmethod
    def calculate_metrics(game):
        end_time = time.time()
        metrics = {}
        
        # Basic timing metrics
        total_time = end_time - game.start_time
        metrics['total_time'] = total_time
        
        # Task completion metrics
        metrics['total_tasks'] = sum(robot.completed_tasks for robot in game.robots)
        metrics['tasks_per_second'] = metrics['total_tasks'] / total_time if total_time > 0 else 0
        
       

        
        # Distance and efficiency metrics
        total_distance = sum(robot.total_distance for robot in game.robots)
        metrics['total_distance'] = total_distance
        metrics['distance_per_task'] = total_distance / metrics['total_tasks'] if metrics['total_tasks'] > 0 else 0
    

        return metrics

    @staticmethod
    def format_metrics(metrics):
        """Format metrics for display"""
        return [
            f"Time: {metrics['total_time']:.1f}s",
            f"Tasks: {metrics['total_tasks']} ({metrics['tasks_per_second']:.2f}/s)",
            f"Distance: {metrics['total_distance']} ({metrics['distance_per_task']:.1f}/task)",
        ] 