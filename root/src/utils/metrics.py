import time

class PerformanceMetrics:
    @staticmethod
    def calculate_metrics(game):
        end_time = time.time()
        total_time = end_time - game.start_time
        total_tasks = sum(robot.completed_tasks for robot in game.robots)
        total_distance = sum(robot.total_distance for robot in game.robots)
        
        # Calculate theoretical minimum distance and time
        theoretical_min_distance = 0
        theoretical_min_time = 0
        
        for robot in game.robots:
            completed_tasks = robot.completed_tasks
            if completed_tasks > 0:
                avg_task_distance = robot.total_distance / completed_tasks
                theoretical_min_distance += avg_task_distance * 0.7  # Assuming 30% optimization
                theoretical_min_time += (end_time - robot.start_time) * 0.7
        
        time_saved = theoretical_min_time - total_time if theoretical_min_time > total_time else 0
        distance_saved = theoretical_min_distance - total_distance if theoretical_min_distance > total_distance else 0
        
        return {
            'total_time': total_time,
            'total_tasks': total_tasks,
            'total_distance': total_distance,
            'time_saved': time_saved,
            'distance_saved': distance_saved,
            'tasks_per_second': total_tasks / total_time if total_time > 0 else 0
        } 