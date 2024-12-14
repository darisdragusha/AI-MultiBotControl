import random
from src.core.constants import EPSILON, LEARNING_RATE, DISCOUNT_FACTOR

class MADQLAgent:
    def __init__(self, game):
        self.game = game
        
    def get_state(self, robot):
        # State includes robot position and other robots' positions
        state = (robot.x, robot.y)
        if robot.target:
            state += robot.target
        return state
    
    def get_reward(self, robot, action, new_pos):
        reward = 0
        
        # Collision penalty
        for other_robot in self.game.robots:
            if other_robot != robot and (other_robot.x, other_robot.y) == new_pos:
                return -10
        
        # Target reward
        if robot.target and new_pos == robot.target:
            reward += 10
            
        # Distance-based reward
        if robot.target:
            old_dist = robot.manhattan_distance((robot.x, robot.y), robot.target)
            new_dist = robot.manhattan_distance(new_pos, robot.target)
            reward += (old_dist - new_dist)
            
        return reward
    
    def choose_action(self, robot, valid_actions):
        if random.random() < EPSILON:
            return random.choice(valid_actions)
            
        state = self.get_state(robot)
        q_values = [robot.q_table[state][a] for a in valid_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def update(self, robot, state, action, reward, new_state):
        old_value = robot.q_table[state][action]
        next_max = max([robot.q_table[new_state][a] for a in [(0,1), (0,-1), (1,0), (-1,0)]])
        robot.q_table[state][action] = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value) 