import heapq
import math

class Node:

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b): #h in A*
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) #Manhattan distance - Perdoret kur levizim vetem horizontalisht dhe vertikalisht

def astar(maze, start, end): #f=g+h
    start_node = Node(None, start)
    end_node = Node(None, end)

    open_list = []
    closed_set = set()

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)

        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        for new_position in [
            (0, -1),  # Up
            (0, 1),   # Down
            (-1, 0),  # Left
            (1, 0)    # Right
        ]:
            node_position = (current_node.position[0] + new_position[0],
                             current_node.position[1] + new_position[1])

            if not (0 <= node_position[0] < len(maze) and 0 <= node_position[1] < len(maze[0])):
                continue

            if maze[node_position[0]][node_position[1]] != 0:
                continue

            new_node = Node(current_node, node_position)

            if new_node.position in closed_set:
                continue

            new_node.g = current_node.g + 1 # Cost deri ne kete moment
            new_node.h = heuristic(new_node.position, end_node.position) #Heuristic cost = vlera e "estimuar" deri ne target te fundit
            new_node.f = new_node.g + new_node.h #Cost totale

            if any(open_node for open_node in open_list if open_node == new_node and open_node.g <= new_node.g):
                continue

            heapq.heappush(open_list, new_node)

    return None

def main():
    maze = [
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    start = (0, 0)
    end = (8, 9)

    path = astar(maze, start, end)
    print("Path:", path)

if __name__ == "__main__":
    main()
