from queue import PriorityQueue
import numpy as np

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

class ThetaStar:
    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def plan(self, start, end, grid, occupations, penalties):
        start_node = Node(None, start)
        end_node = Node(None, end)

        open_list = PriorityQueue()
        closed_list = dict()  

        open_list.put((start_node.f, start_node))
        closed_list[start_node.position] = start_node

        while not open_list.empty():
            current_node = open_list.get()[1]

            if current_node.position == end_node.position:
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1]

            neighbors = [(0,1), (1,0), (0,-1), (-1,0),
                        (1,1), (1,-1), (-1,1), (-1,-1)]

            for new_position in neighbors:
                node_position = (
                    current_node.position[0] + new_position[0],
                    current_node.position[1] + new_position[1]
                )

                if node_position[0] < 0 or node_position[0] >= grid.shape[0]:
                    continue
                if node_position[1] < 0 or node_position[1] >= grid.shape[1]:
                    continue
                if grid[node_position[0]][node_position[1]] == 1:
                    continue
                if current_node.parent and self.line_of_sight(current_node.parent.position, node_position, grid):
                    new_g = current_node.parent.g + self.heuristic(current_node.parent.position, node_position)
                    tentative_node = Node(current_node.parent, node_position)
                else:
                    new_g = current_node.g + self.heuristic(current_node.position, node_position)
                    tentative_node = Node(current_node, node_position)

                base_cost = 1.0
                dynamic_cost = occupations[node_position[0]][node_position[1]]
                penalty = penalties[node_position[0]][node_position[1]]
                new_g += base_cost + 2 * dynamic_cost * penalty
                # if dynamic_cost > 0:
                #     print("new_g += base_cost + 2 * dynamic_cost * penalty")
                #     print(f"{new_g} += {base_cost} + 2* {dynamic_cost} * {penalty}")

                if node_position in closed_list:
                    existing_node = closed_list[node_position]
                    if new_g >= existing_node.g:
                        continue
                    closed_list.pop(node_position)

                tentative_node.g = new_g
                tentative_node.h = self.heuristic(tentative_node.position, end_node.position)
                tentative_node.f = tentative_node.g + tentative_node.h

                open_list.put((tentative_node.f, tentative_node))
                closed_list[tentative_node.position] = tentative_node

        return None

    def line_of_sight(self, start, end, grid):
        x0, y0 = start
        x1, y1 = end
        
        if x0 < 0 or x0 >= grid.shape[0] or y0 < 0 or y0 >= grid.shape[1]:
            return False
        if x1 < 0 or x1 >= grid.shape[0] or y1 < 0 or y1 >= grid.shape[1]:
            return False
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if grid[x0][y0] == 1:
                return False
                
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return True

    def add_to_open(self, open_list, neighbor):
        for item in open_list.queue:
            if neighbor == item[1] and neighbor.g >= item[1].g:
                return False
        return True