import heapq
import time
import numpy as np
from collections import deque

class SearchAlgorithm:
    def __init__(self, graph):
        self.graph = graph
        self.path = []
        self.visited_nodes = 0
        self.execution_time = 0
        self.max_memory = 0

    def get_path(self, start, goal, came_from):
        """Reconstruct path from start to goal using came_from dictionary"""
        if goal not in came_from and start != goal:
            return []
        
        path = [goal]
        current = goal
        
        while current != start:
            current = came_from[current]
            path.append(current)
        
        return list(reversed(path))

    def get_stats(self):
        """Return statistics about the search"""
        return {
            'path_length': len(self.path) - 1 if self.path else 0,
            'path': self.path,
            'visited_nodes': self.visited_nodes,
            'execution_time': self.execution_time,
            'max_memory': self.max_memory
        }


class BFS(SearchAlgorithm):
    def search(self, start, goal):
        """Breadth-First Search algorithm"""
        start_time = time.time()
        
        if start == goal:
            self.path = [start]
            self.execution_time = time.time() - start_time
            return self.path
        
        queue = deque([(start, [start])])
        visited = set([start])
        self.visited_nodes = 0
        max_queue_size = 1
        
        while queue:
            max_queue_size = max(max_queue_size, len(queue))
            current, path = queue.popleft()
            self.visited_nodes += 1
            
            for neighbor in self.graph.get_neighbors(current):
                if neighbor not in visited:
                    if neighbor == goal:
                        self.path = path + [neighbor]
                        self.execution_time = time.time() - start_time
                        self.max_memory = max_queue_size
                        return self.path
                    
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        self.path = []
        self.execution_time = time.time() - start_time
        self.max_memory = max_queue_size
        return self.path


class DFS(SearchAlgorithm):
    def search(self, start, goal):
        """Depth-First Search algorithm"""
        start_time = time.time()
        
        if start == goal:
            self.path = [start]
            self.execution_time = time.time() - start_time
            return self.path
        
        stack = [(start, [start])]
        visited = set([start])
        self.visited_nodes = 0
        max_stack_size = 1
        
        while stack:
            max_stack_size = max(max_stack_size, len(stack))
            current, path = stack.pop()
            self.visited_nodes += 1
            
            for neighbor in self.graph.get_neighbors(current):
                if neighbor not in visited:
                    if neighbor == goal:
                        self.path = path + [neighbor]
                        self.execution_time = time.time() - start_time
                        self.max_memory = max_stack_size
                        return self.path
                    
                    visited.add(neighbor)
                    stack.append((neighbor, path + [neighbor]))
        
        self.path = []
        self.execution_time = time.time() - start_time
        self.max_memory = max_stack_size
        return self.path


class GreedyBestFirstSearch(SearchAlgorithm):
    def search(self, start, goal):
        """Greedy Best-First Search algorithm"""
        start_time = time.time()
        
        if start == goal:
            self.path = [start]
            self.execution_time = time.time() - start_time
            return self.path
        
        # Priority queue with heuristic values
        open_set = [(self.graph.heuristic(start, goal), start)]
        heapq.heapify(open_set)
        
        # Keep track of visited nodes and paths
        came_from = {}
        visited = set([start])
        self.visited_nodes = 0
        max_queue_size = 1
        
        while open_set:
            max_queue_size = max(max_queue_size, len(open_set))
            _, current = heapq.heappop(open_set)
            self.visited_nodes += 1
            
            if current == goal:
                self.path = self.get_path(start, goal, came_from)
                self.execution_time = time.time() - start_time
                self.max_memory = max_queue_size
                return self.path
            
            for neighbor in self.graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (self.graph.heuristic(neighbor, goal), neighbor))
        
        self.path = []
        self.execution_time = time.time() - start_time
        self.max_memory = max_queue_size
        return self.path


class AStar(SearchAlgorithm):
    def search(self, start, goal):
        """A* Search algorithm"""
        start_time = time.time()
        
        if start == goal:
            self.path = [start]
            self.visited_nodes = 1  # Path length (1) + 0
            self.execution_time = time.time() - start_time
            return self.path
        
        # Priority queue with f(n) = g(n) + h(n)
        open_set = [(self.graph.heuristic(start, goal), 0, start)]
        heapq.heapify(open_set)
        
        # For node n, g_score[n] is the cost of the cheapest path from start to n currently known
        g_score = {start: 0}
        
        # For node n, f_score[n] = g_score[n] + h(n)
        f_score = {start: self.graph.heuristic(start, goal)}
        
        # Keep track of visited nodes and paths
        came_from = {}
        closed_set = set()
        self.visited_nodes = 0  # Will be set to path length + 1 when path is found
        max_queue_size = 1
        
        while open_set:
            max_queue_size = max(max_queue_size, len(open_set))
            _, _, current = heapq.heappop(open_set)
            
            if current == goal:
                self.path = self.get_path(start, goal, came_from)
                self.visited_nodes = len(self.path)  # Path length + 1 (since len(path) already includes both start and goal)
                self.execution_time = time.time() - start_time
                self.max_memory = max_queue_size
                return self.path
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            for neighbor in self.graph.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # tentative_g_score is the distance from start to the neighbor through current
                tentative_g_score = g_score[current] + self.graph.get_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.graph.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], neighbor))
        
        self.path = []
        self.visited_nodes = 0  # No path found
        self.execution_time = time.time() - start_time
        self.max_memory = max_queue_size
        return self.path 