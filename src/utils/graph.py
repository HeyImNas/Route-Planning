import math
import numpy as np
import networkx as nx
import string

class Graph:
    def __init__(self):
        """Initialize an empty graph"""
        self.nodes = {}  # Dictionary to store nodes and their coordinates
        self.edges = {}  # Dictionary to store edges and their costs
        self.nx_graph = nx.Graph()  # NetworkX graph for visualization and some operations

    def add_node(self, node_id, x=0, y=0, **attributes):
        """Add a node to the graph with coordinates and optional attributes"""
        self.nodes[node_id] = {'x': x, 'y': y, **attributes}
        self.edges[node_id] = {}
        self.nx_graph.add_node(node_id, pos=(x, y), **attributes)
        return self

    def add_edge(self, node1, node2, cost=None, bidirectional=True):
        """Add an edge between two nodes with optional cost"""
        # If nodes don't exist, add them
        if node1 not in self.nodes:
            self.add_node(node1)
        if node2 not in self.nodes:
            self.add_node(node2)
        
        # Calculate Euclidean distance as cost if not provided
        if cost is None:
            x1, y1 = self.nodes[node1]['x'], self.nodes[node1]['y']
            x2, y2 = self.nodes[node2]['x'], self.nodes[node2]['y']
            cost = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Add edge to our adjacency list
        self.edges[node1][node2] = cost
        self.nx_graph.add_edge(node1, node2, weight=cost)
        
        # If bidirectional, add reverse edge
        if bidirectional:
            self.edges[node2][node1] = cost
        
        return self

    def get_neighbors(self, node):
        """Get all neighbors of a node"""
        if node in self.edges:
            return list(self.edges[node].keys())
        return []

    def get_cost(self, node1, node2):
        """Get the cost between two adjacent nodes"""
        if node1 in self.edges and node2 in self.edges[node1]:
            return self.edges[node1][node2]
        return float('inf')
    
    def heuristic(self, node1, node2):
        """Calculate heuristic (Euclidean distance) between two nodes"""
        if node1 in self.nodes and node2 in self.nodes:
            x1, y1 = self.nodes[node1]['x'], self.nodes[node1]['y']
            x2, y2 = self.nodes[node2]['x'], self.nodes[node2]['y']
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return 0
    
    def get_node_coordinates(self, node):
        """Get the (x,y) coordinates of a node"""
        if node in self.nodes:
            return (self.nodes[node]['x'], self.nodes[node]['y'])
        return None
    
    def get_all_nodes(self):
        """Get all nodes in the graph"""
        return list(self.nodes.keys())
    
    def get_edge_count(self):
        """Get the number of edges in the graph"""
        count = sum(len(neighbors) for neighbors in self.edges.values())
        # If graph is undirected, each edge is counted twice
        return count // 2
    
    def get_node_count(self):
        """Get the number of nodes in the graph"""
        return len(self.nodes)
    
    def get_networkx_graph(self):
        """Get the NetworkX graph representation for visualization"""
        return self.nx_graph
    
    def get_node_positions(self):
        """Get positions dictionary for NetworkX visualization"""
        return {node: (self.nodes[node]['x'], self.nodes[node]['y']) for node in self.nodes}


class MapGraph(Graph):
    """Extension of Graph class for route planning on maps"""
    
    def load_from_coordinates(self, nodes_data, edges_data=None):
        """
        Load graph from a list of nodes and optional edges
        
        nodes_data: List of dictionaries with at least 'id', 'x', and 'y' keys
        edges_data: Optional list of dictionaries with 'from', 'to', and optional 'cost' keys
        """
        # Add nodes
        for node in nodes_data:
            node_id = node['id']
            x, y = node['x'], node['y']
            attributes = {k: v for k, v in node.items() if k not in ['id', 'x', 'y']}
            self.add_node(node_id, x, y, **attributes)
        
        # Add edges if provided
        if edges_data:
            for edge in edges_data:
                from_node = edge['from']
                to_node = edge['to']
                cost = edge.get('cost', None)
                bidirectional = edge.get('bidirectional', True)
                self.add_edge(from_node, to_node, cost, bidirectional)
        
        return self
    
    def generate_grid_graph(self, width, height, diagonal_edges=False):
        """Generate a grid graph with nodes at integer coordinates"""
        # Generate alphabetical labels
        labels = self._generate_alphabetical_labels(width * height)
        label_index = 0
        
        for y in range(height):
            for x in range(width):
                node_id = f"{x},{y}"
                alpha_label = labels[label_index]
                label_index += 1
                self.add_node(node_id, x, y, label=alpha_label)
                
                # Connect to neighbors (excluding diagonals)
                if x > 0:
                    self.add_edge(node_id, f"{x-1},{y}")
                if y > 0:
                    self.add_edge(node_id, f"{x},{y-1}")
                
                # Add diagonal edges if specified
                if diagonal_edges:
                    if x > 0 and y > 0:
                        self.add_edge(node_id, f"{x-1},{y-1}", cost=1.414)  # √2
                    if x > 0 and y < height - 1:
                        self.add_edge(node_id, f"{x-1},{y+1}", cost=1.414)  # √2
        
        return self
    
    def _generate_alphabetical_labels(self, count):
        """Generate alphabetical labels (A, B, C, ... AA, AB, etc.) for a given count"""
        labels = []
        
        # Single letters (A-Z)
        for letter in string.ascii_uppercase:
            labels.append(letter)
            if len(labels) >= count:
                return labels
        
        # Double letters (AA-ZZ)
        for first in string.ascii_uppercase:
            for second in string.ascii_uppercase:
                labels.append(first + second)
                if len(labels) >= count:
                    return labels
        
        # Triple letters if needed (AAA-ZZZ)
        for first in string.ascii_uppercase:
            for second in string.ascii_uppercase:
                for third in string.ascii_uppercase:
                    labels.append(first + second + third)
                    if len(labels) >= count:
                        return labels
        
        return labels
    
    def generate_random_graph(self, num_nodes, connectivity=0.3, bidirectional=True, seed=None):
        """Generate a random graph with the specified number of nodes and connectivity"""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate alphabetical labels
        labels = self._generate_alphabetical_labels(num_nodes)
        
        # Generate random node positions
        for i in range(num_nodes):
            x = np.random.uniform(0, 100)
            y = np.random.uniform(0, 100)
            self.add_node(i, x, y, label=labels[i])
        
        # Generate random edges based on connectivity parameter
        nodes = self.get_all_nodes()
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if np.random.random() < connectivity:
                    self.add_edge(nodes[i], nodes[j], bidirectional=bidirectional)
        
        return self 

    def randomize_edge_costs(self, min_cost=0.1, max_cost=10.0):
        """Randomize the costs of all edges in the graph"""
        for node1 in self.edges:
            for node2 in self.edges[node1]:
                # Generate a random cost between min_cost and max_cost
                new_cost = np.random.uniform(min_cost, max_cost)
                # Update the cost in both directions (for undirected graph)
                self.edges[node1][node2] = new_cost
                self.edges[node2][node1] = new_cost
                # Update the NetworkX graph weights
                self.nx_graph[node1][node2]['weight'] = new_cost
                self.nx_graph[node2][node1]['weight'] = new_cost
        return self 