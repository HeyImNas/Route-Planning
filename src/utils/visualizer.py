import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation

class GraphVisualizer:
    def __init__(self, graph, dark_mode=False):
        self.graph = graph
        self.fig = None
        self.ax = None
        self.node_colors = {}
        self.dark_mode = dark_mode
        
        # Set colors based on theme
        self.update_theme_colors()
        
        self.show_costs = True
        self.show_heuristics = True
        self.goal_node = None
    
    def update_theme_colors(self):
        """Update colors based on current theme"""
        if self.dark_mode:
            self.default_node_color = '#4A6D8C'  # Darker blue in dark mode
            self.highlight_node_color = '#FF5555'  # Brighter red in dark mode
            self.path_node_color = '#55AA55'  # Brighter green in dark mode
            self.default_edge_color = '#777777'  # Lighter gray in dark mode
            self.path_edge_color = '#55DD55'  # Brighter green in dark mode
            self.text_box_color = '#444444'  # Dark background for text
            self.text_color = '#FFFFFF'  # White text
            self.fig_bg_color = '#333333'  # Dark background for figure
        else:
            self.default_node_color = 'lightblue'
            self.highlight_node_color = 'red'
            self.path_node_color = 'green'
            self.default_edge_color = 'gray'
            self.path_edge_color = 'green'
            self.text_box_color = 'white'
            self.text_color = 'black'
            self.fig_bg_color = 'white'
    
    def set_dark_mode(self, enabled):
        """Enable or disable dark mode"""
        self.dark_mode = enabled
        self.update_theme_colors()
    
    def set_goal(self, goal_node):
        """Set the goal node for heuristic calculation"""
        self.goal_node = goal_node
    
    def plot_graph(self, title="Graph Visualization", figsize=(10, 8)):
        """Plot the graph using NetworkX and Matplotlib"""
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.ax.clear()
        
        # Set figure and axes background colors
        self.fig.patch.set_facecolor(self.fig_bg_color)
        self.ax.set_facecolor(self.fig_bg_color)
            
        # Set title color based on theme
        title_color = 'white' if self.dark_mode else 'black'
        self.ax.set_title(title, color=title_color)
        self.ax.axis('equal')
        
        # Set axes text color based on theme
        for spine in self.ax.spines.values():
            spine.set_edgecolor(title_color)
        self.ax.tick_params(colors=title_color)
        
        # Get node positions
        pos = self.graph.get_node_positions()
        
        # Prepare node colors
        node_colors = [self.node_colors.get(node, self.default_node_color) 
                       for node in self.graph.get_all_nodes()]
        
        # Create labels with alphabetical representation
        labels = {}
        for node in self.graph.get_all_nodes():
            # Use the 'label' attribute if available, otherwise use the node ID
            labels[node] = self.graph.nodes[node].get('label', node)
        
        # Draw the graph
        nx_graph = self.graph.get_networkx_graph()
        nx.draw_networkx(
            nx_graph, 
            pos=pos,
            node_color=node_colors,
            edge_color=self.default_edge_color,
            labels=labels,
            with_labels=True,
            node_size=500,
            font_size=10,
            font_color='white' if self.dark_mode else 'black',
            ax=self.ax
        )
        
        # Add edge weights (costs)
        if self.show_costs:
            edge_labels = {}
            for u, v, d in nx_graph.edges(data=True):
                edge_labels[(u, v)] = f"{d.get('weight', 1):.1f}"
            
            nx.draw_networkx_edge_labels(
                nx_graph,
                pos,
                edge_labels=edge_labels,
                font_size=8,
                font_color='white' if self.dark_mode else 'black',
                bbox=dict(facecolor=self.text_box_color, alpha=0.7, edgecolor='none'),
                ax=self.ax
            )
        
        # Add heuristic values if goal is set
        if self.show_heuristics and self.goal_node is not None:
            # Calculate heuristic for each node
            heuristic_labels = {}
            for node in self.graph.get_all_nodes():
                h_value = self.graph.heuristic(node, self.goal_node)
                # Offset the position slightly to not overlap with node labels
                offset = (0.1, 0.1)
                pos_with_offset = (pos[node][0] + offset[0], pos[node][1] + offset[1])
                heuristic_labels[node] = f"h={h_value:.1f}"
                
                self.ax.text(
                    pos_with_offset[0], 
                    pos_with_offset[1], 
                    heuristic_labels[node],
                    fontsize=8,
                    color='red' if not self.dark_mode else '#FF7777',
                    bbox=dict(facecolor=self.text_box_color, alpha=0.7, edgecolor='none')
                )
        
        return self.fig, self.ax
    
    def highlight_path(self, path, node_color=None, edge_color=None):
        """Highlight a path in the graph"""
        if not path or len(path) < 2:
            return
        
        if node_color is None:
            node_color = self.path_node_color
            
        if edge_color is None:
            edge_color = self.path_edge_color
        
        pos = self.graph.get_node_positions()
        
        # Highlight nodes in the path
        nx.draw_networkx_nodes(
            self.graph.get_networkx_graph(),
            pos=pos,
            nodelist=path,
            node_color=node_color,
            node_size=500,
            ax=self.ax
        )
        
        # Highlight edges in the path
        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(
            self.graph.get_networkx_graph(),
            pos=pos,
            edgelist=edges,
            edge_color=edge_color,
            width=2.0,
            ax=self.ax
        )
        
        # Add path costs to each node in the path
        if len(path) > 1:
            start = path[0]
            g_score = {start: 0}  # g(n) = cost from start to n
            
            # Calculate cumulative path cost for each node in the path
            for i in range(1, len(path)):
                prev_node = path[i-1]
                curr_node = path[i]
                g_score[curr_node] = g_score[prev_node] + self.graph.get_cost(prev_node, curr_node)
                
                # Display the cost at each node
                pos_node = pos[curr_node]
                # Offset the g-score to a different position than the heuristic
                offset = (-0.1, 0.1)
                self.ax.text(
                    pos_node[0] + offset[0], 
                    pos_node[1] + offset[1], 
                    f"g={g_score[curr_node]:.1f}",
                    fontsize=8,
                    color='blue' if not self.dark_mode else '#7777FF',
                    bbox=dict(facecolor=self.text_box_color, alpha=0.7, edgecolor='none')
                )
                
                # If we have a goal, we can display f(n) = g(n) + h(n)
                if self.goal_node is not None:
                    h_value = self.graph.heuristic(curr_node, self.goal_node)
                    f_value = g_score[curr_node] + h_value
                    # Offset the f-score to yet another position
                    offset = (0, 0.15)
                    self.ax.text(
                        pos_node[0] + offset[0], 
                        pos_node[1] + offset[1], 
                        f"f={f_value:.1f}",
                        fontsize=8,
                        color='purple' if not self.dark_mode else '#DD77DD',
                        bbox=dict(facecolor=self.text_box_color, alpha=0.7, edgecolor='none')
                    )
        
        if self.fig is not None:
            self.fig.canvas.draw_idle()
    
    def set_node_color(self, node, color):
        """Set color for a specific node"""
        self.node_colors[node] = color
    
    def reset_node_colors(self):
        """Reset all node colors to default"""
        self.node_colors = {}
    
    def save_figure(self, filename):
        """Save the current figure to a file"""
        if self.fig is not None:
            self.fig.savefig(filename, bbox_inches='tight', dpi=300)


class SearchVisualizer:
    def __init__(self, graph, algorithms=None, dark_mode=False):
        self.graph = graph
        self.algorithms = algorithms or {}
        self.results = {}
        self.dark_mode = dark_mode
        self.visualizer = GraphVisualizer(graph, dark_mode=dark_mode)
        self.colors = self._get_theme_colors()
    
    def _get_theme_colors(self):
        """Get colors based on current theme"""
        if self.dark_mode:
            return ['#FF5555', '#55AA55', '#5555FF', '#FFAA55', '#AA55FF', '#55AAAA', '#FF55FF', '#FFFF55']
        else:
            return ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    def set_dark_mode(self, enabled):
        """Enable or disable dark mode for the visualizer"""
        self.dark_mode = enabled
        self.visualizer.set_dark_mode(enabled)
        self.colors = self._get_theme_colors()
    
    def run_algorithm(self, algorithm_name, start, goal):
        """Run a specified algorithm and store results"""
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not found")
        
        algorithm = self.algorithms[algorithm_name]
        path = algorithm.search(start, goal)
        stats = algorithm.get_stats()
        
        self.results[algorithm_name] = {
            'path': path,
            'stats': stats
        }
        
        return stats
    
    def run_all_algorithms(self, start, goal):
        """Run all registered algorithms and collect results"""
        for algorithm_name in self.algorithms:
            self.run_algorithm(algorithm_name, start, goal)
        
        return self.results
    
    def visualize_path(self, algorithm_name, fig=None, ax=None):
        """Visualize the path found by a specific algorithm"""
        if algorithm_name not in self.results:
            raise ValueError(f"No results found for algorithm {algorithm_name}")
        
        path = self.results[algorithm_name]['path']
        
        if fig is not None and ax is not None:
            self.visualizer.fig = fig
            self.visualizer.ax = ax
        
        self.visualizer.plot_graph(title=f"Path found by {algorithm_name}")
        self.visualizer.highlight_path(path)
        
        return self.visualizer.fig, self.visualizer.ax
    
    def visualize_all_paths(self, fig=None, ax=None, separate_figures=False):
        """Visualize paths found by all algorithms"""
        if separate_figures:
            figs = {}
            for i, algorithm_name in enumerate(self.results):
                f, a = plt.subplots(figsize=(12, 10))
                self.visualize_path(algorithm_name, fig=f, ax=a)
                figs[algorithm_name] = (f, a)
            return figs
        else:
            if fig is not None and ax is not None:
                plt.figure(fig.number)
                ax.clear()
            else:
                fig, ax = plt.subplots(figsize=(12, 10))
            
            # Set figure and axes background colors
            fig_bg_color = '#333333' if self.dark_mode else 'white'
            fig.patch.set_facecolor(fig_bg_color)
            ax.set_facecolor(fig_bg_color)
            
            # Set title color based on theme
            title_color = 'white' if self.dark_mode else 'black'
            ax.set_title("Comparison of Different Search Algorithms", color=title_color)
            
            # Set axes text color based on theme
            for spine in ax.spines.values():
                spine.set_edgecolor(title_color)
            ax.tick_params(colors=title_color)
                
            # Draw the graph
            pos = self.graph.get_node_positions()
            nx_graph = self.graph.get_networkx_graph()
            
            # Add labels for nodes (use alphabetical labels if available)
            labels = {}
            for node in self.graph.get_all_nodes():
                label = self.graph.nodes[node].get('label', node)
                labels[node] = label
            
            # Set node and edge colors based on theme
            node_color = 'lightblue' if not self.dark_mode else '#4A6D8C'
            edge_color = 'gray' if not self.dark_mode else '#777777'
            
            nx.draw_networkx(
                nx_graph, 
                pos=pos,
                node_color=node_color,
                edge_color=edge_color,
                labels=labels,
                with_labels=True,
                node_size=500,
                font_size=10,
                font_color=title_color,
                ax=ax
            )
            
            # Add edge weights (costs)
            edge_labels = {}
            for u, v, d in nx_graph.edges(data=True):
                edge_labels[(u, v)] = f"{d.get('weight', 1):.1f}"
            
            text_bg_color = 'white' if not self.dark_mode else '#444444'
            
            nx.draw_networkx_edge_labels(
                nx_graph,
                pos,
                edge_labels=edge_labels,
                font_size=8,
                font_color=title_color,
                bbox=dict(facecolor=text_bg_color, alpha=0.7, edgecolor='none'),
                ax=ax
            )
            
            # Add heuristic values if goal is set
            if self.visualizer.goal_node is not None:
                # Calculate heuristic for each node
                for node in self.graph.get_all_nodes():
                    h_value = self.graph.heuristic(node, self.visualizer.goal_node)
                    # Offset the position slightly to not overlap with node labels
                    offset = (0.1, 0.1)
                    pos_with_offset = (pos[node][0] + offset[0], pos[node][1] + offset[1])
                    
                    text_color = 'red' if not self.dark_mode else '#FF7777'
                    text_bg_color = 'white' if not self.dark_mode else '#333333'
                    
                    ax.text(
                        pos_with_offset[0], 
                        pos_with_offset[1], 
                        f"h={h_value:.1f}",
                        fontsize=8,
                        color=text_color,
                        bbox=dict(facecolor=text_bg_color, alpha=0.7, edgecolor='none')
                    )
            
            # Draw paths for each algorithm with different colors
            for i, (algorithm_name, result) in enumerate(self.results.items()):
                path = result['path']
                if path and len(path) > 1:
                    color = self.colors[i % len(self.colors)]
                    edges = [(path[j], path[j+1]) for j in range(len(path)-1)]
                    
                    # Draw path edges
                    nx.draw_networkx_edges(
                        nx_graph,
                        pos=pos,
                        edgelist=edges,
                        edge_color=color,
                        width=2.0 + i*0.5,  # Make each path slightly wider
                        label=algorithm_name,
                        ax=ax
                    )
                    
                    # Add path costs for this algorithm's path
                    if self.visualizer.goal_node is not None:
                        # Calculate path costs (g-values)
                        start = path[0]
                        g_score = {start: 0}
                        
                        for j in range(1, len(path)):
                            prev_node = path[j-1]
                            curr_node = path[j]
                            g_score[curr_node] = g_score[prev_node] + self.graph.get_cost(prev_node, curr_node)
                            
                            # Display g-scores for this algorithm with slight offset based on algorithm index
                            # to prevent overlap between algorithms
                            pos_node = pos[curr_node]
                            y_offset = -0.1 - (i * 0.05)  # Adjust y-offset for each algorithm
                            offset = (-0.1, y_offset)
                            
                            text_bg_color = 'white' if not self.dark_mode else '#333333'
                            
                            ax.text(
                                pos_node[0] + offset[0], 
                                pos_node[1] + offset[1], 
                                f"{algorithm_name}: g={g_score[curr_node]:.1f}",
                                fontsize=7,
                                color=color,
                                bbox=dict(facecolor=text_bg_color, alpha=0.7, edgecolor='none')
                            )
            
            ax.legend()
            return fig, ax
    
    def compare_results(self):
        """Create a comparison table of algorithm results"""
        if not self.results:
            return "No results to compare"
        
        headers = ["Algorithm", "Path Length", "Visited Nodes", "Execution Time (ms)", "Max Memory"]
        data = []
        
        for name, result in self.results.items():
            stats = result['stats']
            data.append([
                name,
                stats['path_length'],
                stats['visited_nodes'],
                f"{(stats['execution_time'] * 1000):.3f}",  # Convert to milliseconds
                stats['max_memory']
            ])
        
        # Sort by execution time
        data.sort(key=lambda x: float(x[3]))
        
        # Create a formatted table
        col_widths = [max(len(str(row[i])) for row in data + [headers]) for i in range(len(headers))]
        
        # Print header
        header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        separator = "-" * len(header_row)
        
        table = [header_row, separator]
        
        # Print data rows
        for row in data:
            table.append(" | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)))
        
        return "\n".join(table)
    
    def plot_performance_comparison(self, fig=None, axs=None):
        """Plot performance metrics for all algorithms"""
        if not self.results:
            return None
        
        algorithm_names = list(self.results.keys())
        path_lengths = [self.results[name]['stats']['path_length'] for name in algorithm_names]
        visited_nodes = [self.results[name]['stats']['visited_nodes'] for name in algorithm_names]
        execution_times = [self.results[name]['stats']['execution_time'] * 1000 for name in algorithm_names]  # Convert to milliseconds
        memory_usage = [self.results[name]['stats']['max_memory'] for name in algorithm_names]
        
        if fig is None or axs is None:
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        else:
            for ax in axs.flat:
                ax.clear()
                
        fig.suptitle('Algorithm Performance Comparison', fontsize=16, y=0.95)
        
        # Set text color based on theme
        text_color = 'black' if not self.dark_mode else 'white'
        
        # Path length
        axs[0, 0].bar(algorithm_names, path_lengths, color=self.colors[:len(algorithm_names)])
        axs[0, 0].set_title('Path Length', color=text_color, pad=20)
        axs[0, 0].set_ylabel('Number of Nodes', color=text_color)
        axs[0, 0].tick_params(colors=text_color)
        for spine in axs[0, 0].spines.values():
            spine.set_edgecolor(text_color)
        plt.setp(axs[0, 0].get_xticklabels(), rotation=45, ha='right', color=text_color)
        
        # Visited nodes
        axs[0, 1].bar(algorithm_names, visited_nodes, color=self.colors[:len(algorithm_names)])
        axs[0, 1].set_title('Nodes Visited', color=text_color, pad=20)
        axs[0, 1].set_ylabel('Number of Nodes', color=text_color)
        axs[0, 1].tick_params(colors=text_color)
        for spine in axs[0, 1].spines.values():
            spine.set_edgecolor(text_color)
        plt.setp(axs[0, 1].get_xticklabels(), rotation=45, ha='right', color=text_color)
        
        # Execution time
        axs[1, 0].bar(algorithm_names, execution_times, color=self.colors[:len(algorithm_names)])
        axs[1, 0].set_title('Execution Time', color=text_color, pad=20)
        axs[1, 0].set_ylabel('Milliseconds', color=text_color)  # Updated to milliseconds
        axs[1, 0].tick_params(colors=text_color)
        for spine in axs[1, 0].spines.values():
            spine.set_edgecolor(text_color)
        plt.setp(axs[1, 0].get_xticklabels(), rotation=45, ha='right', color=text_color)
        
        # Memory usage
        axs[1, 1].bar(algorithm_names, memory_usage, color=self.colors[:len(algorithm_names)])
        axs[1, 1].set_title('Memory Usage (Max Queue/Stack Size)', color=text_color, pad=20)
        axs[1, 1].set_ylabel('Size', color=text_color)
        axs[1, 1].tick_params(colors=text_color)
        for spine in axs[1, 1].spines.values():
            spine.set_edgecolor(text_color)
        plt.setp(axs[1, 1].get_xticklabels(), rotation=45, ha='right', color=text_color)
        
        # Adjust layout to prevent overlapping
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        return fig, axs 