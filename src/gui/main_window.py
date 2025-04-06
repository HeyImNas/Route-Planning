import sys
import os
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QMessageBox, QFileDialog,
    QTabWidget, QGridLayout, QGroupBox, QLineEdit, QRadioButton,
    QButtonGroup, QSpinBox, QCheckBox, QSplitter, QFrame, QSizePolicy,
    QToolBar, QStyle, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QThreadPool, QRunnable, pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QFont, QPalette, QColor, QAction

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import networkx as nx

import numpy as np
import time
import random

# Import our graph and search algorithm classes
from src.utils.graph import Graph, MapGraph
from src.algorithms.search_algorithms import BFS, DFS, GreedyBestFirstSearch, AStar
from src.utils.visualizer import GraphVisualizer, SearchVisualizer


# Signal class for worker threads
class WorkerSignals(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)


# Worker class for running algorithms in a separate thread
class AlgorithmWorker(QRunnable):
    def __init__(self, algorithm, start, goal):
        super().__init__()
        self.algorithm = algorithm
        self.start = start
        self.goal = goal
        self.signals = WorkerSignals()
    
    def run(self):
        try:
            # Run the algorithm
            path = self.algorithm.search(self.start, self.goal)
            stats = self.algorithm.get_stats()
            
            # Emit the results
            self.signals.finished.emit({
                'path': path,
                'stats': stats
            })
        except Exception as e:
            self.signals.error.emit(str(e))


class MatplotlibCanvas(FigureCanvas):
    """Canvas for matplotlib figures"""
    def __init__(self, fig=None, parent=None, width=5, height=4, dpi=100):
        if fig is None:
            self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        else:
            self.fig = fig
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Make the canvas expandable
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, 
            QSizePolicy.Policy.Expanding
        )
        self.updateGeometry()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set up the window
        self.setWindowTitle("Route Planning with Search Algorithms")
        self.setGeometry(100, 100, 1200, 800)
        
        # Dark mode state
        self.dark_mode_enabled = False
        
        # Initialize graph and algorithms
        self.graph = MapGraph()
        self.algorithms = {
            'BFS': BFS(self.graph),
            'DFS': DFS(self.graph),
            'Greedy Best-First': GreedyBestFirstSearch(self.graph),
            'A*': AStar(self.graph)
        }
        self.search_visualizer = SearchVisualizer(self.graph, self.algorithms)
        
        # Set up thread pool for running algorithms
        self.thread_pool = QThreadPool()
        
        # Create toolbar with theme toggle
        self.create_toolbar()
        
        # Set up the UI
        self.setup_ui()
        
        # Create a default graph for demonstration
        self.create_default_graph()
    
    def create_toolbar(self):
        """Create toolbar with actions like theme toggle"""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # Add theme toggle action
        self.theme_action = QAction("Toggle Dark Mode", self)
        self.theme_action.triggered.connect(self.toggle_theme)
        toolbar.addAction(self.theme_action)
    
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        self.dark_mode_enabled = not self.dark_mode_enabled
        
        if self.dark_mode_enabled:
            self.set_dark_theme()
        else:
            self.set_light_theme()
        
        # Update matplotlib figures for the current theme
        self.update_plot_theme()
    
    def set_dark_theme(self):
        """Set application to dark theme"""
        dark_palette = QPalette()
        
        # Set colors for dark palette
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        
        # Set disabled color explicitly - using ColorGroup.Disabled
        dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(128, 128, 128))
        dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(128, 128, 128))
        dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(128, 128, 128))
        
        # Apply the dark palette
        QApplication.instance().setPalette(dark_palette)
        
        # Apply dark style sheet for additional control
        self.setStyleSheet("""
            QWidget {
                background-color: #333333;
                color: #FFFFFF;
            }
            QFrame, QLabel, QGroupBox {
                background-color: #333333;
                color: #FFFFFF;
            }
            QGroupBox:disabled {
                color: #888888;
                background-color: #2A2A2A;
                border: 1px solid #555555;
            }
            QGroupBox::title:disabled {
                color: #888888;
            }
            QTabWidget::pane {
                background-color: #333333;
                border: 1px solid #555555;
            }
            QTabBar::tab {
                background-color: #444444;
                color: #FFFFFF;
                padding: 8px;
                margin: 2px;
            }
            QTabBar::tab:selected {
                background-color: #555555;
            }
            QPushButton {
                background-color: #555555;
                color: #FFFFFF;
                border: 1px solid #777777;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:disabled {
                background-color: #3A3A3A;
                color: #888888;
                border: 1px solid #555555;
            }
            QComboBox, QSpinBox {
                background-color: #555555;
                color: #FFFFFF;
                border: 1px solid #777777;
                padding: 3px;
                border-radius: 3px;
            }
            QComboBox:disabled, QSpinBox:disabled {
                background-color: #3A3A3A;
                color: #888888;
                border: 1px solid #555555;
            }
            QCheckBox, QRadioButton {
                background-color: transparent;
                color: #FFFFFF;
            }
            QCheckBox:disabled, QRadioButton:disabled {
                color: #888888;
            }
            QToolBar {
                background-color: #333333;
                color: #FFFFFF;
                border-bottom: 1px solid #555555;
            }
            QToolButton {
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #333333;
                width: 14px;
                margin: 15px 3px 15px 3px;
                border: 1px solid #444444;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #555555;
                min-height: 20px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666666;
            }
            QMessageBox {
                background-color: #333333;
                color: #FFFFFF;
            }
            QMessageBox QLabel {
                color: #FFFFFF;
            }
            QMessageBox QPushButton {
                background-color: #555555;
                color: #FFFFFF;
                border: 1px solid #777777;
                padding: 5px;
                min-width: 80px;
                border-radius: 3px;
            }
        """)
        
        # Update matplotlib toolbars
        self.graph_toolbar.setStyleSheet("background-color: #444444; color: white;")
        self.results_toolbar.setStyleSheet("background-color: #444444; color: white;")
        self.perf_toolbar.setStyleSheet("background-color: #444444; color: white;")
        
        # Update theme action text
        self.theme_action.setText("Toggle Light Mode")
    
    def set_light_theme(self):
        """Reset to default light theme"""
        QApplication.instance().setPalette(QApplication.style().standardPalette())
        
        # Clear style sheet
        self.setStyleSheet("")
        
        # Reset matplotlib toolbars
        self.graph_toolbar.setStyleSheet("background-color: #f0f0f0;")
        self.results_toolbar.setStyleSheet("background-color: #f0f0f0;")
        self.perf_toolbar.setStyleSheet("background-color: #f0f0f0;")
        
        # Update theme action text
        self.theme_action.setText("Toggle Dark Mode")
    
    def update_plot_theme(self):
        """Update matplotlib figures to match the current theme"""
        # Set matplotlib style based on current theme
        if self.dark_mode_enabled:
            plt.style.use('dark_background')
            # Force background color for matplotlib figures
            self.graph_figure.patch.set_facecolor('#333333')
            self.results_figure.patch.set_facecolor('#333333')
            self.perf_figure.patch.set_facecolor('#333333')
            
            # Update axes background colors
            for ax in [self.graph_ax, self.results_ax]:
                if ax is not None:
                    ax.set_facecolor('#333333')
            
            if self.perf_axes is not None:
                for ax in self.perf_axes.flat:
                    ax.set_facecolor('#333333')
        else:
            plt.style.use('default')
            # Reset background colors
            self.graph_figure.patch.set_facecolor('white')
            self.results_figure.patch.set_facecolor('white')
            self.perf_figure.patch.set_facecolor('white')
            
            # Reset axes background colors
            for ax in [self.graph_ax, self.results_ax]:
                if ax is not None:
                    ax.set_facecolor('white')
            
            if self.perf_axes is not None:
                for ax in self.perf_axes.flat:
                    ax.set_facecolor('white')
        
        # Update the graph visualization
        self.visualize_graph()
        
        # Update result visualizations if they exist
        if hasattr(self, 'search_visualizer') and self.search_visualizer.results:
            # Update search visualizer theme
            self.search_visualizer.set_dark_mode(self.dark_mode_enabled)
            self.update_result_visualizations()
    
    def setup_ui(self):
        """Set up the user interface"""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Tab 1: Graph Creation and Visualization
        graph_tab = QWidget()
        graph_layout = QVBoxLayout(graph_tab)
        
        # Splitter for graph options and visualization
        graph_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Graph options
        graph_options = QWidget()
        graph_options_layout = QVBoxLayout(graph_options)
        
        # Graph type group
        graph_type_group = QGroupBox("Graph Type")
        graph_type_layout = QVBoxLayout()
        
        self.random_graph_radio = QRadioButton("Random Graph")
        self.grid_graph_radio = QRadioButton("Grid Graph")
        self.custom_graph_radio = QRadioButton("Custom Graph")
        
        graph_type_layout.addWidget(self.random_graph_radio)
        graph_type_layout.addWidget(self.grid_graph_radio)
        graph_type_layout.addWidget(self.custom_graph_radio)
        
        graph_type_group.setLayout(graph_type_layout)
        graph_options_layout.addWidget(graph_type_group)
        
        # Random graph options
        random_graph_options = QGroupBox("Random Graph Options")
        random_layout = QGridLayout()
        
        random_layout.addWidget(QLabel("Number of Nodes:"), 0, 0)
        self.node_count_spin = QSpinBox()
        self.node_count_spin.setRange(2, 100)
        self.node_count_spin.setValue(8)
        random_layout.addWidget(self.node_count_spin, 0, 1)
        
        random_layout.addWidget(QLabel("Connectivity:"), 1, 0)
        self.connectivity_spin = QSpinBox()
        self.connectivity_spin.setRange(1, 100)
        self.connectivity_spin.setValue(20)
        random_layout.addWidget(self.connectivity_spin, 1, 1)
        
        random_graph_options.setLayout(random_layout)
        graph_options_layout.addWidget(random_graph_options)
        
        # Grid graph options
        grid_graph_options = QGroupBox("Grid Graph Options")
        grid_layout = QGridLayout()
        
        grid_layout.addWidget(QLabel("Width:"), 0, 0)
        self.grid_width_spin = QSpinBox()
        self.grid_width_spin.setRange(2, 30)
        self.grid_width_spin.setValue(4)
        grid_layout.addWidget(self.grid_width_spin, 0, 1)
        
        grid_layout.addWidget(QLabel("Height:"), 1, 0)
        self.grid_height_spin = QSpinBox()
        self.grid_height_spin.setRange(2, 30)
        self.grid_height_spin.setValue(4)
        grid_layout.addWidget(self.grid_height_spin, 1, 1)
        
        grid_layout.addWidget(QLabel("Diagonal Edges:"), 2, 0)
        self.diagonal_edges_check = QCheckBox()
        grid_layout.addWidget(self.diagonal_edges_check, 2, 1)
        
        grid_graph_options.setLayout(grid_layout)
        graph_options_layout.addWidget(grid_graph_options)
        
        # Custom graph options
        custom_graph_options = QGroupBox("Custom Graph Options")
        custom_layout = QGridLayout()
        
        custom_layout.addWidget(QLabel("Node X:"), 0, 0)
        self.node_x_spin = QDoubleSpinBox()
        self.node_x_spin.setRange(0, 100)
        self.node_x_spin.setValue(0)
        self.node_x_spin.setSingleStep(1.0)
        custom_layout.addWidget(self.node_x_spin, 0, 1)
        
        custom_layout.addWidget(QLabel("Node Y:"), 1, 0)
        self.node_y_spin = QDoubleSpinBox()
        self.node_y_spin.setRange(0, 100)
        self.node_y_spin.setValue(0)
        self.node_y_spin.setSingleStep(1.0)
        custom_layout.addWidget(self.node_y_spin, 1, 1)
        
        # Node removal
        custom_layout.addWidget(QLabel("Select Node:"), 2, 0)
        self.remove_node_combo = QComboBox()
        custom_layout.addWidget(self.remove_node_combo, 2, 1)
        
        # Button layout
        button_layout = QGridLayout()
        
        # Add node button
        self.add_node_btn = QPushButton("Add Node")
        self.add_node_btn.clicked.connect(self.add_custom_node)
        button_layout.addWidget(self.add_node_btn, 0, 0)
        
        # Remove node button
        self.remove_node_btn = QPushButton("Remove Node")
        self.remove_node_btn.clicked.connect(self.remove_custom_node)
        button_layout.addWidget(self.remove_node_btn, 0, 1)
        
        # Clear custom graph button
        self.clear_custom_graph_btn = QPushButton("Clear Custom Graph")
        self.clear_custom_graph_btn.clicked.connect(self.clear_custom_graph)
        button_layout.addWidget(self.clear_custom_graph_btn, 1, 0, 1, 2)
        
        custom_layout.addLayout(button_layout, 3, 0, 1, 2)
        
        custom_graph_options.setLayout(custom_layout)
        graph_options_layout.addWidget(custom_graph_options)
        
        # Graph generation button
        self.generate_graph_btn = QPushButton("Generate Graph")
        self.generate_graph_btn.clicked.connect(self.generate_graph)
        graph_options_layout.addWidget(self.generate_graph_btn)
        
        # Node selection
        node_selection_group = QGroupBox("Start and Goal Nodes")
        node_selection_layout = QGridLayout()
        
        node_selection_layout.addWidget(QLabel("Start Node:"), 0, 0)
        self.start_node_combo = QComboBox()
        node_selection_layout.addWidget(self.start_node_combo, 0, 1)
        
        node_selection_layout.addWidget(QLabel("Goal Node:"), 1, 0)
        self.goal_node_combo = QComboBox()
        node_selection_layout.addWidget(self.goal_node_combo, 1, 1)
        
        node_selection_group.setLayout(node_selection_layout)
        graph_options_layout.addWidget(node_selection_group)
        
        # Edge Editor
        edge_editor_group = QGroupBox("Edge Cost Editor")
        edge_editor_layout = QGridLayout()
        
        edge_editor_layout.addWidget(QLabel("From Node:"), 0, 0)
        self.from_node_combo = QComboBox()
        edge_editor_layout.addWidget(self.from_node_combo, 0, 1)
        
        edge_editor_layout.addWidget(QLabel("To Node:"), 1, 0)
        self.to_node_combo = QComboBox()
        edge_editor_layout.addWidget(self.to_node_combo, 1, 1)
        
        edge_editor_layout.addWidget(QLabel("Cost:"), 2, 0)
        self.edge_cost_spin = QDoubleSpinBox()
        self.edge_cost_spin.setRange(0.1, 100.0)
        self.edge_cost_spin.setValue(1.0)
        self.edge_cost_spin.setSingleStep(0.1)
        edge_editor_layout.addWidget(self.edge_cost_spin, 2, 1)
        
        # Update/Get buttons
        edge_buttons_layout = QHBoxLayout()
        self.update_edge_btn = QPushButton("Update Edge Cost")
        self.update_edge_btn.clicked.connect(self.update_edge_cost)
        self.get_edge_btn = QPushButton("Get Current Cost")
        self.get_edge_btn.clicked.connect(self.get_edge_cost)
        self.randomize_costs_btn = QPushButton("Randomize All Costs")
        self.randomize_costs_btn.clicked.connect(self.randomize_edge_costs)
        
        edge_buttons_layout.addWidget(self.update_edge_btn)
        edge_buttons_layout.addWidget(self.get_edge_btn)
        edge_buttons_layout.addWidget(self.randomize_costs_btn)
        edge_editor_layout.addLayout(edge_buttons_layout, 3, 0, 1, 2)
        
        edge_editor_group.setLayout(edge_editor_layout)
        graph_options_layout.addWidget(edge_editor_group)
        
        # Add spacer to push everything to the top
        graph_options_layout.addStretch()
        
        # Right panel: Graph visualization
        graph_vis_widget = QWidget()
        graph_vis_layout = QVBoxLayout(graph_vis_widget)
        
        # Create matplotlib canvas
        self.graph_figure, self.graph_ax = plt.subplots(figsize=(6, 6))
        self.graph_canvas = MatplotlibCanvas(self.graph_figure)
        self.graph_toolbar = NavigationToolbar(self.graph_canvas, self)
        self.graph_toolbar.setStyleSheet("background-color: #f0f0f0;")  # Default light style
        
        graph_vis_layout.addWidget(self.graph_toolbar)
        graph_vis_layout.addWidget(self.graph_canvas)
        
        # Add widgets to splitter
        graph_splitter.addWidget(graph_options)
        graph_splitter.addWidget(graph_vis_widget)
        graph_splitter.setSizes([300, 900])
        
        graph_layout.addWidget(graph_splitter)
        
        # Add the graph tab to the tab widget
        tab_widget.addTab(graph_tab, "Graph Creation")
        
        # Tab 2: Algorithm Comparison
        algo_tab = QWidget()
        algo_layout = QVBoxLayout(algo_tab)
        
        # Splitter for algorithm options and results
        algo_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Algorithm options
        algo_options = QWidget()
        algo_options_layout = QVBoxLayout(algo_options)
        
        # Algorithm selection
        algo_selection_group = QGroupBox("Select Algorithms")
        algo_selection_layout = QVBoxLayout()
        
        self.algo_checkboxes = {}
        for algo_name in self.algorithms.keys():
            checkbox = QCheckBox(algo_name)
            checkbox.setChecked(True)
            self.algo_checkboxes[algo_name] = checkbox
            algo_selection_layout.addWidget(checkbox)
        
        algo_selection_group.setLayout(algo_selection_layout)
        algo_options_layout.addWidget(algo_selection_group)
        
        # Run button
        self.run_algorithms_btn = QPushButton("Run Selected Algorithms")
        self.run_algorithms_btn.clicked.connect(self.run_algorithms)
        algo_options_layout.addWidget(self.run_algorithms_btn)
        
        # Visualization options
        vis_options_group = QGroupBox("Visualization Options")
        vis_options_layout = QVBoxLayout()
        
        self.show_all_paths_radio = QRadioButton("Show All Paths")
        self.show_all_paths_radio.setChecked(True)
        self.show_individual_radio = QRadioButton("Show Individual Paths")
        
        vis_options_layout.addWidget(self.show_all_paths_radio)
        vis_options_layout.addWidget(self.show_individual_radio)
        
        vis_options_group.setLayout(vis_options_layout)
        algo_options_layout.addWidget(vis_options_group)
        
        # Add spacer
        algo_options_layout.addStretch()
        
        # Right panel: Results visualization (tabbed)
        results_widget = QTabWidget()
        
        # Path visualization tab
        path_vis_widget = QWidget()
        path_vis_layout = QVBoxLayout(path_vis_widget)
        
        self.results_figure, self.results_ax = plt.subplots(figsize=(6, 6))
        self.results_canvas = MatplotlibCanvas(self.results_figure)
        self.results_toolbar = NavigationToolbar(self.results_canvas, self)
        self.results_toolbar.setStyleSheet("background-color: #f0f0f0;")  # Default light style
        
        path_vis_layout.addWidget(self.results_toolbar)
        path_vis_layout.addWidget(self.results_canvas)
        
        # Performance comparison tab
        perf_widget = QWidget()
        perf_layout = QVBoxLayout(perf_widget)
        
        # Create performance figure with increased height and adjusted spacing
        self.perf_figure, self.perf_axes = plt.subplots(2, 2, figsize=(8, 10))
        self.perf_canvas = MatplotlibCanvas(self.perf_figure)
        self.perf_toolbar = NavigationToolbar(self.perf_canvas, self)
        self.perf_toolbar.setStyleSheet("background-color: #f0f0f0;")  # Default light style
        
        # Add spacing between subplots
        self.perf_figure.subplots_adjust(hspace=0.4, wspace=0.3)
        
        perf_layout.addWidget(self.perf_toolbar)
        perf_layout.addWidget(self.perf_canvas)
        
        # Results table tab
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        
        self.results_table = QLabel("Run algorithms to see results")
        self.results_table.setFont(QFont("Courier New", 10))
        self.results_table.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        table_layout.addWidget(self.results_table)
        
        # Add tabs to results widget
        results_widget.addTab(path_vis_widget, "Path Visualization")
        results_widget.addTab(perf_widget, "Performance Charts")
        results_widget.addTab(table_widget, "Results Table")
        
        # Add widgets to splitter
        algo_splitter.addWidget(algo_options)
        algo_splitter.addWidget(results_widget)
        algo_splitter.setSizes([300, 900])
        
        algo_layout.addWidget(algo_splitter)
        
        # Add the algorithm tab to the tab widget
        tab_widget.addTab(algo_tab, "Algorithm Comparison")
        
        # Add tab widget to main layout
        main_layout.addWidget(tab_widget)
        
        # Set the main widget as the central widget
        self.setCentralWidget(main_widget)
        
        # Connect signals and slots
        self.random_graph_radio.toggled.connect(self.update_graph_options)
        self.grid_graph_radio.toggled.connect(self.update_graph_options)
        self.custom_graph_radio.toggled.connect(self.update_graph_options)
        
        # Set default selection
        self.random_graph_radio.setChecked(True)
        self.update_graph_options()
    
    def update_graph_options(self):
        """Update which graph options are visible based on selection"""
        is_random = self.random_graph_radio.isChecked()
        is_grid = self.grid_graph_radio.isChecked()
        is_custom = self.custom_graph_radio.isChecked()
        
        # Show/hide appropriate option groups
        for widget in self.findChildren(QGroupBox):
            if widget.title() == "Random Graph Options":
                widget.setVisible(is_random)
            elif widget.title() == "Grid Graph Options":
                widget.setVisible(is_grid)
            elif widget.title() == "Custom Graph Options":
                widget.setVisible(is_custom)
            elif widget.title() in ["Start and Goal Nodes", "Edge Cost Editor", "Graph Type"]:
                # These groups should always be visible
                widget.setVisible(True)
        
        # Show/hide generate button based on graph type
        self.generate_graph_btn.setVisible(not is_custom)
    
    def add_custom_node(self):
        """Add a node to the custom graph at the specified coordinates"""
        if not self.custom_graph_radio.isChecked():
            return
            
        # Create a new graph if it doesn't exist
        if not hasattr(self, 'graph') or self.graph is None:
            self.graph = MapGraph()
            
            # Update algorithms with new graph
            for algo in self.algorithms.values():
                algo.graph = self.graph
            
            self.search_visualizer = SearchVisualizer(self.graph, self.algorithms)
        
        # Get coordinates
        x = self.node_x_spin.value()
        y = self.node_y_spin.value()
        
        # Generate node label (A, B, C, ... AA, AB, etc.)
        num_nodes = len(self.graph.nodes)
        if num_nodes < 26:
            # Single letter (A-Z)
            node_id = chr(65 + num_nodes)  # 65 is ASCII for 'A'
        else:
            # Double letters (AA, AB, etc.)
            first = chr(65 + ((num_nodes - 26) // 26))
            second = chr(65 + ((num_nodes - 26) % 26))
            node_id = first + second
        
        # Add node to graph
        self.graph.add_node(node_id, x=x, y=y)
        
        # Update node selection combos
        self.update_node_selection()
        
        # Visualize the graph
        self.visualize_graph()
        
        # Show confirmation message
        self.status_message(f"Added node {node_id} at ({x}, {y})")
    
    def remove_custom_node(self):
        """Remove the selected node from the custom graph"""
        if not self.custom_graph_radio.isChecked() or self.remove_node_combo.count() == 0:
            return
        
        # Get the selected node
        node_idx = self.remove_node_combo.currentIndex()
        node_id = self.remove_node_combo.itemData(node_idx)
        
        if node_id is None:
            return
        
        # Remove the node and all its edges
        if node_id in self.graph.nodes:
            # Remove edges first
            edges_to_remove = []
            for neighbor in self.graph.get_neighbors(node_id):
                edges_to_remove.append((node_id, neighbor))
                edges_to_remove.append((neighbor, node_id))
            
            # Remove from edges dictionary
            for n1, n2 in edges_to_remove:
                if n1 in self.graph.edges and n2 in self.graph.edges[n1]:
                    del self.graph.edges[n1][n2]
            
            # Remove from nodes dictionary
            del self.graph.nodes[node_id]
            
            # Remove from NetworkX graph
            self.graph.nx_graph.remove_node(node_id)
            
            # Update node selection combos
            self.update_node_selection()
            
            # Visualize the graph
            self.visualize_graph()
            
            # Show confirmation message
            self.status_message(f"Removed node {node_id}")
    
    def update_node_selection(self):
        """Update the node selection combo boxes"""
        # Get all nodes
        nodes = self.graph.get_all_nodes()
        if not nodes:
            return
        
        # Clear existing items
        self.start_node_combo.clear()
        self.goal_node_combo.clear()
        self.from_node_combo.clear()
        self.to_node_combo.clear()
        self.remove_node_combo.clear()
        
        # Add nodes to combos with alphabetical labels
        for node in sorted(nodes):
            label = self.graph.nodes[node].get('label', node)
            display_text = f"{node}"
            self.start_node_combo.addItem(display_text, node)
            self.goal_node_combo.addItem(display_text, node)
            self.from_node_combo.addItem(display_text, node)
            self.to_node_combo.addItem(display_text, node)
            self.remove_node_combo.addItem(display_text, node)
        
        # Set default selections (first and last nodes)
        self.start_node_combo.setCurrentIndex(0)
        self.goal_node_combo.setCurrentIndex(len(nodes) - 1)
        
        # Update edge cost if nodes exist
        if len(nodes) >= 2:
            self.from_node_combo.setCurrentIndex(0)
            self.to_node_combo.setCurrentIndex(1)
            self.get_edge_cost()
    
    def get_edge_cost(self):
        """Get the current cost of an edge and update the spinbox"""
        if self.from_node_combo.count() == 0 or self.to_node_combo.count() == 0:
            return
        
        # Get selected nodes
        from_idx = self.from_node_combo.currentIndex()
        to_idx = self.to_node_combo.currentIndex()
        from_node = self.from_node_combo.itemData(from_idx)
        to_node = self.to_node_combo.itemData(to_idx)
        
        # Get the current cost
        cost = self.graph.get_cost(from_node, to_node)
        
        # Update the spinbox value (only if it's a valid edge)
        if cost != float('inf'):
            self.edge_cost_spin.setValue(cost)
        else:
            self.status_message(f"No direct edge exists between these nodes", QMessageBox.Icon.Warning)
    
    def update_edge_cost(self):
        """Update the cost of an edge in the graph or create a new one if it doesn't exist"""
        if self.from_node_combo.count() == 0 or self.to_node_combo.count() == 0:
            return
        
        # Get selected nodes
        from_idx = self.from_node_combo.currentIndex()
        to_idx = self.to_node_combo.currentIndex()
        from_node = self.from_node_combo.itemData(from_idx)
        to_node = self.to_node_combo.itemData(to_idx)
        
        # Get the current cost
        current_cost = self.graph.get_cost(from_node, to_node)
        
        # Get the new cost
        new_cost = self.edge_cost_spin.value()
        
        # Check if this is a new edge or updating an existing one
        is_new_edge = (current_cost == float('inf'))
        
        if is_new_edge:
            # Create a new edge with the specified cost
            self.graph.add_edge(from_node, to_node, new_cost, bidirectional=True)
            message = f"Created new edge from {from_node} to {to_node} with cost: {new_cost}"
        else:
            # Update the edge cost
            self.graph.edges[from_node][to_node] = new_cost
            self.graph.nx_graph[from_node][to_node]['weight'] = new_cost
            
            # Update reverse edge if it exists (for undirected graphs)
            if to_node in self.graph.edges and from_node in self.graph.edges[to_node]:
                self.graph.edges[to_node][from_node] = new_cost
                self.graph.nx_graph[to_node][from_node]['weight'] = new_cost
            
            message = f"Updated edge cost from {from_node} to {to_node}: {new_cost}"
        
        # Re-visualize the graph
        self.visualize_graph()
        
        # Confirmation message
        self.status_message(message)
    
    def randomize_edge_costs(self):
        """Randomize all edge costs in the graph"""
        if not self.graph:
            self.status_message("No graph exists to randomize costs", QMessageBox.Icon.Warning)
            return
            
        # Randomize costs between 0.1 and 10.0
        self.graph.randomize_edge_costs(0.1, 10.0)
        
        # Re-visualize the graph
        self.visualize_graph()
        
        # Update the current edge cost display if an edge is selected
        self.get_edge_cost()
        
        # Show confirmation message
        self.status_message("All edge costs have been randomized")
    
    def visualize_graph(self):
        """Visualize the current graph"""
        # Clear the figure
        self.graph_ax.clear()
        
        # Set figure background color based on theme
        if self.dark_mode_enabled:
            self.graph_figure.set_facecolor('#2D2D2D')
        else:
            self.graph_figure.set_facecolor('white')
        
        # Create a graph visualizer and plot
        visualizer = GraphVisualizer(self.graph)
        visualizer.fig = self.graph_figure
        visualizer.ax = self.graph_ax
        
        # Set the goal node if one is selected
        if self.goal_node_combo.count() > 0:
            goal_idx = self.goal_node_combo.currentIndex()
            goal_node = self.goal_node_combo.itemData(goal_idx)
            visualizer.set_goal(goal_node)
        
        # Adjust node colors based on theme
        if self.dark_mode_enabled:
            visualizer.default_node_color = '#4A6D8C'  # Darker blue in dark mode
            visualizer.default_edge_color = '#555555'  # Darker gray in dark mode
        else:
            visualizer.default_node_color = 'lightblue'
            visualizer.default_edge_color = 'gray'
            
        visualizer.plot_graph()
        
        # Refresh canvas
        self.graph_canvas.draw()
    
    def run_algorithms(self):
        """Run the selected algorithms"""
        # Get start and goal nodes
        if self.start_node_combo.count() == 0 or self.goal_node_combo.count() == 0:
            self.status_message("Please generate a graph first", QMessageBox.Icon.Warning)
            return
        
        # Get the actual node IDs from the combobox data
        start_idx = self.start_node_combo.currentIndex()
        goal_idx = self.goal_node_combo.currentIndex()
        start_node = self.start_node_combo.itemData(start_idx)
        goal_node = self.goal_node_combo.itemData(goal_idx)
        
        # Check which algorithms are selected
        selected_algorithms = {}
        for name, checkbox in self.algo_checkboxes.items():
            if checkbox.isChecked():
                selected_algorithms[name] = self.algorithms[name]
        
        if not selected_algorithms:
            self.status_message("Please select at least one algorithm", QMessageBox.Icon.Warning)
            return
        
        # Create a search visualizer with the selected algorithms
        self.search_visualizer = SearchVisualizer(self.graph, selected_algorithms, dark_mode=self.dark_mode_enabled)
        
        # Set the goal node for heuristic calculation
        self.search_visualizer.visualizer.set_goal(goal_node)
        
        # Run all algorithms and get results
        self.search_visualizer.run_all_algorithms(start_node, goal_node)
        
        # Update visualizations
        self.update_result_visualizations()
    
    def update_result_visualizations(self):
        """Update the visualization of search results"""
        # Set figure background colors based on theme
        if self.dark_mode_enabled:
            self.results_figure.set_facecolor('#2D2D2D')
            self.perf_figure.set_facecolor('#2D2D2D')
        else:
            self.results_figure.set_facecolor('white')
            self.perf_figure.set_facecolor('white')
            
        # Path visualization
        if self.show_all_paths_radio.isChecked():
            # Show all paths on one figure
            self.search_visualizer.visualize_all_paths(
                fig=self.results_figure, 
                ax=self.results_ax, 
                separate_figures=False
            )
        else:
            # Just show the first algorithm for now (we should add algorithm selection later)
            if self.search_visualizer.results:
                algo_name = list(self.search_visualizer.results.keys())[0]
                self.search_visualizer.visualize_path(
                    algo_name, 
                    fig=self.results_figure, 
                    ax=self.results_ax
                )
        
        # Update results canvas
        self.results_canvas.draw()
        
        # Performance charts
        self.search_visualizer.plot_performance_comparison(
            fig=self.perf_figure, 
            axs=self.perf_axes
        )
        
        # Update performance canvas
        self.perf_canvas.draw()
        
        # Results table
        table_text = self.search_visualizer.compare_results()
        self.results_table.setText(table_text)
    
    def create_default_graph(self):
        """Set up the initial state without generating a graph"""
        # Just set random graph as the default selection
        self.random_graph_radio.setChecked(True)
        
        # Initialize an empty graph
        self.graph = MapGraph()
        
        # Update algorithms with empty graph
        for algo in self.algorithms.values():
            algo.graph = self.graph
        
        self.search_visualizer = SearchVisualizer(self.graph, self.algorithms)
        
        # Update node selection combos (will be empty)
        self.update_node_selection()
        
        # Visualize the empty graph
        self.visualize_graph()
    
    def status_message(self, message, icon=QMessageBox.Icon.Information):
        """Show a status message"""
        msg_box = QMessageBox()
        msg_box.setIcon(icon)
        msg_box.setText(message)
        msg_box.setWindowTitle("Route Planning")
        
        # Apply dark mode styling if enabled
        if self.dark_mode_enabled:
            # Create a darker palette specifically for the message box
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
            msg_box.setPalette(dark_palette)
            
            # Apply a specific style for message box buttons
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #333333;
                    color: #FFFFFF;
                }
                QLabel {
                    color: #FFFFFF;
                }
                QPushButton {
                    background-color: #555555;
                    color: #FFFFFF;
                    border: 1px solid #777777;
                    padding: 5px;
                    min-width: 80px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #666666;
                }
            """)
        
        msg_box.exec()
    
    def clear_custom_graph(self):
        """Clear the custom graph and start fresh"""
        if not self.custom_graph_radio.isChecked():
            return
            
        # Create a new empty graph
        self.graph = MapGraph()
        
        # Update algorithms with new graph
        for algo in self.algorithms.values():
            algo.graph = self.graph
        
        self.search_visualizer = SearchVisualizer(self.graph, self.algorithms)
        
        # Update node selection combos
        self.update_node_selection()
        
        # Visualize the graph
        self.visualize_graph()
        
        # Show confirmation message
        self.status_message("Cleared custom graph")
    
    def generate_graph(self):
        """Generate a graph based on the selected options"""
        # Clear existing graph
        self.graph = MapGraph()
        
        # Generate based on selection
        if self.random_graph_radio.isChecked():
            num_nodes = self.node_count_spin.value()
            connectivity = self.connectivity_spin.value() / 100.0
            self.graph.generate_random_graph(num_nodes, connectivity)
            self.status_message(f"Generated random graph with {num_nodes} nodes and {connectivity:.2f} connectivity")
        
        elif self.grid_graph_radio.isChecked():
            width = self.grid_width_spin.value()
            height = self.grid_height_spin.value()
            diagonals = self.diagonal_edges_check.isChecked()
            self.graph.generate_grid_graph(width, height, diagonals)
            self.status_message(f"Generated grid graph of size {width}x{height}" + (" with diagonal edges" if diagonals else ""))
        
        # Update algorithms with new graph
        for algo in self.algorithms.values():
            algo.graph = self.graph
        
        self.search_visualizer = SearchVisualizer(self.graph, self.algorithms)
        
        # Update node selection combos
        self.update_node_selection()
        
        # Visualize the graph
        self.visualize_graph()


# Main application
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 