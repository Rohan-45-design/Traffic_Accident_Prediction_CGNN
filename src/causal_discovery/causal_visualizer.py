"""
Causal Graph Visualizer
======================
Creates visualizations of discovered causal graphs
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path

class CausalVisualizer:
    """Visualizes causal graphs and relationships"""
    
    def __init__(self, figsize=(12, 8)):
        """Initialize visualizer"""
        self.figsize = figsize
        plt.style.use('default')
    
    def create_graph_visualization(self, graph, output_path):
        """Create and save graph visualization"""
        print("📊 Creating causal graph visualization...")
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Create layout
        pos = self._create_layout(graph)
        
        # Draw nodes
        node_colors = ['lightblue' if graph.out_degree(node) > graph.in_degree(node) 
                      else 'lightcoral' if graph.in_degree(node) > graph.out_degree(node)
                      else 'lightgray' for node in graph.nodes()]
        
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=2000, alpha=0.7, ax=ax)
        
        # Draw edges with varying thickness based on strength
        edge_weights = [graph[u][v].get('strength', 1.0) for u, v in graph.edges()]
        edge_widths = [w * 3 for w in edge_weights]  # Scale for visibility
        
        nx.draw_networkx_edges(graph, pos, width=edge_widths, 
                              alpha=0.6, edge_color='gray', 
                              arrowsize=20, arrowstyle='->', ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='bold', ax=ax)
        
        # Add title and legend
        ax.set_title('Discovered Causal Relationships\nTraffic Accident Factors', 
                    fontsize=14, fontweight='bold')
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='Primarily Causes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                      markersize=10, label='Primarily Effects'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                      markersize=10, label='Balanced')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout()
        
        # Save visualization
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved: {output_path}")
        return output_path
    
    def _create_layout(self, graph):
        """Create optimal layout for causal graph"""
        try:
            # Try hierarchical layout if it's a DAG
            if nx.is_directed_acyclic_graph(graph):
                return nx.spring_layout(graph, k=2, iterations=50)
            else:
                return nx.circular_layout(graph)
        except:
            return nx.random_layout(graph)
