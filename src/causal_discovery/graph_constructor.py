"""
Causal Graph Constructor
=======================
Builds structured causal graphs from discovered relationships
"""

import numpy as np
import pandas as pd
import networkx as nx

class GraphConstructor:
    """Constructs causal graphs from discovered relationships"""
    
    def __init__(self):
        """Initialize graph constructor"""
        pass
    
    def build_causal_graph(self, relationships, variable_names):
        """
        Build causal graph from relationships
        
        Parameters:
        -----------
        relationships : list
            List of causal relationships
        variable_names : list
            List of all variable names
            
        Returns:
        --------
        graph : networkx.DiGraph
            Directed graph representing causal structure
        """
        print("🔗 Building causal graph structure...")
        
        # Create directed graph
        graph = nx.DiGraph()
        
        # Add all nodes
        for var in variable_names:
            graph.add_node(var, node_type='variable')
        
        # Add edges from relationships
        for rel in relationships:
            cause = rel['cause']
            effect = rel['effect']
            strength = rel.get('strength', 1.0)
            rel_type = rel.get('type', 'causal')
            
            # Add edge with attributes
            graph.add_edge(cause, effect, 
                          weight=strength,
                          relationship_type=rel_type,
                          strength=strength)
        
        # Calculate graph statistics
        self._calculate_graph_metrics(graph)
        
        print(f"✅ Graph constructed: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        return graph
    
    def _calculate_graph_metrics(self, graph):
        """Calculate various graph metrics"""
        # Basic metrics
        graph.graph['num_nodes'] = len(graph.nodes)
        graph.graph['num_edges'] = len(graph.edges)
        graph.graph['density'] = nx.density(graph)
        
        # Centrality measures
        try:
            in_centrality = nx.in_degree_centrality(graph)
            out_centrality = nx.out_degree_centrality(graph)
            
            # Find most influential variables
            most_influential_cause = max(out_centrality, key=out_centrality.get)
            most_influenced_effect = max(in_centrality, key=in_centrality.get)
            
            graph.graph['most_influential_cause'] = most_influential_cause
            graph.graph['most_influenced_effect'] = most_influenced_effect
            graph.graph['max_out_centrality'] = out_centrality[most_influential_cause]
            graph.graph['max_in_centrality'] = in_centrality[most_influenced_effect]
            
        except:
            graph.graph['most_influential_cause'] = 'unknown'
            graph.graph['most_influenced_effect'] = 'unknown'
        
        # Check for cycles (should be rare in causal graphs)
        try:
            is_dag = nx.is_directed_acyclic_graph(graph)
            graph.graph['is_dag'] = is_dag
            
            if not is_dag:
                cycles = list(nx.simple_cycles(graph))
                graph.graph['cycles'] = cycles[:5]  # Store first 5 cycles
        except:
            graph.graph['is_dag'] = True
    
    def get_graph_summary(self, graph):
        """Get summary statistics of the causal graph"""
        summary = {
            'nodes': len(graph.nodes),
            'edges': len(graph.edges),
            'density': graph.graph.get('density', 0),
            'is_dag': graph.graph.get('is_dag', True),
            'most_influential_cause': graph.graph.get('most_influential_cause', 'none'),
            'most_influenced_effect': graph.graph.get('most_influenced_effect', 'none')
        }
        
        # Edge strength statistics
        edge_weights = [graph[u][v]['weight'] for u, v in graph.edges()]
        if edge_weights:
            summary.update({
                'avg_edge_strength': np.mean(edge_weights),
                'max_edge_strength': np.max(edge_weights),
                'min_edge_strength': np.min(edge_weights)
            })
        
        return summary
    
    def extract_causal_paths(self, graph, source=None, target=None, max_length=3):
        """Extract causal paths between variables"""
        if source and target:
            try:
                paths = list(nx.all_simple_paths(graph, source, target, cutoff=max_length))
                return paths
            except:
                return []
        else:
            # Find all paths up to max_length
            all_paths = []
            for source in graph.nodes():
                for target in graph.nodes():
                    if source != target:
                        try:
                            paths = list(nx.all_simple_paths(graph, source, target, cutoff=max_length))
                            all_paths.extend(paths)
                        except:
                            continue
            return all_paths
