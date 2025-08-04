import networkx as nx
import json
import traceback
from typing import Dict, List, Any, Tuple, Optional
import math
import random
import time
import os
import re


class GraphDataExtractor:
    """
    Extracts and processes NetworkX graph data for interactive visualization.
    
    This class handles the conversion of NetworkX graph objects into JavaScript-compatible
    data structures that can be embedded in HTML templates for client-side rendering.
    """
    
    def __init__(self):
        """Initialize the graph data extractor."""
        self.color_palette = [
            'rgba(59, 130, 246, 0.7)',   # Blue
            'rgba(34, 197, 94, 0.7)',    # Green  
            'rgba(251, 191, 36, 0.7)',   # Yellow
            'rgba(168, 85, 247, 0.7)',   # Purple
            'rgba(236, 72, 153, 0.7)',   # Pink
            'rgba(156, 163, 175, 0.7)',  # Gray
            'rgba(239, 68, 68, 0.7)',    # Red
            'rgba(20, 184, 166, 0.7)',   # Teal
            'rgba(245, 101, 101, 0.7)',  # Light Red
            'rgba(129, 140, 248, 0.7)',  # Indigo
        ]
        
        self.edge_color_palette = [
            'rgba(59, 130, 246, 0.6)',   # Blue
            'rgba(34, 197, 94, 0.6)',    # Green
            'rgba(251, 191, 36, 0.6)',   # Yellow
            'rgba(168, 85, 247, 0.6)',   # Purple
            'rgba(236, 72, 153, 0.6)',   # Pink
            'rgba(156, 163, 175, 0.6)',  # Gray
            'rgba(239, 68, 68, 0.6)',    # Red
            'rgba(20, 184, 166, 0.6)',   # Teal
        ]
    
    def extract_graph_data(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Extract complete graph data from NetworkX graph.
        
        Args:
            graph: NetworkX graph object to extract data from
            
        Returns:
            Dictionary containing metadata, nodes, edges, and legend data
            
        Raises:
            ValueError: If graph is None or empty
            TypeError: If graph is not a NetworkX graph
        """
        if graph is None:
            raise ValueError("Graph cannot be None")
            
        if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            raise TypeError("Input must be a NetworkX graph object")
            
        if len(graph) == 0:
            raise ValueError("Graph cannot be empty")
        
        try:
            # Extract basic graph information
            metadata = self._extract_metadata(graph)
            
            # Extract nodes with positions and attributes
            nodes = self._extract_nodes(graph)
            
            # Extract edges with attributes
            edges = self._extract_edges(graph)
            
            # Generate legend data
            legend = self._generate_legend(nodes, edges)
            
            return {
                'metadata': metadata,
                'nodes': nodes,
                'edges': edges,
                'legend': legend
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract graph data: {str(e)}") from e
    
    def _extract_metadata(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Extract metadata information from the graph.
        
        Args:
            graph: NetworkX graph object
            
        Returns:
            Dictionary containing graph metadata
        """
        num_nodes = len(graph)
        num_edges = graph.number_of_edges()
        
        # Calculate density
        if num_nodes > 1:
            max_edges = num_nodes * (num_nodes - 1)
            if not graph.is_directed():
                max_edges //= 2
            density = num_edges / max_edges if max_edges > 0 else 0
        else:
            density = 0
        
        # Generate title based on graph characteristics
        graph_type = "Directed" if graph.is_directed() else "Undirected"
        has_anchors = any(graph.nodes[n].get('anchor', 0) == 1 for n in graph.nodes())
        anchor_info = " with Anchors" if has_anchors else ""
        
        title = f"{graph_type} Graph{anchor_info}"
        
        return {
            'title': title,
            'nodeCount': num_nodes,
            'edgeCount': num_edges,
            'isDirected': graph.is_directed(),
            'density': round(density, 3)
        }
    
    def _extract_nodes(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        nodes = []
        pos = self._get_node_positions(graph)
        for node_key in graph.nodes():
            node_data = graph.nodes[node_key]
            node_id = str(node_data['id']) if 'id' in node_data and node_data['id'] is not None else str(node_key)
            x, y = pos.get(node_key, (0, 0))
            is_anchor = node_data.get('anchor', 0) == 1

            # Build display label from all attributes except anchor, x, y, id, label
            display_label_parts = []
            for key, value in node_data.items():
                if key not in {'anchor', 'x', 'y'} and value is not None:
                    display_label_parts.append(f"{key}: {value}")
            print(f"Node {node_id} display label parts: {display_label_parts}")
            display_label = "\\n".join(display_label_parts) if display_label_parts else node_id

            node_dict = dict(node_data)
            node_dict['id'] = node_id
            node_dict['x'] = float(x)
            node_dict['y'] = float(y)
            node_dict['anchor'] = is_anchor
            # Ensure 'label' is the type/category (not a display label)
            if 'label' not in node_dict or node_dict['label'] is None:
                node_dict['label'] = self._get_node_type(node_data)
            node_dict['display_label'] = display_label  # <-- Add this line

            nodes.append(node_dict)
        return nodes
    
    def _extract_edges(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        edges = []
        for source, target, edge_data in graph.edges(data=True):
            source_id = str(graph.nodes[source].get('id', source))
            target_id = str(graph.nodes[target].get('id', target))
            edge_dict = dict(edge_data)
            edge_dict['source'] = source_id
            edge_dict['target'] = target_id
            edge_dict['directed'] = graph.is_directed()
            # Ensure 'label' is the type/category (not a display label)
            if 'label' not in edge_dict or edge_dict['label'] is None:
                edge_dict['label'] = self._get_edge_type(edge_data)
            edges.append(edge_dict)
        return edges
    
    def _get_node_positions(self, graph: nx.Graph) -> Dict[str, Tuple[float, float]]:
        """
        Get or generate node positions for layout.
        
        Args:
            graph: NetworkX graph object
            
        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        # Check if positions already exist in node attributes
        has_positions = all('x' in graph.nodes[n] and 'y' in graph.nodes[n] 
                           for n in graph.nodes())
        
        if has_positions:
            return {n: (graph.nodes[n]['x'], graph.nodes[n]['y']) 
                   for n in graph.nodes()}
        
        # Generate layout using spring layout with good parameters
        try:
            pos = nx.spring_layout(graph, k=3.0, iterations=50, seed=42)
            # Scale positions to reasonable canvas coordinates
            scale_factor = 200
            return {n: (pos[n][0] * scale_factor, pos[n][1] * scale_factor) 
                   for n in pos}
        except Exception:
            # Fallback to circular layout
            pos = nx.circular_layout(graph, scale=200)
            return {n: (pos[n][0], pos[n][1]) for n in pos}
    
    def _get_node_type(self, node_data: Dict[str, Any]) -> str:
        """
        Determine node type from node attributes.
        
        Args:
            node_data: Dictionary of node attributes
            
        Returns:
            String representing the node type
        """
        # Priority order for type determination
        type_keys = ['type', 'label', 'category', 'kind', 'class']
        
        for key in type_keys:
            if key in node_data and node_data[key] is not None:
                return str(node_data[key])
        
        return 'default'
    
    def _get_edge_type(self, edge_data: Dict[str, Any]) -> str:
        """
        Determine edge type from edge attributes.

        """
        # Priority order for type determination
        type_keys = ['type', 'label', 'relation', 'category', 'kind']
        
        for key in type_keys:
            if key in edge_data and edge_data[key] is not None:
                return str(edge_data[key])
        
        return 'default'
    
    def _generate_node_label(self, node_id: Any, node_data: Dict[str, Any]) -> str:
        """
        Generate display label for a node.
        """
        # Try to use explicit label first
        if 'label' in node_data and node_data['label'] is not None:
            return str(node_data['label'])
        
        # Use node ID as fallback
        return str(node_id)
    
    def _generate_edge_label(self, edge_data: Dict[str, Any]) -> str:
        """
        Generate display label for an edge.
        """
        # Priority order for label determination
        label_keys = ['label', 'type', 'relation', 'weight']
        
        for key in label_keys:
            if key in edge_data and edge_data[key] is not None:
                value = edge_data[key]
                # Format numeric values nicely
                if isinstance(value, float):
                    return f"{value:.2f}" if abs(value) < 100 else f"{value:.1e}"
                return str(value)
        
        return ''
    
    def _extract_node_metadata(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from node attributes, excluding special keys.
        """
        # Keys to exclude from metadata
        excluded_keys = {'id', 'x', 'y', 'type', 'label', 'anchor'}
        
        metadata = {}
        for key, value in node_data.items():
            if key not in excluded_keys and value is not None:
                # Format values for JSON serialization
                if isinstance(value, (int, float, str, bool, list, dict)):
                    metadata[key] = value
                else:
                    metadata[key] = str(value)
        
        return metadata
    
    def _extract_edge_metadata(self, edge_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from edge attributes, excluding special keys.
        """
        # Keys to exclude from metadata
        excluded_keys = {'type', 'label', 'source', 'target', 'directed'}
        
        metadata = {}
        for key, value in edge_data.items():
            if key not in excluded_keys and value is not None:
                # Format values for JSON serialization
                if isinstance(value, (int, float, str, bool, list, dict)):
                    metadata[key] = value
                else:
                    metadata[key] = str(value)
        
        return metadata
    
    def _generate_legend(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        node_types = set(node['label'] for node in nodes)
        edge_types = set(edge['label'] for edge in edges)

        node_legend = []
        for i, node_type in enumerate(sorted(node_types)):
            color = self.color_palette[i % len(self.color_palette)]
            node_legend.append({
                'label': node_type,
                'color': color,
                'description': f"{node_type.title()} nodes"
            })

        edge_legend = []
        for i, edge_type in enumerate(sorted(edge_types)):
            color = self.edge_color_palette[i % len(self.edge_color_palette)]
            edge_legend.append({
                'label': edge_type,
                'color': color,
                'description': f"{edge_type.replace('_', ' ').title()} edges"
            })

        return {
            'nodeTypes': node_legend,
            'edgeTypes': edge_legend
        }


def extract_graph_data(graph: nx.Graph) -> Dict[str, Any]:
    """
    Convenience function to extract graph data using GraphDataExtractor.
    """
    extractor = GraphDataExtractor()
    return extractor.extract_graph_data(graph)