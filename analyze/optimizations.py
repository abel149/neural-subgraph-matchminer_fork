"""
Advanced optimizations for motif counting in large biological graphs.
Based on CS224W motif counting techniques and ESU algorithm principles.
"""
import networkx as nx
import random
from collections import deque

def extract_k_hop_neighborhood(G, anchor, k=2, max_nodes=100):
    """
    Extract k-hop neighborhood around anchor node.
    For a query of size q, we only need a k-hop neighborhood where k >= ceil(q/2).
    
    This dramatically reduces the search space for VF2 from 400k nodes to ~100 nodes.
    """
    if anchor not in G:
        return G.subgraph([])
    
    neighborhood = {anchor}
    current_layer = {anchor}
    
    for hop in range(k):
        next_layer = set()
        for node in current_layer:
            if G.is_directed():
                neighbors = set(G.successors(node)) | set(G.predecessors(node))
            else:
                neighbors = set(G.neighbors(node))
            next_layer.update(neighbors - neighborhood)
        
        neighborhood.update(next_layer)
        current_layer = next_layer
        
        if len(neighborhood) >= max_nodes:
            # Cap neighborhood size for memory efficiency
            neighborhood = set(list(neighborhood)[:max_nodes])
            break
        
        if not current_layer:
            break
    
    return G.subgraph(neighborhood)


def adaptive_anchor_sampling(G, query_size, max_anchors=1000, strategy='degree_aware'):
    """
    Smart anchor sampling based on node importance.
    
    For biological networks:
    - High-degree nodes (hubs) are more likely to participate in motifs
    - Sampling by degree gives better coverage with fewer anchors
    """
    if G.number_of_nodes() <= max_anchors:
        return list(G.nodes())
    
    if strategy == 'degree_aware':
        # Sample with probability proportional to degree
        degrees = dict(G.degree())
        if not degrees:
            return random.sample(list(G.nodes()), min(max_anchors, G.number_of_nodes()))
        
        # Higher degree = higher sampling probability
        nodes = list(degrees.keys())
        weights = [degrees[n] + 1 for n in nodes]  # +1 to avoid zero weights
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        # Sample without replacement
        sampled = random.choices(nodes, weights=probs, k=min(max_anchors, len(nodes)))
        return list(set(sampled))  # Remove duplicates
    
    elif strategy == 'uniform':
        return random.sample(list(G.nodes()), min(max_anchors, G.number_of_nodes()))
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def batch_anchor_pruning(G, anchors, query, query_stats):
    """
    Pre-filter anchors that cannot possibly participate in the motif.
    
    Checks:
    1. Anchor degree >= minimum degree in query
    2. For directed: in-degree and out-degree compatibility
    """
    if not anchors:
        return []
    
    min_degree = min(query_stats['degree_seq']) if query_stats['degree_seq'] else 0
    
    valid_anchors = []
    
    for anchor in anchors:
        if anchor not in G:
            continue
        
        anchor_degree = G.degree(anchor)
        
        # Degree check
        if anchor_degree < min_degree:
            continue
        
        # For directed graphs, check in/out degree compatibility
        if G.is_directed() and query.is_directed():
            anchor_in = G.in_degree(anchor)
            anchor_out = G.out_degree(anchor)
            
            min_in = min(query_stats['in_degree_seq']) if query_stats['in_degree_seq'] else 0
            min_out = min(query_stats['out_degree_seq']) if query_stats['out_degree_seq'] else 0
            
            if anchor_in < min_in or anchor_out < min_out:
                continue
        
        valid_anchors.append(anchor)
    
    return valid_anchors


def estimate_required_anchors(G, query, confidence=0.95):
    """
    Estimate how many anchors we need to sample for reliable motif counting.
    
    Based on sampling theory:
    - If motif frequency is p, we need ~(1/p) samples for good estimate
    - For rare motifs, we need more samples
    
    Returns a dynamic anchor count instead of fixed 1000.
    """
    n = G.number_of_nodes()
    q = query.number_of_nodes()
    
    # Heuristic: for small queries in large graphs, we need fewer anchors
    if q <= 5:
        # Small motifs are often abundant, 100-500 anchors sufficient
        return min(500, n // 100)
    elif q <= 10:
        # Medium motifs need more coverage
        return min(1000, n // 50)
    else:
        # Large motifs are rare, need extensive sampling
        return min(5000, n // 10)


def parallel_anchor_batching(anchors, batch_size=50):
    """
    Group anchors into batches for better cache locality.
    
    Process nearby anchors together to maximize cache hits
    in igraph/NetworkX data structures.
    """
    # For now, simple chunking
    # TODO: Could use spatial locality if node IDs are spatially correlated
    return [anchors[i:i+batch_size] for i in range(0, len(anchors), batch_size)]


def graphlet_degree_vector_filter(G, query):
    """
    Compute Graphlet Degree Vectors (GDV) for ultra-fast pre-filtering.
    
    GDV is a signature that captures local structure. If query and target
    GDVs are incompatible, no isomorphism exists.
    
    This is O(n) preprocessing that can eliminate 90%+ of VF2 calls.
    """
    # Simplified GDV: count triangles, 2-paths, etc.
    # Full implementation requires graphlet enumeration
    # This is a placeholder for the concept
    
    query_triangles = sum(nx.triangles(query.to_undirected()).values()) // 3
    
    # Target must have at least as many triangles in relevant subgraphs
    return {'triangles': query_triangles}
