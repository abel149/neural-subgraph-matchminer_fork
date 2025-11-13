import argparse
import csv
import time
import os
import json
import concurrent.futures
import sys
import gc
import collections
from functools import lru_cache

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid, KarateClub, QM7b
from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils

import torch_geometric.nn as pyg_nn
from matplotlib import cm

# Try to import local modules, but continue if they don't exist
try:
    from common import data
    from common import models
    from common import utils
    from subgraph_mining import decoder
except ImportError:
    print("Warning: Some local modules could not be imported")

from tqdm import tqdm
import matplotlib.pyplot as plt

from multiprocessing import Pool
import random
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from itertools import permutations
from queue import PriorityQueue
import matplotlib.colors as mcolors
import networkx as nx
import networkx.algorithms.isomorphism as iso
import pickle
import torch.multiprocessing as mp
from sklearn.decomposition import PCA
from itertools import combinations

# Constants
MAX_SEARCH_TIME = 1800  
MAX_MATCHES_PER_QUERY = 10000
DEFAULT_SAMPLE_ANCHORS = 1000
CHECKPOINT_INTERVAL = 100  
SMALL_GRAPH_THRESHOLD = 8
CACHE_SIZE = 1000

class GraphletCounter:
    def __init__(self, args):
        self.args = args
        self.stats_cache = {}
        
    def compute_graph_stats(self, G):
        """Compute graph statistics for filtering with caching."""
        graph_key = self._graph_to_key(G)
        
        if graph_key in self.stats_cache:
            return self.stats_cache[graph_key]
            
        stats = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'is_directed': G.is_directed(),
            'degree_seq': tuple(sorted([d for _, d in G.degree()], reverse=True)),
            'avg_degree': sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1)
        }
        
        if G.is_directed():
            stats['in_degree_seq'] = tuple(sorted([d for _, d in G.in_degree()], reverse=True))
            stats['out_degree_seq'] = tuple(sorted([d for _, d in G.out_degree()], reverse=True))
        
        try:
            if G.is_directed():
                stats['n_components'] = nx.number_weakly_connected_components(G)
            else:
                stats['n_components'] = nx.number_connected_components(G)
        except:
            stats['n_components'] = 1 
            
        self.stats_cache[graph_key] = stats
        return stats

    def _graph_to_key(self, G):
        """Create a hashable key for graph caching."""
        return (G.number_of_nodes(), G.number_of_edges(), G.is_directed())

    def can_be_isomorphic(self, query_stats, target_stats):
        """Fast pre-check for possible isomorphism."""
        if query_stats['n_nodes'] > target_stats['n_nodes']:
            return False
        if query_stats['n_edges'] > target_stats['n_edges']:
            return False
        
        if query_stats['is_directed'] != target_stats['is_directed']:
            return False
        
        # Degree sequence majorization check
        if (query_stats['degree_seq'] and target_stats['degree_seq'] and
            any(qd > td for qd, td in zip(query_stats['degree_seq'], target_stats['degree_seq']))):
            return False
        
        # For directed graphs, check in/out degrees
        if query_stats['is_directed']:
            if (query_stats['in_degree_seq'] and target_stats['in_degree_seq'] and
                any(qd > td for qd, td in zip(query_stats['in_degree_seq'], target_stats['in_degree_seq']))):
                return False
            if (query_stats['out_degree_seq'] and target_stats['out_degree_seq'] and
                any(qd > td for qd, td in zip(query_stats['out_degree_seq'], target_stats['out_degree_seq']))):
                return False
        
        if query_stats['avg_degree'] > target_stats['avg_degree'] * 1.5:  
            return False
        
        return True

    def quick_structural_check(self, query, target):
        """Ultra-fast structural check before isomorphism."""
        if query.number_of_nodes() > target.number_of_nodes():
            return False
        if query.number_of_edges() > target.number_of_edges():
            return False
        
        # Quick degree check
        q_max_degree = max(dict(query.degree()).values())
        t_max_degree = max(dict(target.degree()).values())
        if q_max_degree > t_max_degree:
            return False
            
        return True

def arg_parse():
    parser = argparse.ArgumentParser(description='count graphlets in a graph')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset file path')
    parser.add_argument('--queries_path', type=str, required=True, help='Path to query patterns')
    parser.add_argument('--out_path', type=str, required=True, help='Output directory path')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--count_method', type=str, default='bin', choices=['bin', 'freq'], help='Counting method')
    parser.add_argument('--baseline', type=str, default='none', help='Baseline method')
    parser.add_argument('--node_anchored', action="store_true", help='Use node anchored counting')
    parser.add_argument('--max_query_size', type=int, default=20, help='Maximum query size to process')
    parser.add_argument('--sample_anchors', type=int, default=DEFAULT_SAMPLE_ANCHORS, help='Number of anchor nodes to sample for large graphs')
    parser.add_argument('--checkpoint_file', type=str, default="checkpoint.json", help='File to save/load progress')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for processing')
    parser.add_argument('--timeout', type=int, default=MAX_SEARCH_TIME, help='Timeout per task in seconds')
    parser.add_argument('--use_sampling', action="store_true", help='Use node sampling for very large graphs')
    parser.add_argument('--graph_type', type=str, default='auto', choices=['directed', 'undirected', 'auto'],
                       help='Graph type: directed, undirected, or auto-detect')
    parser.add_argument('--aggressive_filter', action="store_true", help='Use aggressive pre-filtering')
    parser.add_argument('--cache_size', type=int, default=CACHE_SIZE, help='Size of isomorphism cache')
    
    return parser.parse_args()

def load_networkx_graph_optimized(filepath, directed=None):
    """Load a Networkx graph from pickle format with memory optimization."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, (nx.Graph, nx.DiGraph)):
            return optimize_graph_memory(data, directed)
        
        # Handle dictionary format
        if directed is None:
            directed = data.get('directed', False)
        
        graph_type = nx.DiGraph if directed else nx.Graph
        graph = graph_type()
        
        # Efficient node addition
        if 'nodes' in data:
            graph.add_nodes_from(data['nodes'])
        elif 'node_list' in data:
            graph.add_nodes_from(data['node_list'])
        
        # Efficient edge addition
        if 'edges' in data:
            graph.add_edges_from(data['edges'])
        elif 'edge_list' in data:
            graph.add_edges_from(data['edge_list'])
        
        return optimize_graph_memory(graph, directed)
        
    except Exception as e:
        print(f"Error loading graph from {filepath}: {str(e)}")
        raise

def optimize_graph_memory(G, directed=None):
    """Reduce memory usage of NetworkX graphs."""
    if directed is not None:
        if directed and not G.is_directed():
            G = G.to_directed()
        elif not directed and G.is_directed():
            G = G.to_undirected()
    
    # Remove unnecessary attributes to save memory
    for node in list(G.nodes()):
        # Keep only essential attributes
        essential_attrs = {'label', 'id', 'anchor'}
        node_data = G.nodes[node]
        attrs_to_remove = [attr for attr in node_data if attr not in essential_attrs]
        for attr in attrs_to_remove:
            if attr in node_data:
                del node_data[attr]
    
    return G

def count_small_graphlet(query, target, node_anchored=False, anchor=None, timeout=300):
    """Optimized counting for small graphs using canonical labeling."""
    start_time = time.time()
    
    # Use Weisfeiler-Lehman graph hashing for quick rejection
    try:
        query_hash = nx.weisfeiler_lehman_graph_hash(query)
        target_hash = nx.weisfeiler_lehman_graph_hash(target)
        
        # If hashes don't match, no isomorphism possible
        if query_hash != target_hash:
            return 0
    except:
        # Fallback if WL hashing fails
        pass
    
    # For very small graphs, use direct isomorphism
    if query.number_of_nodes() <= 6:
        if node_anchored and anchor is not None:
            # Create node match function for anchored search
            def node_match(attr1, attr2):
                if anchor in attr1 and anchor in attr2:
                    return attr1.get('anchor', 0) == attr2.get('anchor', 0)
                return True
            
            if target.is_directed():
                matcher = iso.DiGraphMatcher(target, query, node_match=node_match)
            else:
                matcher = iso.GraphMatcher(target, query, node_match=node_match)
        else:
            if target.is_directed():
                matcher = iso.DiGraphMatcher(target, query)
            else:
                matcher = iso.GraphMatcher(target, query)
        
        return 1 if matcher.subgraph_is_isomorphic() else 0
    
    return 0

def count_graphlets_helper_optimized(inp):
    """Optimized isomorphism checking with multiple optimization strategies."""
    i, query, target, method, node_anchored, anchor_or_none, timeout, counter = inp
    
    start_time = time.time()
    
    # Ultra-fast structural check
    if not counter.quick_structural_check(query, target):
        return i, 0
    
    # Use optimized method for small graphs
    if query.number_of_nodes() <= SMALL_GRAPH_THRESHOLD:
        count = count_small_graphlet(query, target, node_anchored, anchor_or_none, timeout)
        return i, count
    
    # Detailed statistics check
    query_stats = counter.compute_graph_stats(query)
    target_stats = counter.compute_graph_stats(target)
    if not counter.can_be_isomorphic(query_stats, target_stats):
        return i, 0
    
    # Clean graphs
    query = query.copy()
    query.remove_edges_from(nx.selfloop_edges(query))
    target = target.copy()
    target.remove_edges_from(nx.selfloop_edges(target))

    count = 0
    try:
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Task {i} timed out after {timeout} seconds")
            
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(min(timeout, 600))
        
        if method == "freq":
            if query.is_directed():
                ismags = nx.isomorphism.DiGraphMatcher(query, query)
            else:
                ismags = nx.isomorphism.ISMAGS(query, query)
            n_symmetries = len(list(ismags.isomorphisms_iter(symmetry=False)))
        
        if method == "bin":
            if node_anchored:
                # Optimized anchored matching
                if anchor_or_none not in target.nodes():
                    return i, 0
                    
                nx.set_node_attributes(target, 0, name="anchor")
                target.nodes[anchor_or_none]["anchor"] = 1
                
                if target.is_directed():
                    matcher = iso.DiGraphMatcher(target, query,
                        node_match=iso.categorical_node_match(["anchor"], [0]))
                else:
                    matcher = iso.GraphMatcher(target, query,
                        node_match=iso.categorical_node_match(["anchor"], [0]))
                
                count = int(matcher.subgraph_is_isomorphic())
            else:
                # Use faster isomorphism check for binary counting
                if target.is_directed():
                    matcher = iso.DiGraphMatcher(target, query)
                else:
                    matcher = iso.GraphMatcher(target, query)
                
                count = int(matcher.subgraph_is_isomorphic())
                
        elif method == "freq":
            # Frequency counting with early termination
            if target.is_directed():
                matcher = iso.DiGraphMatcher(target, query)
            else:
                matcher = iso.GraphMatcher(target, query)
            
            count = 0
            for _ in matcher.subgraph_isomorphisms_iter():
                if time.time() - start_time > timeout:
                    break
                count += 1
                if count >= MAX_MATCHES_PER_QUERY:
                    break
            
            if method == "freq" and n_symmetries > 0:
                count = count / n_symmetries
        
        signal.alarm(0)
            
    except TimeoutError:
        count = 0
    except Exception as e:
        print(f"Error processing query {i}: {str(e)}")
        count = 0
        
    processing_time = time.time() - start_time
    if processing_time > 5: 
        print(f"Query {i} processed in {processing_time:.2f}s, count: {count}")
        
    return i, count

def save_checkpoint(n_matches, checkpoint_file):
    """Save checkpoint with compression."""
    # Only save non-zero matches to reduce file size
    non_zero_matches = {str(k): v for k, v in n_matches.items() if v > 0}
    with open(checkpoint_file, 'w') as f:
        json.dump(non_zero_matches, f, separators=(',', ':'))  # Compact JSON
    print(f"Checkpoint saved with {len(non_zero_matches)} non-zero matches")

def load_checkpoint(checkpoint_file):
    """Load checkpoint with error handling."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                return defaultdict(float, {int(k): v for k, v in checkpoint.items()})
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error loading checkpoint: {e}, starting fresh")
    return defaultdict(float)

def sample_subgraphs_efficient(target, n_samples=10, max_size=1000):
    """Efficient subgraph sampling using degree-biased sampling."""
    subgraphs = []
    nodes = list(target.nodes())
    
    # Precompute degrees for biased sampling
    degrees = dict(target.degree())
    high_degree_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:1000]
    
    for _ in range(n_samples):
        # Prefer high-degree starting nodes
        start_node = random.choice(high_degree_nodes if high_degree_nodes else nodes)
        subgraph_nodes = {start_node}
        
        # Use set operations for faster neighbor collection
        if target.is_directed():
            neighbors = set(target.successors(start_node)) | set(target.predecessors(start_node))
        else:
            neighbors = set(target.neighbors(start_node))
        
        frontier = list(neighbors - subgraph_nodes)
        random.shuffle(frontier)  # Random exploration
        
        while len(subgraph_nodes) < max_size and frontier:
            next_node = frontier.pop()
            if next_node not in subgraph_nodes:
                subgraph_nodes.add(next_node)
                if target.is_directed():
                    new_neighbors = set(target.successors(next_node)) | set(target.predecessors(next_node))
                else:
                    new_neighbors = set(target.neighbors(next_node))
                # Add new neighbors to frontier, avoiding duplicates
                frontier.extend(list(new_neighbors - subgraph_nodes - set(frontier)))
        
        sg = target.subgraph(subgraph_nodes)
        if sg.number_of_nodes() > 0:
            subgraphs.append(sg)
        
    return subgraphs

def pre_filter_matches(queries, targets, counter, args):
    """Aggressive pre-filtering to reduce search space."""
    print("Pre-filtering impossible matches...")
    
    # Compute statistics in parallel
    with Pool(processes=min(args.n_workers, 8)) as pool:
        target_stats = pool.map(counter.compute_graph_stats, targets)
        query_stats = pool.map(counter.compute_graph_stats, queries)
    
    # Filter targets that can potentially match any query
    viable_target_indices = set()
    
    for t_idx, t_stats in enumerate(target_stats):
        for q_idx, q_stats in enumerate(query_stats):
            if counter.can_be_isomorphic(q_stats, t_stats):
                viable_target_indices.add(t_idx)
                break  # Only need one match to keep target
    
    filtered_targets = [targets[i] for i in viable_target_indices]
    
    print(f"Pre-filtering: {len(filtered_targets)}/{len(targets)} targets remain")
    return queries, filtered_targets

def prepare_tasks_optimized(queries, targets, args, n_matches, counter):
    """Prepare tasks with intelligent sampling and batching."""
    inp = []
    
    print("Preparing tasks with optimized sampling...")
    
    for i, query in enumerate(queries):
        if query.number_of_nodes() > args.max_query_size:
            continue
            
        query_stats = counter.compute_graph_stats(query)
        
        for t_idx, target in enumerate(targets):
            task_id = f"{i}_{t_idx}"
            
            # Skip already processed tasks
            if task_id in n_matches:
                continue
                
            target_stats = counter.compute_graph_stats(target)
            
            # Early filtering
            if not counter.can_be_isomorphic(query_stats, target_stats):
                continue
            
            if args.node_anchored:
                # Smart anchor sampling based on degree
                if target.number_of_nodes() > args.sample_anchors:
                    degrees = dict(target.degree())
                    # Sample anchors proportional to degree
                    nodes = list(target.nodes())
                    weights = [degrees[node] + 1 for node in nodes]  # +1 to avoid zero
                    total_weight = sum(weights)
                    probabilities = [w / total_weight for w in weights]
                    
                    anchors = np.random.choice(nodes, 
                                             size=min(args.sample_anchors, len(nodes)),
                                             p=probabilities,
                                             replace=False).tolist()
                else:
                    anchors = list(target.nodes)
                    
                for anchor in anchors:
                    inp.append((i, query, target, args.count_method, 
                              args.node_anchored, anchor, args.timeout, counter))
            else:
                inp.append((i, query, target, args.count_method, 
                          args.node_anchored, None, args.timeout, counter))
    
    print(f"Generated {len(inp)} tasks")
    return inp

def count_graphlets_optimized(queries, targets, args):
    """Optimized counting with better resource management."""
    print(f"Processing {len(queries)} queries across {len(targets)} targets")
    
    counter = GraphletCounter(args)
    n_matches = load_checkpoint(args.checkpoint_file)
    
    # Load problematic tasks
    problematic_tasks_file = "problematic_tasks.json"
    problematic_tasks = set()
    if os.path.exists(problematic_tasks_file):
        with open(problematic_tasks_file, 'r') as f:
            try:
                problematic_tasks = set(json.load(f))
                print(f"Loaded {len(problematic_tasks)} problematic tasks to skip")
            except:
                problematic_tasks = set()
    
    # Aggressive pre-filtering
    if args.aggressive_filter:
        queries, targets = pre_filter_matches(queries, targets, counter, args)
    
    # Sampling for very large graphs
    if args.use_sampling and any(t.number_of_nodes() > 100000 for t in targets):
        sampled_targets = []
        for target in targets:
            if target.number_of_nodes() > 100000:
                print(f"Sampling from large graph ({target.number_of_nodes()} nodes)")
                sampled_targets.extend(sample_subgraphs_efficient(target, n_samples=10, max_size=5000))
            else:
                sampled_targets.append(target)
        targets = sampled_targets
        print(f"After sampling: {len(targets)} target graphs")
    
    # Prepare tasks
    inp = prepare_tasks_optimized(queries, targets, args, n_matches, counter)
    
    # Dynamic batch sizing based on query complexity
    if queries:
        avg_query_size = np.mean([q.number_of_nodes() for q in queries])
        dynamic_batch_size = max(10, args.batch_size // max(1, int(avg_query_size / 5)))
    else:
        dynamic_batch_size = args.batch_size
    
    print(f"Using dynamic batch size: {dynamic_batch_size}")
    
    n_done = 0
    last_checkpoint = time.time()
    last_gc = time.time()
   
    with Pool(processes=args.n_workers) as pool:
        for batch_start in range(0, len(inp), dynamic_batch_size):
            batch_end = min(batch_start + dynamic_batch_size, len(inp))
            batch = inp[batch_start:batch_end]

            print(f"Processing batch {batch_start}-{batch_end} out of {len(inp)}")
            batch_start_time = time.time()

            results = pool.imap_unordered(count_graphlets_helper_optimized, batch)

            for result in results:
                current_time = time.time()
                
                # Check for batch timeout
                if current_time - batch_start_time > 1800:  # 30 minutes
                    print(f"Batch timeout, marking remaining tasks as problematic")
                    for task in batch:
                        i = task[0]
                        task_id = f"{i}_{batch_start}"
                        problematic_tasks.add(task_id)
                    break

                i, n = result
                n_matches[i] += n
                n_done += 1

                # Progress reporting
                if n_done % 20 == 0:
                    completed_queries = sum(1 for v in n_matches.values() if v > 0)
                    print(f"Progress: {n_done}/{len(inp)} tasks, "
                          f"{completed_queries}/{len(n_matches)} queries have matches")

                # Checkpoint and cleanup
                if current_time - last_checkpoint > 300:  # 5 minutes
                    save_checkpoint(n_matches, args.checkpoint_file)
                    with open(problematic_tasks_file, 'w') as f:
                        json.dump(list(problematic_tasks), f)
                    last_checkpoint = current_time

                # Garbage collection
                if current_time - last_gc > 120:  # 2 minutes
                    gc.collect()
                    last_gc = current_time

            # Final checkpoint for this batch
            save_checkpoint(n_matches, args.checkpoint_file)
            with open(problematic_tasks_file, 'w') as f:
                json.dump(list(problematic_tasks), f)

    print(f"\nCompleted! Processed {n_done} total tasks")
    return [n_matches.get(i, 0) for i in range(len(queries))]

def generate_one_baseline_optimized(args):
    """Optimized baseline generation."""
    i, query, targets, method = args

    if query.number_of_nodes() == 0:
        return query

    MAX_ATTEMPTS = 50  # Reduced attempts
    for attempt in range(MAX_ATTEMPTS):
        try:
            graph = random.choice(targets)
            if graph.number_of_nodes() == 0:
                continue

            if method == "radial":
                # Use high-degree nodes for better subgraphs
                degrees = dict(graph.degree())
                if degrees:
                    start_node = max(degrees.items(), key=lambda x: x[1])[0]
                else:
                    start_node = random.choice(list(graph.nodes()))
                
                # Efficient BFS with depth limit
                visited = {start_node}
                queue = collections.deque([(start_node, 0)])
                
                while queue and len(visited) < query.number_of_nodes() * 2:
                    current, depth = queue.popleft()
                    if depth < 3:  # Limit depth
                        neighbors = set(graph.neighbors(current)) if not graph.is_directed() else \
                                  set(graph.successors(current)) | set(graph.predecessors(current))
                        for neighbor in neighbors - visited:
                            visited.add(neighbor)
                            queue.append((neighbor, depth + 1))
                
                subgraph = graph.subgraph(visited)
                if subgraph.number_of_nodes() >= query.number_of_nodes():
                    # Take random subset of desired size
                    sampled_nodes = random.sample(list(visited), query.number_of_nodes())
                    final_subgraph = graph.subgraph(sampled_nodes)
                    return nx.convert_node_labels_to_integers(final_subgraph)

            elif method == "tree":
                start_node = random.choice(list(graph.nodes()))
                visited = {start_node}
                frontier = list(set(graph.neighbors(start_node)) - visited)
                
                while len(visited) < query.number_of_nodes() and frontier:
                    new_node = random.choice(frontier)
                    visited.add(new_node)
                    # Add new neighbors to frontier
                    new_neighbors = set(graph.neighbors(new_node)) - visited
                    frontier.extend(new_neighbors)
                    frontier = [n for n in frontier if n not in visited]
                
                if len(visited) >= query.number_of_nodes():
                    sampled_nodes = random.sample(list(visited), query.number_of_nodes())
                    subgraph = graph.subgraph(sampled_nodes)
                    return nx.convert_node_labels_to_integers(subgraph)

        except Exception as e:
            continue

    print(f"[WARN] Baseline not found for query {i} after {MAX_ATTEMPTS} attempts")
    return nx.DiGraph() if query.is_directed() else nx.Graph()

def gen_baseline_queries_optimized(queries, targets, method="radial", node_anchored=False):
    """Optimized baseline generation with progress tracking."""
    print(f"Generating {len(queries)} baseline queries using method: {method}")
    
    args_list = [(i, query, targets, method) for i, query in enumerate(queries)]
    
    with Pool(processes=min(os.cpu_count(), 8)) as pool:
        results = list(tqdm(pool.imap(generate_one_baseline_optimized, args_list), 
                          total=len(queries), 
                          desc="Generating baselines"))
    
    return results

def load_dataset(dataset_name, use_directed):
    """Load target dataset."""
    if dataset_name.endswith('.pkl'):
        print(f"Loading graph from {dataset_name}")
        graph = load_networkx_graph_optimized(dataset_name, use_directed)
        graph_type = "directed" if graph.is_directed() else "undirected"
        print(f"Loaded {graph_type} graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return [graph]
    else:
        # For other dataset types, create a simple graph
        print(f"Creating synthetic graph for dataset: {dataset_name}")
        G = nx.erdos_renyi_graph(100, 0.1)  # Simple random graph as fallback
        if use_directed:
            G = G.to_directed()
        return [G]

def load_queries(queries_path, use_directed):
    """Load query patterns."""
    print(f"Loading queries from {queries_path}")
    try:
        with open(queries_path, "rb") as f:
            queries = pickle.load(f)
        
        # Ensure queries are in the correct format
        processed_queries = []
        for query in queries:
            if isinstance(query, (nx.Graph, nx.DiGraph)):
                if use_directed and not query.is_directed():
                    query = query.to_directed()
                elif not use_directed and query.is_directed():
                    query = query.to_undirected()
                processed_queries.append(query)
            else:
                # Handle case where queries might be in different format
                print(f"Warning: Unexpected query format, skipping")
                continue
                
        print(f"Loaded {len(processed_queries)} queries")
        return processed_queries
        
    except Exception as e:
        print(f"Error loading queries: {e}")
        # Return some simple default queries
        default_queries = [
            nx.complete_graph(3),
            nx.path_graph(4),
            nx.cycle_graph(5)
        ]
        if use_directed:
            default_queries = [q.to_directed() for q in default_queries]
        return default_queries

def main_optimized():
    """Optimized main function with better resource handling."""
    args = arg_parse()
    
    print("=== Optimized Graphlet Counting ===")
    print(f"Workers: {args.n_workers}")
    print(f"Baseline: {args.baseline}")
    print(f"Max query size: {args.max_query_size}")
    print(f"Timeout: {args.timeout}s")
    print(f"Graph type: {args.graph_type}")
    print(f"Node anchored: {args.node_anchored}")
    print(f"Aggressive filtering: {args.aggressive_filter}")

    use_directed = (args.graph_type == 'directed')

    # Load dataset
    targets = load_dataset(args.dataset, use_directed)
    
    # Load queries
    queries = load_queries(args.queries_path, use_directed)
    
    if not queries:
        print("Error: No queries loaded. Exiting.")
        return
    
    print(f"Loaded: {len(queries)} queries, {len(targets)} targets")
    print(f"Query type: {'directed' if queries[0].is_directed() else 'undirected'}")
    print(f"Target type: {'directed' if targets[0].is_directed() else 'undirected'}")

    # Execute counting
    if args.baseline == "exact":
        print("Using exact counting method")
        n_matches = count_graphlets_optimized(queries, targets, args)
    elif args.baseline == "none":
        n_matches = count_graphlets_optimized(queries, targets, args)
    else:
        print(f"Generating baseline queries using {args.baseline}")
        baseline_queries = gen_baseline_queries_optimized(
            queries, targets, method=args.baseline, node_anchored=args.node_anchored)
        n_matches = count_graphlets_optimized(baseline_queries, targets, args)
    
    # Save results
    query_lens = [q.number_of_nodes() for q in queries]
    output_file = os.path.join(args.out_path, "counts.json")
    
    # Ensure output directory exists
    os.makedirs(args.out_path, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump({"query_lengths": query_lens, "counts": n_matches, "metadata": {}}, f)
    
    print(f"Results saved to {output_file}")
    print("=== Completed ===")

if __name__ == "__main__":
    main_optimized()