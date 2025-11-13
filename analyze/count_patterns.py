import argparse
import csv
import time
import os
import json
import concurrent.futures
import sys

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

from common import data
from common import models
from common import utils
from subgraph_mining import decoder

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

MAX_SEARCH_TIME = 1800  
MAX_MATCHES_PER_QUERY = 10000
DEFAULT_SAMPLE_ANCHORS = 1000
CHECKPOINT_INTERVAL = 100  

# Global caches (set in worker initializer)
_GLOBAL_QUERIES = None
_GLOBAL_TARGETS = None
_GLOBAL_QUERY_STATS = None
_GLOBAL_TARGET_STATS = None
_GLOBAL_QUERY_ANCHOR_SIGS = None
_GLOBAL_TARGET_SIGS = None

def _init_worker(queries, targets, query_stats, target_stats):
    """Pool initializer: set module-level globals in each worker and build signatures."""
    global _GLOBAL_QUERIES, _GLOBAL_TARGETS, _GLOBAL_QUERY_STATS, _GLOBAL_TARGET_STATS
    global _GLOBAL_QUERY_ANCHOR_SIGS, _GLOBAL_TARGET_SIGS
    _GLOBAL_QUERIES = queries
    _GLOBAL_TARGETS = targets
    _GLOBAL_QUERY_STATS = query_stats
    _GLOBAL_TARGET_STATS = target_stats

    # Per-query anchor signatures
    _GLOBAL_QUERY_ANCHOR_SIGS = [compute_anchor_signature(q) for q in queries]
    # Per-target per-node signatures
    _GLOBAL_TARGET_SIGS = [compute_target_signatures(t) for t in targets]

def get_node_attr(g, n, keys):
    for k in keys:
        if k in g.nodes[n]:
            return g.nodes[n][k]
    return None

def compute_anchor_signature(q):
    """Return signature for query's anchored node (or arbitrary node if missing anchor)."""
    anchor = None
    for u in q.nodes:
        if q.nodes[u].get('anchor', 0) == 1:
            anchor = u
            break
    if anchor is None:
        anchor = next(iter(q.nodes))
    lbl = get_node_attr(q, anchor, ['label', 'role'])
    deg = q.degree[anchor]
    if q.is_directed():
        nbrs = set(q.successors(anchor)) | set(q.predecessors(anchor))
    else:
        nbrs = set(q.neighbors(anchor))
    neigh_labels = [get_node_attr(q, v, ['label', 'role']) for v in nbrs]
    return {
        'anchor': anchor,
        'label': lbl,
        'degree': deg,
        'neigh_multiset': Counter(neigh_labels)
    }

def compute_target_signatures(t):
    """Return dict: node -> {'label','degree','neigh_multiset'} for target graph t."""
    sigs = {}
    for n in t.nodes:
        lbl = get_node_attr(t, n, ['label', 'role'])
        deg = t.degree[n]
        if t.is_directed():
            nbrs = set(t.successors(n)) | set(t.predecessors(n))
        else:
            nbrs = set(t.neighbors(n))
        neigh_labels = [get_node_attr(t, v, ['label', 'role']) for v in nbrs]
        sigs[n] = {
            'label': lbl,
            'degree': deg,
            'neigh_multiset': Counter(neigh_labels)
        }
    return sigs

def fast_anchor_feasible(q_sig, t_sig, degree_tol=0.2, use_label=False):
    """Cheap necessary checks for anchor mapping feasibility.
    Returns False only when impossible; True means "maybe" (run matcher).
    """
    if use_label:
        if q_sig['label'] is not None and t_sig['label'] is not None and q_sig['label'] != t_sig['label']:
            return False
    if q_sig['degree'] > 0:
        low = max(0, int((1 - degree_tol) * q_sig['degree']))
        high = int((1 + degree_tol) * q_sig['degree']) + 1
        if not (low <= t_sig['degree'] <= high):
            return False
    # neighbor label containment
    for k, v in q_sig['neigh_multiset'].items():
        if k is None:
            continue
        if t_sig['neigh_multiset'].get(k, 0) < v:
            return False
    return True

def compute_graph_stats(G):
    """Compute graph statistics for filtering."""
    stats = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'is_directed': G.is_directed(),
        'degree_seq': sorted([d for _, d in G.degree()], reverse=True),
        'avg_degree': sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1)
    }
    
    if G.is_directed():
        stats['in_degree_seq'] = sorted([d for _, d in G.in_degree()], reverse=True)
        stats['out_degree_seq'] = sorted([d for _, d in G.out_degree()], reverse=True)
    
    try:
        if G.is_directed():
            stats['n_components'] = nx.number_weakly_connected_components(G)
        else:
            stats['n_components'] = nx.number_connected_components(G)
    except:
        stats['n_components'] = 1 
        
    return stats

def can_be_isomorphic(query_stats, target_stats):
    if query_stats['n_nodes'] > target_stats['n_nodes']:
        return False
    if query_stats['n_edges'] > target_stats['n_edges']:
        return False
    
    if query_stats['is_directed'] != target_stats['is_directed']:
        return False
    
    
    if len(query_stats['degree_seq']) > 0 and len(target_stats['degree_seq']) > 0:
        if query_stats['degree_seq'][0] > target_stats['degree_seq'][0]:
            return False
    
    # For directed graphs, check in/out degrees
    if query_stats['is_directed']:
        if (len(query_stats['in_degree_seq']) > 0 and len(target_stats['in_degree_seq']) > 0 and
            query_stats['in_degree_seq'][0] > target_stats['in_degree_seq'][0]):
            return False
        if (len(query_stats['out_degree_seq']) > 0 and len(target_stats['out_degree_seq']) > 0 and
            query_stats['out_degree_seq'][0] > target_stats['out_degree_seq'][0]):
            return False
    
    if query_stats['avg_degree'] > target_stats['avg_degree'] * 1.1:  
        return False
    
    return True

def arg_parse():
    parser = argparse.ArgumentParser(description='count graphlets in a graph')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--queries_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--n_workers', type=int)
    parser.add_argument('--count_method', type=str)
    parser.add_argument('--baseline', type=str)
    parser.add_argument('--node_anchored', action="store_true")
    parser.add_argument('--max_query_size', type=int, default=20, help='Maximum query size to process')
    parser.add_argument('--sample_anchors', type=int, default=DEFAULT_SAMPLE_ANCHORS, help='Number of anchor nodes to sample for large graphs')
    parser.add_argument('--checkpoint_file', type=str, default="checkpoint.json", help='File to save/load progress')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for processing')
    parser.add_argument('--timeout', type=int, default=MAX_SEARCH_TIME, help='Timeout per task in seconds')
    parser.add_argument('--use_sampling', action="store_true", help='Use node sampling for very large graphs')
    parser.add_argument('--graph_type', type=str, default='auto', choices=['directed', 'undirected', 'auto'],
                       help='Graph type: directed, undirected, or auto-detect')
    parser.set_defaults(dataset="enzymes",
                       queries_path="results/out-patterns.p",
                       out_path="results/counts.json",
                       n_workers=4,
                       count_method="bin",
                       baseline="none")
    return parser.parse_args()

def load_networkx_graph(filepath, directed=None):
    """Load a Networkx graph from pickle format with proper attributes handling.
    
    Args:
        filepath: Path to the pickle file
        directed: If True, create DiGraph; if False, create Graph; if None, auto-detect
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        
        if isinstance(data, (nx.Graph, nx.DiGraph)):
            if directed is None:
                return data
            elif directed and not data.is_directed():
                return data.to_directed()
            elif not directed and data.is_directed():
                return data.to_undirected()
            else:
                return data
        
        if directed is None:
            if isinstance(data, dict):
                directed = data.get('directed', False)
        
        graph = nx.DiGraph() if directed else nx.Graph()
        
        for node in data['nodes']:
            if isinstance(node, tuple):
                node_id, attrs = node
                graph.add_node(node_id, **attrs)
            else:
                graph.add_node(node)
        
        for edge in data['edges']:
            if len(edge) == 3:
                src, dst, attrs = edge
                graph.add_edge(src, dst, **attrs)
            else:
                src, dst = edge[:2]
                graph.add_edge(src, dst)
                
        return graph

def count_graphlets_helper(inp):
    """Worker: input is (q_idx, t_idx, method, node_anchored, anchor_or_none, timeout)."""
    q_idx, t_idx, method, node_anchored, anchor_or_none, timeout = inp
    query = _GLOBAL_QUERIES[q_idx]
    target = _GLOBAL_TARGETS[t_idx]
    q_stats = _GLOBAL_QUERY_STATS[q_idx]
    t_stats = _GLOBAL_TARGET_STATS[t_idx]

    start_time = time.time()
    deadline = start_time + min(timeout, 600)

    # Graph-level fast check
    if not can_be_isomorphic(q_stats, t_stats):
        return q_idx, 0

    count = 0
    try:
        # Symmetry estimate only if needed
        if method == "freq":
            ismags = nx.is_directed(query) and nx.isomorphism.DiGraphMatcher(query, query) or nx.isomorphism.ISMAGS(query, query)
            n_symmetries = 0
            for _ in ismags.isomorphisms_iter(symmetry=False):
                n_symmetries += 1
                if time.time() > deadline:
                    break
            n_symmetries = max(1, n_symmetries)

        if node_anchored:
            # Cheap feasibility precheck (never changes counts)
            q_sig = _GLOBAL_QUERY_ANCHOR_SIGS[q_idx]
            t_sig = _GLOBAL_TARGET_SIGS[t_idx].get(anchor_or_none)
            if t_sig is None:
                return q_idx, 0
            # auto-use label/role info only if present on both graphs
            use_label = any('label' in d or 'role' in d for _, d in target.nodes(data=True)) and \
                        any('label' in d or 'role' in d for _, d in query.nodes(data=True))
            if not fast_anchor_feasible(q_sig, t_sig, degree_tol=0.2, use_label=use_label):
                return q_idx, 0

            # Preserve and set anchor attributes
            prev = {}
            for n in target.nodes:
                if 'anchor' in target.nodes[n]:
                    prev[n] = target.nodes[n]['anchor']
            nx.set_node_attributes(target, 0, name='anchor')
            if anchor_or_none in target:
                target.nodes[anchor_or_none]['anchor'] = 1

            try:
                node_match = iso.categorical_node_match(['anchor'], [0])
                matcher = target.is_directed() and iso.DiGraphMatcher(target, query, node_match=node_match) \
                          or iso.GraphMatcher(target, query, node_match=node_match)
                if time.time() > deadline:
                    return q_idx, 0
                if method == 'bin':
                    count = int(matcher.subgraph_is_isomorphic())
                else:
                    c = 0
                    for _ in matcher.subgraph_isomorphisms_iter():
                        if time.time() > deadline or c >= MAX_MATCHES_PER_QUERY:
                            break
                        c += 1
                    count = c / n_symmetries if n_symmetries else c
            finally:
                for n in target.nodes:
                    if n in prev:
                        target.nodes[n]['anchor'] = prev[n]
                    elif 'anchor' in target.nodes[n]:
                        del target.nodes[n]['anchor']
        else:
            matcher = target.is_directed() and iso.DiGraphMatcher(target, query) or iso.GraphMatcher(target, query)
            if time.time() > deadline:
                return q_idx, 0
            if method == 'bin':
                count = int(matcher.subgraph_is_isomorphic())
            else:
                c = 0
                for _ in matcher.subgraph_isomorphisms_iter():
                    if time.time() > deadline or c >= MAX_MATCHES_PER_QUERY:
                        break
                    c += 1
                count = c / n_symmetries if n_symmetries else c

    except Exception as e:
        print(f"Error processing query {q_idx} vs target {t_idx}: {str(e)}")
        count = 0

    took = time.time() - start_time
    if took > 10:
        print(f"Task (q={q_idx}, t={t_idx}) processed in {took:.2f}s with count {count}")

    return q_idx, count

def save_checkpoint(n_matches, checkpoint_file):
    with open(checkpoint_file, 'w') as f:
        json.dump({str(k): v for k, v in n_matches.items()}, f)
    print(f"Checkpoint saved to {checkpoint_file}")

def load_checkpoint(checkpoint_file):
    """Load checkpoint mapping query index -> accumulated count. Returns defaultdict(float)."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            return defaultdict(float, {int(k): v for k, v in checkpoint.items()})
        except Exception as e:
            print(f"Error loading checkpoint file {checkpoint_file}: {e}. Starting fresh")
    return defaultdict(float)

def sample_subgraphs(target, n_samples=10, max_size=1000):
    """Random BFS-like expansions to bounded-size induced subgraphs from target.
    Useful to bound runtime on huge graphs when --use_sampling is set.
    """
    subgraphs = []
    nodes = list(target.nodes())
    if not nodes:
        return [target]

    for _ in range(n_samples):
        start_node = random.choice(nodes)
        sub_nodes = {start_node}
        if target.is_directed():
            frontier = list(set(target.successors(start_node)) | set(target.predecessors(start_node)))
        else:
            frontier = list(target.neighbors(start_node))

        while len(sub_nodes) < max_size and frontier:
            u = frontier.pop(0)
            if u in sub_nodes:
                continue
            sub_nodes.add(u)
            if target.is_directed():
                new_neigh = set(target.successors(u)) | set(target.predecessors(u))
            else:
                new_neigh = set(target.neighbors(u))
            # Enqueue unseen neighbors
            for v in new_neigh:
                if v not in sub_nodes and v not in frontier:
                    frontier.append(v)

        subgraphs.append(target.subgraph(sub_nodes).copy())

    return subgraphs

def count_graphlets(queries, targets, args):
    print(f"Processing {len(queries)} queries across {len(targets)} targets")
    
    is_directed = any(g.is_directed() for g in queries + targets)
    if is_directed:
        print("Detected directed graphs - using DiGraphMatcher")
    
    n_matches = load_checkpoint(args.checkpoint_file)
    
    problematic_tasks_file = "problematic_tasks.json"
    if os.path.exists(problematic_tasks_file):
        with open(problematic_tasks_file, 'r') as f:
            try:
                problematic_tasks = set(json.load(f))
                print(f"Loaded {len(problematic_tasks)} problematic tasks to skip")
            except:
                problematic_tasks = set()
    else:
        problematic_tasks = set()
    
    if args.use_sampling and any(t.number_of_nodes() > 100000 for t in targets):
        sampled_targets = []
        for target in targets:
            if target.number_of_nodes() > 100000:
                print(f"Sampling subgraphs from large graph with {target.number_of_nodes()} nodes")
                sampled_targets.extend(sample_subgraphs(target, n_samples=20, max_size=10000))
            else:
                sampled_targets.append(target)
        targets = sampled_targets
        print(f"After sampling: {len(targets)} target graphs to process")
    
    # Clean graphs once (remove self-loops) to avoid per-task copies
    def _clean(g):
        if any(True for _ in nx.selfloop_edges(g)):
            h = g.copy()
            h.remove_edges_from(list(nx.selfloop_edges(h)))
            return h
        return g

    queries = [_clean(q) for q in queries]
    targets = [_clean(t) for t in targets]

    # Precompute lightweight stats once
    query_stats = [compute_graph_stats(q) for q in queries]
    target_stats = [compute_graph_stats(t) for t in targets]

    # Build task list with indices only
    inp = []
    for qi, q in enumerate(queries):
        if q.number_of_nodes() > args.max_query_size:
            print(f"Skipping query {qi}: exceeds max size {args.max_query_size}")
            continue
        q_stats = query_stats[qi]
        for ti, t in enumerate(targets):
            t_stats = target_stats[ti]
            if not can_be_isomorphic(q_stats, t_stats):
                continue
            task_id = f"{qi}_{ti}"
            if task_id in problematic_tasks:
                print(f"Skipping known problematic task {task_id}")
                continue
            if task_id in n_matches:
                print(f"Skipping already processed task {task_id}")
                continue
            if args.node_anchored:
                nodes = list(t.nodes)
                if len(nodes) > args.sample_anchors:
                    anchors = random.sample(nodes, args.sample_anchors)
                else:
                    anchors = nodes
                for a in anchors:
                    inp.append((qi, ti, args.count_method, args.node_anchored, a, args.timeout))
            else:
                inp.append((qi, ti, args.count_method, args.node_anchored, None, args.timeout))

    print(f"Generated {len(inp)} tasks after filtering")
    n_done = 0
    last_checkpoint = time.time()

    # Initialize workers with global caches
    with Pool(processes=args.n_workers, initializer=_init_worker, initargs=(queries, targets, query_stats, target_stats)) as pool:
        for batch_start in range(0, len(inp), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(inp))
            batch = inp[batch_start:batch_end]

            print(f"Processing batch {batch_start}-{batch_end} out of {len(inp)}")
            batch_start_time = time.time()

            chunksz = max(1, len(batch)//(args.n_workers*4) or 1)
            results = pool.imap_unordered(count_graphlets_helper, batch, chunksize=chunksz)

            for result in results:
                if time.time() - batch_start_time > 3600:
                    print(f"Batch {batch_start}-{batch_end} taking too long, marking remaining tasks problematic")
                    for task in batch:
                        qi = task[0]
                        ti = task[1]
                        problematic_tasks.add(f"{qi}_{ti}")
                    break

                qi, n = result
                n_matches[qi] += n
                n_done += 1

                if n_done % 10 == 0:
                    matched = sum(1 for v in n_matches.values() if v > 0)
                    print(f"Processed {n_done}/{len(inp)} tasks, queries with matches: {matched}/{len(n_matches)}", flush=True)

                if time.time() - last_checkpoint > 300:
                    save_checkpoint(n_matches, args.checkpoint_file)
                    with open(problematic_tasks_file, 'w') as f:
                        json.dump(list(problematic_tasks), f)
                    last_checkpoint = time.time()

            save_checkpoint(n_matches, args.checkpoint_file)
            with open(problematic_tasks_file, 'w') as f:
                json.dump(list(problematic_tasks), f)

    print("\nDone counting")
    return [n_matches[i] for i in range(len(queries))]

def generate_one_baseline(args):
    import networkx as nx
    import random

    i, query, targets, method = args

    if len(query) == 0:
        return query

    MAX_ATTEMPTS = 100 
    for attempt in range(MAX_ATTEMPTS):
        try:
            graph = random.choice(targets)
            if graph.number_of_nodes() == 0:
                continue

            if method == "radial":
                node = random.choice(list(graph.nodes))
                if graph.is_directed():
                    neigh = set([node])
                    visited = {node}
                    queue = [(node, 0)]
                    while queue:
                        current, dist = queue.pop(0)
                        if dist < 3:
                            for neighbor in set(graph.successors(current)) | set(graph.predecessors(current)):
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    neigh.add(neighbor)
                                    queue.append((neighbor, dist + 1))
                    neigh = list(neigh)
                else:
                    neigh = list(nx.single_source_shortest_path_length(graph, node, cutoff=3).keys())
                
                subgraph = graph.subgraph(neigh)
                if subgraph.number_of_nodes() == 0:
                    continue
                
                if graph.is_directed():
                    largest_cc = max(nx.weakly_connected_components(subgraph), key=len)
                else:
                    largest_cc = max(nx.connected_components(subgraph), key=len)
                neigh = subgraph.subgraph(largest_cc)
                neigh = nx.convert_node_labels_to_integers(neigh)
                if len(neigh) == len(query):
                    return neigh

            elif method == "tree":
                start_node = random.choice(list(graph.nodes))
                neigh = [start_node]
                if graph.is_directed():
                    frontier = list(set(graph.successors(start_node)) | set(graph.predecessors(start_node)) - set(neigh))
                else:
                    frontier = list(set(graph.neighbors(start_node)) - set(neigh))
                
                while len(neigh) < len(query) and frontier:
                    new_node = random.choice(frontier)
                    neigh.append(new_node)
                    if graph.is_directed():
                        new_neighbors = list(set(graph.successors(new_node)) | set(graph.predecessors(new_node)))
                    else:
                        new_neighbors = list(graph.neighbors(new_node))
                    frontier += new_neighbors
                    frontier = [x for x in frontier if x not in neigh]
                
                if len(neigh) == len(query):
                    sub = graph.subgraph(neigh)
                    return nx.convert_node_labels_to_integers(sub)

        except Exception as e:
            continue 

    print(f"[WARN] Baseline not found for query {i} after {MAX_ATTEMPTS} attempts.")
    return nx.DiGraph() if query.is_directed() else nx.Graph()

def gen_baseline_queries(queries, targets, method="radial", node_anchored=False):
    print(f"Generating {len(queries)} baseline queries in parallel using method: {method}")
    
    args_list = [(i, query, targets, method) for i, query in enumerate(queries)]
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(generate_one_baseline, args_list)
    
    return results

def main():
    global args
    args = arg_parse()
    print("Using {} workers".format(args.n_workers))
    print("Baseline:", args.baseline)
    print(f"Max query size: {args.max_query_size}")
    print(f"Timeout per task: {args.timeout} seconds")
    print(f"Graph type: {args.graph_type}")

    use_directed = (args.graph_type == 'directed')

    if args.dataset.endswith('.pkl'):
        print(f"Loading Networkx graph from {args.dataset}")
        try:
            if args.graph_type == 'auto':
                graph = load_networkx_graph(args.dataset, directed=None)
            else:
                graph = load_networkx_graph(args.dataset, directed=use_directed)
            
            graph_type = "directed" if graph.is_directed() else "undirected"
            print(f"Loaded {graph_type} graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            dataset = [graph]
        except Exception as e:
            print(f"Error loading graph: {str(e)}")
            raise
    elif args.dataset == 'enzymes':
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    elif args.dataset == 'cox2':
        dataset = TUDataset(root='/tmp/cox2', name='COX2')
    elif args.dataset == 'reddit-binary':
        dataset = TUDataset(root='/tmp/REDDIT-BINARY', name='REDDIT-BINARY')
    elif args.dataset == 'coil':
        dataset = TUDataset(root='/tmp/coil', name='COIL-DEL')
    elif args.dataset == 'ppi-pathways':
        graph = nx.DiGraph() if use_directed else nx.Graph()
        with open("data/ppi-pathways.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                graph.add_edge(int(row[0]), int(row[1]))
        dataset = [graph]
    elif args.dataset in ['diseasome', 'usroads', 'mn-roads', 'infect']:
        fn = {"diseasome": "bio-diseasome.mtx",
            "usroads": "road-usroads.mtx",
            "mn-roads": "mn-roads.mtx",
            "infect": "infect-dublin.edges"}
        graph = nx.DiGraph() if use_directed else nx.Graph()
        with open("data/{}".format(fn[args.dataset]), "r") as f:
            for line in f:
                if not line.strip(): continue
                a, b = line.strip().split(" ")
                graph.add_edge(int(a), int(b))
        dataset = [graph]
    elif args.dataset.startswith('plant-'):
        size = int(args.dataset.split("-")[-1])
        dataset = decoder.make_plant_dataset(size)
    elif args.dataset == "analyze":
        with open("results/analyze.p", "rb") as f:
            cand_patterns, _ = pickle.load(f)
            queries = [q for score, q in cand_patterns[10]][:200]
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

    targets = []
    for i, graph in enumerate(dataset):
        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            if args.graph_type == 'auto':
                graph = pyg_utils.to_networkx(graph).to_undirected()
            elif use_directed:
                graph = pyg_utils.to_networkx(graph, to_undirected=False)
            else:
                graph = pyg_utils.to_networkx(graph).to_undirected()
            for node in graph.nodes():
                if 'label' not in graph.nodes[node]:
                    graph.nodes[node]['label'] = str(node)
                if 'id' not in graph.nodes[node]:
                    graph.nodes[node]['id'] = str(node)
        else:
            if use_directed and not graph.is_directed():
                graph = graph.to_directed()
            elif not use_directed and graph.is_directed():
                graph = graph.to_undirected()
        targets.append(graph)

    if args.dataset != "analyze":
        with open(args.queries_path, "rb") as f:
            queries = pickle.load(f)
    
    if use_directed:
        queries = [q.to_directed() if not q.is_directed() else q for q in queries]
    else:
        queries = [q.to_undirected() if q.is_directed() else q for q in queries]
            
    query_lens = [len(query) for query in queries]
    print(f"Loaded {len(queries)} query patterns")
    print(f"Query graph type: {'directed' if queries[0].is_directed() else 'undirected'}")
    print(f"Target graph type: {'directed' if targets[0].is_directed() else 'undirected'}")

    if args.baseline == "exact":
        print("Using exact counting method")
        n_matches = count_graphlets(queries, targets, args)
    elif args.baseline == "none":
        n_matches = count_graphlets(queries, targets, args)
    else:
        print(f"Generating baseline queries using {args.baseline}")
        baseline_queries = gen_baseline_queries(queries, targets,
            node_anchored=args.node_anchored, method=args.baseline)
        query_lens = [len(q) for q in baseline_queries]
        n_matches = count_graphlets(baseline_queries, targets, args)
            
    # Resolve output path: accept directory or file path
    out_path = args.out_path
    if not out_path or out_path.strip() == "":
        out_path = os.path.join("results", "counts.json")

    # If a directory is provided (or a trailing separator), write a default filename inside it
    if out_path.endswith(os.sep) or (os.path.exists(out_path) and os.path.isdir(out_path)):
        os.makedirs(out_path, exist_ok=True)
        ds_base = os.path.splitext(os.path.basename(args.dataset))[0]
        out_path = os.path.join(out_path, f"counts_{ds_base}.json")
    else:
        # Ensure parent directory exists if a full file path is given
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump((query_lens, n_matches, []), f)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()