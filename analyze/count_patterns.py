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

"""
Optimized version of analyze/count_patterns.py
Main changes:
- Pool initializer loads global graphs/stats to avoid pickling big objects per-task.
- compute_graph_stats returns compact stats (max_degree, avg_degree) instead of full sorted sequences.
- Tasks pass indices and anchors only.
- sample_subgraphs uses deque for BFS.
- Imap with chunksize to reduce IPC overhead.
"""

import argparse
import os
import sys
import time
import json
import random
import pickle
import csv
from collections import defaultdict, deque
from multiprocessing import Pool
import networkx as nx
from networkx.algorithms import isomorphism as iso


# Globals set in pool initializer
_GLOBAL_QUERIES = None
_GLOBAL_TARGETS = None
_GLOBAL_QUERY_STATS = None
_GLOBAL_TARGET_STATS = None


def compute_graph_stats_compact(G):
    """Return compact stats used for quick pruning (lightweight)."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    is_directed = G.is_directed()
    degs = dict(G.degree())
    max_degree = max(degs.values()) if degs else 0
    avg_degree = sum(degs.values()) / max(1, n_nodes)
    max_in = max_out = 0
    if is_directed:
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())
        max_in = max(in_deg.values()) if in_deg else 0
        max_out = max(out_deg.values()) if out_deg else 0
    # components (weak for directed)
    try:
        if is_directed:
            n_components = nx.number_weakly_connected_components(G)
        else:
            n_components = nx.number_connected_components(G)
    except Exception:
        n_components = 1
    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'is_directed': is_directed,
        'max_degree': max_degree,
        'avg_degree': avg_degree,
        'max_in_degree': max_in,
        'max_out_degree': max_out,
        'n_components': n_components
    }


def can_be_isomorphic_compact(qs, ts):
    """Pruning using compact stats."""
    if qs['n_nodes'] > ts['n_nodes']:
        return False
    if qs['n_edges'] > ts['n_edges']:
        return False
    if qs['is_directed'] != ts['is_directed']:
        return False
    if qs['max_degree'] > ts['max_degree']:
        return False
    if qs['is_directed']:
        if qs['max_in_degree'] > ts['max_in_degree']:
            return False
        if qs['max_out_degree'] > ts['max_out_degree']:
            return False
    # Allow slight slack on average degree
    if qs['avg_degree'] > ts['avg_degree'] * 1.1:
        return False
    return True


def arg_parse():
    parser = argparse.ArgumentParser(description='count graphlets in a graph (optimized)')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--queries_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--count_method', type=str, default="bin")
    parser.add_argument('--baseline', type=str, default="none")
    parser.add_argument('--node_anchored', action="store_true")
    parser.add_argument('--max_query_size', type=int, default=20)
    parser.add_argument('--sample_anchors', type=int, default=DEFAULT_SAMPLE_ANCHORS)
    parser.add_argument('--checkpoint_file', type=str, default="checkpoint.json")
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--timeout', type=int, default=MAX_SEARCH_TIME)
    parser.add_argument('--use_sampling', action="store_true")
    parser.add_argument('--graph_type', type=str, default='auto', choices=['directed', 'undirected', 'auto'])
    parser.add_argument('--preserve_labels', action='store_true')
    return parser.parse_args()


def load_networkx_graph(filepath, directed=None):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, (nx.Graph, nx.DiGraph)):
            if directed is None:
                return data
            elif directed and not data.is_directed():
                return data.to_directed()
            elif not directed and data.is_directed():
                return data.to_undirected()
            return data
        if directed is None and isinstance(data, dict):
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


def sample_subgraphs(target, n_samples=10, max_size=1000):
    """Faster BFS-based subgraph sampling using deque."""
    subgraphs = []
    if target.number_of_nodes() == 0:
        return subgraphs
    nodes = list(target.nodes())
    for _ in range(n_samples):
        start_node = random.choice(nodes)
        sub_nodes = set([start_node])
        if target.is_directed():
            frontier = deque(set(target.successors(start_node)) | set(target.predecessors(start_node)))
        else:
            frontier = deque(target.neighbors(start_node))
        while len(sub_nodes) < max_size and frontier:
            next_node = frontier.popleft()
            if next_node in sub_nodes:
                continue
            sub_nodes.add(next_node)
            if target.is_directed():
                new_neighbors = set(target.successors(next_node)) | set(target.predecessors(next_node))
            else:
                new_neighbors = set(target.neighbors(next_node))
            for n in new_neighbors:
                if n not in sub_nodes:
                    frontier.append(n)
        sg = target.subgraph(sub_nodes).copy()  # copy to freeze structure
        subgraphs.append(sg)
    return subgraphs


def save_checkpoint(n_matches, checkpoint_file):
    tmp = checkpoint_file + ".tmp"
    with open(tmp, 'w') as f:
        json.dump({str(k): v for k, v in n_matches.items()}, f)
    os.replace(tmp, checkpoint_file)  # atomic-ish
    # Minimal logging
    print(f"Checkpoint saved ({len(n_matches)} items) -> {checkpoint_file}")


def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            try:
                checkpoint = json.load(f)
                return defaultdict(float, {int(k): v for k, v in checkpoint.items()})
            except json.JSONDecodeError:
                print(f"Bad checkpoint {checkpoint_file}, starting fresh")
    return defaultdict(float)


# Pool initializer to set global graphs & stats in each worker to avoid pickling them per-task
def _pool_initializer(queries, targets, q_stats, t_stats):
    global _GLOBAL_QUERIES, _GLOBAL_TARGETS, _GLOBAL_QUERY_STATS, _GLOBAL_TARGET_STATS
    _GLOBAL_QUERIES = queries
    _GLOBAL_TARGETS = targets
    _GLOBAL_QUERY_STATS = q_stats
    _GLOBAL_TARGET_STATS = t_stats


def count_graphlets_helper_task(task):
    """Worker receives a compact task: (query_idx, target_idx, method, node_anchored, anchor_or_none, timeout)"""
    i, t_idx, method, node_anchored, anchor_or_none, timeout = task
    start_time = time.time()
    effective_timeout = min(timeout, MAX_SEARCH_TIME)

    # Access global graphs/stats
    q = _GLOBAL_QUERIES[i]
    t = _GLOBAL_TARGETS[t_idx]

    # Quick prune using stored stats
    q_stats = _GLOBAL_QUERY_STATS[i]
    t_stats = _GLOBAL_TARGET_STATS[t_idx]
    if not can_be_isomorphic_compact(q_stats, t_stats):
        return i, 0

    # Remove self-loops quickly if any (operate on a view to avoid heavy copying)
    if q.number_of_selfloops():
        q = q.copy()
        q.remove_edges_from(nx.selfloop_edges(q))
    if t.number_of_selfloops():
        # Create a copy only if needed
        t = t.copy()
        t.remove_edges_from(nx.selfloop_edges(t))

    count = 0
    try:
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Task timed out after {effective_timeout}s")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(effective_timeout)

        # node_match: if nodes have same 'label' attribute, use it to speed isomorphism
        node_match = None
        if q.number_of_nodes() > 0:
            # If all query nodes have 'label' attribute and targets probably too, use categorical match
            q_labels = [d.get('label') for _, d in q.nodes(data=True)]
            if all(l is not None for l in q_labels):
                node_match = iso.categorical_node_match('label', None)

        if method == "bin":
            if node_anchored:
                # set anchor attribute on a shallow copy of target if needed
                t_work = t if anchor_or_none is None else t.copy()
                if anchor_or_none is not None:
                    nx.set_node_attributes(t_work, 0, 'anchor')
                    if anchor_or_none in t_work.nodes():
                        t_work.nodes[anchor_or_none]['anchor'] = 1
                    node_match = (iso.categorical_node_match(['anchor', 'label'], [0, None])
                                  if node_match is not None else iso.categorical_node_match('anchor', [0]))
                # Choose matcher depending on directedness
                if t_work.is_directed():
                    matcher = iso.DiGraphMatcher(t_work, q, node_match=node_match)
                else:
                    matcher = iso.GraphMatcher(t_work, q, node_match=node_match)
                count = int(matcher.subgraph_is_isomorphic())
            else:
                matcher = iso.DiGraphMatcher(t, q, node_match=node_match) if t.is_directed() else iso.GraphMatcher(t, q, node_match=node_match)
                count = int(matcher.subgraph_is_isomorphic())

        elif method == "freq":
            matcher = iso.DiGraphMatcher(t, q, node_match=node_match) if t.is_directed() else iso.GraphMatcher(t, q, node_match=node_match)
            for _ in matcher.subgraph_isomorphisms_iter():
                if time.time() - start_time > effective_timeout:
                    break
                count += 1
                if count >= MAX_MATCHES_PER_QUERY:
                    break
            # symmetry correction if needed (we skip symmetry computation here for speed)
        signal.alarm(0)

    except TimeoutError as e:
        # Timeout: return zero
        count = 0
    except Exception as e:
        # Log minimal error
        print(f"[ERROR] processing query {i}, target {t_idx}: {e}", file=sys.stderr)
        count = 0

    proc_time = time.time() - start_time
    if proc_time > 10:
        print(f"[INFO] query {i}, target {t_idx} processed in {proc_time:.2f}s -> {count}")
    return i, count


def gen_baseline_queries(queries, targets, method="radial", node_anchored=False):
    # Keep baseline generation similar to original but parallelize with Pool
    args_list = [(i, q, targets, method) for i, q in enumerate(queries)]
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(generate_one_baseline, args_list)
    return results


def generate_one_baseline(args):
    # kept mostly same as original; lightweight and executed in parallel
    import networkx as nx
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
                    queue = deque([(node, 0)])
                    while queue:
                        current, dist = queue.popleft()
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
                    comp = max(nx.weakly_connected_components(subgraph), key=len)
                else:
                    comp = max(nx.connected_components(subgraph), key=len)
                neigh = subgraph.subgraph(comp)
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
        except Exception:
            continue
    print(f"[WARN] Baseline not found for query {i} after {MAX_ATTEMPTS} attempts.")
    return nx.DiGraph() if query.is_directed() else nx.Graph()


def count_graphlets(queries, targets, args):
    print(f"Processing {len(queries)} queries across {len(targets)} targets")
    # Precompute compact stats for all graphs
    query_stats = [compute_graph_stats_compact(q) for q in queries]
    target_stats = [compute_graph_stats_compact(t) for t in targets]

    n_matches = load_checkpoint(args.checkpoint_file)

    problematic_tasks_file = "problematic_tasks.json"
    if os.path.exists(problematic_tasks_file):
        try:
            with open(problematic_tasks_file, 'r') as f:
                problematic_tasks = set(json.load(f))
                print(f"Loaded {len(problematic_tasks)} problematic tasks")
        except Exception:
            problematic_tasks = set()
    else:
        problematic_tasks = set()

    # Sampling large targets if requested
    if args.use_sampling and any(t.number_of_nodes() > 100000 for t in targets):
        sampled_targets = []
        for target in targets:
            if target.number_of_nodes() > 100000:
                sampled_targets.extend(sample_subgraphs(target, n_samples=20, max_size=10000))
            else:
                sampled_targets.append(target)
        targets = sampled_targets
        target_stats = [compute_graph_stats_compact(t) for t in targets]
        print(f"After sampling, {len(targets)} targets")

    # Build task list using indices only
    tasks = []
    for i, q in enumerate(queries):
        if q.number_of_nodes() > args.max_query_size:
            print(f"[SKIP] query {i} exceeds max size {args.max_query_size}")
            continue
        for t_idx, t in enumerate(targets):
            if not can_be_isomorphic_compact(query_stats[i], target_stats[t_idx]):
                continue
            task_id = f"{i}_{t_idx}"
            if task_id in problematic_tasks:
                continue
            if task_id in n_matches:
                continue
            if args.node_anchored:
                # sample anchors
                nodes_list = list(t.nodes())
                if len(nodes_list) > args.sample_anchors:
                    anchors = random.sample(nodes_list, args.sample_anchors)
                else:
                    anchors = nodes_list
                for anchor in anchors:
                    tasks.append((i, t_idx, args.count_method, True, anchor, args.timeout))
            else:
                tasks.append((i, t_idx, args.count_method, False, None, args.timeout))

    print(f"Generated {len(tasks)} tasks after pruning")

    # Heuristic chunksize to reduce IPC overhead
    chunksize = max(1, len(tasks) // (args.n_workers * 8)) if tasks else 1

    # Run tasks with a pool that has globals loaded
    n_done = 0
    last_checkpoint = time.time()
    with Pool(processes=args.n_workers, initializer=_pool_initializer,
              initargs=(queries, targets, query_stats, target_stats)) as pool:
        it = pool.imap_unordered(count_graphlets_helper_task, tasks, chunksize=chunksize)
        batch_start_time = time.time()
        for result in it:
            i, n = result
            n_matches[i] += n
            n_done += 1
            if n_done % 50 == 0:
                print(f"[PROGRESS] done {n_done}/{len(tasks)}")
            # periodic checkpoint
            if time.time() - last_checkpoint > 300:
                save_checkpoint(n_matches, args.checkpoint_file)
                with open(problematic_tasks_file, 'w') as f:
                    json.dump(list(problematic_tasks), f)
                last_checkpoint = time.time()
    save_checkpoint(n_matches, args.checkpoint_file)
    with open(problematic_tasks_file, 'w') as f:
        json.dump(list(problematic_tasks), f)
    print("[DONE] counting")
    return [n_matches[i] for i in range(len(queries))]


def main():
    args = arg_parse()
    print(f"Workers: {args.n_workers}, method: {args.count_method}, anchored: {args.node_anchored}")
    use_directed = (args.graph_type == 'directed')

    # Load dataset into list 'dataset' (similar to your original logic)
    dataset = []
    if args.dataset.endswith('.pkl') or args.dataset.endswith('.p'):
        print(f"Loading graph from {args.dataset}")
        graph = load_networkx_graph(args.dataset, directed=(None if args.graph_type == 'auto' else use_directed))
        dataset = [graph]
    elif args.dataset == 'enzymes':
        from torch_geometric.datasets import TUDataset
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    elif args.dataset == 'cox2':
        from torch_geometric.datasets import TUDataset
        dataset = TUDataset(root='/tmp/cox2', name='COX2')
    elif args.dataset == 'reddit-binary':
        from torch_geometric.datasets import TUDataset
        dataset = TUDataset(root='/tmp/REDDIT-BINARY', name='REDDIT-BINARY')
    elif args.dataset == 'coil':
        from torch_geometric.datasets import TUDataset
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
                a, b = line.strip().split()
                graph.add_edge(int(a), int(b))
        dataset = [graph]
    elif args.dataset.startswith('plant-'):
        # placeholder for decoder.make_plant_dataset
        size = int(args.dataset.split("-")[-1])
        import decoder
        dataset = decoder.make_plant_dataset(size)
    elif args.dataset == "analyze":
        with open("results/analyze.p", "rb") as f:
            cand_patterns, _ = pickle.load(f)
            queries = [q for score, q in cand_patterns[10]][:200]
        from torch_geometric.datasets import TUDataset
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    else:
        raise ValueError("Unknown dataset")

    # Convert dataset items to networkx graphs (if PyG objects)
    targets = []
    for graph in dataset:
        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            import torch_geometric.utils as pyg_utils
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

    # load queries
    if args.dataset != "analyze":
        with open(args.queries_path, "rb") as f:
            queries = pickle.load(f)

    # ensure directedness
    if use_directed:
        queries = [q.to_directed() if not q.is_directed() else q for q in queries]
    else:
        queries = [q.to_undirected() if q.is_directed() else q for q in queries]

    print(f"Loaded {len(queries)} queries; sample query size: {len(queries[0]) if queries else 0}")
    print(f"Targets: {len(targets)} graphs; sample target size: {targets[0].number_of_nodes() if targets else 0}")

    # Baseline logic preserved
    if args.baseline in ("exact", "none"):
        n_matches = count_graphlets(queries, targets, args)
    else:
        baseline_queries = gen_baseline_queries(queries, targets, method=args.baseline, node_anchored=args.node_anchored)
        n_matches = count_graphlets(baseline_queries, targets, args)

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
