import argparse
import time
import os
import json
import random
import pickle
import networkx as nx
import networkx.algorithms.isomorphism as iso
from multiprocessing import Pool
from collections import defaultdict

# Constants
MAX_SEARCH_TIME = 1800  
MAX_MATCHES_PER_QUERY = 10000
DEFAULT_SAMPLE_ANCHORS = 1000
CHECKPOINT_INTERVAL = 100

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

def quick_anchor_check(query, target, anchor):
    """NEW OPTIMIZATION: Fast pre-check before expensive isomorphism."""
    # Check if anchor has enough neighbors for the query
    anchor_degree = target.degree(anchor)
    query_min_degree = min(dict(query.degree()).values())
    
    if anchor_degree < query_min_degree:
        return False
    
    # Check if anchor's neighborhood is large enough
    if target.is_directed():
        neighbors = set(target.successors(anchor)) | set(target.predecessors(anchor))
    else:
        neighbors = set(target.neighbors(anchor))
    
    if len(neighbors) < query.number_of_nodes() - 1:
        return False
    
    return True

def get_smart_anchors(target, sample_anchors):
    """NEW OPTIMIZATION: Get anchors sorted by degree (highest first)."""
    degrees = dict(target.degree())
    anchors = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    return anchors[:sample_anchors]

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
    """Load a Networkx graph from pickle format with proper attributes handling."""
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
    i, query, target, method, node_anchored, anchor_or_none, timeout = inp
    
    start_time = time.time()
    
    effective_timeout = min(timeout, 600)  
    
    # NEW OPTIMIZATION: Fast anchor pre-check
    if node_anchored and anchor_or_none is not None:
        if not quick_anchor_check(query, target, anchor_or_none):
            return i, 0
    
    query_stats = compute_graph_stats(query)
    target_stats = compute_graph_stats(target)
    if not can_be_isomorphic(query_stats, target_stats):
        return i, 0
    
    query = query.copy()
    query.remove_edges_from(nx.selfloop_edges(query))
    target = target.copy()
    target.remove_edges_from(nx.selfloop_edges(target))

    count = 0
    try:
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Task {i} timed out after {effective_timeout} seconds")
            
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(effective_timeout)
        
        if method == "freq":
            if query.is_directed():
                ismags = nx.isomorphism.DiGraphMatcher(query, query)
            else:
                ismags = nx.isomorphism.ISMAGS(query, query)
            n_symmetries = len(list(ismags.isomorphisms_iter(symmetry=False)))
        
        if method == "bin":
            if node_anchored:
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
                if target.is_directed():
                    matcher = iso.DiGraphMatcher(target, query)
                else:
                    matcher = iso.GraphMatcher(target, query)
                
                count = int(matcher.subgraph_is_isomorphic())
        elif method == "freq":
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
            
    except TimeoutError as e:
        count = 0
    except Exception as e:
        count = 0
        
    processing_time = time.time() - start_time
    if processing_time > 10: 
        print(f"Query {i} processed in {processing_time:.2f} seconds with count {count}")
        
    return i, count

def save_checkpoint(n_matches, checkpoint_file):
    with open(checkpoint_file, 'w') as f:
        json.dump({str(k): v for k, v in n_matches.items()}, f)
    print(f"Checkpoint saved to {checkpoint_file}")

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            try:
                checkpoint = json.load(f)
                return defaultdict(float, {int(k): v for k, v in checkpoint.items()})
            except json.JSONDecodeError:
                print(f"Error loading checkpoint file {checkpoint_file}, starting fresh")
    return defaultdict(float)

def sample_subgraphs(target, n_samples=10, max_size=1000):
    subgraphs = []
    nodes = list(target.nodes())
    
    for _ in range(n_samples):
        start_node = random.choice(nodes)
        subgraph_nodes = {start_node}
        
        if target.is_directed():
            frontier = list(set(target.successors(start_node)) | set(target.predecessors(start_node)))
        else:
            frontier = list(target.neighbors(start_node))
        
        while len(subgraph_nodes) < max_size and frontier:
            next_node = frontier.pop(0)
            if next_node not in subgraph_nodes:
                subgraph_nodes.add(next_node)
                if target.is_directed():
                    new_neighbors = set(target.successors(next_node)) | set(target.predecessors(next_node))
                else:
                    new_neighbors = set(target.neighbors(next_node))
                frontier.extend([n for n in new_neighbors 
                              if n not in subgraph_nodes and n not in frontier])
        
        sg = target.subgraph(subgraph_nodes)
        subgraphs.append(sg)
        
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
    
    with Pool(processes=args.n_workers) as pool:
        target_stats = pool.map(compute_graph_stats, targets)
        query_stats = pool.map(compute_graph_stats, queries)
    
    inp = []
    for i, (query, q_stats) in enumerate(zip(queries, query_stats)):
        if query.number_of_nodes() > args.max_query_size:
            print(f"Skipping query {i}: exceeds max size {args.max_query_size}")
            continue
            
        for t_idx, (target, t_stats) in enumerate(zip(targets, target_stats)):
            if not can_be_isomorphic(q_stats, t_stats):
                continue
            
            task_id = f"{i}_{t_idx}"
            
            if task_id in problematic_tasks:
                print(f"Skipping known problematic task {task_id}")
                continue
                
            if task_id in n_matches:
                print(f"Skipping already processed task {task_id}")
                continue
                
            if args.node_anchored:
                # NEW OPTIMIZATION: Use smart anchor selection
                if target.number_of_nodes() > args.sample_anchors:
                    anchors = get_smart_anchors(target, args.sample_anchors)
                else:
                    anchors = list(target.nodes)
                    
                for anchor in anchors:
                    inp.append((i, query, target, args.count_method, args.node_anchored, anchor, 
                             args.timeout))
            else:
                inp.append((i, query, target, args.count_method, args.node_anchored, None, 
                         args.timeout))
    
    print(f"Generated {len(inp)} tasks after filtering")
    n_done = 0
    last_checkpoint = time.time()
   
    with Pool(processes=args.n_workers) as pool:
        for batch_start in range(0, len(inp), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(inp))
            batch = inp[batch_start:batch_end]

            print(f"Processing batch {batch_start}-{batch_end} out of {len(inp)}")
            batch_start_time = time.time()

            results = pool.imap_unordered(count_graphlets_helper, batch)

            for result in results:
                if time.time() - batch_start_time > 3600:  
                    print(f"Batch {batch_start}-{batch_end} taking too long, marking remaining tasks problematic")
                    for task in batch:
                        i = task[0]
                        task_id = f"{i}_{batch_start}"
                        problematic_tasks.add(task_id)
                    break

                i, n = result
                n_matches[i] += n
                n_done += 1

                if n_done % 50 == 0:  # Reduced verbosity
                    print(f"Processed {n_done}/{len(inp)} tasks, queries with matches: {sum(1 for v in n_matches.values() if v > 0)}/{len(n_matches)}")

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
    i, query, targets, method = args

    if query.number_of_nodes() == 0:
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
                if neigh.number_of_nodes() == query.number_of_nodes():
                    return neigh

            elif method == "tree":
                start_node = random.choice(list(graph.nodes))
                neigh = [start_node]
                if graph.is_directed():
                    frontier = list(set(graph.successors(start_node)) | set(graph.predecessors(start_node)) - set(neigh))
                else:
                    frontier = list(set(graph.neighbors(start_node)) - set(neigh))
                
                while len(neigh) < query.number_of_nodes() and frontier:
                    new_node = random.choice(frontier)
                    neigh.append(new_node)
                    if graph.is_directed():
                        new_neighbors = list(set(graph.successors(new_node)) | set(graph.predecessors(new_node)))
                    else:
                        new_neighbors = list(graph.neighbors(new_node))
                    frontier += new_neighbors
                    frontier = [x for x in frontier if x not in neigh]
                
                if len(neigh) == query.number_of_nodes():
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
        import csv
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
        from subgraph_mining import decoder
        dataset = decoder.make_plant_dataset(size)
    elif args.dataset == "analyze":
        with open("results/analyze.p", "rb") as f:
            cand_patterns, _ = pickle.load(f)
            queries = [q for score, q in cand_patterns[10]][:200]
        from torch_geometric.datasets import TUDataset
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

    targets = []
    for i, graph in enumerate(dataset):
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

    if args.dataset != "analyze":
        with open(args.queries_path, "rb") as f:
            queries = pickle.load(f)
    
    if use_directed:
        queries = [q.to_directed() if not q.is_directed() else q for q in queries]
    else:
        queries = [q.to_undirected() if q.is_directed() else q for q in queries]
            
    query_lens = [q.number_of_nodes() for q in queries]
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
        query_lens = [q.number_of_nodes() for q in baseline_queries]
        n_matches = count_graphlets(baseline_queries, targets, args)
            
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
    main()