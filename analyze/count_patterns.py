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

def arg_parse():
    parser = argparse.ArgumentParser(description='count graphlets in a graph')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--queries_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--count_method', type=str, default='bin', choices=['bin', 'freq'])
    parser.add_argument('--node_anchored', action="store_true")
    parser.add_argument('--max_query_size', type=int, default=20)
    parser.add_argument('--sample_anchors', type=int, default=DEFAULT_SAMPLE_ANCHORS)
    parser.add_argument('--timeout', type=int, default=MAX_SEARCH_TIME)
    parser.add_argument('--graph_type', type=str, default='auto', choices=['directed', 'undirected', 'auto'])
    return parser.parse_args()

def load_networkx_graph(filepath, directed=None):
    """Load a Networkx graph from pickle format."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, (nx.Graph, nx.DiGraph)):
        if directed is not None:
            if directed and not data.is_directed():
                return data.to_directed()
            elif not directed and data.is_directed():
                return data.to_undirected()
        return data
    
    # Handle other formats
    if directed is None:
        directed = data.get('directed', False)
    
    graph = nx.DiGraph() if directed else nx.Graph()
    
    if 'nodes' in data:
        graph.add_nodes_from(data['nodes'])
    if 'edges' in data:
        graph.add_edges_from(data['edges'])
    
    return graph

def quick_anchor_check(query, target, anchor):
    """Fast pre-check before expensive isomorphism."""
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
    """Get anchors sorted by degree (highest first)."""
    degrees = dict(target.degree())
    anchors = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    return anchors[:sample_anchors]

def count_graphlets_helper(inp):
    """Counting function for parallel processing."""
    i, query, target, method, node_anchored, anchor_or_none, timeout = inp
    
    start_time = time.time()
    
    # FAST PRE-FILTER for anchored queries
    if node_anchored and anchor_or_none is not None:
        if not quick_anchor_check(query, target, anchor_or_none):
            return i, 0
    
    # Basic size checks
    if query.number_of_nodes() > target.number_of_nodes():
        return i, 0
    if query.number_of_edges() > target.number_of_edges():
        return i, 0
    
    query = query.copy()
    query.remove_edges_from(nx.selfloop_edges(query))
    target = target.copy()
    target.remove_edges_from(nx.selfloop_edges(target))

    count = 0
    try:
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Task timed out")
            
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
            
    except TimeoutError:
        count = 0
    except Exception as e:
        count = 0
        
    return i, count

def count_graphlets(queries, targets, args):
    """Main counting function with performance optimizations."""
    print(f"Processing {len(queries)} queries across {len(targets)} targets")
    
    n_matches = defaultdict(float)
    
    inp = []
    for i, query in enumerate(queries):
        if query.number_of_nodes() > args.max_query_size:
            print(f"Skipping query {i} - exceeds max size {args.max_query_size}")
            continue
            
        print(f"Processing query {i} with {query.number_of_nodes()} nodes, {query.number_of_edges()} edges")
            
        for target in targets:
            if args.node_anchored:
                # Use smart anchor selection
                if target.number_of_nodes() > args.sample_anchors:
                    anchors = get_smart_anchors(target, args.sample_anchors)
                else:
                    anchors = list(target.nodes)
                    
                for anchor in anchors:
                    inp.append((i, query, target, args.count_method, args.node_anchored, anchor, args.timeout))
            else:
                inp.append((i, query, target, args.count_method, args.node_anchored, None, args.timeout))
    
    print(f"Generated {len(inp)} total tasks")
    
    if len(inp) == 0:
        print("Warning: No tasks generated! Check query/target compatibility")
        return [0] * len(queries)
    
    completed_tasks = 0
    last_progress = time.time()
    
    with Pool(processes=args.n_workers) as pool:
        results = pool.imap_unordered(count_graphlets_helper, inp)
        
        for i, n in results:
            n_matches[i] += n
            completed_tasks += 1
            
            # Progress reporting every 30 seconds or 100 tasks
            current_time = time.time()
            if current_time - last_progress > 30 or completed_tasks % 100 == 0:
                print(f"Progress: {completed_tasks}/{len(inp)} tasks completed")
                last_progress = current_time
    
    # Return counts in the original order
    final_counts = [n_matches[i] for i in range(len(queries))]
    print(f"Final counts: {final_counts}")
    return final_counts

def main():
    args = arg_parse()
    
    print("=== Graphlet Counting ===")
    print(f"Dataset: {args.dataset}")
    print(f"Queries: {args.queries_path}")
    print(f"Output: {args.out_path}")
    print(f"Workers: {args.n_workers}")
    print(f"Count method: {args.count_method}")
    print(f"Node anchored: {args.node_anchored}")
    print(f"Graph type: {args.graph_type}")

    use_directed = (args.graph_type == 'directed')

    # Load target graph
    print(f"Loading target graph from {args.dataset}")
    target_graph = load_networkx_graph(args.dataset, use_directed)
    targets = [target_graph]
    
    print(f"Loaded target - {target_graph.number_of_nodes()} nodes, {target_graph.number_of_edges()} edges")
    print(f"Target type: {'directed' if target_graph.is_directed() else 'undirected'}")

    # Load queries
    print(f"Loading queries from {args.queries_path}")
    with open(args.queries_path, "rb") as f:
        queries = pickle.load(f)
    
    if not queries:
        print("Error: No queries loaded!")
        return
    
    # Ensure correct graph direction
    converted_queries = []
    for query in queries:
        if not isinstance(query, (nx.Graph, nx.DiGraph)):
            continue
            
        if use_directed:
            if not query.is_directed():
                converted_query = query.to_directed()
            else:
                converted_query = query
        else:
            if query.is_directed():
                converted_query = query.to_undirected()
            else:
                converted_query = query
        converted_queries.append(converted_query)
    
    queries = converted_queries
    print(f"Loaded {len(queries)} queries")

    # Count graphlets
    print("Starting graphlet counting...")
    n_matches = count_graphlets(queries, targets, args)
    
    # Prepare output in original format
    query_lens = [q.number_of_nodes() for q in queries]
    output_data = [query_lens, n_matches, []]
    
    # Save results
    output_file = os.path.join(args.out_path, "counts.json")
    with open(output_file, "w") as f:
        json.dump(output_data, f)
    
    print(f"Results saved to {output_file}")
    print("=== Completed ===")

if __name__ == "__main__":
    main()