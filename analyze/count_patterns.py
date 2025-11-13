import argparse
import time
import os
import json
import gc
import collections
import random
import pickle
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
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
    print(f"DEBUG: Loading graph from {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"DEBUG: Loaded data type: {type(data)}")
    
    if isinstance(data, (nx.Graph, nx.DiGraph)):
        print(f"DEBUG: Direct NetworkX graph, directed: {data.is_directed()}")
        if directed is not None:
            if directed and not data.is_directed():
                result = data.to_directed()
                print(f"DEBUG: Converted to directed")
            elif not directed and data.is_directed():
                result = data.to_undirected()
                print(f"DEBUG: Converted to undirected")
            else:
                result = data
        else:
            result = data
        print(f"DEBUG: Final graph: {result.number_of_nodes()} nodes, {result.number_of_edges()} edges")
        return result
    
    # Handle other formats if needed
    print(f"DEBUG: Processing dictionary/data format")
    if directed is None:
        directed = data.get('directed', False)
        print(f"DEBUG: Auto-detected directed: {directed}")
    
    graph = nx.DiGraph() if directed else nx.Graph()
    
    if 'nodes' in data:
        graph.add_nodes_from(data['nodes'])
        print(f"DEBUG: Added {len(data['nodes'])} nodes")
    if 'edges' in data:
        graph.add_edges_from(data['edges'])
        print(f"DEBUG: Added {len(data['edges'])} edges")
    
    print(f"DEBUG: Final graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph

def count_graphlets_helper(inp):
    """Counting function with detailed debugging."""
    i, query, target, method, node_anchored, anchor_or_none, timeout = inp
    
    start_time = time.time()
    
    print(f"DEBUG_TASK: Processing query {i}, nodes: {query.number_of_nodes()}, edges: {query.number_of_edges()}, "
          f"anchored: {node_anchored}, anchor: {anchor_or_none}")
    
    # Basic size checks
    if query.number_of_nodes() > target.number_of_nodes():
        print(f"DEBUG_TASK: Query {i} skipped - too many nodes")
        return i, 0
    if query.number_of_edges() > target.number_of_edges():
        print(f"DEBUG_TASK: Query {i} skipped - too many edges")
        return i, 0
    
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
        
        print(f"DEBUG_TASK: Query {i} - Starting isomorphism check")
        
        if method == "freq":
            print(f"DEBUG_TASK: Query {i} - Computing symmetries")
            if query.is_directed():
                ismags = nx.isomorphism.DiGraphMatcher(query, query)
            else:
                ismags = nx.isomorphism.ISMAGS(query, query)
            n_symmetries = len(list(ismags.isomorphisms_iter(symmetry=False)))
            print(f"DEBUG_TASK: Query {i} - Found {n_symmetries} symmetries")
        
        if method == "bin":
            if node_anchored:
                print(f"DEBUG_TASK: Query {i} - Anchored matching with anchor {anchor_or_none}")
                nx.set_node_attributes(target, 0, name="anchor")
                target.nodes[anchor_or_none]["anchor"] = 1
                
                if target.is_directed():
                    matcher = iso.DiGraphMatcher(target, query,
                        node_match=iso.categorical_node_match(["anchor"], [0]))
                else:
                    matcher = iso.GraphMatcher(target, query,
                        node_match=iso.categorical_node_match(["anchor"], [0]))
                
                count = int(matcher.subgraph_is_isomorphic())
                print(f"DEBUG_TASK: Query {i} - Anchored result: {count}")
            else:
                print(f"DEBUG_TASK: Query {i} - Regular binary matching")
                if target.is_directed():
                    matcher = iso.DiGraphMatcher(target, query)
                else:
                    matcher = iso.GraphMatcher(target, query)
                
                count = int(matcher.subgraph_is_isomorphic())
                print(f"DEBUG_TASK: Query {i} - Binary result: {count}")
                
        elif method == "freq":
            print(f"DEBUG_TASK: Query {i} - Frequency counting")
            if target.is_directed():
                matcher = iso.DiGraphMatcher(target, query)
            else:
                matcher = iso.GraphMatcher(target, query)
            
            count = 0
            for match_idx, _ in enumerate(matcher.subgraph_isomorphisms_iter()):
                if time.time() - start_time > timeout:
                    print(f"DEBUG_TASK: Query {i} - Timeout during iteration")
                    break
                count += 1
                if count >= MAX_MATCHES_PER_QUERY:
                    print(f"DEBUG_TASK: Query {i} - Reached max matches")
                    break
            
            print(f"DEBUG_TASK: Query {i} - Raw count: {count}")
            
            if method == "freq" and n_symmetries > 0:
                count = count / n_symmetries
                print(f"DEBUG_TASK: Query {i} - Normalized count: {count}")
        
        signal.alarm(0)
            
    except TimeoutError:
        print(f"DEBUG_TASK: Query {i} - TIMEOUT")
        count = 0
    except Exception as e:
        print(f"DEBUG_TASK: Query {i} - ERROR: {str(e)}")
        count = 0
        
    processing_time = time.time() - start_time
    print(f"DEBUG_TASK: Query {i} - Completed in {processing_time:.2f}s, final count: {count}")
    
    return i, count

def count_graphlets(queries, targets, args):
    """Counting function with detailed progress tracking."""
    print(f"DEBUG: Starting count_graphlets with {len(queries)} queries and {len(targets)} targets")
    print(f"DEBUG: Count method: {args.count_method}, node_anchored: {args.node_anchored}")
    
    n_matches = defaultdict(float)
    
    inp = []
    for i, query in enumerate(queries):
        if query.number_of_nodes() > args.max_query_size:
            print(f"DEBUG: Skipping query {i} - exceeds max size {args.max_query_size}")
            continue
            
        print(f"DEBUG: Processing query {i} with {query.number_of_nodes()} nodes, {query.number_of_edges()} edges")
            
        for target_idx, target in enumerate(targets):
            print(f"DEBUG: Query {i} with target {target_idx} ({target.number_of_nodes()} nodes)")
            
            if args.node_anchored:
                if target.number_of_nodes() > args.sample_anchors:
                    anchors = random.sample(list(target.nodes), args.sample_anchors)
                    print(f"DEBUG: Query {i} - Sampling {len(anchors)} anchors from {target.number_of_nodes()} nodes")
                else:
                    anchors = list(target.nodes)
                    print(f"DEBUG: Query {i} - Using all {len(anchors)} anchors")
                    
                for anchor_idx, anchor in enumerate(anchors):
                    inp.append((i, query, target, args.count_method, args.node_anchored, anchor, args.timeout))
                    if anchor_idx < 3:  # Print first few anchors for debugging
                        print(f"DEBUG: Query {i} - Anchor {anchor_idx}: {anchor}")
            else:
                inp.append((i, query, target, args.count_method, args.node_anchored, None, args.timeout))
                print(f"DEBUG: Query {i} - No anchoring")
    
    print(f"DEBUG: Generated {len(inp)} total tasks")
    
    if len(inp) == 0:
        print("DEBUG: WARNING - No tasks generated! Check query/target compatibility")
        return [0] * len(queries)
    
    completed_tasks = 0
    with Pool(processes=args.n_workers) as pool:
        results = pool.imap_unordered(count_graphlets_helper, inp)
        
        for result_idx, (i, n) in enumerate(results):
            n_matches[i] += n
            completed_tasks += 1
            
            if completed_tasks % 10 == 0:
                print(f"DEBUG_PROGRESS: Completed {completed_tasks}/{len(inp)} tasks")
                print(f"DEBUG_PROGRESS: Current counts: {dict(n_matches)}")
    
    # Return counts in the original order
    final_counts = [n_matches[i] for i in range(len(queries))]
    print(f"DEBUG: Final counts: {final_counts}")
    return final_counts

def main():
    args = arg_parse()
    
    print("=" * 60)
    print("GRAPH COUNTING DEBUG VERSION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Queries: {args.queries_path}")
    print(f"Output: {args.out_path}")
    print(f"Workers: {args.n_workers}")
    print(f"Count method: {args.count_method}")
    print(f"Node anchored: {args.node_anchored}")
    print(f"Graph type: {args.graph_type}")
    print(f"Max query size: {args.max_query_size}")
    print(f"Sample anchors: {args.sample_anchors}")
    print(f"Timeout: {args.timeout}")
    print("=" * 60)

    use_directed = (args.graph_type == 'directed')
    print(f"DEBUG: use_directed = {use_directed}")

    # Load target graph
    print(f"\nDEBUG: Loading target graph from {args.dataset}")
    target_graph = load_networkx_graph(args.dataset, use_directed)
    targets = [target_graph]
    
    print(f"DEBUG: Loaded target - {target_graph.number_of_nodes()} nodes, {target_graph.number_of_edges()} edges")
    print(f"DEBUG: Target type: {'directed' if target_graph.is_directed() else 'undirected'}")
    
    # Check target properties
    degrees = dict(target_graph.degree())
    if degrees:
        avg_degree = sum(degrees.values()) / len(degrees)
        print(f"DEBUG: Target degree stats - min: {min(degrees.values())}, max: {max(degrees.values())}, avg: {avg_degree:.2f}")
    else:
        print("DEBUG: Target has no nodes or edges")

    # Load queries
    print(f"\nDEBUG: Loading queries from {args.queries_path}")
    try:
        with open(args.queries_path, "rb") as f:
            queries = pickle.load(f)
        print(f"DEBUG: Successfully loaded {len(queries)} queries")
    except Exception as e:
        print(f"DEBUG: ERROR loading queries: {e}")
        return
    
    if not queries:
        print("DEBUG: ERROR - No queries loaded!")
        return
    
    # Check query properties before conversion
    print(f"DEBUG: Query types before conversion: {[type(q).__name__ for q in queries]}")
    print(f"DEBUG: Query directed flags before: {[q.is_directed() if hasattr(q, 'is_directed') else 'N/A' for q in queries]}")
    
    # Ensure correct graph direction
    converted_queries = []
    for i, query in enumerate(queries):
        if not isinstance(query, (nx.Graph, nx.DiGraph)):
            print(f"DEBUG: WARNING - Query {i} is not a NetworkX graph: {type(query)}")
            continue
            
        if use_directed:
            if not query.is_directed():
                converted_query = query.to_directed()
                print(f"DEBUG: Query {i} converted to directed")
            else:
                converted_query = query
        else:
            if query.is_directed():
                converted_query = query.to_undirected()
                print(f"DEBUG: Query {i} converted to undirected")
            else:
                converted_query = query
        converted_queries.append(converted_query)
    
    queries = converted_queries
    print(f"DEBUG: Final query count: {len(queries)}")
    
    # Print detailed query information
    print(f"\nDEBUG: Query details:")
    for i, query in enumerate(queries):
        print(f"  Query {i}: {query.number_of_nodes()} nodes, {query.number_of_edges()} edges, "
              f"directed: {query.is_directed()}")
        if i >= 5:  # Limit output
            print(f"  ... and {len(queries) - 5} more queries")
            break
    
    query_lens = [q.number_of_nodes() for q in queries]
    print(f"DEBUG: Query lengths: {query_lens}")

    # Count graphlets
    print(f"\nDEBUG: Starting graphlet counting...")
    n_matches = count_graphlets(queries, targets, args)
    
    # Prepare output in original format
    output_data = [query_lens, n_matches, []]
    
    # Save results
    output_file = os.path.join(args.out_path, "counts.json")
    print(f"\nDEBUG: Saving results to {output_file}")
    with open(output_file, "w") as f:
        json.dump(output_data, f)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print("=" * 60)
    print(f"Query lengths: {query_lens}")
    print(f"Counts: {n_matches}")
    print(f"Total queries: {len(queries)}")
    print(f"Queries with matches: {sum(1 for count in n_matches if count > 0)}")
    print(f"Total matches: {sum(n_matches)}")
    print(f"Output file: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()