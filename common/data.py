import os
import pickle
import random

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.datasets import TUDataset, PPI, QM9
import torch_geometric.utils as pyg_utils
from tqdm import tqdm
import scipy.stats as stats
from common import combined_syn
from common import feature_preprocess
from common import utils

def load_dataset(name):
    """ Load real-world datasets, available in PyTorch Geometric.

    Used as a helper for DiskDataSource.
    """
    task = "graph"
    if name == "enzymes":
        dataset = TUDataset(root="/tmp/ENZYMES", name="ENZYMES")
    elif name == "proteins":
        dataset = TUDataset(root="/tmp/PROTEINS", name="PROTEINS")
    elif name == "cox2":
        dataset = TUDataset(root="/tmp/cox2", name="COX2")
    elif name == "aids":
        dataset = TUDataset(root="/tmp/AIDS", name="AIDS")
    elif name == "reddit-binary":
        dataset = TUDataset(root="/tmp/REDDIT-BINARY", name="REDDIT-BINARY")
    elif name == "imdb-binary":
        dataset = TUDataset(root="/tmp/IMDB-BINARY", name="IMDB-BINARY")
    elif name == "firstmm_db":
        dataset = TUDataset(root="/tmp/FIRSTMM_DB", name="FIRSTMM_DB")
    elif name == "dblp":
        dataset = TUDataset(root="/tmp/DBLP_v1", name="DBLP_v1")
    elif name == "ppi":
        dataset = PPI(root="/tmp/PPI")
    elif name == "qm9":
        dataset = QM9(root="/tmp/QM9")
    elif name == "atlas":
        dataset = [g for g in nx.graph_atlas_g()[1:] if nx.is_connected(g)]
    if task == "graph":
        train_len = int(0.8 * len(dataset))
        train, test = [], []
        dataset = list(dataset)
        random.shuffle(dataset)
        has_name = hasattr(dataset[0], "name")
        for i, graph in tqdm(enumerate(dataset)):
            if not type(graph) == nx.Graph:
                if has_name: del graph.name
                graph = pyg_utils.to_networkx(graph).to_undirected()
            if i < train_len:
                train.append(graph)
            else:
                test.append(graph)
    return train, test, task


def sample_subgraph(g_obj, anchors=None, radius=2, hard_neg_idxs=None):
    """
    Sample a subgraph around an anchor node from a graph wrapped in DSGraph.

    Args:
        g_obj (DSGraph): Graph object containing a NetworkX graph (G).
        anchors (list[int], optional): List of anchor nodes per graph index.
        radius (int): Number of hops from the anchor node to include.
        hard_neg_idxs (set, optional): Set of graph indices to apply hard negative logic.

    Returns:
        Tuple[DSGraph, DSGraph]: (Original graph, sampled subgraph)
    """
    G = g_obj.G  # NetworkX graph inside DSGraph
    idx = G.graph.get("idx", 0)  # Unique index for the graph

    # Hard negative logic: return entire graph
    if hard_neg_idxs is not None and idx in hard_neg_idxs:
        subgraph = G.copy()
    else:
        # Choose anchor node
        if anchors is not None:
            anchor = anchors[idx]
        else:
            anchor = random.choice(list(G.nodes))

        # k-hop neighborhood around anchor
        nodes = nx.single_source_shortest_path_length(G, anchor, cutoff=radius).keys()
        subgraph = G.subgraph(nodes).copy()

    # Ensure node_feature exists for all nodes
    for v in subgraph.nodes:
        if "node_feature" not in subgraph.nodes[v]:
            subgraph.nodes[v]["node_feature"] = torch.ones(1)

    return g_obj, DSGraph(subgraph)

class DataSource:
    def gen_batch(batch_target, batch_neg_target, batch_neg_query, train):
        raise NotImplementedError

class GeneGraphDataSource:
  def __init__(self, graph_pkl_path, node_anchored=True, num_queries=32, subgraph_hops=1):
    import pickle
    import networkx as nx
    import torch
    from deepsnap.graph import Graph as DSGraph

    with open(graph_pkl_path, "rb") as f:
        raw_data = pickle.load(f)

    G = nx.Graph()
    G.add_nodes_from(raw_data['nodes'])

    # Clean edge attributes before adding
    cleaned_edges = []
    for edge in raw_data['edges']:
        if len(edge) == 3:
            u, v, attr = edge
            cleaned_attr = {}
            for key, val in attr.items():
                if isinstance(val, (int, float)):  # Only keep tensor-compatible values
                    cleaned_attr[key] = val
                # optionally: encode string attributes like this
                # elif key == 'type':
                #     cleaned_attr[key] = 0  # or a mapping if you have multiple types
            cleaned_edges.append((u, v, cleaned_attr))
        else:
            cleaned_edges.append(edge)  # No attribute case
    G.add_edges_from(cleaned_edges)

    # Ensure node features exist
    for node in G.nodes():
        if 'node_feature' not in G.nodes[node]:
            G.nodes[node]['node_feature'] = torch.tensor([1.0])

    self.full_graph = DSGraph(G)
    self.node_anchored = node_anchored
    self.num_queries = num_queries
    self.subgraph_hops = subgraph_hops
  
  def gen_batch(self, batch_target, batch_neg_target, batch_neg_query, train):
    import random
    import networkx as nx
    from deepsnap.graph import Graph as DSGraph
    from torch_geometric.data import Batch

    # 1. Sample random query nodes from the underlying NetworkX graph
    query_nodes = random.sample(list(self.full_graph.G.nodes), self.num_queries)

    # 2. Prepare positive target (whole graph)
    pos_target_graph = DSGraph(self.full_graph.G.copy())
    pos_target_graph.idx = 0  # Optional: custom attribute
    pos_target = Batch.from_data_list([pos_target_graph])

    # 3. Prepare positive query graphs (subgraphs around sampled nodes)
    query_graphs = []
    for i, node in enumerate(query_nodes):
        sub_nodes = nx.single_source_shortest_path_length(
            self.full_graph.G, node, cutoff=self.subgraph_hops).keys()

        subgraph_nx = self.full_graph.G.subgraph(sub_nodes).copy()
        if subgraph_nx.number_of_edges() == 0:
            # Force at least one edge to avoid DeepSnap crash
            subgraph_nx.add_edge(node, node)

        g = DSGraph(subgraph_nx)
        g.idx = i
        query_graphs.append(g)

    pos_query = Batch.from_data_list(query_graphs)

    # 4. Create valid negative samples (each must have at least one edge)
    def make_valid_dummy_graph(idx):
        G = nx.Graph()
        G.add_edge(0, 1)  # Minimal valid graph with 1 edge
        g = DSGraph(G)
        g.idx = idx
        return g

    neg_target = Batch.from_data_list([make_valid_dummy_graph(i) for i in range(len(query_graphs))])
    neg_query = Batch.from_data_list([make_valid_dummy_graph(i + len(query_graphs)) for i in range(len(query_graphs))])

    return pos_target, pos_query, neg_target, neg_query


  def gen_data_loaders(self, size, batch_size, train=True, use_distributed_sampling=False):
        dummy_loader = [None] * (size // batch_size)
        return [dummy_loader, dummy_loader, dummy_loader]

class OTFSynDataSource(DataSource):
    """ On-the-fly generated synthetic data for training the subgraph model.

    At every iteration, new batch of graphs (positive and negative) are generated
    with a pre-defined generator (see combined_syn.py).

    DeepSNAP transforms are used to generate the positive and negative examples.
    """
    def __init__(self, max_size=29, min_size=5, n_workers=4,
        max_queue_size=256, node_anchored=False):
        self.closed = False
        self.max_size = max_size
        self.min_size = min_size
        self.node_anchored = node_anchored
        self.generator = combined_syn.get_generator(np.arange(
            self.min_size + 1, self.max_size + 1))

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loaders = []
        for i in range(2):
            dataset = combined_syn.get_dataset("graph", size // 2,
                np.arange(self.min_size + 1, self.max_size + 1))
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=hvd.size(), rank=hvd.rank()) if \
                    use_distributed_sampling else None
            loaders.append(TorchDataLoader(dataset,
                collate_fn=Batch.collate([]), batch_size=batch_size // 2 if i
                == 0 else batch_size // 2,
                sampler=sampler, shuffle=False))
        loaders.append([None]*(size // batch_size))
        return loaders

    def gen_batch(self, batch_target, batch_neg_target, batch_neg_query,
        train):
        def sample_subgraph(graph, offset=0, use_precomp_sizes=False,
            filter_negs=False, supersample_small_graphs=False, neg_target=None,
            hard_neg_idxs=None):
            if neg_target is not None: graph_idx = graph.G.graph["idx"]
            use_hard_neg = (hard_neg_idxs is not None and graph.G.graph["idx"]
                in hard_neg_idxs)
            done = False
            n_tries = 0
            while not done:
                if use_precomp_sizes:
                    size = graph.G.graph["subgraph_size"]
                else:
                    if train and supersample_small_graphs:
                        sizes = np.arange(self.min_size + offset,
                            len(graph.G) + offset)
                        ps = (sizes - self.min_size + 2) ** (-1.1)
                        ps /= ps.sum()
                        size = stats.rv_discrete(values=(sizes, ps)).rvs()
                    else:
                        d = 1 if train else 0
                        size = random.randint(self.min_size + offset - d,
                            len(graph.G) - 1 + offset)
                start_node = random.choice(list(graph.G.nodes))
                neigh = [start_node]
                frontier = list(set(graph.G.neighbors(start_node)) - set(neigh))
                visited = set([start_node])
                while len(neigh) < size:
                    new_node = random.choice(list(frontier))
                    assert new_node not in neigh
                    neigh.append(new_node)
                    visited.add(new_node)
                    frontier += list(graph.G.neighbors(new_node))
                    frontier = [x for x in frontier if x not in visited]
                if self.node_anchored:
                    anchor = neigh[0]
                    for v in graph.G.nodes:
                        graph.G.nodes[v]["node_feature"] = (torch.ones(1) if
                            anchor == v else torch.zeros(1))
                        #print(v, graph.G.nodes[v]["node_feature"])
                neigh = graph.G.subgraph(neigh)
                if use_hard_neg and train:
                    neigh = neigh.copy()
                    if random.random() < 1.0 or not self.node_anchored: # add edges
                        non_edges = list(nx.non_edges(neigh))
                        if len(non_edges) > 0:
                            for u, v in random.sample(non_edges, random.randint(1,
                                min(len(non_edges), 5))):
                                neigh.add_edge(u, v)
                    else:                         # perturb anchor
                        anchor = random.choice(list(neigh.nodes))
                        for v in neigh.nodes:
                            neigh.nodes[v]["node_feature"] = (torch.ones(1) if
                                anchor == v else torch.zeros(1))

                if (filter_negs and train and len(neigh) <= 6 and neg_target is
                    not None):
                    matcher = nx.algorithms.isomorphism.GraphMatcher(
                        neg_target[graph_idx], neigh)
                    if not matcher.subgraph_is_isomorphic(): done = True
                else:
                    done = True

            return graph, DSGraph(neigh)

        augmenter = feature_preprocess.FeatureAugment()

        pos_target = batch_target
        pos_target, pos_query = pos_target.apply_transform_multi(sample_subgraph)
        neg_target = batch_neg_target
        # TODO: use hard negs
        hard_neg_idxs = set(random.sample(range(len(neg_target.G)),
            int(len(neg_target.G) * 1/2)))
        #hard_neg_idxs = set()
        batch_neg_query = Batch.from_data_list(
            [DSGraph(self.generator.generate(size=len(g))
                if i not in hard_neg_idxs else g)
                for i, g in enumerate(neg_target.G)])
        for i, g in enumerate(batch_neg_query.G):
            g.graph["idx"] = i
        _, neg_query = batch_neg_query.apply_transform_multi(sample_subgraph,
            hard_neg_idxs=hard_neg_idxs)
        if self.node_anchored:
            def add_anchor(g, anchors=None):
                if anchors is not None:
                    anchor = anchors[g.G.graph["idx"]]
                else:
                    anchor = random.choice(list(g.G.nodes))
                for v in g.G.nodes:
                    if "node_feature" not in g.G.nodes[v]:
                        g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
                            else torch.zeros(1))
                return g
            neg_target = neg_target.apply_transform(add_anchor)
        pos_target = augmenter.augment(pos_target).to(utils.get_device())
        pos_query = augmenter.augment(pos_query).to(utils.get_device())
        neg_target = augmenter.augment(neg_target).to(utils.get_device())
        neg_query = augmenter.augment(neg_query).to(utils.get_device())
        #print(len(pos_target.G[0]), len(pos_query.G[0]))
        return pos_target, pos_query, neg_target, neg_query

class OTFSynImbalancedDataSource(OTFSynDataSource):
    """ Imbalanced on-the-fly synthetic data.

    Unlike the balanced dataset, this data source does not use 1:1 ratio for
    positive and negative examples. Instead, it randomly samples 2 graphs from
    the on-the-fly generator, and records the groundtruth label for the pair (subgraph or not).
    As a result, the data is imbalanced (subgraph relationships are rarer).
    This setting is a challenging model inference scenario.
    """
    def __init__(self, max_size=29, min_size=5, n_workers=4,
        max_queue_size=256, node_anchored=False):
        super().__init__(max_size=max_size, min_size=min_size,
            n_workers=n_workers, node_anchored=node_anchored)
        self.batch_idx = 0

    def gen_batch(self, graphs_a, graphs_b, _, train):
        def add_anchor(g):
            anchor = random.choice(list(g.G.nodes))
            for v in g.G.nodes:
                g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
                    or not self.node_anchored else torch.zeros(1))
            return g
        pos_a, pos_b, neg_a, neg_b = [], [], [], []
        fn = "data/cache/imbalanced-{}-{}".format(str(self.node_anchored),
            self.batch_idx)
        if not os.path.exists(fn):
            graphs_a = graphs_a.apply_transform(add_anchor)
            graphs_b = graphs_b.apply_transform(add_anchor)
            for graph_a, graph_b in tqdm(list(zip(graphs_a.G, graphs_b.G))):
                matcher = nx.algorithms.isomorphism.GraphMatcher(graph_a, graph_b,
                    node_match=(lambda a, b: (a["node_feature"][0] > 0.5) ==
                    (b["node_feature"][0] > 0.5)) if self.node_anchored else None)
                if matcher.subgraph_is_isomorphic():
                    pos_a.append(graph_a)
                    pos_b.append(graph_b)
                else:
                    neg_a.append(graph_a)
                    neg_b.append(graph_b)
            if not os.path.exists("data/cache"):
                os.makedirs("data/cache")
            with open(fn, "wb") as f:
                pickle.dump((pos_a, pos_b, neg_a, neg_b), f)
            print("saved", fn)
        else:
            with open(fn, "rb") as f:
                print("loaded", fn)
                pos_a, pos_b, neg_a, neg_b = pickle.load(f)
        print(len(pos_a), len(neg_a))
        if pos_a:
            pos_a = utils.batch_nx_graphs(pos_a)
            pos_b = utils.batch_nx_graphs(pos_b)
        neg_a = utils.batch_nx_graphs(neg_a)
        neg_b = utils.batch_nx_graphs(neg_b)
        self.batch_idx += 1
        return pos_a, pos_b, neg_a, neg_b

class DiskDataSource(DataSource):
    """ Uses a set of graphs saved in a dataset file to train the subgraph model.

    At every iteration, new batch of graphs (positive and negative) are generated
    by sampling subgraphs from a given dataset.

    See the load_dataset function for supported datasets.
    """
    def __init__(self, dataset_name, node_anchored=False, min_size=5,
        max_size=29):
        self.node_anchored = node_anchored
        self.dataset = load_dataset(dataset_name)
        self.min_size = min_size
        self.max_size = max_size

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loaders = [[batch_size]*(size // batch_size) for i in range(3)]
        return loaders

    def gen_batch(self, a, b, c, train, max_size=15, min_size=5, seed=None,
        filter_negs=False, sample_method="tree-pair"):
        batch_size = a
        train_set, test_set, task = self.dataset
        graphs = train_set if train else test_set
        if seed is not None:
            random.seed(seed)

        pos_a, pos_b = [], []
        pos_a_anchors, pos_b_anchors = [], []
        for i in range(batch_size // 2):
            if sample_method == "tree-pair":
                size = random.randint(min_size+1, max_size)
                graph, a = utils.sample_neigh(graphs, size)
                b = a[:random.randint(min_size, len(a) - 1)]
            elif sample_method == "subgraph-tree":
                graph = None
                while graph is None or len(graph) < min_size + 1:
                    graph = random.choice(graphs)
                a = graph.nodes
                _, b = utils.sample_neigh([graph], random.randint(min_size,
                    len(graph) - 1))
            if self.node_anchored:
                anchor = list(graph.nodes)[0]
                pos_a_anchors.append(anchor)
                pos_b_anchors.append(anchor)
            neigh_a, neigh_b = graph.subgraph(a), graph.subgraph(b)
            pos_a.append(neigh_a)
            pos_b.append(neigh_b)

        neg_a, neg_b = [], []
        neg_a_anchors, neg_b_anchors = [], []
        while len(neg_a) < batch_size // 2:
            if sample_method == "tree-pair":
                size = random.randint(min_size+1, max_size)
                graph_a, a = utils.sample_neigh(graphs, size)
                graph_b, b = utils.sample_neigh(graphs, random.randint(min_size,
                    size - 1))
            elif sample_method == "subgraph-tree":
                graph_a = None
                while graph_a is None or len(graph_a) < min_size + 1:
                    graph_a = random.choice(graphs)
                a = graph_a.nodes
                graph_b, b = utils.sample_neigh(graphs, random.randint(min_size,
                    len(graph_a) - 1))
            if self.node_anchored:
                neg_a_anchors.append(list(graph_a.nodes)[0])
                neg_b_anchors.append(list(graph_b.nodes)[0])
            neigh_a, neigh_b = graph_a.subgraph(a), graph_b.subgraph(b)
            if filter_negs:
                matcher = nx.algorithms.isomorphism.GraphMatcher(neigh_a, neigh_b)
                if matcher.subgraph_is_isomorphic(): # a <= b (b is subgraph of a)
                    continue
            neg_a.append(neigh_a)
            neg_b.append(neigh_b)

        pos_a = utils.batch_nx_graphs(pos_a, anchors=pos_a_anchors if
            self.node_anchored else None)
        pos_b = utils.batch_nx_graphs(pos_b, anchors=pos_b_anchors if
            self.node_anchored else None)
        neg_a = utils.batch_nx_graphs(neg_a, anchors=neg_a_anchors if
            self.node_anchored else None)
        neg_b = utils.batch_nx_graphs(neg_b, anchors=neg_b_anchors if
            self.node_anchored else None)
        return pos_a, pos_b, neg_a, neg_b

class DiskImbalancedDataSource(OTFSynDataSource):
    """ Imbalanced on-the-fly real data.

    Unlike the balanced dataset, this data source does not use 1:1 ratio for
    positive and negative examples. Instead, it randomly samples 2 graphs from
    the on-the-fly generator, and records the groundtruth label for the pair (subgraph or not).
    As a result, the data is imbalanced (subgraph relationships are rarer).
    This setting is a challenging model inference scenario.
    """
    def __init__(self, dataset_name, max_size=29, min_size=5, n_workers=4,
        max_queue_size=256, node_anchored=False):
        super().__init__(max_size=max_size, min_size=min_size,
            n_workers=n_workers, node_anchored=node_anchored)
        self.batch_idx = 0
        self.dataset = load_dataset(dataset_name)
        self.train_set, self.test_set, _ = self.dataset
        self.dataset_name = dataset_name

    def gen_data_loaders(self, size, batch_size, train=True,
        use_distributed_sampling=False):
        loaders = []
        for i in range(2):
            neighs = []
            for j in range(size // 2):
                graph, neigh = utils.sample_neigh(self.train_set if train else
                    self.test_set, random.randint(self.min_size, self.max_size))
                neighs.append(graph.subgraph(neigh))
            dataset = GraphDataset(neighs)
            loaders.append(TorchDataLoader(dataset,
                collate_fn=Batch.collate([]), batch_size=batch_size // 2 if i
                == 0 else batch_size // 2,
                sampler=None, shuffle=False))
        loaders.append([None]*(size // batch_size))
        return loaders

    def gen_batch(self, graphs_a, graphs_b, _, train):
        def add_anchor(g):
            anchor = random.choice(list(g.G.nodes))
            for v in g.G.nodes:
                g.G.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v
                    or not self.node_anchored else torch.zeros(1))
            return g
        pos_a, pos_b, neg_a, neg_b = [], [], [], []
        fn = "data/cache/imbalanced-{}-{}-{}".format(self.dataset_name.lower(),
            str(self.node_anchored), self.batch_idx)
        if not os.path.exists(fn):
            graphs_a = graphs_a.apply_transform(add_anchor)
            graphs_b = graphs_b.apply_transform(add_anchor)
            for graph_a, graph_b in tqdm(list(zip(graphs_a.G, graphs_b.G))):
                matcher = nx.algorithms.isomorphism.GraphMatcher(graph_a, graph_b,
                    node_match=(lambda a, b: (a["node_feature"][0] > 0.5) ==
                    (b["node_feature"][0] > 0.5)) if self.node_anchored else None)
                if matcher.subgraph_is_isomorphic():
                    pos_a.append(graph_a)
                    pos_b.append(graph_b)
                else:
                    neg_a.append(graph_a)
                    neg_b.append(graph_b)
            if not os.path.exists("data/cache"):
                os.makedirs("data/cache")
            with open(fn, "wb") as f:
                pickle.dump((pos_a, pos_b, neg_a, neg_b), f)
            print("saved", fn)
        else:
            with open(fn, "rb") as f:
                print("loaded", fn)
                pos_a, pos_b, neg_a, neg_b = pickle.load(f)
        print(len(pos_a), len(neg_a))
        if pos_a:
            pos_a = utils.batch_nx_graphs(pos_a)
            pos_b = utils.batch_nx_graphs(pos_b)
        neg_a = utils.batch_nx_graphs(neg_a)
        neg_b = utils.batch_nx_graphs(neg_b)
        self.batch_idx += 1
        return pos_a, pos_b, neg_a, neg_b
    

import pickle
import networkx as nx
import random
import torch

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch

class PKLGraphDataSource:
    """
    DataSource for a single graph loaded from a pickle file with 'nodes' and 'edges'.
    Generates positive and negative subgraph pairs for training.
    """
    def __init__(self, pkl_path, node_anchored=False, min_size=5, max_size=29):
        self.node_anchored = node_anchored
        self.min_size = min_size
        self.max_size = max_size
        self.graph = self._load_graph(pkl_path)

    def _load_graph(self, pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        G = nx.Graph()
        for n, attrs in data['nodes']:
            G.add_node(n, **attrs)
        for u, v, attrs in data['edges']:
            G.add_edge(u, v, **attrs)
        G = self._sanitize_edge_attrs(G)
        return G

    def _sanitize_edge_attrs(self, g):
        for u, v, attrs in g.edges(data=True):
            if attrs is None or not isinstance(attrs, dict):
                g[u][v].clear()
        return g

    def _sample_subgraph(self, graph, size, max_tries=10):
        for _ in range(max_tries):
            nodes = random.sample(list(graph.nodes), min(size, len(graph.nodes)))
            subg = graph.subgraph(nodes).copy()
            if subg.number_of_edges() > 0:
                return subg
        # If after max_tries still no edges, return the largest connected component
        if subg.number_of_edges() == 0 and subg.number_of_nodes() > 1:
            components = list(nx.connected_components(subg))
            if components:
                largest_cc = max(components, key=len)
                subg = subg.subgraph(largest_cc).copy()
        return subg

    def _bfs_sample_subgraph(self, graph, size, max_tries=10):
        """
        Sample a connected subgraph of given size using BFS.
        """
        for _ in range(max_tries):
            start_node = random.choice(list(graph.nodes))
            visited = {start_node}
            queue = [start_node]
            while queue and len(visited) < size:
                current = queue.pop(0)
                neighbors = list(set(graph.neighbors(current)) - visited)
                random.shuffle(neighbors)
                for neighbor in neighbors:
                    if len(visited) >= size:
                        break
                    visited.add(neighbor)
                    queue.append(neighbor)
            subg = graph.subgraph(visited).copy()
            subg = self._sanitize_edge_attrs(subg)
            if subg.number_of_edges() > 0 and nx.is_connected(subg):
                return subg
        # fallback: largest connected component
        if subg.number_of_edges() == 0 and subg.number_of_nodes() > 1:
            components = list(nx.connected_components(subg))
            if components:
                largest_cc = max(components, key=len)
                subg = subg.subgraph(largest_cc).copy()
        return subg

    def _add_anchor(self, g, anchor=None):
        if anchor is None:
            anchor = random.choice(list(g.nodes))
        for v in g.nodes:
            g.nodes[v]["node_feature"] = (torch.ones(1) if anchor == v else torch.zeros(1))
        return g

    def gen_batch(self, batch_size, *_args, train=True, **kwargs):
        tries = 0
        max_tries = 20
        while tries < max_tries:
            pos_a, pos_b, neg_a, neg_b = [], [], [], []
            for _ in range(batch_size // 2):
                size_a = random.randint(self.min_size + 1, self.max_size)
                size_b = random.randint(self.min_size, size_a - 1)
                sub_a = self._bfs_sample_subgraph(self.graph, size_a)
                sub_b = self._bfs_sample_subgraph(sub_a, size_b)
                sub_b = self._sanitize_edge_attrs(sub_b)

                if sub_a.number_of_edges() == 0 or sub_b.number_of_edges() == 0:
                    continue
                if self.node_anchored:
                    anchor = random.choice(list(sub_a.nodes))
                    sub_a = self._add_anchor(sub_a, anchor)
                    sub_b = self._add_anchor(sub_b, anchor if anchor in sub_b.nodes else random.choice(list(sub_b.nodes)))
                pos_a.append(sub_a)
                pos_b.append(sub_b)

            for _ in range(batch_size // 2):
                size_a = random.randint(self.min_size + 1, self.max_size)
                size_b = random.randint(self.min_size, self.max_size)
                sub_a = self._bfs_sample_subgraph(self.graph, size_a)
                sub_b = self._bfs_sample_subgraph(self.graph, size_b)
                if sub_a.number_of_edges() == 0 or sub_b.number_of_edges() == 0:
                    continue
                if nx.is_isomorphic(sub_a, sub_b):
                    continue
                if self.node_anchored:
                    anchor_a = random.choice(list(sub_a.nodes))
                    anchor_b = random.choice(list(sub_b.nodes))
                    sub_a = self._add_anchor(sub_a, anchor_a)
                    sub_b = self._add_anchor(sub_b, anchor_b)
                neg_a.append(sub_a)
                neg_b.append(sub_b)

            if pos_a and pos_b and neg_a and neg_b:
                break
            tries += 1

        if not (pos_a and pos_b and neg_a and neg_b):
            raise RuntimeError("Could not generate a non-empty batch after {} tries.".format(max_tries))

        # Convert to DeepSNAP batches
        pos_a = Batch.from_data_list([DSGraph(g) for g in pos_a])
        pos_b = Batch.from_data_list([DSGraph(g) for g in pos_b])
        neg_a = Batch.from_data_list([DSGraph(g) for g in neg_a])
        neg_b = Batch.from_data_list([DSGraph(g) for g in neg_b])
        return pos_a, pos_b, neg_a, neg_b

    def gen_data_loaders(self, size, batch_size, train=True, use_distributed_sampling=False):
        """
        Returns three loaders (lists of batch sizes) for compatibility with the training loop.
        This matches the interface of DiskDataSource and OTFSynDataSource.
        """
        num_batches = size // batch_size
        # Each loader is a list of batch sizes, as expected by the training loop.
        loaders = [[batch_size] * num_batches for _ in range(3)]
        return loaders

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14})
    for name in ["enzymes", "reddit-binary", "cox2"]:
        data_source = DiskDataSource(name)
        train, test, _ = data_source.dataset
        i = 11
        neighs = [utils.sample_neigh(train, i) for j in range(10000)]
        clustering = [nx.average_clustering(graph.subgraph(nodes)) for graph,
            nodes in neighs]
        path_length = [nx.average_shortest_path_length(graph.subgraph(nodes))
            for graph, nodes in neighs]
        #plt.subplot(1, 2, i-9)
        plt.scatter(clustering, path_length, s=10, label=name)
    plt.legend()
    plt.savefig("plots/clustering-vs-path-length.png")