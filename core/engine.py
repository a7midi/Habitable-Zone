import networkx as nx
from itertools import product, combinations
import math
import random
from collections import Counter
from typing import Set, Dict, Tuple, Any, List

# --- Data Structure for Equivalence Classes ---
class DSU:
    """Disjoint Set Union (DSU) or Union-Find data structure."""
    def __init__(self, elements):
        self.parent = {el: el for el in elements}
    def find(self, i):
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i
            return True
        return False

# --- Core Utility ---
def get_closure(G: nx.DiGraph, S: Set, direction: str = 'backward') -> nx.DiGraph:
    """
    Computes the closure of a set S.
    'backward' (default) is predecessor-closure, as per the paper.
    'forward' is successor-closure, for modeling forward growth.
    """
    closure_nodes = set(S)
    get_neighbors = G.predecessors if direction == 'backward' else G.successors
    
    while True:
        neighbors_to_add = {neighbor for node in closure_nodes for neighbor in get_neighbors(node) if neighbor not in closure_nodes}
        if not neighbors_to_add: break
        closure_nodes.update(neighbors_to_add)
    return G.subgraph(closure_nodes)

def fuse(G: nx.DiGraph, P: nx.DiGraph, Q: nx.DiGraph, direction: str = 'backward') -> nx.DiGraph:
    """Corrected fuse function that keeps all fused components."""
    union_nodes = set(P.nodes()) | set(Q.nodes())
    closure_subgraph = get_closure(G, union_nodes, direction=direction)
    undirected = closure_subgraph.to_undirected()
    keep = set()
    for comp in nx.connected_components(undirected):
        if union_nodes & comp:
            keep |= comp
    return G.subgraph(keep) if keep else G.subgraph(union_nodes)

def tick(G: nx.DiGraph, config: Dict, lambdas: Dict) -> Dict:
    """Performs one synchronous (parallel) update tick."""
    new_config = {}
    for v in G.nodes():
        preds = sorted(list(G.predecessors(v)))
        pred_tags = tuple(config.get(p, 0) for p in preds)
        if v in lambdas and pred_tags in lambdas[v]:
            new_config[v] = lambdas[v][pred_tags]
        else:
            new_config[v] = config.get(v, 0)
    return new_config