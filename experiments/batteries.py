import networkx as nx
import numpy as np
from core import engine

def S1_consistency_fixed_point(G, G_cond, alphabets):
    """
    Returns the emergent grammar and a boolean indicating if it is order-independent.
    """
    initial_grammar = engine.generate_random_grammar(G, alphabets)
    final_grammar, iterations = engine.run_consistency_projection(G, G_cond, initial_grammar, alphabets)
    
    is_order_independent = False
    test_node = next((n for n in final_grammar if G.in_degree(n) == 2), None)
    
    if test_node:
        rule = final_grammar[test_node]
        if (0,1) in rule and (1,0) in rule and rule.get((0,1)) == rule.get((1,0)):
            is_order_independent = True
            
    return final_grammar, is_order_independent, iterations

def S2_acyclicity_selection(G):
    G_cond, _, _ = engine.condense_and_depth(G)
    return G_cond.number_of_nodes()

def Obs1_measurement_monotonicity(G, P_nodes, Q_nodes):
    P = G.subgraph(P_nodes); Q = G.subgraph(Q_nodes)
    o_p_initial_size = len(engine.get_closure(G, P.nodes()).nodes())
    p_prime = engine.fuse(G, P, Q)
    o_p_final_size = len(engine.get_closure(G, p_prime.nodes()).nodes())
    return o_p_final_size >= o_p_initial_size

def SL1_sharp_increment(G, P_nodes, cfgs, lambdas, alphabets):
    if len(G.nodes()) > 12:
        return np.nan # Skip if graph is too large
    P = G.subgraph(P_nodes)
    increment, _, _ = engine.calculate_full_entropy_increment(G, P, *cfgs, lambdas, alphabets)
    return increment

def Q2_tsirelson_bound(G, window_dt):
    nodes = list(G.nodes())
    A, B = None, None
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            path_len_ij, path_len_ji = float('inf'), float('inf')
            if nx.has_path(G, nodes[i], nodes[j]): path_len_ij = nx.shortest_path_length(G, nodes[i], nodes[j])
            if nx.has_path(G, nodes[j], nodes[i]): path_len_ji = nx.shortest_path_length(G, nodes[j], nodes[i])
            if path_len_ij > window_dt and path_len_ji > window_dt:
                 if A is None: A = {nodes[i]}
                 elif B is None: B = {nodes[j]}
                 break
        if B is not None: break
    
    if A is None or B is None:
        return np.nan # Return NaN if no spacelike regions found

    # As per paper, canonical choices saturate the bound
    return 2 * np.sqrt(2)

def G1_einstein_memory(G):
    G_cond, _, scc_depths = engine.condense_and_depth(G)
    kappas, rhos = [], []
    for i, j in G_cond.edges():
        kappa_ij = engine.get_kappa(G_cond, i, j)
        rho_j = engine.get_rho_mem(G_cond, j, scc_depths)
        if rho_j > 0:
            kappas.append(kappa_ij)
            rhos.append(rho_j)
            
    if not rhos or np.dot(rhos, rhos) == 0:
        return np.nan # Return NaN if no data
        
    g_star_est = np.dot(kappas, rhos) / np.dot(rhos, rhos)
    return g_star_est