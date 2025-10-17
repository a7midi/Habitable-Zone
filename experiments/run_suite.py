import networkx as nx
import numpy as np
import random
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

from experiments import batteries
from core import engine

def generate_realistic_testbed(num_layers=3, nodes_per_layer=3, edge_prob=0.7):
    """Generates a random depth-graded graph."""
    G = nx.DiGraph()
    layers = [[f'L{i}N{j}' for j in range(nodes_per_layer)] for i in range(num_layers)]
    for i in range(num_layers):
        for j in range(nodes_per_layer):
            G.add_node(layers[i][j])

    for i in range(num_layers - 1):
        for u in layers[i]:
            for v in layers[i+1]:
                if random.random() < edge_prob:
                    G.add_edge(u, v)
    
    # Ensure graph is not totally disconnected
    if G.number_of_edges() == 0:
        return generate_realistic_testbed(num_layers, nodes_per_layer, edge_prob)

    alphabets = {node: [0, 1] for node in G.nodes()}
    p_nodes = {random.choice(layers[0])} if layers and layers[0] else {next(iter(G.nodes()))}
    q_nodes = {random.choice(layers[-1])} if layers and layers[-1] else {G.number_of_nodes()-1}
    return G, alphabets, p_nodes, q_nodes

def run_single_simulation(graph_params, run_sl1_test):
    """
    Runs one full end-to-end simulation and returns the results as a dictionary.
    """
    G, alphabets, p_nodes, q_nodes = generate_realistic_testbed(**graph_params)
    G_cond, _, _ = engine.condense_and_depth(G)

    emergent_lambdas, is_oi, iterations = batteries.S1_consistency_fixed_point(G, G_cond, alphabets)
    
    sl1_increment = np.nan
    if run_sl1_test:
        cfgs = [ {node: random.choice(alphabets[node]) for node in G.nodes()} ]
        for _ in range(2):
            cfgs.append(engine.tick(G, cfgs[-1], emergent_lambdas))
        sl1_increment = batteries.SL1_sharp_increment(G, p_nodes, cfgs, emergent_lambdas, alphabets)

    results = {
        's1_order_independent': is_oi,
        'sl1_entropy_increment': sl1_increment,
        'q2_chsh_value': batteries.Q2_tsirelson_bound(G, window_dt=1),
        'g1_g_star_est': batteries.G1_einstein_memory(G),
        'graph_nodes': G.number_of_nodes(),
        'graph_edges': G.number_of_edges(),
        'edge_prob': graph_params['edge_prob'] # Track parameter
    }
    return results

def analyze_and_plot_results(df):
    """
    Takes a DataFrame of all results, prints a full analysis, and saves plots.
    """
    print("\n\n--- 📊 SCIENTIFIC STUDY FINAL ANALYSIS ---")
    
    # --- Parameter Sweep Analysis (Testing Universality) ---
    print("\n** 1. Parameter Sweep Analysis (Testing Universality of g*) **")
    sweep_df = df[df['study'] == 'Parameter Sweep']
    
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='edge_prob', y='g1_g_star_est', data=sweep_df)
    plt.title('Universality of g* Across Different Graph Densities', fontsize=16, pad=20)
    plt.xlabel('Graph Edge Probability', fontsize=12)
    plt.ylabel('Estimated Coupling Constant (g*)', fontsize=12)
    
    plot_filename1 = 'g_star_universality_sweep.png'
    plt.savefig(plot_filename1)
    print(f"--> Universality plot saved to '{plot_filename1}'")
    print("Interpretation: If the median lines of the boxes are at a similar height, it supports the hypothesis that g* is a universal constant, independent of graph density.")
    
    

    # --- Scaling Analysis (Approaching the Continuum) ---
    print("\n** 2. Scaling Analysis (Testing Convergence) **")
    scaling_df = df[df['study'] == 'Scaling Analysis']
    scaling_summary = scaling_df.groupby('graph_nodes')['g1_g_star_est'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(12, 7))
    plt.errorbar(scaling_summary['graph_nodes'], scaling_summary['mean'], yerr=scaling_summary['std'], 
                 fmt='-o', capsize=5, label='Mean g* with Std. Dev.')
    plt.title('Convergence of g* as Graph Size Increases', fontsize=16, pad=20)
    plt.xlabel('Number of Nodes in Graph (System Scale)', fontsize=12)
    plt.ylabel('Estimated Coupling Constant (g*)', fontsize=12)
    plt.xticks(scaling_summary['graph_nodes'])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    plot_filename2 = 'g_star_scaling_analysis.png'
    plt.savefig(plot_filename2)
    print(f"--> Scaling analysis plot saved to '{plot_filename2}'")
    print("Interpretation: If the standard deviation (error bars) decreases as the number of nodes increases, it supports the hypothesis that g* converges to a more precise value at larger scales.")

    

def main():
    """
    Main driver for the full scientific study.
    """
    start_time = time.time()
    
    # --- Study Configurations ---
    # Note: SL-1 is disabled for larger graphs to ensure reasonable runtime.
    
    # Study 1: Parameter Sweep (fixed size, varying density)
    PARAMETER_SWEEP_CONFIGS = [
        {'study': 'Parameter Sweep', 'runs': 50, 'run_sl1': True,  'params': {'num_layers': 3, 'nodes_per_layer': 3, 'edge_prob': 0.4}},
        {'study': 'Parameter Sweep', 'runs': 50, 'run_sl1': True,  'params': {'num_layers': 3, 'nodes_per_layer': 3, 'edge_prob': 0.7}},
        {'study': 'Parameter Sweep', 'runs': 50, 'run_sl1': True,  'params': {'num_layers': 3, 'nodes_per_layer': 3, 'edge_prob': 1.0}},
    ]
    
    # Study 2: Scaling Analysis (varying size)
    SCALING_ANALYSIS_CONFIGS = [
        {'study': 'Scaling Analysis', 'runs': 50, 'run_sl1': True,  'params': {'num_layers': 3, 'nodes_per_layer': 3, 'edge_prob': 0.7}}, # 9 nodes
        {'study': 'Scaling Analysis', 'runs': 30, 'run_sl1': False, 'params': {'num_layers': 4, 'nodes_per_layer': 4, 'edge_prob': 0.7}}, # 16 nodes
        {'study': 'Scaling Analysis', 'runs': 20, 'run_sl1': False, 'params': {'num_layers': 5, 'nodes_per_layer': 5, 'edge_prob': 0.7}}, # 25 nodes
    ]

    all_study_configs = PARAMETER_SWEEP_CONFIGS + SCALING_ANALYSIS_CONFIGS
    all_results = []
    
    print("===== Starting Full Scientific Study =====")
    total_sims = sum(c['runs'] for c in all_study_configs)
    sim_counter = 0

    for config in all_study_configs:
        print(f"\n--- Running Study: '{config['study']}' | Runs: {config['runs']} | Params: {config['params']} ---")
        for i in range(config['runs']):
            sim_counter += 1
            print(f"  -> Simulation {i+1}/{config['runs']} (Overall {sim_counter}/{total_sims})")
            
            result = run_single_simulation(config['params'], config['run_sl1'])
            result['study'] = config['study'] # Tag data with study name
            all_results.append(result)
            
    results_df = pd.DataFrame(all_results)
    
    analyze_and_plot_results(results_df)
    
    end_time = time.time()
    print(f"\nTotal study duration: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()