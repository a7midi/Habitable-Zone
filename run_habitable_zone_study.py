import networkx as nx
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Dict
from core import engine

def classify_zones(G: nx.DiGraph) -> Dict:
    """Classify nodes using SCCs (robust and scalable)."""
    print("Classifying cyclic and acyclic zones...")
    sccs = list(nx.strongly_connected_components(G))
    cyclic_nodes = set().union(*(s for s in sccs if len(s) > 1))
    cyclic_nodes |= {n for n in G if G.has_edge(n, n)} # Add self-loops
    print("...classification complete.")
    return {n: ('Cyclic' if n in cyclic_nodes else 'Acyclic') for n in G.nodes()}

def create_universe_soup(num_nodes, edge_prob, seed):
    """Creates a single, large, random graph with mixed regions."""
    print(f"Generating a 'universe soup' with {num_nodes} nodes...")
    G = nx.gnp_random_graph(num_nodes, edge_prob, directed=True, seed=seed)
    return G

def analyze_and_plot_results(df):
    """Takes a DataFrame of all results, prints a full analysis, and saves plots."""
    print("\n\n--- 📊 SCIENTIFIC STUDY FINAL ANALYSIS ---")
    
    # --- New, More Insightful Metrics ---
    print("\n** Growth Dynamics Analysis:")
    final_sizes = df[df['Tick'] == df['Tick'].max()]
    
    avg_final_size_acyclic = final_sizes[final_sizes['Zone'] == 'Acyclic']['Size'].mean()
    avg_final_size_cyclic = final_sizes[final_sizes['Zone'] == 'Cyclic']['Size'].mean()

    # Calculate initial growth rate (change in size over first 5 ticks)
    initial_growth = df[df['Tick'] <= 5].groupby(['Observer ID', 'Zone'])['Size'].agg(['first', 'last']).reset_index()
    initial_growth['rate'] = (initial_growth['last'] - initial_growth['first']) / 5
    
    avg_rate_acyclic = initial_growth[initial_growth['Zone'] == 'Acyclic']['rate'].mean()
    avg_rate_cyclic = initial_growth[initial_growth['Zone'] == 'Cyclic']['rate'].mean()

    print(f"Average Initial Growth Rate (Acyclic Zone): {avg_rate_acyclic:.2f} nodes/tick")
    print(f"Average Initial Growth Rate (Cyclic Zone):  {avg_rate_cyclic:.2f} nodes/tick")
    print(f"Average Final Size (Acyclic Zone):          {avg_final_size_acyclic:.2f} nodes")
    print(f"Average Final Size (Cyclic Zone):           {avg_final_size_cyclic:.2f} nodes")

    # --- Plotting ---
    plt.figure(figsize=(14, 8))
    sns.lineplot(x='Tick', y='Size', hue='Zone', data=df, errorbar='sd', palette={'Acyclic': 'blue', 'Cyclic': 'red'})
    plt.title('Growth of Observers in Different Universal Zones', fontsize=16, pad=20)
    plt.xlabel('Simulation Time (Ticks)', fontsize=12)
    plt.ylabel('Observer Size (Number of Nodes)', fontsize=12)
    plt.legend(title='Initial Zone')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plot_filename = 'habitable_zone_emergence_FINAL.png'
    plt.savefig(plot_filename)
    plt.close()
    
    print(f"\n** Plot saved to '{plot_filename}'")
    

def main():
    start_time = time.time()
    
    # --- Study Parameters ---
    NUM_NODES = 150
    EDGE_PROB = 0.01  # Lower probability for more distinct zones
    NUM_SEEDS_PER_ZONE = 25
    SIMULATION_TICKS = 30
    RANDOM_SEED = 42

    # --- Seeding for Reproducibility ---
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("===== Starting 'Emergence by Habitable Zone' Study (FINAL VERSION) =====")

    # 1. Create the Universe and classify its regions
    G = create_universe_soup(NUM_NODES, EDGE_PROB, seed=RANDOM_SEED)
    node_zones = classify_zones(G)

    # 2. Stratified Seeding: Deliberately seed observers in each zone
    acyclic_nodes = [n for n, zone in node_zones.items() if zone == 'Acyclic']
    cyclic_nodes = [n for n, zone in node_zones.items() if zone == 'Cyclic']
    print(f"Found {len(acyclic_nodes)} Acyclic nodes and {len(cyclic_nodes)} Cyclic nodes.")
    
    observers = []
    # Seed in Acyclic zones
    for i in range(NUM_SEEDS_PER_ZONE):
        if len(acyclic_nodes) < 2: break
        seed_nodes = random.sample(acyclic_nodes, 2)
        observers.append({'id': i, 'zone': 'Acyclic', 'nodes': set(seed_nodes), 'history': [2]})
    # Seed in Cyclic zones
    for i in range(NUM_SEEDS_PER_ZONE):
        if len(cyclic_nodes) < 2: break
        seed_nodes = random.sample(cyclic_nodes, 2)
        observers.append({'id': i + NUM_SEEDS_PER_ZONE, 'zone': 'Cyclic', 'nodes': set(seed_nodes), 'history': [2]})

    # 3. Simulate growth using FORWARD-ONLY rule
    print(f"Simulating growth of {len(observers)} observers over {SIMULATION_TICKS} ticks...")
    for tick in range(SIMULATION_TICKS):
        for obs in observers:
            if not obs['nodes']: continue
            potential_targets = {neighbor for node in obs['nodes'] for neighbor in G.successors(node)}
            potential_targets -= obs['nodes']
            if potential_targets:
                target_node = random.choice(list(potential_targets))
                fused_nodes = engine.fuse(G, G.subgraph(obs['nodes']), G.subgraph([target_node]), direction='forward').nodes()
                obs['nodes'] = set(fused_nodes)
            obs['history'].append(len(obs['nodes']))
        if (tick + 1) % 5 == 0:
            print(f"  -> Tick {tick+1}/{SIMULATION_TICKS} complete.")

    # 4. Analyze and Plot the Results
    plot_data = [{'Observer ID': obs['id'], 'Zone': obs['zone'], 'Tick': tick, 'Size': size}
                 for obs in observers for tick, size in enumerate(obs['history'])]
    df = pd.DataFrame(plot_data)
    analyze_and_plot_results(df)
    
    end_time = time.time()
    print(f"\nTotal study duration: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()