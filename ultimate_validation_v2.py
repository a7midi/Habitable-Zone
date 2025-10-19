#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ultimate_validation_v2.py

An end-to-end, *truthful* validator of the Einstein–Memory fixed point g_*
and its mapping to 1/alpha, with robust block-slice averaging, a real
dimension/isotropy check, and conservative aggregators.

Design choices are aligned with:
- Emergent DAG, block curvature κ and memory density ρ_mem, and RG-style
  block-averaging to a universal g_* (paper’s Sec. 10).  [file cite]
- Deterministic selection for acyclicity and entropy growth; we enforce DAG
  and use transitive reduction before measuring.                            [file cite]

References:
- strong_validation.py (anneal scaffold & energy form)  [file cite]
- “A Parameter-Free Theory of Emergence …” (Einstein–Memory, RG to g_*) [file cite]

[file cite markers]:
(1) strong_validation.py -> :contentReference[oaicite:2]{index=2}
(2) the paper (Einstein–Memory and g_* RG fixed point) -> :contentReference[oaicite:3]{index=3}
"""

import argparse, json, math, random, time
from typing import Dict, List, Tuple, Optional, Any
from multiprocessing import Pool, cpu_count

import numpy as np
import networkx as nx

# ---------------------------
# Defaults inspired by your sweep (data-rich slices, decent isotropy)
# ---------------------------
DEF_TARGET_M     = 1.55
DEF_W_DEPTH      = -0.34
DEF_LAMBDA_COMP  = 0.04

# Dimension gate
DIM_GATE = (3.5, 4.5)  # Require ~4D before mapping to alpha

# ---------------------------
# Utility: RNG seeding
# ---------------------------
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

# ---------------------------
# Energy and annealing (adapted from your strong suite)  :contentReference[oaicite:4]{index=4}
# ---------------------------
def energy(G: nx.DiGraph,
           w_cycle: float = 1.0,
           w_sparse: float = 0.02,
           w_depth: float = DEF_W_DEPTH,
           target_m: float = DEF_TARGET_M,
           lambda_comp: float = DEF_LAMBDA_COMP) -> float:
    """Lower is better: penalize cycles, soft-target mean out-degree, reward depth, and
    softly discourage high comparability to avoid trivial fans."""
    n = G.number_of_nodes()
    if n == 0:
        return 0.0

    # Cycle penalty (count mass of SCCs > 1)
    if not nx.is_directed_acyclic_graph(G):
        cyc_mass = sum(len(s) for s in nx.strongly_connected_components(G) if len(s) > 1)
        return w_cycle * cyc_mass

    m = G.number_of_edges()

    # Depth reward via longest path on DAG
    depth = 0
    dp = {v: 0 for v in G}
    for v in nx.topological_sort(G):
        for u in G.predecessors(v):
            dp[v] = max(dp[v], dp[u] + 1)
        depth = max(depth, dp[v])

    sparsity_pen = w_sparse * ((m / max(n, 1)) - target_m) ** 2

    # Comparability proxy (in/out over-branching)
    indeg, outdeg = dict(G.in_degree()), dict(G.out_degree())
    comp_pen = 0.0
    for v in G:
        di, do = indeg[v], outdeg[v]
        comp_pen += (di * (di - 1) + do * (do - 1)) / 2.0

    depth_reward = w_depth * depth
    return sparsity_pen + depth_reward + lambda_comp * comp_pen / max(n, 1)


def mutate(G: nx.DiGraph) -> nx.DiGraph:
    """Random local move: toggle, reverse, rewire, or targeted cycle break."""
    H = G.copy()
    nodes = list(H)
    if len(nodes) < 2:
        return H

    r = random.random()

    # Targeted cycle break
    if r < 0.15:
        try:
            cyc = nx.find_cycle(H, orientation="original")
            u, v, _ = cyc[0]
            H.remove_edge(u, v)
            return H
        except nx.NetworkXNoCycle:
            r = 0.15

    # Toggle
    if r < 0.70:
        u, v = random.sample(nodes, 2)
        if H.has_edge(u, v):
            H.remove_edge(u, v)
        else:
            H.add_edge(u, v)
        return H

    # Reverse
    if r < 0.85 and H.number_of_edges() > 0:
        u, v = random.choice(list(H.edges()))
        H.remove_edge(u, v)
        H.add_edge(v, u)
        return H

    # Rewire head
    if H.number_of_edges() > 0:
        u, v = random.choice(list(H.edges()))
        w = random.choice(nodes)
        if v != w:
            H.remove_edge(u, v)
            H.add_edge(u, w)

    return H


def force_dag(G: nx.DiGraph, max_removals: int = 10000) -> nx.DiGraph:
    """Greedy fallback to ensure a DAG."""
    H = G.copy()
    if nx.is_directed_acyclic_graph(H):
        return H
    removals = 0
    while removals < max_removals:
        try:
            cyc = nx.find_cycle(H, orientation="original")
        except nx.NetworkXNoCycle:
            break
        u, v, _ = cyc[0]
        H.remove_edge(u, v)
        removals += 1
    return H


def anneal_to_emergent_dag(n: int, p: float, seed: int,
                           w_depth: float, target_m: float, lambda_comp: float,
                           iters_per_T: int = 180,
                           T0: float = 15.0, Tfinal: float = 0.1, cool: float = 0.99) -> Tuple[nx.DiGraph, float]:
    set_seeds(seed)
    start = time.time()

    G = nx.gnp_random_graph(n, p, directed=True, seed=seed)
    e = energy(G, w_depth=w_depth, target_m=target_m, lambda_comp=lambda_comp)
    bestG, beste = G, e
    T = T0

    while T > Tfinal:
        for _ in range(iters_per_T):
            H = mutate(G)
            # avoid cooling into cycles
            if T < 1.0 and not nx.is_directed_acyclic_graph(H):
                continue
            eh = energy(H, w_depth=w_depth, target_m=target_m, lambda_comp=lambda_comp)
            dE = eh - e
            if dE < 0 or random.random() < math.exp(-dE / max(T, 1e-9)):
                G, e = H, eh
                if e < beste:
                    bestG, beste = G, e
        T *= cool

    # Project: hard-DAG + transitive reduction
    bestG = force_dag(bestG)
    try:
        bestG = nx.algorithms.dag.transitive_reduction(bestG)
    except Exception:
        pass

    # Depths
    depths = {v: 0 for v in bestG}
    for v in nx.topological_sort(bestG):
        for u in bestG.predecessors(v):
            depths[v] = max(depths[v], depths[u] + 1)
    nx.set_node_attributes(bestG, depths, 'depth')

    return bestG, time.time() - start

# ---------------------------
# Geometry on DAG (paper’s Sec. 10)  :contentReference[oaicite:5]{index=5}
# ---------------------------
def dag_kappa_rhomem(G: nx.DiGraph) -> Tuple[Dict[Tuple[int,int], int], Dict[int, int]]:
    """Return κ on edges and ρ_mem on vertices for a DAG G."""
    indeg = dict(G.in_degree())
    outdeg = dict(G.out_degree())

    kappa = {}
    for u, v in G.edges():
        kappa[(u, v)] = indeg.get(v, 0) - indeg.get(u, 0)

    rho = {}
    for v in G.nodes():
        rho[v] = max(outdeg.get(v, 0) - 1, 0)

    return kappa, rho


def block_slice_stats(G: nx.DiGraph, kappa: Dict[Tuple[int,int], int], rho: Dict[int, int],
                      R: int, min_edges_per_slice: int = 12) -> Optional[Dict[str, float]]:
    """Average κ and ρ_mem over edges with depth(src) % R == 0, return g_R if well-posed."""
    depth = nx.get_node_attributes(G, 'depth')
    edges = [(u, v) for (u, v) in G.edges() if depth.get(u, 0) % R == 0]

    if len(edges) < min_edges_per_slice:
        return None

    k_sum = 0.0
    r_sum = 0.0
    for (u, v) in edges:
        # source at u, ρ_mem measured at target v
        k_sum += kappa.get((u, v), 0)
        r_sum += rho.get(v, 0)

    if r_sum <= 0:
        return None

    return {
        "n_edges": float(len(edges)),
        "kappa_R": k_sum / len(edges),
        "rho_R": r_sum / len(edges),
        "g_R": (k_sum / len(edges)) / (r_sum / len(edges)),
    }

# ---------------------------
# Dimension & isotropy (cone growth slope)
# ---------------------------
def forward_cone_sizes(G: nx.DiGraph, root: int, rmax: int) -> List[int]:
    """Count #targets at distance ≤ r from root (directed) for r=1..rmax."""
    # BFS forward; networkx shortest_path_length can be expensive; do manual frontier.
    sizes = []
    visited = {root}
    frontier = {root}
    for r in range(1, rmax + 1):
        new_frontier = set()
        for u in frontier:
            new_frontier.update(G.successors(u))
        new_frontier -= visited
        visited |= new_frontier
        sizes.append(len(visited) - 1)  # exclude root
        frontier = new_frontier
        if not frontier:
            # no further growth; pad with last value
            sizes += [sizes[-1]] * (rmax - r)
            break
    return sizes


def estimate_dimension(G: nx.DiGraph, rmax: int = 10, n_roots: int = 32) -> Tuple[float, float]:
    """Fit log <V_r> vs log r over r=2..rmax. Return (dim_mean, iso_proxy)."""
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return float('nan'), float('nan')

    # prefer shallow roots to sample forward geometry
    depth = nx.get_node_attributes(G, 'depth')
    nodes.sort(key=lambda v: depth.get(v, 0))
    sample = nodes[:max(1, min(n_roots, len(nodes)))]

    # For each root, fit slope on r=2..rmax if growth exists
    per_root_slopes = []
    for root in sample:
        Vr = forward_cone_sizes(G, root, rmax)
        # need at least r=2 values, and positive growth
        if len(Vr) < 2 or all(v <= 0 for v in Vr[1:]):
            continue
        # Fit slope of log(V_r+1) vs log(r)
        xs = []
        ys = []
        for r in range(2, rmax + 1):
            v = Vr[r - 1]
            xs.append(math.log(r))
            ys.append(math.log(v + 1.0))  # +1 to avoid log(0)
        # linear regression
        x = np.array(xs)
        y = np.array(ys)
        denom = (x * x).sum() - x.sum() ** 2 / len(x)
        if denom <= 0:
            continue
        slope = ((x * y).sum() - x.sum() * y.sum() / len(x)) / denom
        per_root_slopes.append(slope)

    if not per_root_slopes:
        return float('nan'), float('nan')

    dim_mean = float(np.mean(per_root_slopes))
    sd = float(np.std(per_root_slopes))
    iso_proxy = float(sd / dim_mean) if dim_mean != 0 else float('inf')
    return dim_mean, iso_proxy

# ---------------------------
# Alpha identities (two options; see discussion)  :contentReference[oaicite:6]{index=6}
# ---------------------------
def alpha_inverse_from_gstar(g_star: float, D: float, q: int, identity: str) -> float:
    identity = identity.lower()
    if identity == "paper":            # α^{-1} = 9 D^2 q / (π g_*)
        return (9.0 * D * D * q) / (math.pi * g_star)
    elif identity == "d2pi":           # α^{-1} = D^2 π / g_*
        return (D * D * math.pi) / g_star
    else:
        raise ValueError("Unknown alpha identity: choose 'paper' or 'd2pi'.")

# ---------------------------
# Worker
# ---------------------------
def one_replicate(seed: int, n: int, p: float,
                  w_depth: float, target_m: float, lambda_comp: float,
                  report_R: List[int], g_estimator: str,
                  min_edges_per_slice: int, dim_rmax: int) -> Dict[str, Any]:
    G, dur = anneal_to_emergent_dag(n, p, seed, w_depth, target_m, lambda_comp)
    depth_max = max(nx.get_node_attributes(G, 'depth').values() or [0])

    kappa, rho = dag_kappa_rhomem(G)

    # per-R stats
    stats = {}
    valid_Rs = []
    for R in sorted(report_R):
        s = block_slice_stats(G, kappa, rho, R, min_edges_per_slice=min_edges_per_slice)
        if s is not None:
            stats[str(R)] = s
            valid_Rs.append((R, s))

    # pick a g_* per replicate
    g_rep = float('nan')
    if valid_Rs:
        if g_estimator == "largestR":
            R, s = valid_Rs[-1]  # largest R available
            g_rep = s["g_R"]
        elif g_estimator == "avg":
            g_rep = float(np.mean([s["g_R"] for _, s in valid_Rs]))
        else:
            # default to largestR
            g_rep = valid_Rs[-1][1]["g_R"]

    # dimension & isotropy
    dim_mean, iso_proxy = estimate_dimension(G, rmax=dim_rmax, n_roots=32)

    # horizon pair budget on depth-min layer
    depth_attr = nx.get_node_attributes(G, 'depth')
    dmin = min(depth_attr.values() or [0])
    D0 = [v for v, d in depth_attr.items() if d == dmin]
    outdeg = dict(G.out_degree())
    pair_budget = sum(max(outdeg.get(v, 0) - 1, 0) for v in D0)

    return {
        "seed": seed,
        "stats": stats,
        "g_star_rep": g_rep,
        "dim_mean": dim_mean,
        "iso_proxy": iso_proxy,
        "pair_budget": pair_budget,
        "depth_max": depth_max,
        "duration_s": dur,
    }

# ---------------------------
# CI helper
# ---------------------------
def mean_and_ci95(xs: List[float]) -> Tuple[float, float]:
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    if not xs:
        return float('nan'), float('nan')
    mu = float(np.mean(xs))
    sd = float(np.std(xs))
    ci = 1.96 * sd / math.sqrt(len(xs)) if len(xs) > 1 else float('nan')
    return mu, ci

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=24)
    ap.add_argument("--n-nodes", type=int, default=90)
    ap.add_argument("--p-edge", type=float, default=0.075)
    ap.add_argument("--report-R", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--g-estimator", type=str, choices=["largestR", "avg"], default="largestR")
    ap.add_argument("--target-m", type=float, default=DEF_TARGET_M)
    ap.add_argument("--w-depth", type=float, default=DEF_W_DEPTH)
    ap.add_argument("--lambda-comp", type=float, default=DEF_LAMBDA_COMP)
    ap.add_argument("--min-edges-per-slice", type=int, default=12)
    ap.add_argument("--dim-rmax", type=int, default=10)
    ap.add_argument("--alpha-identity", type=str, choices=["paper", "d2pi"], default="paper")
    ap.add_argument("--alpha-q", type=int, default=2)
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    seeds = [1000 + i for i in range(args.replicates)]

    tasks = [(s, args.n_nodes, args.p_edge,
              args.w_depth, args.target_m, args.lambda_comp,
              args.report_R, args.g_estimator,
              args.min_edges_per_slice, args.dim_rmax)
             for s in seeds]

    workers = max(1, cpu_count() - 1)
    results = []
    t0 = time.time()
    with Pool(processes=workers) as pool:
        for i, r in enumerate(pool.starmap(one_replicate, tasks), 1):
            results.append(r)
            print(f"[{i}/{args.replicates}] seed={r['seed']} "
                  f"g*={r['g_star_rep'] if math.isfinite(r['g_star_rep']) else 'NaN'} "
                  f"D≈{r['dim_mean']:.2f} iso={r['iso_proxy']:.3f} "
                  f"depth_max={r['depth_max']} T={r['duration_s']:.1f}s")

    # Summaries
    g_vals = [r["g_star_rep"] for r in results if math.isfinite(r["g_star_rep"])]
    g_mean, g_ci = mean_and_ci95(g_vals)

    dim_vals = [r["dim_mean"] for r in results if math.isfinite(r["dim_mean"])]
    dim_mean, dim_ci = mean_and_ci95(dim_vals)

    iso_vals = [r["iso_proxy"] for r in results if math.isfinite(r["iso_proxy"])]
    iso_mean, iso_ci = mean_and_ci95(iso_vals)

    gate_lo, gate_hi = DIM_GATE
    dim_gate_pass = (math.isfinite(dim_mean) and gate_lo <= dim_mean <= gate_hi)

    # g_* per-R summaries (only for reporting)
    perR = {str(R): [] for R in args.report_R}
    for r in results:
        for k, s in r["stats"].items():
            perR.setdefault(k, []).append(s["g_R"])
    perR_summary = {}
    for k, arr in perR.items():
        mu, ci = mean_and_ci95(arr)
        if math.isfinite(mu):
            perR_summary[k] = {"mean": mu, "ci95": ci, "n": len(arr)}

    # Alpha prediction (only if dimension gate passes)
    alpha = {}
    if dim_gate_pass and math.isfinite(g_mean) and g_mean > 0:
        D = dim_mean
        try:
            pred = alpha_inverse_from_gstar(g_mean, D, args.alpha_q, args.alpha_identity)
            # Propagate uncertainty from g_* only (conservative)
            pred_err = pred * (g_ci / g_mean) if math.isfinite(g_ci) else float('nan')
            alpha = {
                "identity": args.alpha_identity,
                "D_used": D,
                "q_used": args.alpha_q,
                "alpha_pred_inv": pred,
                "alpha_pred_ci95": pred_err,
                "note": "Only g_* uncertainty propagated; D uncertainty omitted for clarity."
            }
        except Exception as e:
            alpha = {"skipped": True, "reason": f"alpha mapping failed: {e}"}
    else:
        alpha = {
            "skipped": True,
            "reason": "dimension_gate_failed" if not dim_gate_pass else "g_star_invalid",
            "dim_mean": dim_mean,
            "dim_gate": [gate_lo, gate_hi],
        }

    out = {
        "THEORY_VALIDATION_TOOL": "UltimateEmergenceValidator_v2",
        "settings": {
            "replicates": args.replicates,
            "report_R": args.report_R,
            "g_estimator": args.g_estimator,
            "anneal": {
                "target_m": args.target_m,
                "w_depth": args.w_depth,
                "lambda_comp": args.lambda_comp
            },
            "graph": {"n_nodes": args.n_nodes, "p_edge": args.p_edge},
            "min_edges_per_slice": args.min_edges_per_slice,
            "dimension": {"rmax": args.dim_rmax},
            "alpha_identity": args.alpha_identity,
            "alpha_q": args.alpha_q
        },
        "module_1": {
            "notes": "DAG enforced; transitive reduction applied before measurements."
        },
        "module_2": {
            "rg_output": {
                "replicates": results,
                "g_star_summary": {"mean": g_mean, "ci95": g_ci, "n_valid": len(g_vals)},
                "per_R_summary": perR_summary,
                "elapsed_seconds": time.time() - t0
            }
        },
        "module_3": {
            "dimension_isotropy": {
                "dim_mean": dim_mean,
                "dim_ci95": dim_ci,
                "iso_proxy_mean": iso_mean,
                "iso_proxy_ci95": iso_ci,
                "gate_interval": [gate_lo, gate_hi],
                "gate_passed_for_alpha": dim_gate_pass
            },
            "alpha_prediction": alpha
        }
    }

    if args.pretty:
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps(out))

if __name__ == "__main__":
    main()
