#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine_structure_prediction.py

A complete, runnable implementation of the amended plan to predict the
fine‑structure constant α from an emergent, parameter‑free graph model.

Design choices map directly onto the paper’s definitions:
  • Substrate: finite directed multigraph; we *work on its condensation DAG*.
  • Curvature: κ(i→j) := |Pred(j)| − |Pred(i)| (edge indegree difference).
  • Memory density: ρ_mem(v) ≈ deg⁺(v) − 1 (minimal layer exact; block‑averaged elsewhere).
  • Block averaging: slice edges at depths that are multiples of R and average;
    take g_R := κ_R / (ρ_mem)_R, then estimate the RG fixed point g_* by
    tail‑averaging g_R over large R (Theorem 10.13).
  • Microscopic rule: adopt the simplest order‑independent binary fusion
    (XOR/OR‑like) → single‑coordinate degeneracy σ(d)=1 ⇒ κ₀=1, hence c₀=1 bit/tick
    (Theorem 8.4 and Corollary 8.5).
  • Final mapping to α: test a minimal identity and a small integer family:
      α⁻¹ ≈ π·D / (c₀·g_*)
      and   α⁻¹ ≈ A·π^a·D^b·q^c·c₀^e·g_*^(−d)  with tiny integer exponents.
    This is *hypothesis generation*, not a proof.

Two backends are provided:
  1) "pure"  — no third‑party deps. Builds a layered DAG with bounded out‑degree.
  2) "nx"    — uses networkx (if available) to sample a random digraph, project to a DAG
               (by removing/back‑edge filtering), compute depths, then g_R.

References to the attached paper:
  • Edge curvature and Einstein–Memory fixed‑ratio g_* (Defs. 10.5–10.12; Thm. 10.13).
  • Memory density and minimal‑layer normalization (Defs. 3.4, 10.7).
  • Deterministic Second Law and c₀ from σ, κ₀ (Sec. 8; Thm. 8.4, Cor. 8.5).
  • 4D Lorentzian limit (Thm. 10.19).

CLI quickstart (fast, pure backend):
  python fine_structure_prediction.py --pretty --replicates 3 --backend pure --use-kappa-positive

Typical full run (still fast):
  python fine_structure_prediction.py --replicates 10 --layers 64 --layer-size 128 \
      --layer-schedule linear --rmax 4 --use-kappa-positive --pretty

If networkx is installed, you may try:
  python fine_structure_prediction.py --backend nx --replicates 10 --nx-nodes 120 --nx-p 0.075 --pretty

Outputs a JSON summary with: settings, per‑R g_R rows, g_*, α prediction and error.
"""

from __future__ import annotations
import argparse, json, math, random, statistics, sys, time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional

ALPHA_INV_EMPIRICAL = 137.035999084  # CODATA-like reference for comparison (dimensionless)

# =========================================================
# Common utilities
# =========================================================

def tail_average(values: List[float], k: int = 3) -> float:
    vals = [v for v in values if v == v and math.isfinite(v)]
    if not vals:
        return float("nan")
    tail = vals[-k:] if len(vals) >= k else vals
    return statistics.fmean(tail)

def confint_mean(values: List[float], z: float = 1.96) -> Tuple[float, float]:
    vals = [v for v in values if v == v and math.isfinite(v)]
    if not vals:
        return (float("nan"), float("nan"))
    m = statistics.fmean(vals)
    if len(vals) <= 1:
        return (m, float("nan"))
    s = statistics.pstdev(vals) if len(vals) <= 1 else statistics.stdev(vals)
    ci = z * (s / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")
    return (m, ci)

# =========================================================
# PURE backend (dependency‑free): layered DAG with fixed out‑fan bound
# =========================================================

@dataclass
class Node:
    id: int
    layer: int
    preds: List[int] = field(default_factory=list)
    succs: List[int] = field(default_factory=list)

@dataclass
class LayeredDAG:
    nodes: List[Node]
    layers: List[List[int]]  # node ids per layer
    rmax: int

    def indeg(self, v: int) -> int:
        return len(self.nodes[v].preds)

    def outdeg(self, v: int) -> int:
        return len(self.nodes[v].succs)

    def edges(self) -> Iterable[Tuple[int, int]]:
        for v in range(len(self.nodes)):
            for w in self.nodes[v].succs:
                yield (v, w)

    def depth(self, v: int) -> int:
        return self.nodes[v].layer

    @property
    def max_depth(self) -> int:
        return len(self.layers) - 1

def make_layer_sizes(L: int, base: int, schedule: str) -> List[int]:
    sizes = []
    for ell in range(L):
        if schedule == "constant":
            sizes.append(max(1, base))
        elif schedule == "linear":
            sizes.append(max(1, base + ell))
        elif schedule == "quadratic":
            sizes.append(max(1, base + (ell * (ell + 1)) // 2))
        else:
            raise ValueError(f"unknown schedule {schedule}")
    return sizes

def build_layered_dag(L: int, layer_sizes: List[int], rmax: int, seed: int = 0,
                      allow_parallel: bool = True) -> LayeredDAG:
    rng = random.Random(seed)
    nodes: List[Node] = []
    layers: List[List[int]] = []
    vid = 0
    for ell in range(L):
        ids = []
        for _ in range(layer_sizes[ell]):
            nodes.append(Node(id=vid, layer=ell))
            ids.append(vid)
            vid += 1
        layers.append(ids)

    for ell in range(L - 1):
        this_ids = layers[ell]
        next_ids = layers[ell + 1]
        for v in this_ids:
            deg = rmax if rmax > 0 else 0
            if not next_ids or deg == 0:
                continue
            if allow_parallel:
                choices = [rng.choice(next_ids) for _ in range(deg)]
            else:
                k = min(deg, len(next_ids))
                choices = rng.sample(next_ids, k=k)
            for w in choices:
                nodes[v].succs.append(w)
                nodes[w].preds.append(v)
    return LayeredDAG(nodes=nodes, layers=layers, rmax=rmax)

def kappa_on_edge_pure(G: LayeredDAG, e: Tuple[int, int]) -> int:
    i, j = e
    return G.indeg(j) - G.indeg(i)  # κ(i→j)

def rho_mem_site_pure(G: LayeredDAG, v: int) -> int:
    return max(0, G.outdeg(v) - 1)

def block_slice_edges_pure(G: LayeredDAG, R: int) -> List[Tuple[int, int]]:
    return [(i, j) for (i, j) in G.edges() if G.depth(i) % R == 0]

def g_R_pure(G: LayeredDAG, R: int, use_kappa_positive: bool) -> Tuple[float, float, float, int]:
    E = block_slice_edges_pure(G, R)
    if not E:
        return (float("nan"), float("nan"), float("nan"), 0)
    kappas, rhos = [], []
    for (i, j) in E:
        k = kappa_on_edge_pure(G, (i, j))
        if use_kappa_positive:
            k = max(0, k)
        r = rho_mem_site_pure(G, j)
        if r > 0:
            kappas.append(k)
            rhos.append(r)
    if not rhos:
        return (float("nan"), float("nan"), float("nan"), 0)
    kR = statistics.fmean(kappas)
    rR = statistics.fmean(rhos)
    return (kR, rR, (kR / rR) if rR != 0 else float("nan"), len(rhos))

def estimate_g_star_pure(L: int, layer_size: int, schedule: str, rmax: int,
                         seed: int, report_R: List[int], use_kappa_positive: bool) -> Dict[str, object]:
    sizes = make_layer_sizes(L, layer_size, schedule)
    G = build_layered_dag(L, sizes, rmax, seed=seed, allow_parallel=True)
    rows = []
    for R in report_R:
        kR, rR, gR, n = g_R_pure(G, R, use_kappa_positive=use_kappa_positive)
        rows.append({"R": R, "kappa_R": kR, "rho_R": rR, "g_R": gR, "n_edges": n})
    vals = [row["g_R"] for row in rows if row["n_edges"] > 0 and math.isfinite(row["g_R"])]
    g_star = tail_average(vals, k=3) if vals else float("nan")
    return {"rows": rows, "g_star": g_star}

# =========================================================
# Optional NetworkX backend (if available)
# =========================================================

def nx_backend_available() -> bool:
    try:
        import networkx as nx  # noqa
        return True
    except Exception:
        return False

def estimate_g_star_nx(n_nodes: int, p_edge: float, seed: int,
                       report_R: List[int], use_kappa_positive: bool) -> Dict[str, object]:
    """
    Simple NX formulation: sample a random digraph G(n,p) (oriented),
    strip backwards edges to obtain a DAG consistent with depth (greedy),
    compute depths, indegrees/outdegrees, and then g_R as in the pure backend.
    """
    import random as _random
    import networkx as nx  # type: ignore

    rng = _random.Random(seed)

    # Sample directed Erdos-Renyi and remove self-loops
    G0 = nx.DiGraph()
    G0.add_nodes_from(range(n_nodes))
    for u in range(n_nodes):
        for v in range(n_nodes):
            if u == v:
                continue
            if rng.random() < p_edge:
                G0.add_edge(u, v)

    # Greedy depth assignment by repeated removal of sources (Kahn)
    try:
        order = list(nx.topological_sort(G0))
        # Already a DAG
        H = G0.copy()
    except nx.NetworkXUnfeasible:
        # Break cycles: greedily remove one back edge from each cycle found
        H = G0.copy()
        removed = 0
        while True:
            try:
                cyc = nx.find_cycle(H, orientation="original")
                u, v, _ = cyc[0]
                if H.has_edge(u, v):
                    H.remove_edge(u, v)
                    removed += 1
                if removed > 5 * n_nodes:  # safety
                    break
            except nx.NetworkXNoCycle:
                break

    # Now compute depths (longest path from sources)
    depth = {v: 0 for v in H.nodes()}
    try:
        topo = list(nx.topological_sort(H))
    except nx.NetworkXUnfeasible:
        topo = list(H.nodes())  # fallback
    for v in topo:
        for u in H.predecessors(v):
            depth[v] = max(depth[v], depth[u] + 1)

    indeg = {v: H.in_degree(v) for v in H.nodes()}
    outdeg = {v: H.out_degree(v) for v in H.nodes()}

    def kappa(u, v):  # edge curvature
        return indeg[v] - indeg[u]

    def rho(v):       # memory density proxy
        return max(0, outdeg[v] - 1)

    rows = []
    for R in report_R:
        kappas, rhos = [], []
        for (u, v) in H.edges():
            if depth.get(u, 0) % R == 0:
                kv = kappa(u, v)
                if use_kappa_positive:
                    kv = max(0, kv)
                rv = rho(v)
                if rv > 0:
                    kappas.append(kv)
                    rhos.append(rv)
        if rhos:
            kR = statistics.fmean(kappas)
            rR = statistics.fmean(rhos)
            gR = kR / rR if rR != 0 else float("nan")
            rows.append({"R": R, "kappa_R": kR, "rho_R": rR, "g_R": gR, "n_edges": len(rhos)})
        else:
            rows.append({"R": R, "kappa_R": float("nan"), "rho_R": float("nan"),
                         "g_R": float("nan"), "n_edges": 0})
    vals = [row["g_R"] for row in rows if row["n_edges"] > 0 and math.isfinite(row["g_R"])]
    g_star = tail_average(vals, k=3) if vals else float("nan")
    return {"rows": rows, "g_star": g_star}

# =========================================================
# Degeneracy constants for the XOR‑style, binary rule
# =========================================================

def degeneracy_constants() -> Dict[str, int]:
    # For XOR over binary tags: σ(d)=1 ⇒ κ₀ = 1; with |A_h|min = 2 ⇒ c₀ = 1
    kappa0 = 1
    c0 = 1
    return {"kappa0": kappa0, "c0": c0}

# =========================================================
# Mapping g_* → α (minimal identity + tiny integer search)

# =========================================================

def predict_alpha_from_gstar(
    g_star: float,
    D: int = 4,
    q: int = 2,
    c0: int = 1,
    do_integer_search: bool = True,
    tol: float = 1e-4,
) -> Dict[str, object]:
    """
    Returns a dict with a default minimal identity and (optionally)
    a small integer‑family search for a slightly better match.
    """
    if not (g_star and g_star > 0 and math.isfinite(g_star)):
        default_alpha_inv = float("nan")
    else:
        default_alpha_inv = (math.pi * D) / (c0 * g_star)

    default = {
        "form": "alpha_inv = π·D / (c0·g*)",
        "A": 1, "a": 1, "b": 1, "c": 0, "e": 1, "d": 1,
        "alpha_inv": default_alpha_inv,
        "alpha": (1.0 / default_alpha_inv) if (default_alpha_inv and default_alpha_inv != 0) else float("nan"),
        "abs_err": abs(default_alpha_inv - ALPHA_INV_EMPIRICAL) if math.isfinite(default_alpha_inv) else float("inf"),
        "note": "Minimal identity per Phase 4 hypothesis."
    }

    if not do_integer_search or not (g_star and g_star > 0 and math.isfinite(g_star)):
        return {"best": default, "candidates": [default]}

    # Tiny integer search as hypothesis generator
    best = default
    candidates = [default]
    pi = math.pi
    exp_range = range(-3, 4)
    for A in range(1, 17):
        for a in exp_range:
            for b in exp_range:
                for c in exp_range:
                    for e in exp_range:
                        for d in range(0, 4):  # exponent on g*
                            try:
                                val = A * (pi ** a) * (D ** b) * (q ** c) * (c0 ** e) * (g_star ** (-d if d != 0 else 0))
                                if not math.isfinite(val) or val <= 0:
                                    continue
                                err = abs(val - ALPHA_INV_EMPIRICAL)
                                complexity = (abs(a) + abs(b) + abs(c) + abs(e) + abs(d) + (0 if A == 1 else 1))
                                cand = {
                                    "form": "alpha_inv = A·π^a·D^b·q^c·c0^e·g*^{-d}",
                                    "A": A, "a": a, "b": b, "c": c, "e": e, "d": d,
                                    "alpha_inv": val,
                                    "alpha": 1.0 / val,
                                    "abs_err": err,
                                    "complexity": complexity
                                }
                                candidates.append(cand)
                                better = (err < best["abs_err"] - tol) or (abs(err - best["abs_err"]) <= tol and complexity < best.get("complexity", 999))
                                if better:
                                    best = cand
                            except OverflowError:
                                continue

    # Also include an alternative single‑line identity the user sketched:
    # α⁻¹ ≈ (D²·π)/g*
    if (g_star and g_star > 0 and math.isfinite(g_star)):
        alt = (D * D * math.pi) / g_star
        alt_cand = {
            "form": "alpha_inv = D^2 · π / g*",
            "A": 1, "a": 1, "b": 2, "c": 0, "e": 0, "d": 1,
            "alpha_inv": alt,
            "alpha": 1.0 / alt,
            "abs_err": abs(alt - ALPHA_INV_EMPIRICAL),
            "complexity": 3
        }
        candidates.append(alt_cand)
        if alt_cand["abs_err"] + tol < best["abs_err"]:
            best = alt_cand

    # Keep the top few by absolute error
    top = sorted(candidates, key=lambda x: (x["abs_err"], x.get("complexity", 999)))[:10]
    return {"best": best, "candidates": top}

# =========================================================
# Experiment runner (replicates) and CLI
# =========================================================

def run_once_backend_pure(args, seed: int) -> Dict[str, object]:
    R_values = args.report_R or [1, 2, 4, 8, 16, 32]
    est = estimate_g_star_pure(
        L=args.layers,
        layer_size=args.layer_size,
        schedule=args.layer_schedule,
        rmax=args.rmax,
        seed=seed,
        report_R=R_values,
        use_kappa_positive=args.use_kappa_positive,
    )
    return est

def run_once_backend_nx(args, seed: int) -> Dict[str, object]:
    R_values = args.report_R or [1, 2, 4, 8, 16, 32]
    est = estimate_g_star_nx(
        n_nodes=args.nx_nodes,
        p_edge=args.nx_p,
        seed=seed,
        report_R=R_values,
        use_kappa_positive=args.use_kappa_positive,
    )
    return est

def main():
    parser = argparse.ArgumentParser(description="Predict α from emergent DAGs via RG fixed ratio g* and low‑complexity identities.")
    # Common knobs
    parser.add_argument("--backend", type=str, default="pure", choices=["pure", "nx"], help="graph backend: 'pure' (no deps) or 'nx' (networkx if installed)")
    parser.add_argument("--replicates", type=int, default=6, help="number of independent replicates (different seeds)")
    parser.add_argument("--seed", type=int, default=7, help="base seed (replicate i uses seed+ i)")
    parser.add_argument("--use-kappa-positive", action="store_true", help="use κ⁺ = max(0, κ) to avoid sign cancellations in g_R")
    parser.add_argument("--report-R", type=int, nargs="*", default=None, help="R block sizes to report, e.g. 2 4 8 16 32")
    parser.add_argument("--pretty", action="store_true", help="pretty‑print JSON")
    # Pure backend controls
    parser.add_argument("--layers", type=int, default=64, help="number of layers (depth) for pure backend")
    parser.add_argument("--layer-size", type=int, default=128, help="base layer size for pure backend (schedule shapes deeper layers)")
    parser.add_argument("--layer-schedule", type=str, default="linear", choices=["constant", "linear", "quadratic"], help="layer size schedule for pure backend")
    parser.add_argument("--rmax", type=int, default=4, help="max out‑degree per node in pure backend")
    # NX backend controls
    parser.add_argument("--nx-nodes", type=int, default=120, help="number of nodes for networkx random digraph")
    parser.add_argument("--nx-p", type=float, default=0.075, help="edge probability for networkx random digraph")
    # Mapping and search
    parser.add_argument("--no-integer-search", action="store_true", help="disable tiny integer family search; only print default identity result")

    args = parser.parse_args()

    if args.backend == "nx" and not nx_backend_available():
        print("Warning: networkx not available; falling back to 'pure' backend.", file=sys.stderr)
        args.backend = "pure"

    # Compute per‑replicate g_* estimates
    replicate_results: List[Dict[str, object]] = []
    gstars: List[float] = []
    start = time.time()
    for k in range(args.replicates):
        this_seed = args.seed + k
        if args.backend == "pure":
            est = run_once_backend_pure(args, seed=this_seed)
        else:
            est = run_once_backend_nx(args, seed=this_seed)
        replicate_results.append({"seed": this_seed, **est})
        gstars.append(est["g_star"])

    mean_g_star, ci_g_star = confint_mean(gstars, z=1.96)

    # Degeneracy constants and mapping to α
    deg = degeneracy_constants()
    pred = predict_alpha_from_gstar(
        mean_g_star,
        D=4, q=2, c0=deg["c0"],
        do_integer_search=not args.no_integer_search,
    )

    # Compose output
    out = {
        "settings": {
            "backend": args.backend,
            "replicates": args.replicates,
            "use_kappa_positive": args.use_kappa_positive,
            "report_R": args.report_R or [1,2,4,8,16,32],
            "pure": {
                "layers": args.layers, "layer_size": args.layer_size,
                "layer_schedule": args.layer_schedule, "rmax": args.rmax
            },
            "nx": {"nx_nodes": args.nx_nodes, "nx_p": args.nx_p},
        },
        "degeneracy_constants": deg,
        "replicates": replicate_results,
        "g_star_summary": {
            "values": gstars,
            "mean": mean_g_star,
            "ci95": ci_g_star
        },
        "alpha_empirical_inv": ALPHA_INV_EMPIRICAL,
        "alpha_prediction": pred,
        "elapsed_seconds": round(time.time() - start, 3),
        "notes": [
            "κ(i→j) = |Pred(j)| − |Pred(i)|;  ρ_mem ≈ deg⁺ − 1.",
            "Block‑slice average over edges with depth(i) ≡ 0 (mod R); g_R := κ_R/(ρ_mem)_R.",
            "Tail‑average of g_R over the largest R values ≈ g_* (Theorem 10.13).",
            "Binary XOR‑style rule: σ(d)=1 ⇒ κ₀=1; with |A_h|min=2 ⇒ c₀=1 (Sec. 8).",
            "D=4 from the continuum Lorentz limit (Theorem 10.19).",
            "Integer‑search identity is a *hypothesis generator*; not a proof."
        ]
    }

    if args.pretty:
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps(out))

if __name__ == "__main__":
    main()
