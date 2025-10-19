#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict the fine-structure constant α from the emergent, parameter-free
graph model described in the attached paper:

  A. Alayar, "A Parameter-Free Theory of Emergence: Unifying Spacetime,
  Quantum Statistics, and the Arrow of Time from Deterministic Relations"
  (see sections cited in the README at the bottom of this script).

This script implements a *computable* surrogate of the plan provided by the user:

Phase 1 (Model & Rule):
  - Build an acyclic layered directed multigraph (condensation DAG = graph).
  - Binary alphabet (q_min = 2). Local deterministic update is the XOR rule
    over predecessor tags (order-independent, non-injective globally but
    single-coordinate degeneracy σ(d)=1 so c0 = 1 bit/tick).

Phase 2 (Selection/Stability constants):
  - Compute κ0 := max_d σ(d) for the XOR rule (κ0 = 1).
  - Compute the per-tick entropy quantum c0 = ceil(log2(|Ah|/κ0)) on the
    minimal hidden layer (with |Ah| = 2 ⇒ c0 = 1).

Phase 3 (Block RG and g*):
  - On the condensation DAG, define edge curvature κ(i→j) = |Pred(j)| - |Pred(i)|.
  - Define the (coarse) memory density ρ_mem(v) ≈ deg⁺(v) - 1.
  - For increasing block sizes R, average κ and ρ_mem over slice-edges
    with depth(i) ≡ 0 mod R to obtain g_R := κ_R / ρ_R. Report a tail average
    as g* (and a version using κ⁺ to avoid sign cancellations).

Phase 4 (Mapping to α):
  - Provide a default minimal identity: α^{-1} ≈ π D / (c0 g*), with D=4.
  - Also search (tiny) integer-combinatorial identities of the form
       α^{-1} ≈ A · π^a · D^b · q^c · c0^e · g*^{-d}
    for small integers (A ∈ 1..16, exponents in [-3,3]) and pick the closest
    if it improves over the default. This is a *hypothesis generator*, not a proof.

The script prints a JSON summary including g_R, g*, the chosen identity,
and α_pred with error against α_empirical ≈ 1/137.035999084.

USAGE (CLI):
  python predict_alpha_from_emergence.py \
      --layers 64 --layer-schedule linear --layer-size 128 \
      --rmax 4 --seed 7 --use-kappa-positive

The defaults are sensible and fast.

This code uses only the Python standard library; no third-party deps.
"""

from __future__ import annotations
import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional

# -----------------------------
# Graph construction utilities
# -----------------------------

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
        # We count *edges* as defined; it's a multigraph, so parallel edges count.
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


def build_layered_dag(
    L: int,
    layer_sizes: List[int],
    rmax: int,
    allow_parallel: bool = True,
    seed: int = 0
) -> LayeredDAG:
    """
    Build an acyclic layered directed multigraph:
      - layers 0..L-1 with given sizes
      - each node picks deg in [1, rmax] random successors in next layer
      - allows parallel edges when allow_parallel=True (multigraph semantics)
    """
    assert len(layer_sizes) == L, "layer_sizes must have length = L"
    rng = random.Random(seed)

    # Create nodes
    nodes: List[Node] = []
    layers: List[List[int]] = []
    id_counter = 0
    for ell in range(L):
        layer_ids = []
        for _ in range(layer_sizes[ell]):
            nodes.append(Node(id=id_counter, layer=ell))
            layer_ids.append(id_counter)
            id_counter += 1
        layers.append(layer_ids)

    # Wire edges forward
    for ell in range(L - 1):
        this_ids = layers[ell]
        next_ids = layers[ell + 1]
        for v in this_ids:
            deg = rmax if rmax > 0 else 0
            if deg == 0:
                continue
            # Guarantee at least 1 out-edge, at most rmax
            # Use exactly rmax for stability of ρ_mem unless next layer has < rmax nodes
            if len(next_ids) == 0:
                continue
            # Choose deg successors with replacement if allow_parallel else without
            if allow_parallel:
                choices = [rng.choice(next_ids) for _ in range(deg)]
            else:
                # Without replacement (clip if layer is small)
                deg_eff = min(deg, len(next_ids))
                choices = rng.sample(next_ids, k=deg_eff)
            for w in choices:
                nodes[v].succs.append(w)
                nodes[w].preds.append(v)

    return LayeredDAG(nodes=nodes, layers=layers, rmax=rmax)


def make_layer_sizes(L: int, base: int, schedule: str) -> List[int]:
    """
    Simple layer-size schedules:
      - 'constant': base
      - 'linear'  : base + ell
      - 'quadratic': base + ell*(ell+1)//2
    These schedules influence indegree growth and thus κ statistics.
    """
    sizes = []
    for ell in range(L):
        if schedule == "constant":
            sizes.append(base)
        elif schedule == "linear":
            sizes.append(max(1, base + ell))
        elif schedule == "quadratic":
            sizes.append(max(1, base + (ell * (ell + 1)) // 2))
        else:
            raise ValueError(f"unknown schedule {schedule}")
    return sizes


# -----------------------------
# Emergent observables: κ, ρ, g_R
# -----------------------------

def kappa_on_edge(G: LayeredDAG, e: Tuple[int, int]) -> int:
    i, j = e
    return G.indeg(j) - G.indeg(i)  # κ(i→j) = |Pred(j)| - |Pred(i)|


def rho_mem_site(G: LayeredDAG, v: int) -> int:
    # Coarse τ=1 approximation: ρ_mem(v) ≈ deg⁺(v) - 1 on all layers.
    # (Exact def: minimal layer with block averaging elsewhere; for our stationary,
    # bounded-outdegree model, this coarse value is stable.)
    return max(0, G.outdeg(v) - 1)


def block_slice_edges(G: LayeredDAG, R: int) -> List[Tuple[int, int]]:
    return [(i, j) for (i, j) in G.edges() if G.depth(i) % R == 0]


def g_R(G: LayeredDAG, R: int, use_kappa_positive: bool = False) -> Tuple[float, float, float, int]:
    """
    Return (kappa_R, rho_R, g_R, count_edges) for the depth-R slice.
    """
    E = block_slice_edges(G, R)
    if not E:
        return (float("nan"), float("nan"), float("nan"), 0)

    kappas = []
    rhos = []
    for (i, j) in E:
        k = kappa_on_edge(G, (i, j))
        if use_kappa_positive:
            k = max(0, k)
        kappas.append(k)
        rhos.append(rho_mem_site(G, j))

    # Avoid zero-division: drop zero-ρ edges
    valid = [(k, r) for (k, r) in zip(kappas, rhos) if r > 0]
    if not valid:
        return (float("nan"), float("nan"), float("nan"), len(E))

    kappa_R = statistics.fmean(k for (k, _) in valid)
    rho_R = statistics.fmean(r for (_, r) in valid)
    gR = kappa_R / rho_R if rho_R != 0 else float("nan")
    return (kappa_R, rho_R, gR, len(valid))


def estimate_g_star(G: LayeredDAG, R_values: List[int], use_kappa_positive: bool=False) -> Dict[str, object]:
    rows = []
    for R in R_values:
        kR, rR, gR, n = g_R(G, R, use_kappa_positive=use_kappa_positive)
        rows.append({"R": R, "kappa_R": kR, "rho_R": rR, "g_R": gR, "n_edges": n})

    # Tail average over the last ~3 valid entries
    valids = [row for row in rows if row["n_edges"] > 0 and math.isfinite(row["g_R"])]
    tail = valids[-3:] if len(valids) >= 3 else valids
    if tail:
        g_star = statistics.fmean(row["g_R"] for row in tail)
    else:
        g_star = float("nan")

    return {
        "rows": rows,
        "g_star": g_star,
        "use_kappa_positive": use_kappa_positive
    }


# -----------------------------
# Degeneracy σ, κ0 and entropy quantum c0 for XOR
# -----------------------------

def sigma_single_coordinate_for_xor(num_preds: int) -> int:
    """
    For XOR across num_preds inputs over binary alphabet, fixing other coordinates:
      - output toggles when we flip the distinguished coordinate
      - For any desired y and fixed others, exactly one value of that coordinate produces y
      ⇒ fibre size = 1
    """
    if num_preds <= 0:
        return 1  # by convention (no preds), but we won't use it
    return 1


def kappa0_for_xor(G: LayeredDAG) -> int:
    # Maximum σ(d) over sites d reachable by at least one edge
    # For XOR, σ(d)=1 whenever Pred(d) non-empty.
    reachable = set(j for (_, j) in G.edges())
    if not reachable:
        return 1
    return 1


def c0_for_binary_alphabet(ah_size: int, kappa0: int) -> int:
    # c0 = ceil(log2(|Ah|/κ0)) on minimal hidden layer (use |Ah|=2 here).
    ratio = max(1.0e-12, ah_size / float(kappa0))
    return int(math.ceil(math.log(ratio, 2)))


# -----------------------------
# Mapping g* to α (hypothesis search)
# -----------------------------

ALPHA_INV_EMPIRICAL = 137.035999084  # CODATA-like

def predict_alpha_from_gstar(
    g_star: float,
    D: int = 4,
    q: int = 2,
    c0: int = 1,
    do_integer_search: bool = True,
    tol: float = 1e-4,
) -> Dict[str, object]:
    """
    Provide a default minimal identity and optionally search a tiny integer-combinatorial
    family for an improved match:
       α^{-1} ≈ A · π^a · D^b · q^c · c0^e · g*^{-d}
    with A ∈ {1..16}, exponents in [-3,3].
    """
    # Default identity (minimal): α^{-1} ~ π D / (c0 g*)
    default_alpha_inv = (math.pi * D) / (c0 * g_star) if g_star > 0 else float("nan")
    default = {
        "form": "alpha_inv = π·D / (c0·g*)",
        "A": 1, "a": 1, "b": 1, "c": 0, "e": 1, "d": 1,
        "alpha_inv": default_alpha_inv,
        "alpha": 1.0 / default_alpha_inv if default_alpha_inv != 0 else float("nan"),
        "abs_err": abs(default_alpha_inv - ALPHA_INV_EMPIRICAL) if math.isfinite(default_alpha_inv) else float("inf")
    }

    if not do_integer_search or not math.isfinite(g_star) or g_star <= 0:
        return {"best": default, "candidates": [default]}

    # Tiny integer search
    best = default
    candidates = [default]
    pi = math.pi
    # Exponents: a,b,c,e,d in [-3,3]; A in 1..16
    exp_range = range(-3, 4)
    for A in range(1, 17):
        for a in exp_range:
            for b in exp_range:
                for c in exp_range:
                    for e in exp_range:
                        for d in range(0, 4):  # exponent on g* (non-negative)
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
                                # rank by (error, complexity)
                                better = (err < best["abs_err"] - tol) or (abs(err - best["abs_err"]) <= tol and complexity < best.get("complexity", 999))
                                if better:
                                    best = cand
                            except OverflowError:
                                continue

    return {"best": best, "candidates": sorted(candidates, key=lambda x: (x["abs_err"], x.get("complexity", 999)))[:10]}


# -----------------------------
# Main orchestration
# -----------------------------

def run_experiment(
    L: int = 64,
    layer_size: int = 128,
    layer_schedule: str = "linear",
    rmax: int = 4,
    seed: int = 7,
    use_kappa_positive: bool = True,
    report_R: Optional[List[int]] = None,
) -> Dict[str, object]:
    if report_R is None:
        # Reasonable geometric series up to the max valid slice size
        report_R = [1, 2, 4, 8, 16, 32]

    sizes = make_layer_sizes(L, layer_size, layer_schedule)
    G = build_layered_dag(L=L, layer_sizes=sizes, rmax=rmax, allow_parallel=True, seed=seed)

    # Compute g* from κ/ρ on depth-R slices
    est_signed = estimate_g_star(G, R_values=report_R, use_kappa_positive=False)
    est_pos = estimate_g_star(G, R_values=report_R, use_kappa_positive=use_kappa_positive)

    # Degeneracy constants for XOR and c0 for binary
    kappa0 = kappa0_for_xor(G)
    c0 = c0_for_binary_alphabet(ah_size=2, kappa0=kappa0)

    # Map to alpha (search tiny integer forms)
    pred_signed = predict_alpha_from_gstar(est_signed["g_star"], D=4, q=2, c0=c0, do_integer_search=True)
    pred_pos = predict_alpha_from_gstar(est_pos["g_star"], D=4, q=2, c0=c0, do_integer_search=True)

    summary = {
        "model": {
            "layers": L,
            "layer_schedule": layer_schedule,
            "layer_size_base": layer_size,
            "rmax": rmax,
            "seed": seed,
            "q_min": 2,
            "rule": "XOR over predecessor tags (order-independent)"
        },
        "degeneracy": {
            "kappa0": kappa0,
            "c0_bits_per_tick": c0
        },
        "rg_estimate_signed": est_signed,
        "rg_estimate_positive": est_pos,
        "alpha_empirical_inv": ALPHA_INV_EMPIRICAL,
        "alpha_prediction_signed": pred_signed,
        "alpha_prediction_positive": pred_pos
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Predict α from emergent graph model via RG-fixed ratio g* and simple integer identities.")
    parser.add_argument("--layers", type=int, default=64, help="number of layers (depth)")
    parser.add_argument("--layer-size", type=int, default=128, help="base size for layer 0 (schedule determines others)")
    parser.add_argument("--layer-schedule", type=str, default="linear", choices=["constant", "linear", "quadratic"], help="layer size schedule")
    parser.add_argument("--rmax", type=int, default=4, help="max out-degree per node")
    parser.add_argument("--seed", type=int, default=7, help="PRNG seed")
    parser.add_argument("--use-kappa-positive", action="store_true", help="use κ⁺ = max(0, κ) in g_R averages")
    parser.add_argument("--report-R", type=int, nargs="*", default=None, help="block sizes R to report (e.g., 2 4 8 16 32)")
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    args = parser.parse_args()

    summary = run_experiment(
        L=args.layers,
        layer_size=args.layer_size,
        layer_schedule=args.layer_schedule,
        rmax=args.rmax,
        seed=args.seed,
        use_kappa_positive=args.use_kappa_positive,
        report_R=args.report_R,
    )

    if args.pretty:
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary))

if __name__ == "__main__":
    main()


# -----------------------------
# README / Provenance Notes
# -----------------------------
#
# This script adheres to the definitions and constructions in the paper:
#   • Edge curvature κ(i→j) = V1(j) − V1(i) = |Pred(j)| − |Pred(i)|
#     (Def. 10.5; see also the "curvature normalisation" remark). After block
#     averaging over depth-R slices, κ_R pairs with the block-averaged memory
#     density (ρ_mem)_R to form the Einstein–Memory ratio (Thm. 10.13).
#   • Memory density ρ_mem on the minimal layer equals out-degree − 1 and
#     is extended to deeper layers by block averages (Defs. 3.4, 10.7).
#     Here we use the coarse approximation ρ_mem ≈ deg⁺ − 1 uniformly; this
#     is appropriate for bounded-degree acyclic layered DAGs used here.
#   • The block map contracts and the ratio g_R := κ_R/(ρ_mem)_R converges
#     to a universal (graph-class) constant g* > 0 (Thm. 10.13); we estimate g*
#     by a tail average across increasing R.
#   • The observer-entropy increment bound uses single-coordinate degeneracy
#     σ(d), with κ0 := max_d σ(d) and c0 := min_{h∈H_min} ceil(log2(|Ah|/κ0))
#     (Defs. 8.1–8.2; Thm. 8.4). For the XOR rule over binary tags, σ(d)=1,
#     so κ0=1 and c0=1 bit/tick.
#   • The continuum/Lorentz 4D limit motivates the use of D=4 in the
#     final α-identity (Thm. 10.19).
#
# Mapping to α here is explicitly *hypothesis generation*: we test a minimal
# form α^{-1} ≈ π D / (c0 g*) and then search for low-complexity integer
# combinations of {π, D, q, c0, g*} that best match 1/137.035999084.
# This mirrors the "Phase 4" proposal in the user's plan, but is not a proof.
#
