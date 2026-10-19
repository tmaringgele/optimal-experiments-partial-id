"""Erdős-Rényi random DAG generator with latent confounders.

The topological order is fixed as (V_1, ..., V_N).  Directed and bidirected
edges are drawn independently for every ordered pair, so the density is
controlled by two scalar probabilities:

- ``p`` — each directed edge  i → j  (i < j) is included independently with
  probability ``p``.
- ``q`` — each bidirected edge  i ↔ j  (i < j, equivalently a latent
  confounder U_{ij} → V_i, U_{ij} → V_j) is included independently with
  probability ``q``.

Useful parameter regimes
------------------------
- ``p = 1/N``          → sparse, forest-like DAG
- ``p = log(N)/N``     → connected DAG near the connectivity threshold
- ``q = c/N`` (small c) → many small bidirected components (districts)
"""
import numpy as np

from ..causal_graph import CausalGraph


def generate_er_graph(N, p, q, seed):
    """Generate an Erdős-Rényi random DAG with latent confounders.

    Args:
        N: number of observed variables.
        p: probability of each directed edge i → j (i < j).
        q: probability of each bidirected edge i ↔ j (latent confounder).
        seed: integer random seed for reproducibility.

    Returns:
        cg: :class:`~algorithm_evaluation.causal_graph.CausalGraph` instance
            whose ``.hidden`` list already contains the sampled confounders.
        graph_name: human-readable identifier, e.g. ``"ER_N10_p0.3000_q0.1000"``.
    """
    rng = np.random.default_rng(seed)

    variables = [f"V_{i}" for i in range(1, N + 1)]

    directed_edges = [
        (variables[i], variables[j])
        for i in range(N)
        for j in range(i + 1, N)
        if rng.random() < p
    ]

    hidden_confounders = [
        (variables[i], variables[j])
        for i in range(N)
        for j in range(i + 1, N)
        if rng.random() < q
    ]

    graph_name = f"ER_N{N}_p{p:.4f}_q{q:.4f}"
    cg = CausalGraph(variables, directed_edges, hidden_confounders)
    return cg, graph_name
