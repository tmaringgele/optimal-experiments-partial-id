"""Graph generator interface for the evaluation pipeline.

Two generators are available:
- ``bnlearn``: loads a DAG from a BIF file (bnlearn repository).
  Confounders are added later by :func:`evaluate.evaluate`.
- ``erdos_renyi``: generates a random DAG with latent confounders
  directly from (N, p, q) parameters.

Both are exposed here for convenience:
"""
from .bnlearn import load_bnlearn_graph
from .erdos_renyi import generate_er_graph

__all__ = ["load_bnlearn_graph", "generate_er_graph"]
