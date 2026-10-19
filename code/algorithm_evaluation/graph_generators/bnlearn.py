"""Load a causal DAG from a BIF file (bnlearn format).

The graph structure (variables and directed edges) is extracted; probability
tables are ignored. Confounders are NOT added here — that step is handled by
the evaluation loop so that confounder density can be varied independently.
"""
import os

from ..bif_parser import parse_bif


def load_bnlearn_graph(bif_path):
    """Load a DAG from a BIF file.

    Args:
        bif_path: path to a ``.bif`` file.

    Returns:
        variables: list of observed variable names.
        edges: list of ``(parent, child)`` tuples.
        graph_name: the basename of ``bif_path`` (e.g. ``"alarm.bif"``).
    """
    variables, edges = parse_bif(bif_path)
    graph_name = os.path.basename(bif_path)
    return variables, edges, graph_name
