"""Visualization tools for saved simulation reconstructions.

Loads a simulation from a graphs.json file (produced by evaluate(..., save_graphs=...))
and provides:
  - plot_simulation: draws the causal graph with colored query nodes and
    undirected edges for confounders
  - print_experiments: lists experiments partitioned into useful /
    ID-pruned / blocking-pruned
"""
import json
import textwrap

import matplotlib.pyplot as plt
import networkx as nx

from .causal_graph import CausalGraph
from .get_useless import get_useless_experiments
from .id_algorithm import is_identifiable


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load(json_path, graph_name, sim):
    """Load a single simulation record from a JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    sim_id = f"{graph_name}__sim{sim}"
    if sim_id not in data:
        available = [k for k in data if k.startswith(graph_name)]
        raise KeyError(
            f"sim_id '{sim_id}' not found. "
            f"Available sims for '{graph_name}': {available}"
        )
    return data[sim_id]


def _reconstruct(record):
    """Turn a JSON record into (CausalGraph, query_worlds, experiments)."""
    variables = record["variables"]
    edges = [tuple(e) for e in record["edges"]]
    confounders = [tuple(c) for c in record["confounders"]]

    cg = CausalGraph(variables, edges, confounders)

    query_worlds = [
        (frozenset(w["Y"]), frozenset(w["X"]))
        for w in record["query_worlds"]
    ]

    experiments = [
        tuple(
            (frozenset(world["W"]), frozenset(world["Z"]))
            for world in experiment
        )
        for experiment in record["experiments"]
    ]

    return cg, query_worlds, experiments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_simulation(json_path, graph_name, sim, figsize=(16, 10), seed=42):
    """Plot the causal graph for a saved simulation.

    Directed edges are drawn as arrows. Confounders are shown as curved
    undirected (dashed) edges between the two observed variables they connect.
    Query Y nodes are highlighted in red, X nodes in blue, and nodes that are
    both in purple. All other nodes are light grey.

    Args:
        json_path: path to the JSON file produced by evaluate(..., save_graphs=...)
        graph_name: graph filename, e.g. "asia.bif"
        sim: simulation index (int)
        figsize: matplotlib figure size
        seed: layout seed for reproducibility
    """
    record = _load(json_path, graph_name, sim)
    cg, query_worlds, _ = _reconstruct(record)

    # Collect all Y and X nodes across worlds
    all_Y = set()
    all_X = set()
    for Y, X in query_worlds:
        all_Y |= Y
        all_X |= X

    # Node colours
    def node_color(v):
        in_y = v in all_Y
        in_x = v in all_X
        if in_y and in_x:
            return "#9b59b6"   # purple
        if in_y:
            return "#e74c3c"   # red
        if in_x:
            return "#2980b9"   # blue
        return "#d0d0d0"       # grey

    node_colors = [node_color(v) for v in cg.observed]

    # Layout on directed graph of observed variables only
    obs_dag = nx.DiGraph()
    obs_dag.add_nodes_from(cg.observed)
    obs_dag.add_edges_from(cg.directed_edges)

    try:
        pos = nx.nx_agraph.graphviz_layout(obs_dag, prog="dot")
    except Exception:
        pos = nx.spring_layout(obs_dag, seed=seed)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw directed edges
    nx.draw_networkx_edges(
        obs_dag, pos, ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        edge_color="#555555",
        width=1.2,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        obs_dag, pos, ax=ax,
        node_color=node_colors,
        node_size=800,
    )
    nx.draw_networkx_labels(obs_dag, pos, ax=ax, font_size=7)

    # Draw confounders as curved dashed undirected edges
    confounder_graph = nx.Graph()
    confounder_graph.add_nodes_from(cg.observed)
    for _, (v1, v2) in cg.hidden_children.items():
        confounder_graph.add_edge(v1, v2)

    nx.draw_networkx_edges(
        confounder_graph, pos, ax=ax,
        style="dashed",
        edge_color="#e67e22",
        width=1.5,
        connectionstyle="arc3,rad=0.3",
    )

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
                   markersize=10, label="Y (query outcome)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2980b9",
                   markersize=10, label="X (query intervention)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#9b59b6",
                   markersize=10, label="Y and X"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d0d0d0",
                   markersize=10, label="Other"),
        plt.Line2D([0], [0], linestyle="--", color="#e67e22",
                   linewidth=2, label="Confounder (bidirected)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    n_worlds = len(query_worlds)
    world_strs = "; ".join(
        f"Y={set(Y)}, X={set(X)}" for Y, X in query_worlds
    )
    ax.set_title(
        f"{graph_name}  |  sim {sim}  |  "
        f"|V|={len(cg.observed)}, |U|={len(cg.hidden)}\n"
        f"Query ({'CF' if n_worlds == 2 else 'single-world'}): {world_strs}",
        fontsize=9,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def print_experiments(json_path, graph_name, sim):
    """Print experiments for a saved simulation, grouped by pruning status.

    Shows three sections:
      1. Useful experiments (not pruned)
      2. Pruned by ID check (single-world, identifiable from P(V))
      3. Pruned by AllPathsBlocked

    Args:
        json_path: path to the JSON file produced by evaluate(..., save_graphs=...)
        graph_name: graph filename, e.g. "asia.bif"
        sim: simulation index (int)
    """
    record = _load(json_path, graph_name, sim)
    cg, query_worlds, experiments = _reconstruct(record)

    useless, stats = get_useless_experiments(cg, query_worlds, experiments)

    # Partition experiment indices
    id_pruned_idx = set()
    blocked_pruned_idx = set()
    for i, experiment in enumerate(experiments):
        if i not in useless:
            continue
        if len(experiment) == 1:
            W, Z = experiment[0]
            if is_identifiable(cg, list(W), list(Z)):
                id_pruned_idx.add(i)
                continue
        blocked_pruned_idx.add(i)

    useful_idx = [i for i in range(len(experiments)) if i not in useless]

    def fmt_experiment(i, experiment):
        worlds = "; ".join(
            f"W={set(W)}, Z={set(Z)}" for W, Z in experiment
        )
        kind = "CF" if len(experiment) == 2 else "interventional"
        return f"  [{i:3d}] ({kind}) {worlds}"

    header = (
        f"{'='*70}\n"
        f"Simulation: {graph_name}  sim={sim}\n"
        f"|V|={len(cg.observed)}, |U|={len(cg.hidden)}, "
        f"|A|={len(experiments)}\n"
        f"Query: " + "; ".join(
            f"Y={set(Y)}, X={set(X)}" for Y, X in query_worlds
        ) + f"\n{'='*70}"
    )
    print(header)

    print(f"\n--- Useful ({len(useful_idx)}) ---")
    for i in useful_idx:
        print(fmt_experiment(i, experiments[i]))

    print(f"\n--- Pruned by ID check ({len(id_pruned_idx)}) ---")
    for i in sorted(id_pruned_idx):
        print(fmt_experiment(i, experiments[i]))

    print(f"\n--- Pruned by AllPathsBlocked ({len(blocked_pruned_idx)}) ---")
    for i in sorted(blocked_pruned_idx):
        print(fmt_experiment(i, experiments[i]))

    print(f"\n{'='*70}")
