"""Main evaluation function for the GetUselessExperiments algorithm.

Supports three graph sources:

- ``"bnlearn"`` — loads a DAG from a BIF file and adds latent confounders
  randomly during each simulation (original approach).
- ``"erdos_renyi"`` — generates a fresh Erdős-Rényi random DAG with latent
  confounders already embedded for each simulation; the confounder-addition
  step is skipped.
- ``"custom"`` — uses a user-supplied DAG (variables, edges, confounders)
  reused across every simulation; the confounder-addition step is skipped.
  The query and experiment set may be overridden directly via
  ``query_worlds_override`` / ``experiments_override``.
"""
import json
import os
import networkx as nx
import numpy as np
import pandas as pd

from .causal_graph import CausalGraph
from .get_useless import (
    sample_query,
    sample_experiments,
    get_useless_experiments,
    get_useless_experiments_separate,
)
from .graph_generators import load_bnlearn_graph, generate_er_graph


DEFAULT_THETA_CONFIG = {
    'CF_fraction': 0.0,
    'W_sizes': [1],
    'Z_sizes': [1],
}

DEFAULT_EXPERIMENT_CONFIG = {
    'CF_fraction': 0.0,
    'W_sizes': [1, 2, 3],
    'Z_sizes': [1, 2, 3],
}


def evaluate(
    graph_source="bnlearn",
    graph_config=None,
    confounder_ratio_range=(0.1, 0.5),
    n_simulations=100,
    seed=42,
    experiment_set_size=200,
    theta_config=None,
    experiment_config=None,
    verbose=False,
    save_graphs=None,
    confounder_strategy="random",
    ID_only=False,
    query_worlds_override=None,
    experiments_override=None,
):
    """Evaluate GetUselessExperiments on a causal graph.

    Supports three graph sources controlled by ``graph_source``:

    **bnlearn** (``graph_source="bnlearn"``)
        ``graph_config`` must contain ``{"bif_path": "<path>.bif"}``.
        For each simulation the same DAG structure is reused, and fresh latent
        confounders are sampled according to ``confounder_ratio_range``.  A
        guaranteed confounder between every ``x ∈ X^(1)`` and a random
        ``y ∈ Y^(1)`` is added before any random confounders.

    **Erdős-Rényi** (``graph_source="erdos_renyi"``)
        ``graph_config`` must contain ``{"N": <int>, "p": <float>, "q": <float>}``.
        Each simulation generates an independent random DAG (different seed per
        sim).  Confounders are already part of the generated graph (controlled
        by ``q``), so ``confounder_ratio_range`` and the forced-confounder step
        are skipped.

    **Custom** (``graph_source="custom"``)
        ``graph_config`` must contain ``{"variables": [...], "edges": [(p, c), ...],
        "confounders": [(v1, v2), ...], "graph_name": "<str>"}``.  The same DAG is
        reused across every simulation; the confounder-addition step is
        skipped.  Pair this with ``query_worlds_override`` / ``experiments_override``
        to evaluate a fixed query and a hand-specified or enumerated experiment
        set on a known graph (e.g. for case-study analyses).

    Args:
        graph_source: ``"bnlearn"`` or ``"erdos_renyi"``.
        graph_config: dict of source-specific parameters (see above).
        confounder_ratio_range: ``(a, b)`` interval for ``|U|/|V|`` ratio.
            Only used when ``graph_source="bnlearn"``.
        n_simulations: number of random simulations.
        seed: master random seed for reproducibility.
        experiment_set_size: number of candidate experiments ``|A|`` per sim.
        theta_config: dict controlling query sampling.
        experiment_config: dict controlling candidate experiment sampling.
        verbose: if True, print progress.
        save_graphs: if set to a file path, saves full reconstruction data for
            every simulation to a JSON file (merged with any existing content).
        confounder_strategy: ``"random"`` (default) or ``"non_ancestor"``.
            Controls which variable pairs are eligible as additional latent
            confounders when ``graph_source="bnlearn"``.  ``"random"`` allows
            any pair; ``"non_ancestor"`` restricts candidates to pairs where
            neither variable is an observed ancestor of the other in the bare
            DAG, following the standard ADMG convention that hidden common
            causes arise between causally unrelated variables.  The forced
            confounder between X^(1) and Y^(1) is always added regardless of
            this setting.
        ID_only: if True, run the ID-only ablation of GetUselessExperiments
            (Algorithm 3) — the AllPathsBlocked criterion is skipped and
            only the ID check is applied.  Two-world counterfactual
            experiments are never pruned in this mode.
        query_worlds_override: if not None, skip query sampling and use this
            list of ``(Y, X)`` pairs as the query for every simulation.  Each
            pair is normalised to ``(frozenset(Y), frozenset(X))``.  Most
            useful with ``graph_source="custom"``.
        experiments_override: if not None, skip experiment sampling and use
            this list as the candidate experiment set for every simulation.
            Each entry must be an iterable of one or two ``(W, Z)`` pairs;
            sets are normalised to frozensets.  Most useful with
            ``graph_source="custom"``.

    Returns:
        pd.DataFrame with one row per simulation.
    """
    if graph_config is None:
        graph_config = {}
    if theta_config is None:
        theta_config = DEFAULT_THETA_CONFIG
    if experiment_config is None:
        experiment_config = DEFAULT_EXPERIMENT_CONFIG

    if graph_source not in ("bnlearn", "erdos_renyi", "custom"):
        raise ValueError(f"Unknown graph_source {graph_source!r}. "
                         "Use 'bnlearn', 'erdos_renyi', or 'custom'.")
    if confounder_strategy not in ("random", "non_ancestor"):
        raise ValueError(f"Unknown confounder_strategy {confounder_strategy!r}. "
                         "Use 'random' or 'non_ancestor'.")

    rng = np.random.default_rng(seed)

    # Normalise overrides once (frozensets) so they can be reused per sim.
    query_override = None
    if query_worlds_override is not None:
        query_override = [
            (frozenset(Y), frozenset(X)) for Y, X in query_worlds_override
        ]
    exp_override = None
    if experiments_override is not None:
        exp_override = [
            tuple((frozenset(W), frozenset(Z)) for W, Z in exp)
            for exp in experiments_override
        ]

    # custom: build the full DAG (with confounders) once and reuse it.
    if graph_source == "custom":
        custom_variables = list(graph_config["variables"])
        custom_edges = [tuple(e) for e in graph_config.get("edges", [])]
        custom_confounders = [tuple(c) for c in graph_config.get("confounders", [])]
        custom_graph_name = graph_config.get("graph_name", "custom")
        cg_custom = CausalGraph(custom_variables, custom_edges,
                                custom_confounders)
        if verbose:
            print(f"Graph: {custom_graph_name}, |V|={len(custom_variables)}, "
                  f"|U|={len(custom_confounders)}, "
                  f"|A|={experiment_set_size if exp_override is None else len(exp_override)}")

    # bnlearn: load the bare graph once; confounders are added per simulation.
    if graph_source == "bnlearn":
        bif_path = graph_config["bif_path"]
        variables, edges, graph_name = load_bnlearn_graph(bif_path)
        bare_cg = CausalGraph(variables, edges, [])
        if verbose:
            print(f"Graph: {graph_name}, |V|={len(variables)}, "
                  f"|A|={experiment_set_size}, "
                  f"ratio_range={confounder_ratio_range}")

        # Precompute eligible confounder pairs (static for this DAG).
        if confounder_strategy == "non_ancestor":
            _ancestor_of = {v: nx.ancestors(bare_cg.graph, v) for v in variables}
            _non_anc_pairs = frozenset(
                (variables[i], variables[j])
                for i in range(len(variables))
                for j in range(i + 1, len(variables))
                if variables[j] not in _ancestor_of[variables[i]]
                and variables[i] not in _ancestor_of[variables[j]]
            )
        else:
            _non_anc_pairs = None

    results = []
    graph_records = {}

    for sim in range(n_simulations):

        # ── Graph construction ───────────────────────────────────────────────
        if graph_source == "bnlearn":
            if query_override is not None:
                query_worlds = query_override
            else:
                # Sample query on the bare DAG (before confounders)
                query_worlds = sample_query(bare_cg, theta_config, rng)
                if query_worlds is None:
                    continue

            # Forced confounder: every x ∈ X^(1) gets a confounder with some y ∈ Y^(1)
            Y1, X1 = query_worlds[0]
            y1_list = sorted(Y1)
            forced_confounders = {
                (x, rng.choice(y1_list))
                for x in X1
            }

            # Sample target |U|/|V| ratio, clamped to the forced-confounder floor
            r_lo = max(confounder_ratio_range[0],
                       len(forced_confounders) / len(variables))
            r_hi = confounder_ratio_range[1]
            r = rng.uniform(r_lo, r_hi) if r_lo <= r_hi else r_lo
            target_u = max(len(forced_confounders), int(r * len(variables)))

            candidate_pairs = [
                (v1, v2)
                for i, v1 in enumerate(variables)
                for v2 in variables[i + 1:]
                if (v1, v2) not in forced_confounders
                and (v2, v1) not in forced_confounders
                and (_non_anc_pairs is None or (v1, v2) in _non_anc_pairs)
            ]
            n_additional = min(target_u - len(forced_confounders),
                               len(candidate_pairs))
            if n_additional > 0:
                indices = rng.choice(len(candidate_pairs),
                                     size=n_additional, replace=False)
                confounders = list(forced_confounders) + [
                    candidate_pairs[i] for i in indices
                ]
            else:
                confounders = list(forced_confounders)

            cg = CausalGraph(variables, edges, confounders)

        elif graph_source == "erdos_renyi":
            graph_seed = int(rng.integers(0, 2**31))
            cg, graph_name = generate_er_graph(
                N=graph_config["N"],
                p=graph_config["p"],
                q=graph_config["q"],
                seed=graph_seed,
            )
            variables = cg.observed
            edges = cg.directed_edges
            confounders = list(cg.hidden_children.values())

            if verbose and sim == 0:
                print(f"Graph: {graph_name}, |V|={len(variables)}, "
                      f"|A|={experiment_set_size}")

            if query_override is not None:
                query_worlds = query_override
            else:
                # Sample query on the graph (confounders already present)
                query_worlds = sample_query(cg, theta_config, rng)
                if query_worlds is None:
                    continue

        else:  # custom
            cg = cg_custom
            variables = custom_variables
            edges = custom_edges
            confounders = custom_confounders
            graph_name = custom_graph_name

            if query_override is not None:
                query_worlds = query_override
            else:
                query_worlds = sample_query(cg, theta_config, rng)
                if query_worlds is None:
                    continue

        # ── Experiment sampling & pruning ────────────────────────────────────
        if exp_override is not None:
            experiments = exp_override
        else:
            experiments = sample_experiments(
                cg, experiment_config, experiment_set_size, rng
            )

        if save_graphs is not None:
            sim_id = f"{graph_name}__sim{sim}"
            graph_records[sim_id] = {
                "sim_id": sim_id,
                "graph": graph_name,
                "sim": sim,
                "variables": list(variables),
                "edges": [list(e) for e in edges],
                "confounders": [list(c) for c in confounders],
                "query_worlds": [
                    {"Y": sorted(Y), "X": sorted(X)}
                    for Y, X in query_worlds
                ],
                "experiments": [
                    [{"W": sorted(W), "Z": sorted(Z)} for W, Z in experiment]
                    for experiment in experiments
                ],
            }

        R_theta = cg.get_R_theta(query_worlds)
        R_theta_star = cg.get_R_theta_star(query_worlds)

        useless, stats = get_useless_experiments(
            cg, query_worlds, experiments, ID_only=ID_only
        )

        n_experiments = len(experiments)
        n_cf = sum(1 for e in experiments if len(e) == 2)
        query_str = "; ".join(
            f"Y={set(Y)}, X={set(X)}" for Y, X in query_worlds
        )

        obs_undirected = nx.Graph()
        obs_undirected.add_nodes_from(variables)
        obs_undirected.add_edges_from(edges)
        min_xy_dist = float('inf')
        for Y, X in query_worlds:
            for x in X:
                for y in Y:
                    if nx.has_path(obs_undirected, x, y):
                        d = nx.shortest_path_length(obs_undirected, x, y)
                        min_xy_dist = min(min_xy_dist, d)
        if min_xy_dist == float('inf'):
            min_xy_dist = None

        result = {
            'graph': graph_name,
            'graph_source': graph_source,
            'n_vars': len(variables),
            'u_v_ratio': len(confounders) / len(variables),
            'sim': sim,
            'n_confounders': len(confounders),
            'n_districts': len(cg.get_districts()),
            'query': query_str,
            'query_is_cf': len(query_worlds) == 2,
            'min_intervention_outcome_dist': min_xy_dist,
            'n_R_theta': len(R_theta),
            'n_R_theta_star': len(R_theta_star),
            'n_experiments': n_experiments,
            'n_cf_experiments': n_cf,
            'fraction_cf': n_cf / n_experiments if n_experiments else 0,
            'n_useless': len(useless),
            'n_useful': n_experiments - len(useless),
            'fraction_pruned': len(useless) / n_experiments if n_experiments else 0,
            'id_pruned': stats['id_pruned'],
            'blocked_pruned': stats['blocked_pruned'],
        }
        results.append(result)

        if verbose and (sim + 1) % 10 == 0:
            print(f"  u/v={result['u_v_ratio']:.2f}, sim {sim+1}/{n_simulations}: "
                  f"pruned {result['fraction_pruned']:.1%}")

    if save_graphs is not None and graph_records:
        existing = {}
        if os.path.exists(save_graphs):
            with open(save_graphs, "r") as f:
                existing = json.load(f)
        existing.update(graph_records)
        with open(save_graphs, "w") as f:
            json.dump(existing, f, indent=2)

    return pd.DataFrame(results)


def evaluate_separate(
    graph_config,
    n_simulations=100,
    seed=42,
    experiment_set_size=200,
    theta_config=None,
    experiment_config=None,
    verbose=False,
    return_per_experiment=True,
):
    """Evaluate the two pruning mechanisms *independently* on Erdős-Rényi DAGs.

    Mirrors :func:`evaluate` for ``graph_source="erdos_renyi"``, but uses
    :func:`get_useless_experiments_separate` so that each candidate experiment
    is checked against AllPathsBlocked *and* the ID criterion regardless of
    the order. This exposes the complementarity (overlap) of the two methods.

    Args:
        graph_config: dict with keys ``N``, ``p``, ``q``.
        n_simulations: number of random simulations.
        seed: master random seed.
        experiment_set_size: number of candidate experiments per simulation.
        theta_config, experiment_config: same semantics as :func:`evaluate`.
        verbose: print progress per simulation.
        return_per_experiment: if True (default), also return a long-form
            dataframe with one row per (simulation, experiment).

    Returns:
        sim_df: DataFrame with one row per simulation (overlap counts).
        exp_df: DataFrame with one row per experiment (only when
            ``return_per_experiment=True``); otherwise ``None``.
    """
    if theta_config is None:
        theta_config = DEFAULT_THETA_CONFIG
    if experiment_config is None:
        experiment_config = DEFAULT_EXPERIMENT_CONFIG

    rng = np.random.default_rng(seed)

    sim_rows = []
    exp_rows = []

    for sim in range(n_simulations):
        graph_seed = int(rng.integers(0, 2**31))
        cg, graph_name = generate_er_graph(
            N=graph_config["N"],
            p=graph_config["p"],
            q=graph_config["q"],
            seed=graph_seed,
        )
        variables = cg.observed
        confounders = list(cg.hidden_children.values())

        query_worlds = sample_query(cg, theta_config, rng)
        if query_worlds is None:
            continue

        experiments = sample_experiments(
            cg, experiment_config, experiment_set_size, rng
        )

        R_theta_star = cg.get_R_theta_star(query_worlds)
        per_exp, stats = get_useless_experiments_separate(
            cg, query_worlds, experiments
        )

        n = stats['n_experiments']
        n_cf = sum(1 for e in experiments if len(e) == 2)

        sim_rows.append({
            'graph': graph_name,
            'n_vars': len(variables),
            'u_v_ratio': len(confounders) / len(variables),
            'sim': sim,
            'n_confounders': len(confounders),
            'n_districts': len(cg.get_districts()),
            'query_is_cf': len(query_worlds) == 2,
            'n_R_theta_star': len(R_theta_star),
            'n_experiments': n,
            'n_cf_experiments': n_cf,
            'fraction_cf': n_cf / n if n else 0,
            **stats,
        })

        if return_per_experiment:
            for idx, e in enumerate(per_exp):
                exp_rows.append({
                    'graph': graph_name,
                    'sim': sim,
                    'experiment_idx': idx,
                    **e,
                })

        if verbose and (sim + 1) % 10 == 0:
            print(f"  sim {sim+1}/{n_simulations}: "
                  f"blocked={stats['n_blocked']}/{n}, "
                  f"id={stats['n_id']}/{n}, "
                  f"both={stats['n_both']}/{n}")

    sim_df = pd.DataFrame(sim_rows)
    exp_df = pd.DataFrame(exp_rows) if return_per_experiment else None
    return sim_df, exp_df
