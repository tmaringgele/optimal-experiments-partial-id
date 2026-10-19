"""Implementation of GetUselessExperiments (Algorithm 3 from the paper).

Combines two pruning criteria:
1. AllPathsBlocked: experiments where all paths from R*_theta to outcomes
   pass through interventions have zero potency
2. ID check: single-world experiments whose outcome is identifiable from
   P(V) have zero potency (only checked if AllPathsBlocked did not prune)

Also provides sampling functions for queries and experiments.
"""
import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import combinations

from .causal_graph import CausalGraph
from .id_algorithm import is_identifiable


# ---------------------------------------------------------------------------
#  Sampling helpers
# ---------------------------------------------------------------------------

def _pick_size(sizes, rng):
    """Pick a size uniformly at random from a list of allowed sizes."""
    return sizes[rng.integers(len(sizes))]


def _sample_world(causal_graph, variables, w_sizes, z_sizes, rng,
                   use_distance=False, obs_undirected=None,
                   distances=None, **_kwargs):
    """Sample a single (Y_set, X_set) world with X ⊆ ancestors(Y).

    When use_distance is True:
      1. Pick Y uniformly from V
      2. Compute anc(Y) \\ Y and their undirected distances to Y
      3. Pick target distance uniformly from the `distances` list
      4. Pick seed x from ancestors at that distance; if none exist,
         pick from ancestors at the closest available distance
      5. Fill remaining X from anc(Y) \\ Y

    Returns (frozenset(Y), frozenset(X)) or None on failure.
    """
    n = len(variables)
    max_retries = 50
    w_size = _pick_size(w_sizes, rng)
    z_size = _pick_size(z_sizes, rng)

    for _retry in range(max_retries):
        # Pick Y
        w_actual = min(w_size, n)
        w_indices = rng.choice(n, size=w_actual, replace=False)
        Y_set = {variables[i] for i in w_indices}

        anc_Y = causal_graph.observed_ancestors(Y_set)
        if not anc_Y:
            continue

        if use_distance:
            target_dist = distances[rng.integers(len(distances))]

            # Compute min undirected distance from each ancestor to any y ∈ Y
            anc_by_dist = defaultdict(list)
            for a in anc_Y:
                min_d = min(
                    (nx.shortest_path_length(obs_undirected, a, y)
                     for y in Y_set
                     if obs_undirected.has_node(a) and obs_undirected.has_node(y)
                     and nx.has_path(obs_undirected, a, y)),
                    default=None,
                )
                if min_d is not None:
                    anc_by_dist[min_d].append(a)

            if not anc_by_dist:
                anc_list = list(anc_Y)
            else:
                available = sorted(anc_by_dist.keys())
                best_dist = min(available,
                                key=lambda d: abs(d - target_dist))
                candidates = anc_by_dist[best_dist]
                x_seed = candidates[rng.integers(len(candidates))]

                X_set = {x_seed}
                anc_pool = list(anc_Y - X_set)
                needed = z_size - 1
                if needed > 0 and anc_pool:
                    extra = min(needed, len(anc_pool))
                    chosen = rng.choice(len(anc_pool), size=extra,
                                        replace=False)
                    for i in chosen:
                        X_set.add(anc_pool[i])
                return (frozenset(Y_set), frozenset(X_set))
        else:
            anc_list = list(anc_Y)

        z_actual = min(z_size, len(anc_list))
        z_indices = rng.choice(len(anc_list), size=z_actual, replace=False)
        X_set = {anc_list[i] for i in z_indices}
        return (frozenset(Y_set), frozenset(X_set))

    # Fallback
    for v in variables:
        anc = causal_graph.observed_ancestors({v})
        if anc:
            return (frozenset({v}), frozenset({rng.choice(list(anc))}))
    return None


def _build_obs_undirected(causal_graph):
    """Build undirected graph over observed variables for distance queries."""
    obs_undirected = nx.Graph()
    obs_undirected.add_nodes_from(causal_graph.observed)
    obs_undirected.add_edges_from(causal_graph.directed_edges)
    return obs_undirected


def sample_query(causal_graph, theta_config, rng):
    """Sample a query theta according to theta_config.

    Returns a list of (Y_set, X_set) tuples representing counterfactual
    worlds (1 or 2).  X is always a subset of ancestors(Y).

    CF_fraction controls the probability of a 2-world counterfactual query.
    For 2-world queries the second world shares at least one intervention
    variable with the first.

    Args:
        causal_graph: CausalGraph object
        theta_config: dict with keys CF_fraction, W_sizes (list of int),
                      Z_sizes (list of int), and optionally
                      intervention_outcome_distances (list of int)
        rng: numpy random Generator

    Returns:
        list of (frozenset(Y), frozenset(X)) tuples, or None if no valid
        query could be constructed
    """
    variables = causal_graph.observed

    distances = theta_config.get('intervention_outcome_distances')
    use_distance = distances is not None
    obs_undirected = _build_obs_undirected(causal_graph) if use_distance else None

    cf_fraction = theta_config.get('CF_fraction', 0.0)
    is_cf = rng.random() < cf_fraction
    n_worlds = 2 if is_cf else 1

    dist_kwargs = dict(
        use_distance=use_distance, obs_undirected=obs_undirected,
        distances=distances,
    )

    w_sizes = theta_config['W_sizes']
    z_sizes = theta_config['Z_sizes']

    # --- First world ---
    world1 = _sample_world(causal_graph, variables, w_sizes, z_sizes, rng,
                           **dist_kwargs)
    if world1 is None:
        return None

    worlds = [world1]

    if n_worlds == 2:
        # Second world: seed W₂ from W₁, seed Z₂ from Z₁
        W1, X1 = world1
        world2 = _sample_cf_world(causal_graph, variables, w_sizes, z_sizes,
                                  W1, X1, rng, **dist_kwargs)
        if world2 is not None:
            worlds.append(world2)

    return worlds


def _sample_cf_world(causal_graph, variables, w_sizes, z_sizes,
                     W_prev, Z_prev, rng,
                     use_distance=False, obs_undirected=None,
                     distances=None, **_kwargs):
    """Sample world 2 for a counterfactual experiment.

    Seeds W₂ from W₁ and Z₂ from Z₁ to guarantee shared variables:
      1. Pick first w₂ uniformly from W₁
      2. Expand W₂ with additional variables from V (like world 1)
      3. Pick first z₂ from Z₁ ∩ (anc(W₂) \\ W₂), using distance control
         if enabled (preferring the target distance)
      4. Expand Z₂ with additional variables from anc(W₂) \\ W₂
    """
    max_retries = 50
    W_prev_list = list(W_prev)
    Z_prev_list = list(Z_prev)
    w_size = _pick_size(w_sizes, rng)
    z_size = _pick_size(z_sizes, rng)

    for _retry in range(max_retries):
        # Seed W₂ from W₁
        w_seed = W_prev_list[rng.integers(len(W_prev_list))]
        W_set = {w_seed}

        # Expand W₂ with additional variables
        if w_size > 1:
            pool = [v for v in variables if v != w_seed]
            if pool:
                extra = min(w_size - 1, len(pool))
                chosen = rng.choice(len(pool), size=extra, replace=False)
                for i in chosen:
                    W_set.add(pool[i])

        anc_W = causal_graph.observed_ancestors(W_set)
        if not anc_W:
            continue

        # Seed Z₂ from Z₁ ∩ anc(W₂)
        shared_candidates = [z for z in Z_prev_list if z in anc_W]
        if not shared_candidates:
            continue

        if use_distance:
            target_dist = distances[rng.integers(len(distances))]

            # Compute min undirected distance from each candidate to any w ∈ W₂
            cand_by_dist = defaultdict(list)
            for z in shared_candidates:
                min_d = min(
                    (nx.shortest_path_length(obs_undirected, z, w)
                     for w in W_set
                     if obs_undirected.has_node(z) and obs_undirected.has_node(w)
                     and nx.has_path(obs_undirected, z, w)),
                    default=None,
                )
                if min_d is not None:
                    cand_by_dist[min_d].append(z)

            if cand_by_dist:
                available = sorted(cand_by_dist.keys())
                best_dist = min(available,
                                key=lambda d: abs(d - target_dist))
                candidates = cand_by_dist[best_dist]
                z_seed = candidates[rng.integers(len(candidates))]
            else:
                z_seed = shared_candidates[rng.integers(len(shared_candidates))]
        else:
            z_seed = shared_candidates[rng.integers(len(shared_candidates))]

        Z_set = {z_seed}

        # Expand Z₂ with additional ancestors of W₂
        if z_size > 1:
            anc_pool = list(anc_W - Z_set)
            if anc_pool:
                extra = min(z_size - 1, len(anc_pool))
                chosen = rng.choice(len(anc_pool), size=extra, replace=False)
                for i in chosen:
                    Z_set.add(anc_pool[i])

        return (frozenset(W_set), frozenset(Z_set))

    return None


def sample_experiments(causal_graph, experiment_config, n_experiments, rng):
    """Sample candidate experiments according to experiment_config.

    Each experiment is a tuple of (W, Z) frozenset pairs (1 or 2 worlds).
    Z is always a subset of observed ancestors of W.

    CF_fraction controls the probability that an experiment is a 2-world
    counterfactual.  For counterfactual experiments, the second world shares
    at least one intervention variable with the first.

    Args:
        causal_graph: CausalGraph object
        experiment_config: dict with keys CF_fraction, W_sizes (list of int),
                           Z_sizes (list of int), and optionally
                           intervention_outcome_distances (list of int)
        n_experiments: number of experiments to sample
        rng: numpy random Generator

    Returns:
        list of experiment tuples; each experiment is a tuple of
        (frozenset(W), frozenset(Z)) pairs
    """
    variables = causal_graph.observed

    distances = experiment_config.get('intervention_outcome_distances')
    use_distance = distances is not None
    obs_undirected = _build_obs_undirected(causal_graph) if use_distance else None

    cf_fraction = experiment_config.get('CF_fraction', 0.0)
    w_sizes = experiment_config['W_sizes']
    z_sizes = experiment_config['Z_sizes']

    dist_kwargs = dict(
        use_distance=use_distance, obs_undirected=obs_undirected,
        distances=distances,
    )

    experiments = []

    for _ in range(n_experiments):
        is_cf = rng.random() < cf_fraction
        n_worlds = 2 if is_cf else 1

        world1 = _sample_world(causal_graph, variables, w_sizes, z_sizes, rng,
                               **dist_kwargs)
        if world1 is None:
            continue

        worlds = [world1]

        if n_worlds == 2:
            W1, Z1 = world1
            world2 = _sample_cf_world(causal_graph, variables, w_sizes,
                                      z_sizes, W1, Z1, rng,
                                      **dist_kwargs)
            if world2 is not None:
                worlds.append(world2)

        experiments.append(tuple(worlds))

    return experiments


# ---------------------------------------------------------------------------
#  Exhaustive enumeration (alternative to sampling)
# ---------------------------------------------------------------------------

def _nonempty_subsets(items):
    """All non-empty subsets of `items` as frozensets."""
    items = list(items)
    return [
        frozenset(c)
        for r in range(1, len(items) + 1)
        for c in combinations(items, r)
    ]


def enumerate_all_experiments(causal_graph, include_counterfactuals=True):
    """Enumerate every (W, Z) experiment satisfying the framework's constraints.

    Returns the union of:
      - All single-world experiments (W, Z) with W ⊆ V non-empty and
        Z ⊆ obs_anc(W) \\ W non-empty.
      - If ``include_counterfactuals=True``: all 2-world experiments formed
        as unordered pairs of single-world experiments (including self-pairs,
        which represent ATE-style counterfactuals where both worlds share the
        same (W, Z) signature but differ in intervention values).

    Args:
        causal_graph: CausalGraph object.
        include_counterfactuals: include 2-world experiments (default True).

    Returns:
        list of experiments; each experiment is a tuple of one or two
        ``(frozenset(W), frozenset(Z))`` pairs.
    """
    variables = causal_graph.observed
    single_world = []
    for W in _nonempty_subsets(variables):
        anc_W = causal_graph.observed_ancestors(W)
        if not anc_W:
            continue
        for Z in _nonempty_subsets(anc_W):
            single_world.append((W, Z))

    experiments = [(w,) for w in single_world]
    if include_counterfactuals:
        for i, w1 in enumerate(single_world):
            for w2 in single_world[i:]:
                experiments.append((w1, w2))
    return experiments


# ---------------------------------------------------------------------------
#  GetUselessExperiments (Algorithm 3)
# ---------------------------------------------------------------------------

def get_useless_experiments(causal_graph, query_worlds, experiments,
                            ID_only=False):
    """Run GetUselessExperiments (Algorithm 3).

    For each experiment checks:
    1. If AllPathsBlocked for every world -> useless (Corollary 1)
    2. If single-world and P(W|do(Z)) is identifiable -> useless (Thm 4)

    Args:
        causal_graph: CausalGraph object
        query_worlds: list of (Y_set, X_set) tuples (the query theta)
        experiments: list of experiments, each a tuple of (W, Z) pairs
        ID_only: if True, skip the AllPathsBlocked criterion and only apply
            the ID check.  Two-world counterfactual experiments are then
            never pruned (the ID check is undefined for them).

    Returns:
        useless: set of experiment indices that have zero potency
        stats: dict with pruning statistics
    """
    R_theta_star = causal_graph.get_R_theta_star(query_worlds)

    useless = set()
    id_pruned = 0
    blocked_pruned = 0

    for i, experiment in enumerate(experiments):
        # 1. AllPathsBlocked: must hold for ALL worlds in the experiment
        if not ID_only:
            all_blocked = True
            for W, Z in experiment:
                if not causal_graph.all_paths_blocked(R_theta_star, W, Z):
                    all_blocked = False
                    break
            if all_blocked:
                useless.add(i)
                blocked_pruned += 1
                continue

        # 2. ID check: only for single-world interventional experiments
        if len(experiment) == 1:
            W, Z = experiment[0]
            if is_identifiable(causal_graph, list(W), list(Z)):
                useless.add(i)
                id_pruned += 1

    stats = {
        'id_pruned': id_pruned,
        'blocked_pruned': blocked_pruned,
        'total_pruned': id_pruned + blocked_pruned,
    }
    return useless, stats


def get_useless_experiments_separate(causal_graph, query_worlds, experiments):
    """Run AllPathsBlocked and the ID check independently for every experiment.

    Unlike :func:`get_useless_experiments`, which short-circuits the ID check
    whenever AllPathsBlocked already prunes an experiment, this variant always
    evaluates *both* criteria so that their overlap can be measured.

    The ID check is only well-defined for single-world (interventional)
    experiments; for two-world counterfactuals we set ``id`` to ``False``.

    Args:
        causal_graph: CausalGraph object
        query_worlds: list of (Y_set, X_set) tuples (the query theta)
        experiments: list of experiments, each a tuple of (W, Z) pairs

    Returns:
        per_experiment: list of dicts (one per experiment) with keys
            ``blocked``, ``id``, ``is_cf``, ``w_size``, ``z_size``.
        stats: dict with aggregate counts: ``n_experiments``, ``n_blocked``,
            ``n_id``, ``n_both``, ``n_blocked_only``, ``n_id_only``,
            ``n_neither``, ``n_useless`` (= union of the two criteria), and
            single-world-only counterparts suffixed ``_sw``.
    """
    R_theta_star = causal_graph.get_R_theta_star(query_worlds)

    per_experiment = []
    for experiment in experiments:
        is_cf = len(experiment) > 1

        all_blocked = True
        for W, Z in experiment:
            if not causal_graph.all_paths_blocked(R_theta_star, W, Z):
                all_blocked = False
                break

        if is_cf:
            id_blocked = False
        else:
            W, Z = experiment[0]
            id_blocked = is_identifiable(causal_graph, list(W), list(Z))

        w_size = sum(len(W) for W, _ in experiment)
        z_size = sum(len(Z) for _, Z in experiment)

        per_experiment.append({
            'blocked': all_blocked,
            'id': id_blocked,
            'is_cf': is_cf,
            'w_size': w_size,
            'z_size': z_size,
        })

    def _count(pred):
        return sum(1 for e in per_experiment if pred(e))

    stats = {
        'n_experiments':    len(per_experiment),
        'n_blocked':        _count(lambda e: e['blocked']),
        'n_id':             _count(lambda e: e['id']),
        'n_both':           _count(lambda e: e['blocked'] and e['id']),
        'n_blocked_only':   _count(lambda e: e['blocked'] and not e['id']),
        'n_id_only':        _count(lambda e: e['id'] and not e['blocked']),
        'n_neither':        _count(lambda e: not e['blocked'] and not e['id']),
        'n_useless':        _count(lambda e: e['blocked'] or e['id']),
        # Single-world-only views (where both criteria can apply)
        'n_sw':             _count(lambda e: not e['is_cf']),
        'n_blocked_sw':     _count(lambda e: not e['is_cf'] and e['blocked']),
        'n_id_sw':          _count(lambda e: not e['is_cf'] and e['id']),
        'n_both_sw':        _count(lambda e: not e['is_cf'] and e['blocked'] and e['id']),
        'n_blocked_only_sw':_count(lambda e: not e['is_cf'] and e['blocked'] and not e['id']),
        'n_id_only_sw':     _count(lambda e: not e['is_cf'] and e['id'] and not e['blocked']),
        'n_neither_sw':     _count(lambda e: not e['is_cf'] and not e['blocked'] and not e['id']),
    }
    return per_experiment, stats
