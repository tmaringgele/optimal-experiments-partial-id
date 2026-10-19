"""Deterministic instance generation for the runtime experiments.

Samples a (graph, query, P(V)) instance and a candidate experiment list:

  * an Erdos-Renyi ADMG under guards that keep the canonical response-type
    model tractable (max in-degree, max c-component size, non-degenerate
    query),
  * a two-world counterfactual estimand over that graph,
  * a Dirichlet-pushforward P(V) that is consistent with the model, and
  * scalar candidate experiments of the form `E(C=1)=1`.

`instances.py` drives this module; `run_rerun.py` measures the potency
evaluation on the instances it produces.
"""
from __future__ import annotations
import os, sys, math, itertools, re

# Reproducibility: the shared algorithm_evaluation package iterates sets of variable
# *name strings*, so list(set) ordering (and thus seeded rng.choice) depends on Python's
# per-process string-hash randomization. Pin PYTHONHASHSEED so the same seed always yields
# the same (graph, query, candidates). Re-exec once if it isn't already fixed.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from algorithm_evaluation.graph_generators.erdos_renyi import generate_er_graph
from algorithm_evaluation.get_useless import sample_experiments, sample_query

from autobound.causalProblem import causalProblem
from autobound.DAG import DAG

# ============================================================================
#  Instance configuration.  `instances.py` overrides the run-specific entries.
# ============================================================================
config = dict(
    run_id="er5",
    seed=42,
    n_vars=5, p="log(N)/N",   # q = c/N, set via `c` below
    dirichlet_alpha=1.0, p_min=1e-4,
    n_candidates=20,
    w_size=(1, 1), z_size=(1, 1), candidate_class="fixed_z",
    grid_points=20,
    witness_tol=1e-6,
    solver_timeout_s=60,
    w_empty_timeout_s=20,
    candidate_cap_s=720,
    n_workers=1,
    max_hours=1.5,
    # instance guards
    w_empty_min=0.05,       # regenerate if the query is (near-)identified
    require_interception=False,
    max_indegree=3,         # regenerate: a node with >3 observed parents blows up
                            # the canonical model
    max_group_params=20000, # regenerate: a c-component this large is intractable
    max_regen=200,
    # ER density: q = c / N (set `c`); p = log(N)/N
    c=1.5,
)

PV_PATH = HERE / "er5_pv.csv"      # instances.py redirects this per instance


# ============================================================================
#  Instance construction: ER graph -> autobound DAG + Dirichlet-pushforward P(V)
# ============================================================================
def resolve_pq(config):
    N = config['n_vars']
    p = math.log(N) / N if config['p'] == "log(N)/N" else float(config['p'])
    q = config['c'] / N if 'c' in config else float(config['q'])   # bidirected density c/N
    return p, q


def to_autobound(cg):
    """Map CausalGraph -> (structure_str, unob_str, name_map). Observed vars are
    renamed to single uppercase letters (autobound concatenates var name + digits)."""
    letters = [chr(ord('A') + i) for i in range(len(cg.observed))]
    name_map = {v: letters[i] for i, v in enumerate(cg.observed)}
    edges = [f"{name_map[a]}->{name_map[b]}" for a, b in cg.directed_edges]
    u_terms, u_names = [], []
    for i, (u, (v1, v2)) in enumerate(cg.hidden_children.items()):
        un = f"U{i}"
        u_names.append(un)
        u_terms += [f"{un}->{name_map[v1]}", f"{un}->{name_map[v2]}"]
    structure = ", ".join(edges + u_terms)
    return structure, ",".join(u_names), name_map, letters


def indegree(cg, v):
    return sum(1 for a, b in cg.directed_edges if b == v)


def free_first_nodes(problem, letters):
    """Roots with no confounder: autobound gives them empty-monomial queries; their
    marginal is free/data-supplied and must be injected separately in the pushforward."""
    free = []
    for L in letters:
        if problem.query(f"{L}=1")._lst == [(1, [])]:
            free.append(L)
    return free


def _resp_seg(seg):
    m = re.match(r'^(.*?)(\d*)$', seg)
    return m.group(1), m.group(2)


def sample_group_q(group, dag, alpha, rng, latent_states=2):
    """Response-type joint for ONE district, drawn from independent latents.

        q(R_D = r) = sum_u prod_e p_e(u_e) prod_v theta_v(r_v | u_pa(v))

    One independent latent per bidirected edge, plus private per-node
    randomness theta_v.  Every independence the ADMG implies (two nodes of a
    district sharing no latent parent) therefore holds by construction.  A flat
    Dirichlet over the district's joint response types -- what this module did
    before -- imposes none of them, so the resulting P(V) is off-model whenever
    the ADMG implies one, and no model of that ADMG can reproduce it.
    """
    decoded = []
    for pname in group:
        d = {}
        for seg in pname.split('.'):
            name, val = _resp_seg(seg)
            d[name] = val
        decoded.append((pname, d))
    nodes = sorted(decoded[0][1])
    vals = {v: sorted({d[v] for _, d in decoded}) for v in nodes}
    lat_of = {v: sorted(dag.find_parents_u(v)) for v in nodes}
    latents = sorted({u for v in nodes for u in lat_of[v]})

    p_lat = {u: rng.dirichlet(alpha * np.ones(latent_states)) for u in latents}
    theta = {}
    for v in nodes:
        for a in itertools.product(range(latent_states), repeat=len(lat_of[v])):
            theta[(v, a)] = rng.dirichlet(alpha * np.ones(len(vals[v])))

    q = {pname: 0.0 for pname, _ in decoded}
    for full in itertools.product(range(latent_states), repeat=len(latents)):
        u = dict(zip(latents, full))
        w = 1.0
        for uu in latents:
            w *= p_lat[uu][u[uu]]
        if w == 0.0:
            continue
        for pname, d in decoded:
            m = w
            for v in nodes:
                a = tuple(u[x] for x in lat_of[v])
                m *= theta[(v, a)][vals[v].index(d[v])]
            q[pname] += m
    return q


def sample_pv(problem, letters, alpha, p_min, rng, max_tries=60):
    """Dirichlet pushforward: independent Dirichlet per c-component over its canonical
    response functions (+ independent Bernoulli per free first-node). P(V) is feasible
    for the autobound program by construction. Returns dict cell(tuple)->prob.

    The per-cell monomial expansion is parsed ONCE (autobound's parser is the costly part)
    and cached; each try only re-samples the Dirichlets and re-evaluates the cached
    monomials. p_min is a soft preference: we return the best-min-cell sample after
    `max_tries` (the pushforward is exact/feasible even if a cell dips below p_min)."""
    groups = problem.Parser.c_parameters
    free = free_first_nodes(problem, letters)
    non_free = [L for L in letters if L not in free]
    # parse the canonical monomials for each non-free sub-assignment ONCE
    nf_assigns = list(itertools.product([0, 1], repeat=len(non_free)))
    mono = {}
    for nf in nf_assigns:
        cmap = dict(zip(non_free, nf))
        expr = "&".join(f"{L}={cmap[L]}" for L in non_free) if non_free else "1"
        mono[nf] = [(coef, list(params)) for coef, params in problem.query(expr)._lst]
    cells = list(itertools.product([0, 1], repeat=len(letters)))
    best, best_min = None, -1.0
    for _ in range(max_tries):
        val = {}
        for g in groups:
            val.update(sample_group_q(g, problem.dag, alpha, rng))
        pi = {L: float(rng.dirichlet(alpha * np.ones(2))[1]) for L in free}
        out = {}
        for cell in cells:
            cmap = dict(zip(letters, cell))
            nf = tuple(cmap[L] for L in non_free)
            canon = 0.0
            for coef, params in mono[nf]:
                m = coef
                for pnm in params:
                    m *= val.get(pnm, 1.0 if pnm == '1' else 0.0)
                canon += m
            ff = 1.0
            for L in free:
                ff *= pi[L] if cmap[L] == 1 else (1 - pi[L])
            out[cell] = canon * ff
        total = sum(out.values())
        assert abs(total - 1.0) < 1e-9, f"P(V) sums to {total}, not 1"
        mn = min(out.values())
        if mn >= p_min:
            return out
        if mn > best_min:
            best_min, best = mn, out
    return best
    return None


def build_instance(config, base_seed):
    """Sample a valid (graph, query, P(V)) instance, regenerating on guard failures.
    Returns dict with structure, name_map, letters, query worlds, theta_str, pv cells,
    cg, and the chosen seed."""
    p, q = resolve_pq(config)
    N = config['n_vars']
    theta_config = dict(CF_fraction=1.0, W_sizes=[1], Z_sizes=[1],
                        intervention_outcome_distances=[1, 2, 3, 4, 5])
    for k in range(config['max_regen']):
        seed = base_seed + k
        cg, name = generate_er_graph(N, p, q, seed=seed)
        if not cg.hidden_children:
            continue                                    # guard: no confounder
        if any(indegree(cg, v) > config['max_indegree'] for v in cg.observed):
            continue                                    # guard: canonical-model blowup
        structure, unob, name_map, letters = to_autobound(cg)
        problem = causalProblem(DAG_from(structure, unob))
        if any(len(g) > config['max_group_params'] for g in problem.Parser.c_parameters):
            continue                                    # guard: intractable c-component
        # query
        qrng = np.random.default_rng(seed ^ 0x5DEECE66D)
        qworlds = None
        for _ in range(200):
            cand = sample_query(cg, theta_config, qrng)
            if cand and len(cand) == 2:
                qworlds = cand
                break
        if qworlds is None:
            continue
        theta_str = estimand_str(qworlds, name_map)
        # P(V)
        pv = sample_pv(problem, letters, config['dirichlet_alpha'], config['p_min'],
                       np.random.default_rng(seed ^ 0xABCD1234))
        if pv is None:
            continue
        write_pv(pv, letters, PV_PATH)
        return dict(cg=cg, name=name, structure=structure, unob=unob,
                    name_map=name_map, letters=letters, qworlds=qworlds,
                    theta_str=theta_str, seed=seed)
    raise RuntimeError("could not build a valid instance within max_regen")


def DAG_from(structure, unob):
    d = DAG(); d.from_structure(structure, unob=unob); return d


def estimand_str(qworlds, name_map):
    """Two-world counterfactual: outcomes=1 in both worlds; world-1 interventions=1,
    world-2 interventions=0.  e.g. 'E(B=1)=1&E(B=0)=1'."""
    def term(world, ival):
        Y, X = world
        xs = "&".join(f"{name_map[x]}={ival}" for x in X)
        return "&".join(f"{name_map[y]}({xs})=1" for y in Y)
    return term(qworlds[0], 1) + "&" + term(qworlds[1], 0)


def write_pv(pv_cells, letters, path):
    rows = [{**dict(zip(letters, cell)), 'prob': prob} for cell, prob in pv_cells.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


# ============================================================================
#  Candidate sampling (single-world |W|=|Z|=1 + intervention value z)
# ============================================================================
def sample_candidates(cg, name_map, n, rng, cf_fraction=0.0):
    """Sample scalar candidate experiments (deduped).

    Two classes, mixed by `cf_fraction`:
      * fixed_z (single-world): p_a = P(W=1 | do(Z=z)), pruning experiment
        ((W,Z),).  Can be interception- or ID-pruned.
      * cf (two-world counterfactual scalar): p_a = P(W(Z=0)=1 & W(Z=1)=0),
        pruning experiment ((W,Z),(W,Z)).  The ID check is undefined for a
        two-world experiment, so a cf candidate is never ID-pruned.
    """
    sw_config = dict(CF_fraction=0.0, W_sizes=[1], Z_sizes=[1])
    seen, cands = set(), []
    tries = 0
    while len(cands) < n and tries < n * 400:
        tries += 1
        exps = sample_experiments(cg, sw_config, 1, rng)
        if not exps:
            continue
        world = exps[0][0]
        W, Z = list(world[0]), list(world[1])
        if len(W) != 1 or len(Z) != 1:
            continue
        Wl, Zl = name_map[W[0]], name_map[Z[0]]
        make_cf = rng.random() < cf_fraction
        if make_cf:
            key = ('cf', W[0], Z[0])
            if key in seen:
                continue
            seen.add(key)
            cands.append(dict(
                kind='cf', W=frozenset(W), Z=frozenset(Z), z_value='cf',
                W_str=W[0], Z_str=Z[0],
                param_str=f"{Wl}({Zl}=0)=1&{Wl}({Zl}=1)=0",       # cross-world stratum
                worlds=(world, world)))       # two-world experiment for pruning
        else:
            z_value = int(rng.integers(2))
            key = ('fz', W[0], Z[0], z_value)
            if key in seen:
                continue
            seen.add(key)
            cands.append(dict(
                kind='fixed_z', W=frozenset(W), Z=frozenset(Z), z_value=z_value,
                W_str=W[0], Z_str=Z[0],
                param_str=f"{Wl}({Zl}={z_value})=1",
                worlds=(world,)))             # single-world experiment for pruning
    return cands
