"""Deterministic reconstruction of the runtime benchmark instances.

Given `PYTHONHASHSEED=0`, an instance is fully determined by its density
parameter `c` and its base seed: it is
`P.build_instance(base_cfg(5, c), base_seed=seed)`, and its candidate list is
`P.sample_candidates(cg, name_map, 80, default_rng((42 ^ 0xC0FFEE) + seed),
cf_fraction=0.0)`.  `verify()` checks the reconstruction against the shipped
`results/single_candidates.csv`, so a re-run measures exactly the same
26 instances / 152 candidates.

Each instance's P(V) is written to `pv/`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SUPP_ROOT = HERE.parent
PV_DIR = HERE / "pv"
PV_DIR.mkdir(exist_ok=True)
RESULTS = HERE / "results"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SUPP_ROOT))
import pruned_vs_unpruned as P  # noqa: E402
from algorithm_evaluation.get_useless import (  # noqa: E402
    get_useless_experiments_separate)

# ---- benchmark configuration -------------------------------------------
HEADLINE = dict(
    n_vars=5, c_list=[0.5, 1.0], cf_fraction=0.0, n_candidate_target=80,
    grid_points=50, solver_timeout_s=60, w_empty_timeout_s=20,
    candidate_cap_s=720, witness_tol=1e-3, w_empty_min=0.05, dirichlet_alpha=1.0,
    p_min=1e-4, max_indegree=3, max_group_params=5000, max_regen=400, seed=42,
)

# The 26 instances of the benchmark: (c, base seed), in run order.
INSTANCES = [
    (0.5, 42), (0.5, 48), (0.5, 53), (0.5, 57), (0.5, 74),
    (1.0, 42), (1.0, 48), (1.0, 57), (1.0, 81), (1.0, 86),
    (0.5, 49), (0.5, 52), (0.5, 61), (0.5, 63), (0.5, 70),
    (0.5, 87), (0.5, 93), (0.5, 147),
    (1.0, 47), (1.0, 49), (1.0, 50), (1.0, 52), (1.0, 53),
    (1.0, 61), (1.0, 63), (1.0, 70),
]


def base_cfg(N, c):
    import copy
    cfg = copy.deepcopy(P.config)
    cfg.update(n_vars=N, c=c, dirichlet_alpha=HEADLINE['dirichlet_alpha'],
               p_min=HEADLINE['p_min'], max_indegree=HEADLINE['max_indegree'],
               max_group_params=HEADLINE['max_group_params'],
               max_regen=HEADLINE['max_regen'], grid_points=HEADLINE['grid_points'],
               witness_tol=HEADLINE['witness_tol'],
               solver_timeout_s=HEADLINE['solver_timeout_s'],
               w_empty_timeout_s=HEADLINE['w_empty_timeout_s'],
               candidate_cap_s=HEADLINE['candidate_cap_s'],
               w_empty_min=HEADLINE['w_empty_min'], seed=HEADLINE['seed'])
    return cfg


def pv_path(c, seed):
    return PV_DIR / f"pv_c{c}_s{seed}.csv"


def get_instance(c, seed):
    """Rebuild instance (c, seed) and its classified candidate list."""
    cfg = base_cfg(HEADLINE['n_vars'], c)
    P.PV_PATH = pv_path(c, seed)                      # redirect BEFORE generating
    inst = P.build_instance(cfg, base_seed=seed)
    assert inst['seed'] == seed, (
        f"build_instance(base_seed={seed}) returned seed {inst['seed']}")
    crng = np.random.default_rng((cfg['seed'] ^ 0xC0FFEE) + seed)
    cands = P.sample_candidates(inst['cg'], inst['name_map'],
                                HEADLINE['n_candidate_target'], crng,
                                cf_fraction=HEADLINE['cf_fraction'])
    per_exp, _ = get_useless_experiments_separate(
        inst['cg'], inst['qworlds'], [x['worlds'] for x in cands])
    for x, pe in zip(cands, per_exp):
        x['pruned'] = 'interception' if pe['blocked'] else ('id' if pe['id'] else 'none')
    inst['pv_path'] = P.PV_PATH
    return inst, cands


def timed_pruning_pass(c, seed):
    """Time the graph-only pruning pass (GetUselessExperiments) alone."""
    import time
    inst, _ = get_instance(c, seed)
    cfg = base_cfg(HEADLINE['n_vars'], c)
    crng = np.random.default_rng((cfg['seed'] ^ 0xC0FFEE) + seed)
    cands = P.sample_candidates(inst['cg'], inst['name_map'],
                                HEADLINE['n_candidate_target'], crng,
                                cf_fraction=HEADLINE['cf_fraction'])
    exps = [x['worlds'] for x in cands]
    t0 = time.time()
    get_useless_experiments_separate(inst['cg'], inst['qworlds'], exps)
    ms = (time.time() - t0) * 1000.0
    return len(cands), ms


def verify(verbose=True):
    """Check the reconstruction against results/single_candidates.csv."""
    ref = pd.read_csv(RESULTS / "single_candidates.csv")
    problems = []
    for c, seed in INSTANCES:
        inst, cands = get_instance(c, seed)
        g = ref[(ref['c'] == c) & (ref['seed'] == seed)].sort_values('candidate_id')
        if len(g) != len(cands):
            problems.append(f"c={c} seed={seed}: {len(cands)} rebuilt vs {len(g)} recorded")
            continue
        for (_, r), x in zip(g.iterrows(), cands):
            if r['param_str'] != x['param_str'] or r['pruned'] != x['pruned']:
                problems.append(
                    f"c={c} seed={seed} cid={r['candidate_id']}: "
                    f"recorded ({r['param_str']},{r['pruned']}) != "
                    f"rebuilt ({x['param_str']},{x['pruned']})")
        if verbose:
            print(f"  c={c} seed={seed}: {len(cands)} candidates OK  theta={inst['theta_str']}")
    return problems


if __name__ == "__main__":
    print("verifying instance/candidate reconstruction...")
    probs = verify()
    if probs:
        print(f"\n{len(probs)} MISMATCH(es):")
        for p in probs[:40]:
            print("  " + p)
        sys.exit(1)
    n = sum(1 for _ in INSTANCES)
    print(f"\nall {n} instances / 152 candidates reproduce exactly.")
