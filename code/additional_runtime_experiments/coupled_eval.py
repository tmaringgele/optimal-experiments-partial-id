"""Coupled single-program potency evaluator for arbitrary ADMGs and estimands.

`../coupled_potency.py` implements Algorithm EvaluatePotency for the NHANES ATE
example: a fixed DAG, a fixed data file, and `causalProblem.set_ate('X','Y')` as
the target functional.  The runtime harness here instead uses

  * arbitrary Erdos-Renyi ADMGs (usually MULTI-district, so the response-type
    probabilities appear in products and the programs are genuinely non-convex
    -- the NHANES model is a single district and therefore linear),
  * an arbitrary two-world counterfactual estimand string, set via
    `causalProblem.set_estimand(problem.query(theta_str))`, and
  * `fixed_z`-style candidate experiments, e.g. `E(C=1)=1`.

This module keeps every piece of `coupled_potency` that does real work
(`namespaced_program`, `query_terms`, `parse_couenne_log`, `solve`) and only
generalises the problem construction.  In particular `parse_couenne_log`'s
behaviour is preserved verbatim: couenne sometimes prints "Infeasible problem"
after proving a 0% gap, so the *printed* bounds, not pyomo's termination status,
decide whether an answer is trustworthy.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

SUPP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUPP_ROOT))

import coupled_potency as CP  # noqa: E402

from autobound.causalProblem import causalProblem  # noqa: E402
from autobound.DAG import DAG  # noqa: E402


# --------------------------------------------------------------------------
# problem construction (generalised)
# --------------------------------------------------------------------------

def dag_from(structure, unob):
    d = DAG()
    d.from_structure(structure, unob=unob)
    return d


def build_problem(dag, pv_path, estimand_str):
    """P_empty = (T, C_empty) for an arbitrary ADMG / estimand / P(V)."""
    problem = causalProblem(dag)
    problem.load_data(str(pv_path))
    problem.add_prob_constraints()
    problem.set_estimand(problem.query(estimand_str))
    return problem


# Response-type independence constraints are part of the model, not an option;
# without them the program is a relaxation that merges each district into a
# single latent.  Settable to 0 only to reproduce the relaxation for comparison.
INDEPENDENCE = os.environ.get('INDEPENDENCE', '1') != '0'


def base_program(dag, pv_path, estimand_str):
    prob = build_problem(dag, pv_path, estimand_str)
    prog = prob.write_program()
    params, cons = list(prog.parameters), [list(c) for c in prog.constraints]
    if INDEPENDENCE:
        CP.add_independence_constraints(prob, params, cons)
    return params, cons


def build_coupled(dag, pv_path, estimand_str, experiment_queries,
                  prefix_u='U_', prefix_l='L_'):
    """The single coupled program of Algorithm EvaluatePotency.

        max  T(R^U) - T(R^L)
        s.t. C_empty^U, C_empty^L,
             f_at(R^U) = f_at(R^L)   for every experiment at in a.

    Returns (parameters, constraints, obj_terms).
    """
    prob_u = build_problem(dag, pv_path, estimand_str)
    prob_l = build_problem(dag, pv_path, estimand_str)

    # coupling terms BEFORE write_program() (which mutates constraints in place)
    coupling = []
    for expr in experiment_queries:
        terms_u = CP.query_terms(prob_u, expr, prefix_u, sign=1)
        terms_l = CP.query_terms(prob_l, expr, prefix_l, sign=-1)
        coupling.append(terms_u + terms_l + [['==']])

    params_u, cons_u = CP.namespaced_program(prob_u, prefix_u)
    params_l, cons_l = CP.namespaced_program(prob_l, prefix_l)
    if INDEPENDENCE:
        CP.add_independence_constraints(prob_u, params_u, cons_u, prefix_u)
        CP.add_independence_constraints(prob_l, params_l, cons_l, prefix_l)

    parameters = params_u + params_l
    constraints = cons_u + cons_l + coupling
    obj_terms = [[prefix_u + 'objvar'], ['-1', prefix_l + 'objvar']]
    return parameters, constraints, obj_terms


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def observational_width(dag, pv_path, estimand_str, time_limit=600, workdir=None):
    """W_theta(empty) = max T - min T on the base program (two solves)."""
    params, cons = base_program(dag, pv_path, estimand_str)
    obj = [['objvar']]
    t0 = time.time()
    hi = CP.solve(params, cons, obj, 'max', time_limit=time_limit, workdir=workdir)
    lo = CP.solve(params, cons, obj, 'min', time_limit=time_limit, workdir=workdir)
    wall = time.time() - t0
    ok = hi.get('ok') and lo.get('ok')
    return {
        'lb': lo.get('value'), 'ub': hi.get('value'),
        'W_empty': (hi['value'] - lo['value']) if ok else None,
        'ok': bool(ok), 'wall': wall, 'programs': 2,
        'n_vars': len(params), 'n_constraints': len(cons),
        'max_run': hi, 'min_run': lo,
    }


def evaluate_potency(dag, pv_path, estimand_str, experiment_queries, W_empty,
                     time_limit=600, workdir=None):
    """ONE coupled program -> W_theta(a) and pot(a) = W_theta(empty) - W_theta(a)."""
    t0 = time.time()
    params, cons, obj = build_coupled(dag, pv_path, estimand_str, experiment_queries)
    build_s = time.time() - t0
    run = CP.solve(params, cons, obj, 'max', objvar_bounds=(-1.0, 1.0),
                   time_limit=time_limit, workdir=workdir)
    W_a = run.get('value')
    ok = bool(run.get('ok'))
    pot = None
    if ok and W_a is not None:
        # W_a <= W_empty in theory (a constraint can only shrink the feasible
        # set); clamp couenne's tolerance-level overshoot.
        W_a = min(W_a, W_empty)
        pot = W_empty - W_a
    return {
        'experiments': list(experiment_queries),
        'W_a': W_a, 'potency': pot, 'ok': ok,
        'programs': 1,
        'n_vars': len(params), 'n_constraints': len(cons),
        'build_seconds': build_s,
        'wall': build_s + run.get('wall', run.get('seconds', 0.0)),
        'solve_wall': run.get('wall', run.get('seconds', 0.0)),
        'status': run.get('status'), 'termination': run.get('termination'),
        'proved_by_log': run.get('proved_by_log'),
        'couenne_lb': run.get('couenne_lb'), 'couenne_ub': run.get('couenne_ub'),
        'error': run.get('error'),
    }
