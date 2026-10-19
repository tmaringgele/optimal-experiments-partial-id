"""Single-program (coupled) implementation of EvaluatePotency.

The worst-case post-experimental width

    W_theta(a) = max_{p_a} [ U_theta(a, p_a) - L_theta(a, p_a) ]

collapses into ONE polynomial program.  Unfold the two inner bounds as two
independent optimizers P^U and P^L, both constrained by C_empty and both
required to reproduce the same experimental outcome.  Then p_a appears only as
the common value of f(P^U) and f(P^L); dropping it as a decision variable and
replacing its two binding constraints by the single coupling equality
f(P^U) = f(P^L) gives

    W_theta(a) = max [ T(R^U) - T(R^L) ]
                 s.t.  C_empty^U,  C_empty^L,
                       f_at(R^U) = f_at(R^L)  for every at in a.

This module builds that program on top of `autobound`'s `causalProblem` /
`Program` representation and solves it with couenne through pyomo.

Representation notes (autobound internals)
------------------------------------------
`causalProblem.write_program()` returns a `Program` whose
  * `parameters` is a list of response-type probability names (plus 'objvar'),
  * `constraints` is a list of constraints; each constraint is a list of terms
    followed by a final one-element list holding the relational symbol
    (e.g. ['==']).  A term is a list of strings: an optional leading numeric
    coefficient followed by the parameter names that are multiplied together.
    A constraint asserts   sum(terms)  <symbol>  0.

Namespacing a program therefore only requires renaming the strings that are
parameter names, leaving numeric coefficients alone.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
import time
from copy import deepcopy

from autobound.DAG import DAG
from autobound.Query import Query
from autobound.causalProblem import (
    causalProblem,
    replace_first_nodes,
    transform_constraint,
)

# --------------------------------------------------------------------------
# NHANES ATE problem
# --------------------------------------------------------------------------

NHANES_STRUCTURE = 'Z->X, X->Y, U1->Z, U1->X, U2->X, U2->Y'
NHANES_UNOB = 'U1,U2'


def nhanes_dag() -> DAG:
    dag = DAG()
    dag.from_structure(NHANES_STRUCTURE, unob=NHANES_UNOB)
    return dag


def build_base_problem(data_path, dag=None, extra_constraints=None):
    """Base causalProblem P_empty = (T, C_empty) for the NHANES ATE example.

    `extra_constraints` is a list of (query_string, value) pairs.
    """
    problem = causalProblem(dag if dag is not None else nhanes_dag())
    problem.load_data(str(data_path))
    problem.add_prob_constraints()
    problem.set_ate('X', 'Y')
    if extra_constraints:
        for lhs_q, rhs_val in extra_constraints:
            problem.add_constraint(problem.query(lhs_q) - Query(float(rhs_val)))
    return problem


# --------------------------------------------------------------------------
# Namespacing helpers
# --------------------------------------------------------------------------

def _rename_terms(terms, params, prefix):
    """Prefix every string in `terms` that is a parameter name."""
    out = []
    for term in terms:
        out.append([prefix + s if s in params else s for s in term])
    return out


def namespaced_program(problem, prefix):
    """Return (parameters, constraints) of `problem`'s program, namespaced.

    Every response-type probability P(r) (and 'objvar') gets `prefix` prepended
    so that two copies of the same program can live in one model.
    """
    prog = problem.write_program()
    params = set(prog.parameters)
    new_params = [prefix + p for p in prog.parameters]
    new_cons = []
    for c in prog.constraints:
        terms, symbol = c[:-1], c[-1]
        new_cons.append(_rename_terms(terms, params, prefix) + [symbol])
    return new_params, new_cons


def query_terms(problem, expr, prefix, sign=1):
    """Transformed + namespaced terms of the functional f_expr(R).

    Mirrors what `causalProblem.add_constraint` / `write_program` would do to
    `problem.query(expr)`, so the resulting terms are in `Program` format.
    """
    q = problem.query(expr, sign)
    raw = [[coef, list(varlist)] for coef, varlist in q._lst]
    raw = [replace_first_nodes(problem.unconf_first_nodes, t) for t in raw]
    zero_parameters = [x[1] for x in problem.parameters if x[0] == 0]
    terms = transform_constraint(raw, zero_parameters)
    params = set(x[1] for x in problem.parameters if x[0] == 1)
    return _rename_terms(terms, params, prefix)


# --------------------------------------------------------------------------
# Response-type independence constraints
# --------------------------------------------------------------------------

def _split_segment(seg):
    """'X01' -> ('X', '01');  'mY00' -> ('mY', '00')."""
    m = re.match(r'^(.*?)(\d*)$', seg)
    return m.group(1), m.group(2)


def response_marginal(params, var, prefix=''):
    """Map each value of R_`var` to the parameters that sum to q(R_var = value).

    Parameter names are dotted response-type products, e.g. 'X01.Y10.Z0'; under
    a namespaced copy the first segment carries `prefix`.  Returns {} if `var`
    has no response-type variable in the program (e.g. it was substituted out
    as an unconfounded first node).
    """
    groups = {}
    for prm in params:
        if prefix and not prm.startswith(prefix):
            continue
        if prm.endswith('objvar'):
            continue
        bare = prm[len(prefix):] if prefix else prm
        for seg in bare.split('.'):
            name, digits = _split_segment(seg)
            if name == var:
                # Parameters of the other districts factor separately and are
                # not part of this marginal, so they are skipped rather than
                # treated as a missing response type.
                groups.setdefault(digits, []).append(prm)
                break
    return groups


def add_independence_constraints(problem, parameters, constraints, prefix=''):
    """Impose q(R_u, R_v) = q(R_u) q(R_v) for every independent response pair.

    Within a district, two response-type variables are independent exactly when
    their nodes share no latent parent -- `causalProblem.check_indep` returns
    those pairs.  The district-level response-type parameterization does not
    encode this, so without these constraints the program is a relaxation: it
    admits a single latent confounding the whole district, and the bounds it
    returns are valid but not sharp for the ADMG that was specified.

    Mutates `parameters` and `constraints` in place; returns the number of
    bilinear constraints added.  Auxiliary marginal variables are introduced so
    each product stays bilinear rather than a product of two long sums.
    """
    added = 0
    for comp in problem.canModel.c_comp:
        for pair in problem.check_indep(comp):
            u, v = sorted(pair)
            gu = response_marginal(parameters, u, prefix)
            gv = response_marginal(parameters, v, prefix)
            if not gu or not gv:
                continue
            for var, groups in ((u, gu), (v, gv)):
                for val in sorted(groups):
                    aux = f'{prefix}m_{var}{val}'
                    if aux in parameters:
                        continue
                    parameters.append(aux)
                    constraints.append([[aux]]
                                       + [['-1', q] for q in groups[val]]
                                       + [['==']])
            for uval in sorted(gu):
                su = set(gu[uval])
                for vval in sorted(gv):
                    joint = su.intersection(gv[vval])
                    constraints.append(
                        [[f'{prefix}m_{u}{uval}', f'{prefix}m_{v}{vval}']]
                        + [['-1', q] for q in sorted(joint)] + [['==']])
                    added += 1
    return added


# --------------------------------------------------------------------------
# The coupled program
# --------------------------------------------------------------------------

class CoupledProgram:
    """The single program of Algorithm EvaluatePotency."""

    def __init__(self, parameters, constraints, obj_terms,
                 objvar_bounds=(-1.0, 1.0)):
        self.parameters = parameters
        self.constraints = constraints
        self.obj_terms = obj_terms          # terms of T(R^U) - T(R^L)
        self.objvar_bounds = objvar_bounds

    @property
    def n_vars(self):
        return len(self.parameters)

    @property
    def n_constraints(self):
        return len(self.constraints)


def build_coupled_program(data_path, experiment_queries, dag=None,
                          prefix_u='U_', prefix_l='L_'):
    """Build the coupled single program for the experiment set `a`.

    `experiment_queries` is a list of autobound query strings, one per
    experiment in the set.
    """
    prob_u = build_base_problem(data_path, dag=dag)
    prob_l = build_base_problem(data_path, dag=dag)

    # Coupling terms must be computed BEFORE write_program(), which mutates
    # the problem's constraints in place.
    coupling = []
    for expr in experiment_queries:
        terms_u = query_terms(prob_u, expr, prefix_u, sign=1)
        terms_l = query_terms(prob_l, expr, prefix_l, sign=-1)
        coupling.append(terms_u + terms_l + [['==']])

    params_u, cons_u = namespaced_program(prob_u, prefix_u)
    params_l, cons_l = namespaced_program(prob_l, prefix_l)
    add_independence_constraints(prob_u, params_u, cons_u, prefix_u)
    add_independence_constraints(prob_l, params_l, cons_l, prefix_l)

    parameters = params_u + params_l
    constraints = cons_u + cons_l + coupling

    # objective: T(R^U) - T(R^L) = objvar^U - objvar^L
    obj_terms = [[prefix_u + 'objvar'], ['-1', prefix_l + 'objvar']]
    return CoupledProgram(parameters, constraints, obj_terms)


# --------------------------------------------------------------------------
# pyomo / couenne driver
# --------------------------------------------------------------------------

def _term_expr(term, param_set, M):
    import pyomo.environ as pyo  # noqa: F401  (kept for clarity)
    from functools import reduce
    return reduce(lambda a, b: a * b,
                  [getattr(M, s) if s in param_set else float(s) for s in term])


def _build_pyomo_model(parameters, constraints, obj_terms, sense,
                       objvar_bounds=(-1.0, 1.0), initial=None):
    import pyomo.environ as pyo
    from autobound.Program import pyomo_symb

    M = pyo.ConcreteModel()
    param_set = set(parameters)
    initial = initial or {}
    for p in parameters:
        kw = {'initialize': initial[p]} if p in initial else {}
        if p.endswith('objvar'):
            setattr(M, p, pyo.Var(bounds=objvar_bounds, **kw))
        else:
            setattr(M, p, pyo.Var(bounds=(0, 1), **kw))
    for i, c in enumerate(constraints):
        terms, symbol = c[:-1], c[-1]
        setattr(M, 'c' + str(i), pyo.Constraint(
            expr=pyomo_symb[symbol[0]](
                sum(_term_expr(t, param_set, M) for t in terms), 0)))
    obj_expr = sum(_term_expr(t, param_set, M) for t in obj_terms)
    M.obj = pyo.Objective(expr=obj_expr,
                          sense=pyo.maximize if sense == 'max' else pyo.minimize)
    return M


GOOD_TERMINATIONS = {'optimal', 'locallyOptimal', 'globallyOptimal', 'feasible'}


def parse_couenne_log(text):
    """Extract couenne's own proven bounds from its stdout.

    couenne sometimes ends a run it has actually solved to a 0% gap with
    'Infeasible problem' in the .sol file: its root NLP incumbent is rejected
    by the feasibility tolerance, so cbc finishes with no accepted incumbent
    even though the lower and upper bound have met.  pyomo then reports
    termination_condition 'infeasible'.  The printed bounds are the reliable
    signal, so we read them back.
    """
    out = {'couenne_lb': None, 'couenne_ub': None, 'couenne_finished': False}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('Lower bound:'):
            try:
                out['couenne_lb'] = float(s.split(':')[1].split('(')[0])
            except ValueError:
                pass
        elif s.startswith('Upper bound:'):
            try:
                out['couenne_ub'] = float(s.split(':')[1].split('(')[0])
            except ValueError:
                pass
        elif '"Finished"' in s:
            out['couenne_finished'] = True
    lb, ub = out['couenne_lb'], out['couenne_ub']
    out['couenne_gap'] = (abs(ub - lb) if (lb is not None and ub is not None)
                          else None)
    return out


def _solve_worker(parameters, constraints, obj_terms, sense, objvar_bounds,
                  time_limit, verbose, workdir, initial, q):
    try:
        import contextlib
        import io
        import tempfile

        import pyomo.environ as pyo
        os.chdir(workdir)
        # couenne reads its options from ./couenne.opt
        with open('couenne.opt', 'w') as f:
            f.write(f'time_limit {time_limit}\n')
        M = _build_pyomo_model(parameters, constraints, obj_terms, sense,
                               objvar_bounds, initial=initial)
        solver = pyo.SolverFactory('couenne')

        # Capture couenne's stdout (fd level: it is a subprocess) so its own
        # proven bounds can be read back, see parse_couenne_log.
        logfile = tempfile.NamedTemporaryFile('w+', suffix='.couenne.log',
                                              delete=False)
        # pyomo's tee also echoes to sys.stdout, which is not fd 1 under a
        # notebook kernel; capture both so nothing leaks into cell output.
        buf = io.StringIO()
        saved = os.dup(1)
        os.dup2(logfile.fileno(), 1)
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(buf):
                res = solver.solve(M, tee=True)
        finally:
            elapsed = time.time() - t0
            os.dup2(saved, 1)
            os.close(saved)
            logfile.flush()
            logfile.seek(0)
            log_text = open(logfile.name).read() + buf.getvalue()
            os.unlink(logfile.name)
        if verbose:
            print(log_text)
        term = str(res.solver.termination_condition)
        try:
            value = float(pyo.value(M.obj))
        except Exception:
            value = None
        parsed = parse_couenne_log(log_text)
        # couenne proved optimality iff it finished with a closed gap; the
        # objective value read back then matches that bound.
        proved = (parsed['couenne_finished']
                  and parsed['couenne_gap'] is not None
                  and parsed['couenne_gap'] <= 1e-6)
        out = {
            'value': value,
            'status': str(res.solver.status),
            'termination': term,
            'seconds': elapsed,
            # `ok` means: the answer is trustworthy.  couenne reports
            # 'infeasible' on programs that are feasible by construction
            # (rejected root incumbent), so we also accept a run whose own
            # printed bounds closed.
            'ok': (value is not None)
                  and (term in GOOD_TERMINATIONS or proved),
            'proved_by_log': proved,
        }
        out.update(parsed)
        q.put(out)
    except Exception as exc:  # pragma: no cover
        q.put({'ok': False, 'error': f'{type(exc).__name__}: {exc}',
               'status': 'error', 'termination': 'error'})


def solve(parameters, constraints, obj_terms, sense='max',
          objvar_bounds=(-1.0, 1.0), time_limit=300, verbose=False,
          workdir=None, initial=None):
    """Solve one program with couenne, hard-killed after `time_limit` seconds.

    Returns a dict with value/status/termination/seconds (or an error entry).
    `initial` optionally warm-starts the variables.
    """
    workdir = workdir or os.getcwd()
    ctx = mp.get_context('fork')
    q = ctx.Queue()
    p = ctx.Process(target=_solve_worker,
                    args=(parameters, constraints, obj_terms, sense,
                          objvar_bounds, time_limit, verbose, workdir,
                          initial, q))
    t0 = time.time()
    p.start()
    p.join(time_limit + 30)          # grace period on top of couenne's own limit
    wall = time.time() - t0
    if p.is_alive():
        p.terminate()
        p.join()
        return {'ok': False, 'status': 'killed',
                'termination': 'wallclock_timeout', 'seconds': wall,
                'error': f'killed after {wall:.1f}s'}
    out = q.get() if not q.empty() else {
        'ok': False, 'status': 'no_result', 'termination': 'no_result',
        'seconds': wall, 'error': 'worker produced no result'}
    out['wall'] = wall
    return out


# --------------------------------------------------------------------------
# Parallel batch solving (grid sweeps)
# --------------------------------------------------------------------------

_BATCH = {}


def _batch_init(data_path, structure, unob, time_limit):
    import tempfile
    _BATCH.update(data_path=data_path, structure=structure, unob=unob,
                  time_limit=time_limit,
                  workdir=tempfile.mkdtemp(prefix='cp_batch_'))


def _batch_one(constraints):
    """(lb, ub) for one set of extra constraints, solved in-process.

    Deliberately does NOT go through `solve`: that forks a watchdog process per
    solve, which cannot be nested inside a pool worker.  The grid programs here
    are small and need no wall-clock kill.
    """
    import contextlib
    import io as _io
    import pyomo.environ as pyo

    dag = DAG()
    dag.from_structure(_BATCH['structure'], unob=_BATCH['unob'])
    problem = build_base_problem(_BATCH['data_path'], dag=dag,
                                 extra_constraints=constraints)
    prog = problem.write_program()
    params = list(prog.parameters)
    cons = [list(c) for c in prog.constraints]
    add_independence_constraints(problem, params, cons)

    cwd = os.getcwd()
    out = []
    try:
        os.chdir(_BATCH['workdir'])
        with open('couenne.opt', 'w') as f:
            f.write(f"time_limit {_BATCH['time_limit']}\n")
        for sense in ('min', 'max'):
            M = _build_pyomo_model(params, cons, [['objvar']], sense)
            try:
                with contextlib.redirect_stdout(_io.StringIO()):
                    pyo.SolverFactory('couenne').solve(M, tee=False)
                out.append(float(pyo.value(M.obj)))
            except Exception:
                out.append(float('nan'))
    finally:
        os.chdir(cwd)
    return tuple(out)


def bounds_batch(data_path, constraint_sets, structure=None, unob=None,
                 workers=None, time_limit=300, chunksize=4):
    """[(lb, ub), ...] for many extra-constraint sets, evaluated in parallel.

    Each solve is independent and deterministic, so the result does not depend
    on the worker count.
    """
    from concurrent.futures import ProcessPoolExecutor
    sets = [list(c) if c else [] for c in constraint_sets]
    structure = structure or NHANES_STRUCTURE
    unob = unob or NHANES_UNOB
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers, initializer=_batch_init,
                             initargs=(str(data_path), structure, unob,
                                       time_limit)) as ex:
        return list(ex.map(_batch_one, sets, chunksize=chunksize))


# --------------------------------------------------------------------------
# High-level entry points
# --------------------------------------------------------------------------

def observational_bounds(data_path, dag=None, time_limit=300, workdir=None):
    """W_theta(empty) via the plain base program (two solves)."""
    problem = build_base_problem(data_path, dag=dag)
    prog = problem.write_program()
    params = list(prog.parameters)
    cons = [list(c) for c in prog.constraints]
    add_independence_constraints(problem, params, cons)
    obj = [['objvar']]
    t0 = time.time()
    hi = solve(params, cons, obj, 'max', time_limit=time_limit, workdir=workdir)
    lo = solve(params, cons, obj, 'min', time_limit=time_limit, workdir=workdir)
    wall = time.time() - t0
    return {
        'lb': lo.get('value'), 'ub': hi.get('value'),
        'W_empty': (hi.get('value') - lo.get('value'))
        if (hi['ok'] and lo['ok']) else None,
        'n_vars': len(params), 'n_constraints': len(cons),
        'wall': wall, 'max_run': hi, 'min_run': lo,
    }


def duplicated_base_point(data_path, dag=None, sense='max',
                          prefix_u='U_', prefix_l='L_'):
    """A point that is feasible for the coupled program, by construction.

    Any P satisfying C_empty gives the coupled-feasible point P^U = P^L = P
    (the coupling equality f(P^U) = f(P^L) then holds trivially).  We take P
    from a plain base solve.  Useful as a warm start and as a feasibility
    certificate.
    """
    import pyomo.environ as pyo
    problem = build_base_problem(data_path, dag=dag)
    prog = problem.write_program()
    params = list(prog.parameters)
    cons = [list(c) for c in prog.constraints]
    add_independence_constraints(problem, params, cons)
    M = _build_pyomo_model(params, cons, [['objvar']], sense)
    pyo.SolverFactory('couenne').solve(M, tee=False)
    sol = {p: pyo.value(getattr(M, p)) for p in params}
    point = {}
    for p, v in sol.items():
        point[prefix_u + p] = v
        point[prefix_l + p] = v
    return point


def evaluate_potency(data_path, experiment_queries, W_empty, dag=None,
                     time_limit=300, verbose=False, workdir=None,
                     warm_start=False):
    """Algorithm EvaluatePotency: one coupled program -> W_theta(a), pot(a)."""
    t_build = time.time()
    cp = build_coupled_program(data_path, experiment_queries, dag=dag)
    initial = (duplicated_base_point(data_path, dag=dag)
               if warm_start else None)
    build_seconds = time.time() - t_build
    run = solve(cp.parameters, cp.constraints, cp.obj_terms, 'max',
                objvar_bounds=(-1.0, 1.0), time_limit=time_limit,
                verbose=verbose, workdir=workdir, initial=initial)
    W_a = run.get('value')
    return {
        'experiments': list(experiment_queries),
        'W_a': W_a,
        'warm_start': warm_start,
        'potency': (W_empty - W_a) if run.get('ok') else None,
        'n_vars': cp.n_vars, 'n_constraints': cp.n_constraints,
        'build_seconds': build_seconds,
        'run': run,
    }
