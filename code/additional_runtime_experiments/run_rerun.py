"""Runtime and scalability measurements for the coupled potency evaluator.

Phases
------
  pruning : time the graph-only pruning pass on each benchmark instance.
  single  : the 26 benchmark instances / 152 candidates (identical seeds and
            candidate lists -- verified by `instances.verify()`).  One
            evaluation pass per candidate; BOTH accountings (unpruned = every
            candidate, pruned = survivors only) are derived from that one pass.
  subset  : one instance (c=1.0, seed 70; 8 experiments, unit costs, B=2).
            Enumerates every budget-feasible set (8 singletons + 28 pairs) and
            records its potency, so greedy / top-singleton / random fractions
            of the exhaustive optimum can be recomputed.

Everything is resumable: each instance and each candidate is appended to CSV in
`results/` as soon as it finishes, and a restart skips whatever is already
recorded.  The shipped CSVs are the measured run, so delete the relevant file to
recompute a phase from scratch.  Strictly sequential, single-threaded --
concurrent couenne would inflate the solve times this script exists to measure.

Usage:
    PYTHONHASHSEED=0 python run_rerun.py pruning
    PYTHONHASHSEED=0 python run_rerun.py single
    PYTHONHASHSEED=0 python run_rerun.py subset
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import instances as I  # noqa: E402
import coupled_eval as CE  # noqa: E402

WORK = HERE / "work"
WORK.mkdir(exist_ok=True)
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

SINGLE_CAND_CSV = RESULTS / "single_candidates.csv"
SINGLE_INST_CSV = RESULTS / "single_instances.csv"
SUBSET_CSV = RESULTS / "subset_sets.csv"
PRUNING_CSV = RESULTS / "pruning_pass.csv"
LOG = HERE / "run.log"

TIME_LIMIT = 600            # per-solve cap (seconds); every hit is logged
SUBSET_C, SUBSET_SEED = 1.0, 70
SUBSET_MAX_CANDIDATES, SUBSET_BUDGET = 12, 2
YIELD_CSV = RESULTS / "yield_curve.csv"
# (N, base) -> resolved seed.  Rebuilding an instance just to learn which seed
# it resolves to costs ~70 s at |V|=7 (autobound's expression parser), so the
# mapping is cached and a resume skips the rebuild for rows already measured.
YIELD_SEEDMAP = RESULTS / "yield_seedmap.csv"
YIELD = dict(N_list=[5, 6, 7], c=0.5, attempts=25, gate_s=60)


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def append(path, row):
    header = not path.exists() or path.stat().st_size == 0
    pd.DataFrame([row]).to_csv(path, mode="w" if header else "a",
                               header=header, index=False)


def done_keys(path, cols):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    d = pd.read_csv(path)
    if any(col not in d.columns for col in cols):
        return set()
    return set(tuple(str(v) for v in r)
               for r in d[cols].itertuples(index=False, name=None))


def cached_w_empty(c, seed):
    if not SINGLE_INST_CSV.exists() or SINGLE_INST_CSV.stat().st_size == 0:
        return None
    d = pd.read_csv(SINGLE_INST_CSV)
    g = d[(d['c'] == c) & (d['seed'] == seed)]
    if g.empty or not bool(g['w_empty_ok'].iloc[0]):
        return None
    return float(g['W_empty'].iloc[0])


# =====================================================================
#  PHASE: pruning pass timing (graph-only, no solver)
# =====================================================================
def phase_pruning():
    log("\n########## PHASE: pruning-pass timing (graph-only) ##########")
    done = done_keys(PRUNING_CSV, ['c', 'seed'])
    for c, seed in I.INSTANCES:
        if (str(c), str(seed)) in done:
            continue
        n_cand, ms = I.timed_pruning_pass(c, seed)
        append(PRUNING_CSV, dict(c=c, seed=seed, n_candidates=n_cand,
                                 pruning_pass_ms=round(ms, 4),
                                 ms_per_instance=round(ms, 4)))
        log(f"  c={c} seed={seed}: {n_cand} candidates, pruning pass {ms:.2f} ms")
    d = pd.read_csv(PRUNING_CSV)
    log(f"  >>> pruning pass over {len(d)} instances: "
        f"mean {d.pruning_pass_ms.mean():.2f} ms/instance, "
        f"median {d.pruning_pass_ms.median():.2f} ms, "
        f"total {d.pruning_pass_ms.sum():.1f} ms")


# =====================================================================
#  PHASE: single-experiment arm
# =====================================================================
def phase_single(limit_instances=None, tag=""):
    log(f"\n########## PHASE: single-experiment arm {tag} ##########")
    targets = I.INSTANCES
    if limit_instances:
        targets = targets[:limit_instances]
    t_phase = time.time()
    for c, seed in targets:
        cand_done = done_keys(SINGLE_CAND_CSV, ['c', 'seed', 'candidate_id'])
        inst, cands = I.get_instance(c, seed)
        pending = [i for i in range(len(cands))
                   if (str(c), str(seed), str(i)) not in cand_done]
        W_empty = cached_w_empty(c, seed)
        if W_empty is None:
            dag = CE.dag_from(inst['structure'], inst['unob'])
            we = CE.observational_width(dag, inst['pv_path'], inst['theta_str'],
                                        time_limit=TIME_LIMIT, workdir=str(WORK))
            # drop any stale row for this instance, then record
            if SINGLE_INST_CSV.exists() and SINGLE_INST_CSV.stat().st_size:
                d = pd.read_csv(SINGLE_INST_CSV)
                d = d[~((d['c'] == c) & (d['seed'] == seed))]
                d.to_csv(SINGLE_INST_CSV, index=False)
            append(SINGLE_INST_CSV, dict(
                c=c, seed=seed, theta=inst['theta_str'],
                n_districts=len(inst['cg'].get_districts()),
                n_candidates=len(cands),
                n_int=sum(1 for x in cands if x['pruned'] == 'interception'),
                n_id=sum(1 for x in cands if x['pruned'] == 'id'),
                n_surv=sum(1 for x in cands if x['pruned'] == 'none'),
                W_empty=we['W_empty'], w_empty_ok=we['ok'],
                w_empty_wall_s=round(we['wall'], 3), w_empty_programs=2,
                base_n_vars=we['n_vars'], base_n_constraints=we['n_constraints']))
            log(f"  [inst c={c} seed={seed}] W_empty={we['W_empty']} ok={we['ok']} "
                f"({we['wall']:.1f}s, {len(cands)} candidates, {len(pending)} pending)")
            W_empty = we['W_empty']
            if not we['ok']:
                log(f"  [inst c={c} seed={seed}] W_empty FAILED -- skipping candidates")
                continue
        else:
            log(f"  [inst c={c} seed={seed}] W_empty={W_empty} (cached), "
                f"{len(cands)} candidates, {len(pending)} pending")
        dag = CE.dag_from(inst['structure'], inst['unob'])
        for i in pending:
            x = cands[i]
            r = CE.evaluate_potency(dag, inst['pv_path'], inst['theta_str'],
                                    [x['param_str']], W_empty,
                                    time_limit=TIME_LIMIT, workdir=str(WORK))
            capped = (r['termination'] in ('maxTimeLimit', 'wallclock_timeout')
                      or r['solve_wall'] >= TIME_LIMIT)
            append(SINGLE_CAND_CSV, dict(
                c=c, seed=seed, candidate_id=i, param_str=x['param_str'],
                pruned=x['pruned'], W_empty=W_empty, W_a=r['W_a'],
                potency=r['potency'], ok=r['ok'], programs=r['programs'],
                wall_s=round(r['wall'], 3), solve_s=round(r['solve_wall'], 3),
                build_s=round(r['build_seconds'], 3),
                n_vars=r['n_vars'], n_constraints=r['n_constraints'],
                termination=r['termination'], proved_by_log=r['proved_by_log'],
                cap_hit=bool(capped), error=r['error']))
            tag_p = f"pot={r['potency']:.5f}" if r['potency'] is not None else "pot=NA"
            log(f"      [{i:2d}] {x['param_str']:14s} pruned={x['pruned']:12s} "
                f"{tag_p} {r['wall']:.2f}s ok={r['ok']}"
                f"{'  *CAP*' if capped else ''}")
            if capped:
                log(f"      !! PER-SOLVE CAP ({TIME_LIMIT}s) HIT: c={c} seed={seed} cid={i}")
    log(f"  >>> single-experiment phase wall this run: {time.time()-t_phase:.1f}s")
    summarise_single()


def summarise_single():
    if not SINGLE_CAND_CSV.exists() or SINGLE_CAND_CSV.stat().st_size == 0:
        return
    d = pd.read_csv(SINGLE_CAND_CSV)
    surv = d[d['pruned'] == 'none']
    log("\n  --- single-experiment accounting ---")
    log(f"  candidates evaluated   : {len(d)} / 152")
    log(f"  UNPRUNED  wall {d.wall_s.sum():10.2f} s   programs {int(d.programs.sum())}")
    log(f"  PRUNED    wall {surv.wall_s.sum():10.2f} s   programs {int(surv.programs.sum())}")
    if surv.wall_s.sum() > 0:
        log(f"  wall speedup {d.wall_s.sum()/surv.wall_s.sum():.2f}x   "
            f"program speedup {d.programs.sum()/max(surv.programs.sum(),1):.2f}x")
    log(f"  solves not ok: {int((~d['ok'].astype(bool)).sum())}   "
        f"cap hits: {int(d['cap_hit'].astype(bool).sum())}")


# =====================================================================
#  PHASE: subset arm
# =====================================================================
def subset_candidates():
    inst, cands = I.get_instance(SUBSET_C, SUBSET_SEED)
    survs = [x for x in cands if x['pruned'] == 'none']
    prun = [x for x in cands if x['pruned'] != 'none']
    chosen = (survs + prun)[:SUBSET_MAX_CANDIDATES]
    for j, x in enumerate(chosen):
        x['cid'] = j
    return inst, chosen


def phase_subset():
    log("\n########## PHASE: subset arm (c=1.0, seed 70; B=2, unit costs) ##########")
    inst, chosen = subset_candidates()
    dag = CE.dag_from(inst['structure'], inst['unob'])
    log(f"  {len(chosen)} experiments: " +
        ", ".join(f"{x['cid']}:{x['param_str']}({x['pruned']})" for x in chosen))

    W_empty = cached_w_empty(SUBSET_C, SUBSET_SEED)
    if W_empty is None:
        we = CE.observational_width(dag, inst['pv_path'], inst['theta_str'],
                                    time_limit=TIME_LIMIT, workdir=str(WORK))
        W_empty = we['W_empty']
        log(f"  W_empty={W_empty} ({we['wall']:.1f}s)")
    else:
        log(f"  W_empty={W_empty} (cached from the single-experiment phase)")

    # every budget-feasible set under B=2 with unit costs: singletons + pairs
    sets = [(x['cid'],) for x in chosen]
    sets += [(a['cid'], b['cid']) for a, b in itertools.combinations(chosen, 2)]
    by_cid = {x['cid']: x for x in chosen}
    done = done_keys(SUBSET_CSV, ['members'])
    for s in sets:
        key = str(list(s))
        if (key,) in done:
            continue
        queries = [by_cid[i]['param_str'] for i in s]
        r = CE.evaluate_potency(dag, inst['pv_path'], inst['theta_str'], queries,
                                W_empty, time_limit=TIME_LIMIT, workdir=str(WORK))
        capped = (r['termination'] in ('maxTimeLimit', 'wallclock_timeout')
                  or r['solve_wall'] >= TIME_LIMIT)
        append(SUBSET_CSV, dict(
            kind='single' if len(s) == 1 else 'pair', c=SUBSET_C, seed=SUBSET_SEED,
            members=key,
            member_params=" + ".join(queries),
            pruned="+".join(by_cid[i]['pruned'] for i in s),
            has_id=any(by_cid[i]['pruned'] == 'id' for i in s),
            W_empty=W_empty, W_a=r['W_a'], potency=r['potency'], ok=r['ok'],
            programs=r['programs'], wall_s=round(r['wall'], 3),
            solve_s=round(r['solve_wall'], 3), build_s=round(r['build_seconds'], 3),
            n_vars=r['n_vars'], n_constraints=r['n_constraints'],
            termination=r['termination'], cap_hit=bool(capped), error=r['error']))
        log(f"    {key:10s} {' + '.join(queries):32s} "
            f"pot={r['potency'] if r['potency'] is None else round(r['potency'],6)} "
            f"{r['wall']:.2f}s ok={r['ok']}{'  *CAP*' if capped else ''}")
        if capped:
            log(f"    !! PER-SOLVE CAP ({TIME_LIMIT}s) HIT on set {key}")
    summarise_subset()


def summarise_subset():
    if not SUBSET_CSV.exists() or SUBSET_CSV.stat().st_size == 0:
        return
    d = pd.read_csv(SUBSET_CSV)
    noid = d[~d['has_id'].astype(bool)]
    log("\n  --- subset accounting ---")
    log(f"  sets evaluated            : {len(d)}")
    log(f"  FULL enumeration     wall {d.wall_s.sum():10.2f} s  programs {int(d.programs.sum())}")
    log(f"  ID-pruned removed    wall {noid.wall_s.sum():10.2f} s  programs {int(noid.programs.sum())}"
        f"  ({len(noid)} sets)")

    pot = {tuple(eval(m)): p for m, p in zip(d['members'], d['potency'])}
    singles = {k[0]: v for k, v in pot.items() if len(k) == 1}
    opt_full = max(v for v in pot.values() if v is not None)
    id_cids = {tuple(eval(m))[0] for m, h in zip(d['members'], d['has_id'])
               if h and len(tuple(eval(m))) == 1}
    opt_noid = max(v for k, v in pot.items()
                   if v is not None and not (set(k) & id_cids))
    top_singleton = max(singles.values())
    best_single = max(singles, key=singles.get)
    greedy = top_singleton
    for k, v in pot.items():
        if len(k) == 2 and best_single in k and v is not None:
            greedy = max(greedy, v)
    rand = float(np.mean([v for v in pot.values() if v is not None]))
    log(f"  exhaustive optimum (full)      : {opt_full:.6f}")
    log(f"  exhaustive optimum (ID-removed): {opt_noid:.6f}  "
        f"(equal: {abs(opt_full-opt_noid) < 1e-9})")
    log(f"  greedy         = {greedy:.6f}  -> frac {greedy/opt_full:.4f}")
    log(f"  top singleton  = {top_singleton:.6f}  -> frac {top_singleton/opt_full:.4f}")
    log(f"  random (mean)  = {rand:.6f}  -> frac {rand/opt_full:.4f}")
    pd.DataFrame([dict(
        c=SUBSET_C, seed=SUBSET_SEED, n_sets=len(d),
        wall_full_s=round(float(d.wall_s.sum()), 2),
        wall_idremoved_s=round(float(noid.wall_s.sum()), 2),
        programs_full=int(d.programs.sum()), programs_idremoved=int(noid.programs.sum()),
        optimum_full=round(opt_full, 6), optimum_idremoved=round(opt_noid, 6),
        greedy=round(greedy, 6), top_singleton=round(top_singleton, 6),
        random_mean=round(rand, 6),
        greedy_frac=round(greedy / opt_full, 4),
        top_singleton_frac=round(top_singleton / opt_full, 4),
        random_frac=round(rand / opt_full, 4))]).to_csv(
            RESULTS / "subset_summary.csv", index=False)


# =====================================================================
# Some (graph, query) pairs make autobound's parser allocate without bound:
# `causalProblem.query` on a two-world counterfactual whose node has several
# parents expands a cross-product in Parser.intersect_expr.  One |V|=7 instance
# grew past 8 GB with no plateau and took the whole machine down.  The budget is
# a FIXED constant, not a fraction of the host's RAM, so an instance is
# classified the same way on any machine.
MEM_BUDGET_GB = 4
BUILD_TIMEOUT_S = 600   # a build slower than this is pathological; skip it


def _capped_worker(fn, args, kwargs, budget_gb, q):
    import resource
    # set AFTER the fork: the parent has already imported numpy/pyomo, so this
    # limits the work, not the interpreter start-up.
    lim = int(budget_gb * 1024 ** 3)
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    try:
        q.put(("ok", fn(*args, **kwargs)))
    except MemoryError:
        q.put(("mem", None))
    except Exception as exc:
        q.put(("err", f"{type(exc).__name__}: {exc}"))


def run_capped(fn, *args, budget_gb=MEM_BUDGET_GB, timeout=None, **kwargs):
    """Run `fn` in a forked child limited to `budget_gb` of address space.

    Returns (status, value) with status in {"ok", "mem", "err", "timeout"}.
    Memory overruns come back as "mem" instead of taking down the host.
    """
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=_capped_worker,
                       args=(fn, args, kwargs, budget_gb, q))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate(); proc.join()
        return ("timeout", None)
    if q.empty():
        return ("err", "worker produced no result")
    return q.get()


def _yield_build(cfg, base):
    """Only primitives cross the process boundary."""
    inst = I.P.build_instance(cfg, base_seed=base)
    return dict(structure=inst["structure"], unob=inst["unob"],
                theta_str=inst["theta_str"], seed=inst["seed"])


def _yield_solve(structure, unob, pv_path, theta_str, gate_s, workdir):
    dag = CE.dag_from(structure, unob)
    we = CE.observational_width(dag, pv_path, theta_str,
                                time_limit=gate_s, workdir=workdir)
    return dict(ok=bool(we["ok"]), wall=we["wall"], W_empty=we["W_empty"],
                timed_out=bool(we.get("timed_out")))


def phase_yield():
    """Are the observational bounds alone solvable, as a function of |V|?

    Independent of the potency evaluator: this solves the base program only.
    """
    log("\n########## PHASE: yield curve ##########")
    done = done_keys(YIELD_CSV, ["N", "seed"])
    seedmap = {}
    if YIELD_SEEDMAP.exists() and YIELD_SEEDMAP.stat().st_size:
        _sm = pd.read_csv(YIELD_SEEDMAP)
        seedmap = {(str(r.N), str(r.base)): int(r.seed) for r in _sm.itertuples()}
    for N in YIELD["N_list"]:
        cfg = I.base_cfg(N, YIELD["c"])
        cfg["w_empty_timeout_s"] = YIELD["gate_s"]
        base, got = cfg["seed"], 0
        while got < YIELD["attempts"]:
            cached = seedmap.get((str(N), str(base)))
            if cached is not None and (str(N), str(cached)) in done:
                base = cached + 1        # already measured: skip the rebuild
                got += 1
                continue
            I.P.PV_PATH = I.PV_DIR / f"pv_yield_N{N}_s{base}.csv"
            # A build has no natural bound either: autobound's parser can grind
            # for hours on some (graph, query) pairs while staying under the
            # memory cap, so it gets a wall-clock bound of its own.
            bstat, inst = run_capped(_yield_build, cfg, base,
                                     timeout=BUILD_TIMEOUT_S)
            if bstat != "ok":
                # cannot even construct this instance within the budget
                log(f"  [yield N={N} base={base}] build {bstat}; skipped")
                base += 1
                continue
            if seedmap.get((str(N), str(base))) != inst["seed"]:
                append(YIELD_SEEDMAP, dict(N=N, base=base, seed=inst["seed"]))
                seedmap[(str(N), str(base))] = inst["seed"]
            base = inst["seed"] + 1
            got += 1
            if (str(N), str(inst["seed"])) in done:
                continue
            t_inst = time.time()
            sstat, we = run_capped(_yield_solve, inst["structure"], inst["unob"],
                                   str(I.P.PV_PATH), inst["theta_str"],
                                   YIELD["gate_s"], str(WORK),
                                   timeout=4 * YIELD["gate_s"] + 60)
            if sstat == "ok":
                wall = we["wall"]
                status = "solved" if we["ok"] else (
                    "timeout" if we["timed_out"] or wall >= YIELD["gate_s"] else "err")
                w_emp = round(we["W_empty"], 4) if we["ok"] else None
            else:
                wall = time.time() - t_inst
                status = "mem" if sstat == "mem" else (
                    "timeout" if sstat == "timeout" else "err")
                w_emp = None
            append(YIELD_CSV, dict(
                N=N, seed=inst["seed"], status=status,
                we_time=round(wall, 1), W_empty=w_emp))
            log(f"  [yield N={N} seed={inst['seed']}] {status} {wall:.1f}s")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    log(f"===== phase={phase} start {time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"(per-solve cap {TIME_LIMIT}s, sequential) =====")
    t0 = time.time()
    if phase in ("pruning", "all"):
        phase_pruning()
    if phase in ("yield", "all"):
        phase_yield()
    if phase in ("single", "all"):
        phase_single(limit_instances=n)
    if phase in ("subset", "all"):
        phase_subset()
    if phase == "summary":
        summarise_single()
        summarise_subset()
    log(f"===== done phase={phase} in {time.time()-t0:.1f}s =====")
