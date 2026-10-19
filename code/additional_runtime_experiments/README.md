# Runtime and Scalability

Erdos-Renyi ADMGs with two-world counterfactual queries and sampled `P(V)`. No
NHANES data is involved.

## Running

```bash
PYTHONHASHSEED=0 python run_rerun.py [pruning|yield|single|subset|all]
```

`PYTHONHASHSEED=0` is required: instance generation iterates over sets of
variable names, so the sampled instances depend on the string-hash seed.

Everything is **sequential and single-threaded** by design. These are timing
measurements, and concurrent couenne processes inflate solve times and
manufacture false timeouts.

Each phase appends to `results/` and skips rows already recorded, so a phase is
resumable and re-running one is a no-op until the relevant CSV is deleted.

Per-solve caps: 60 s for `yield`, 600 s for `single` and `subset`.

In `yield` each instance additionally runs in child processes limited to 4 GB
of memory, with 600 s to build it and 300 s to solve it, because for some graph and query pairs the step that expands a query
into response-type variables allocates without bound. An instance that hits the
memory limit is recorded with status `mem`. The other phases run on a fixed
instance list that does not contain such a case, so they are not wrapped.

## Files

| | |
| --- | --- |
| `coupled_potency.py` (repo root) | the evaluator: one coupled program per candidate set |
| `coupled_eval.py` | that evaluator applied to arbitrary ADMGs |
| `instances.py` | deterministic instance and candidate generation |
| `pruned_vs_unpruned.py` | graph, query and `P(V)` sampling used by `instances.py` |
| `run_rerun.py` | phase driver |

`P(V)` is drawn by sampling one independent latent per bidirected edge and
pushing the result forward, so it satisfies the independences the ADMG implies.
Drawing a flat Dirichlet over a district's joint response types instead would
put `P(V)` outside the model whenever the ADMG implies one, and no model of that
graph could reproduce it.

## Results

| Paper claim | File | Value |
| --- | --- | --- |
| solvable within 60 s, by size | `results/yield_curve.csv` | 12/25 at \|V\|=5, 3/25 at 6, 1/25 at 7 |
| pruning pass | `results/pruning_pass.csv` | ~6 ms per instance |
| evaluate every candidate | `results/single_candidates.csv` | >= 26,753 s over 152 candidates |
| evaluate survivors only | same, `pruned == none` | 3,849 s over 48 candidates |
| solve time by candidate group | same | median 2.5 s kept, 4.7 s ID-pruned, 612 s interception-pruned |
| subset enumeration | `results/subset_sets.csv` | 105.7 s / 36 programs; 62.7 s / 21 with ID-pruned removed |
| selection baselines | `results/subset_summary.csv` | greedy 1.0000, best single 0.5289, random 0.2222 |
| incompleteness | `results/single_instances.csv` | 10 of 26 instances fully pruned; 48 of 102 candidates survive |

39 of the 152 candidates did not finish within the 600 s cap and contribute the
time actually spent on them, so the unpruned total and the speedup derived from
it are lower bounds. 29 of those 39 are interception-pruned.

`results/yield_seedmap.csv` records which seed each search started from and
which it resolved to. It only makes a resumed run cheaper and holds no results.

## Cost to reproduce

`pruning` ~2 min, `subset` ~2 min, `yield` ~2 h, `single` ~7.5 h.
