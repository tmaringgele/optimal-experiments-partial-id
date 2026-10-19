# Additional NHANES Experiments

Same graph, data and solver as `../nhanes_pipeline.ipynb`; each notebook varies
one thing.

| Notebook | Varies |
| --- | --- |
| `nhanes_alternative_expected.ipynb` | expected width instead of worst case |
| `nhanes_alternative_best_case.ipynb` | best-case width instead of worst case |
| `nhanes_bootstrap.ipynb` | 50 bootstrap resamples of the 5,833 records |
| `nhanes_graph_misspecification.ipynb` | one edit to the assumed graph |

`bootstrap_results.csv` holds the per-replicate values. Delete it to force a
recomputation; the notebook resumes from it otherwise.

## Criteria and method

The worst-case criterion collapses into a single polynomial program
(`../coupled_potency.py`), so the main results, the bootstrap and the
graph-misspecification run each solve one program per candidate set.

The expected and best-case criteria do not collapse -- the best case is a
min-max, the expected width an integral -- so those two notebooks sweep a grid
of feasible outcomes and solve a program at each point: 800 points per scalar
outcome, 30x30 for outcome pairs, and for `P(X,Y|do(Z))` 30 sampled feasible
`do(Z=0)` distributions crossed with the 20 feasible distributions on a 1/30
enumeration grid for `do(Z=1)`. Those values carry the corresponding
discretization error; the worst-case values do not.

Grid points are independent solves, so they are evaluated in parallel
(`BATCH_WORKERS`). The worker count changes only the wall time.

## Graph misspecification

Adding `Z -> Y` drops the assumption that health insurance affects diabetes only
through physical activity. It leaves the query's district hull unchanged, so
every pruning decision survives it. It also makes `Z` a parent of `Y`, which
lies inside the hull, so the potencies are not protected and they do change: the
observational width goes from 0.86 to 1.00 and every nonzero potency rises with it. The
notebook recomputes both graphs from scratch and checks the optimal choice at
each budget, which is the same under both.

## Data

These notebooks read `../data/`. Nothing is duplicated here.
