# Supplemental Code

Code and data for all experiments in the paper.

## Layout

```
coupled_potency.py                  EvaluatePotency: one coupled program per candidate set
algorithm_evaluation/               pruning rules, ID check, graph generators
er_evaluation.ipynb                 pruning on Erdos-Renyi graphs
bnlearn_evaluation.ipynb            pruning on bnlearn networks
id_only_evaluation.ipynb            pruning with the ID check alone
nhanes_pipeline.ipynb               NHANES case study (Table 1)
additional_nhanes_experiments/      criterion sensitivity, bootstrap, graph misspecification
additional_runtime_experiments/     runtime and scalability
data/                               NHANES 2017-2018 source files and P(Z,X,Y)
plots/                              figures used by the evaluation notebooks
```

## Entry points

| Paper result | Run |
| --- | --- |
| Erdos-Renyi pruning rates | `er_evaluation.ipynb` |
| bnlearn pruning rates | `bnlearn_evaluation.ipynb` |
| Subset search-space reduction | `id_only_evaluation.ipynb` |
| Table 1 | `nhanes_pipeline.ipynb` |
| Criterion sensitivity | `additional_nhanes_experiments/nhanes_alternative_{expected,best_case}.ipynb` |
| Finite-sample stability | `additional_nhanes_experiments/nhanes_bootstrap.ipynb` |
| Graph misspecification | `additional_nhanes_experiments/nhanes_graph_misspecification.ipynb` |
| Runtime and scalability | `additional_runtime_experiments/` |

## Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

This suffices for the three pruning notebooks. Everything that computes potency
also needs:

1. **autobound** (Duarte et al., 2023) - not on PyPI. Download from that paper's
   supplement and `pip install -e .` from the unpacked folder.
2. **pyomo** - `pip install pyomo`
3. **couenne** - a system binary, not a Python package. Debian/Ubuntu:
   `apt-get install coinor-couenne`; macOS: `brew install coin-or-tools/coinor/couenne`;
   or prebuilt binaries from COIN-OR. Must be on `PATH`.

Couenne runs at its default tolerances. Set `PYTHONHASHSEED=0` for the runtime
experiments: instance generation iterates over sets of variable names, so it
depends on the string-hash seed.

## A note on the programs

Within a district, two response-type variables are independent when their nodes
share no latent parent. `coupled_potency.py` adds those constraints
(`add_independence_constraints`). Without them the program is a relaxation that
admits a single latent confounding the whole district, and its bounds are valid
but not sharp for the ADMG that was specified.

## Approximate runtimes

Single-threaded, on a consumer laptop, except where noted.

| | |
| --- | --- |
| pruning notebooks | seconds by default; hours with `RECOMPUTE = True` |
| `nhanes_pipeline.ipynb` | seconds (six coupled programs) |
| `nhanes_bootstrap.ipynb` | ~3 min (50 replicates) |
| `nhanes_graph_misspecification.ipynb` | ~30 s |
| alternative-criteria notebooks | ~3 min each; these sweep an outcome grid and solve its points in parallel |
| runtime experiments | see that folder's README |

The potency notebooks ship with their outputs, so those numbers can be read off
without re-running anything. The three pruning notebooks ship cleared. By
default (`RECOMPUTE = False`) they load the results in
`algorithm_evaluation/results/`; setting `RECOMPUTE = True` reruns the sweeps.
The sampled graphs and queries depend on `PYTHONHASHSEED`, which was not fixed
for the shipped results, so a rerun gives equivalent but not identical numbers.
The plot cells render with LaTeX (`text.usetex`) and need a LaTeX installation.

## Data

`data/` holds the public NHANES 2017-2018 files (`DEMO_J`, `PAQ_J`, `HIQ_J`,
`DIQ_J`) from the U.S. CDC National Center for Health Statistics, and the
discretized joint `P_V.csv` produced by `nhanes_pipeline.ipynb`.

## Third-party code

- AutoBounds (MIT) - see `LICENSES/autobound-LICENSE`
- Ananke (Apache 2.0) - see `LICENSES/ananke-LICENSE`
