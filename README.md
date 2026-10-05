# OpenGLT

This repository provides the released material for our OpenGLT submission. Thank you so much for reviewing our work.

OpenGLT is a unified benchmark for training, tuning, and evaluating graph
learning methods across graph-property prediction, cardinality estimation,
and distribution-shift settings. This repository contains the public
implementations, dataset interfaces, evaluation protocols, and selected
hyperparameter configurations.

## Benchmark scope

| Setting | Datasets | Supported models |
|---|---|---|
| Graph classification and regression | OGB: `ogbg-molbace`, `ogbg-molhiv`, `ogbg-molpcba`, `ogbg-moltox21`; TU: `IMDB-BINARY`, `IMDB-MULTI`, `PROTEINS`, `ENZYMES`, `MUTAG`, `NCI1`, `DD`, `COLLAB`, `REDDIT-BINARY`; LRGB: `Peptides-func`, `Peptides-struct`; PyG molecular: `ZINC`, `AQSOL`; Motif Counting: `Count-{3,4,5,6,7,8}-Cycle`, `Count-{4,5,6}-Path`, `Count-4-Clique`, `Count-Tailed-Tri`, `Count-Chordal-Cyc`, and `Count-Tri-Rect` | `GCN`, `GCNPlus`, `GIN`, `GINE`, `SAGE`, `SAGEPlus`, `PNA`, `GraphGPS`, `HGT`, `NAG`, `EdgePool`, `TopkPool`, `GMT`, `HMN`, `AK`, `ECS`, `I2GNN`, `MO`, `HGP`, `VIB`, `RGC`, `MVG`, and `GCA` |
| Cardinality estimation | G-CARE: `AIDS`, `YAGO`; GNCE: `SWDF`, `LUBM`, `YAGO`, and `Wikidata`, each with path and star queries; CardBench: 20 single-table and 20 binary-join workloads, plus binary-join transfer | `GINE`, `PNA`, `GraphGPS`, `GMT`, `AK`, `MO`, `GCA`, `MLP-F`, `MLP-S`, `LGBM` |
| Distribution shift | `GOOD-HIV` and `GOOD-ZINC`: no-shift, scaffold-covariate, scaffold-concept, size-covariate, and size-concept | `GINE`, `PNA`, `GraphGPS`, `GMT`, `AK`, `MO`, `GCA` |

## Repository layout

| Path | Details |
|---|---|
| `example/` | Runners entries |
| `tuning/` | Tuning entries and search spaces |
| `best_hparams/` | Selected configurations |
| `datasets/` | Unified loading, feature construction, and split |
| `models/` | Model implementations |
| `engine/` | Training losses, metrics, and reproducibility utilities |
| `configs/` | Runtime configuration |
| `data/` | Bundled Counting data and download/layout details |
| `selection/` | Dataset-level model selection |

## Environment

The reference environment uses PyTorch 2.0.1, CUDA 11.8, and
PyG 2.5.3. All direct runtime dependencies are pinned in
[`requirements.txt`](requirements.txt).

```bash
conda create -n openglt python=3.10 -y
conda activate openglt
pip install -r requirements.txt
```

Run commands below from the repository root. Use `--gpu -1` for CPU-only execution.

## Data

Official links and the complete directory layout are given in [`data/README.md`](data/README.md).

## Evaluate selected configurations

Common entries are under `example/run.py`, use them like the following examples :

```bash
python example/run.py --dataset X --model Y --use_best_hyper_params --gpu 0 (-- ... other hyper-params that you want to try)

# E.g.,
# TU
python example/run.py --dataset NCI1 --model GINE --use_best_hyper_params --gpu 0

# OGB
python example/run.py --dataset ogbg-molhiv --model GraphGPS --use_best_hyper_params --gpu 0

# Counting
python example/run.py --dataset Count-3-Cycle --model I2GNN --use_best_hyper_params --gpu 0

# LRGB
python example/run.py --dataset peptides-func --model PNA --use_best_hyper_params --gpu 0

# ZINC-12K
python example/run.py --dataset ZINC --model AK --use_best_hyper_params --gpu 0

# G-CARE
python example/run.py --dataset GCARE-AIDS --model LGBM --use_best_hyper_params --gpu -1

# GNCE
python example/run.py --dataset GNCE-SWDF-path --model GCA --use_best_hyper_params --gpu 0

# GOOD
python example/run.py --dataset good-hiv-scaffold-covariate --model GMT --use_best_hyper_params --gpu 0

# CardBench
python example/cardbench.py --npz_path data/CardBench/single_table/airline_single_table.npz --model_name gine_root --use_best_hyper_params --n_runs 3 --gpu 0
```

## Hyperparameter tuning

`tuning.run` dispatches to the model-specific Optuna tuner. Trials select checkpoints on validation data and do not evaluate the test split.

```bash
# E.g., 
# TU
python -m tuning.run --dataset NCI1 --model GINE --n_trials 200 --gpu 0

# OGB
python -m tuning.run --dataset ogbg-molhiv --model GraphGPS --n_trials 200 --gpu 0

# Counting
python -m tuning.run --dataset Count-4-Clique --model I2GNN --n_trials 200 --gpu 0

# GNCE
python -m tuning.run --dataset GNCE-SWDF-path --model GCA --n_trials 200 --gpu 0

# Distribution shift
python -m tuning.run --dataset good-hiv-size-concept --model GMT --n_trials 200 --gpu 0
```

The executable search spaces are in `tuning/*_tuning.py`.

## Evaluation protocol

- TU datasets use stratified 10-fold evaluation; G-CARE uses topology-balanced 5-fold evaluation.
- Fixed-split datasets use their widely-adopted train/validation/test splits and report repeated-run statistics.

## Model selection and records

Run `python selection/run_selection.py` from the repository root to evaluate the MedQ and P95 targets under LOGO and LODO.
The bundled [records](selection/data/openglt_selection.npz) contain selection-target entries and dataset descriptors for all workloads; no graph-dataset downloads are needed for this experiment.
Results are written to `selection/results/` (`main_results.csv`, `fold_results.csv`, and `domain_results.csv`).
See [selection/README.md](selection/README.md) for details; dependencies are included in the root [requirements.txt](requirements.txt).

Thank you again for reviewing our work.
