import argparse
import copy
import json
import logging
import math
import sys
from pathlib import Path

import optuna
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import init_cfg
from example.i2gnn import run_i2gnn, supported_dataset_type
from models.i2gnn.data import default_i2gnn_h
from tuning.main_baselines_tuning import (
    GCARE_REGRESSION,
    PYG_MOL_REGRESSION,
    compact_stats,
    default_val_metric,
    metric_direction,
    normalize_metric,
)
from tuning.gnce_protocol import optimize_to_total, run_final_evaluation
from tuning.main_effectiveness_protocol import (
    apply_main_effectiveness_trial_protocol,
    is_main_effectiveness_fixed_split,
    main_effectiveness_default_metric,
    print_main_effectiveness_protocol,
)
from tuning.evaluation import is_tu_dataset, run_tu_cv_final_evaluation


SEARCH_SPACE_VERSION = 'i2gnn_v2_main30'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ogbg-molhiv')
    parser.add_argument('--val_metric', default='auto')
    parser.add_argument('--n_trials', type=int, default=None)
    parser.add_argument('--train_epochs', type=int, default=500)
    parser.add_argument('--train_patience', type=int, default=50)
    parser.add_argument('--train_runs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold_k', type=int, default=0, help='0 uses TU=10 and G-CARE=5.')
    parser.add_argument('--fold_seed', type=int, default=2024)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--study_name', default=None)
    parser.add_argument('--storage', default=None)
    return parser.parse_args()


def default_study_name(args, optimize_metric):
    dataset = args.dataset.replace('/', '_')
    return f"i2gnn_{SEARCH_SPACE_VERSION}_{dataset}_{optimize_metric}"


def dataset_type(dataset):
    return supported_dataset_type(dataset)


def dataset_size_bucket(dataset):
    dataset_lower = dataset.lower()
    if dataset == 'ogbg-molpcba' or dataset_lower in {'peptides-func', 'peptides-struct'}:
        return 'large'
    if dataset in {'COLLAB', 'REDDIT-BINARY', 'DD'}:
        return 'large_tu'
    if dataset.startswith('Count-'):
        return 'counting'
    if dataset_lower in GCARE_REGRESSION or dataset_lower.startswith('gnce-'):
        return 'gcare'
    if dataset_lower in {'zinc', 'aqsol'}:
        return 'mol_regression'
    return 'default'


def hidden_choices(dataset):
    return [64, 128, 256, 384]


def layer_choices(dataset):
    return list(range(2, 7))


def h_choices(dataset):
    if dataset_type(dataset) == 'counting':
        return [1, 2, 3, 4, 5]
    return [1, 2, 3, 4]


def max_nodes_per_hop_choices(dataset):
    return [16, 32, 64, 0]


def node_label_choices(dataset):
    return ['spd', 'drnl']


def use_rd_choices(dataset):
    return [False] if dataset_type(dataset) == 'counting' else [False, True]


def subgraph_pooling_choices(dataset):
    dtype = dataset_type(dataset)
    if dtype == 'counting':
        return ['mean', 'center']
    if dtype == 'pyg_mol':
        return ['mean', 'add']
    if dtype == 'tu':
        return ['mean', 'add', 'mean-context']
    return ['mean']


def subgraph2_pooling_choices(dataset):
    dtype = dataset_type(dataset)
    if dtype in {'ogb', 'lrgb'}:
        return ['mean', 'center', 'mean-center-side']
    if dtype in {'pyg_mol', 'tu'}:
        return ['mean', 'center', 'mean-center-side']
    return ['mean-center-side']


def graph_pooling_choices(dataset):
    dtype = dataset_type(dataset)
    if dtype in {'ogb', 'lrgb'}:
        return ['mean', 'max', 'attention']
    return ['mean', 'sum', 'add', 'max']


def gnn_type_choices(dataset):
    return ['gin', 'gcn'] if dataset_type(dataset) in {'ogb', 'lrgb'} else ['gin']


def bool_choices_with_default_false():
    return [False, True]


def set_fixed_i2gnn_defaults(cfg):
    cfg.use_augm_gca = False
    cfg.model.use_fingerprints = False
    cfg.model.use_rwpe = False
    cfg.model.pooling_method = 'none'
    cfg.i2gnn.force_reload = False
    cfg.i2gnn.self_loop = False
    cfg.i2gnn.rni = False


def suggest_pooling_cfg(trial, cfg, dataset):
    dtype = dataset_type(dataset)

    if dtype in {'pyg_mol', 'tu'}:
        pair_choices = []
        for subgraph_pool in subgraph_pooling_choices(dataset):
            for subgraph2_pool in subgraph2_pooling_choices(dataset):
                pair_choices.append(f'{subgraph_pool}|{subgraph2_pool}')
        pair = trial.suggest_categorical(f'i2_pool_pair_{dtype}', pair_choices)
        cfg.i2gnn.subgraph_pooling, cfg.i2gnn.subgraph2_pooling = pair.split('|')
    elif dtype == 'counting':
        cfg.i2gnn.subgraph_pooling = trial.suggest_categorical(
            'i2_subgraph_pool_counting',
            subgraph_pooling_choices(dataset),
        )
        cfg.i2gnn.subgraph2_pooling = 'mean-center-side'
    else:
        cfg.i2gnn.subgraph_pooling = 'mean'
        pool_nn_pair = trial.suggest_categorical(
            f'i2_subgraph2_pool_nn_{dtype}',
            ['mean|0', 'mean|1', 'center|0', 'center|1', 'mean-center-side|1'],
        )
        cfg.i2gnn.subgraph2_pooling, use_pooling_nn = pool_nn_pair.split('|')
        cfg.i2gnn.use_pooling_nn = bool(int(use_pooling_nn))

    if dtype in {'ogb', 'lrgb', 'tu'}:
        cfg.i2gnn.graph_pooling = trial.suggest_categorical(
            f'i2_graph_pool_{dtype}', graph_pooling_choices(dataset)
        )
    else:
        # CountI2GNN and ZincI2GNN implement their graph readout internally.
        cfg.i2gnn.graph_pooling = 'mean'

    if dtype in {'pyg_mol', 'tu'}:
        cfg.i2gnn.use_pooling_nn = trial.suggest_categorical(
            f'i2_use_pooling_nn_{dtype}',
            bool_choices_with_default_false(),
        )
    elif dtype == 'counting':
        cfg.i2gnn.use_pooling_nn = True

    if dtype in {'ogb', 'lrgb', 'pyg_mol', 'tu'}:
        cfg.i2gnn.gate = trial.suggest_categorical(
            f'i2_gate_{dtype}',
            bool_choices_with_default_false(),
        )
    else:
        cfg.i2gnn.gate = False

    if dtype == 'pyg_mol':
        cfg.i2gnn.double_pooling = trial.suggest_categorical(
            'i2_double_pooling_pyg_mol',
            bool_choices_with_default_false(),
        )
    else:
        cfg.i2gnn.double_pooling = False


def suggest_cfg(trial, base_cfg, args):
    cfg = copy.deepcopy(base_cfg)
    dtype = dataset_type(args.dataset)

    cfg.dataset = args.dataset
    cfg.device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    cfg.seed = args.seed
    cfg.cv.fold_k = args.fold_k
    cfg.cv.seed = args.fold_seed
    cfg.deterministic = args.deterministic
    cfg.deterministic_warn_only = True
    cfg.load_best_params = False
    set_fixed_i2gnn_defaults(cfg)

    hidden = hidden_choices(args.dataset)
    cfg.model.hidden_size = trial.suggest_categorical('i2_hidden', hidden)

    cfg.model.num_layers = trial.suggest_categorical(
        'i2_layers',
        layer_choices(args.dataset),
    )
    if dtype == 'counting':
        # The imported official CountI2GNN has no dropout argument.
        cfg.i2gnn.drop_ratio = 0.0
    else:
        cfg.i2gnn.drop_ratio = trial.suggest_categorical(
            'i2_dropout', [0.0, 0.1, 0.2, 0.35]
        )

    cfg.i2gnn.h = trial.suggest_categorical(
        'i2_h',
        h_choices(args.dataset),
    )
    cfg.i2gnn.node_label = trial.suggest_categorical(
        'i2_node_label',
        node_label_choices(args.dataset),
    )
    cfg.i2gnn.max_nodes_per_hop = trial.suggest_categorical(
        'i2_max_nodes_per_hop',
        max_nodes_per_hop_choices(args.dataset),
    )

    cfg.i2gnn.use_rd_auto = False
    cfg.i2gnn.use_rd = trial.suggest_categorical(
        'i2_use_rd',
        use_rd_choices(args.dataset),
    )

    cfg.i2gnn.gnn_type = (
        trial.suggest_categorical('i2_gnn_type', gnn_type_choices(args.dataset))
        if dtype in {'ogb', 'lrgb'} else 'gin'
    )
    if dtype in {'ogb', 'lrgb'}:
        cfg.i2gnn.virtual_node = trial.suggest_categorical(
            f'i2_virtual_node_{dtype}',
            bool_choices_with_default_false(),
        )
        cfg.i2gnn.residual = trial.suggest_categorical(
            f'i2_residual_{dtype}',
            bool_choices_with_default_false(),
        )
    elif dtype == 'tu':
        cfg.i2gnn.virtual_node = False
        cfg.i2gnn.residual = trial.suggest_categorical(
            'i2_residual_tu',
            bool_choices_with_default_false(),
        )
    elif dtype == 'pyg_mol':
        cfg.i2gnn.virtual_node = trial.suggest_categorical(
            'i2_virtual_node_pyg_mol', bool_choices_with_default_false()
        )
        cfg.i2gnn.residual = True
    else:
        cfg.i2gnn.virtual_node = False
        cfg.i2gnn.residual = True

    if dtype == 'pyg_mol':
        cfg.i2gnn.rni = trial.suggest_categorical(
            f'i2_rni_{dtype}',
            bool_choices_with_default_false(),
        )
    else:
        cfg.i2gnn.rni = False

    suggest_pooling_cfg(trial, cfg, args.dataset)

    cfg.train.lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    cfg.train.wd = trial.suggest_categorical(
        'weight_decay',
        [0.0, 1e-5, 1e-4, 5e-4, 1e-3],
    )
    cfg.train.scheduler = trial.suggest_categorical(
        'scheduler',
        [None, 'StepLR', 'ReduceLR', 'CosRestart'],
    )
    cfg.train.epochs = args.train_epochs
    cfg.train.patience = args.train_patience
    cfg.train.runs = args.train_runs
    cfg.train.batch_size = args.batch_size

    return apply_main_effectiveness_trial_protocol(cfg, args)


def close_trial_logger(name):
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        handler.flush()
        logger.removeHandler(handler)
        if isinstance(handler, logging.FileHandler):
            handler.close()


def objective(trial, base_cfg, args, optimize_metric):
    cfg = suggest_cfg(trial, base_cfg, args)
    config_string = f"tuning_trial_{trial.number:04d}_{cfg.dataset}_I2GNN"

    def prune_callback(value, step):
        trial.report(value, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    try:
        result = run_i2gnn(
            cfg=cfg,
            key_metric_override=optimize_metric,
            config_string=config_string,
            prune_callback=prune_callback,
            evaluate_test=False,
        )
    except optuna.TrialPruned:
        close_trial_logger(config_string)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    except FloatingPointError as exc:
        close_trial_logger(config_string)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise optuna.TrialPruned(f"Numerically unstable trial: {exc}") from exc
    except ValueError as exc:
        if 'nan' in str(exc).lower() or 'non-finite' in str(exc).lower():
            close_trial_logger(config_string)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.TrialPruned(f"Numerically unstable trial: {exc}") from exc
        raise
    except RuntimeError as exc:
        if 'out of memory' in str(exc).lower():
            close_trial_logger(config_string)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.TrialPruned(f"CUDA out of memory: {exc}") from exc
        raise

    val_stats = result['best_val_perf']
    efficiency = result['efficiency']
    if optimize_metric not in val_stats:
        raise KeyError(f"Metric {optimize_metric} not found in validation metrics: {sorted(val_stats)}")

    value = val_stats[optimize_metric]['mean']
    if not math.isfinite(float(value)):
        raise optuna.TrialPruned(f"Non-finite objective value: {value}")
    trial.set_user_attr('best_val_perf', compact_stats(val_stats))
    trial.set_user_attr('efficiency', compact_stats(efficiency))

    print(f"\n[Trial {trial.number}] optimize {optimize_metric}={value:.6f}")
    print("  Val:", json.dumps(compact_stats(val_stats), sort_keys=True))
    print("  Test: skipped (validation-only tuning trial)")
    print("  Efficiency:", json.dumps(compact_stats(efficiency), sort_keys=True))

    return value


def main():
    args = parse_args()
    # Fail early for unsupported names and keep error text close to the runner.
    supported_dataset_type(args.dataset)

    base_cfg = init_cfg()
    optimize_metric = normalize_metric(args.val_metric, args.dataset)
    if args.val_metric == 'auto':
        optimize_metric = (
            main_effectiveness_default_metric(args.dataset)
            if is_main_effectiveness_fixed_split(args.dataset)
            else default_val_metric(args.dataset)
        )
    direction = metric_direction(optimize_metric)
    study_name = args.study_name or default_study_name(args, optimize_metric)
    n_trials = 200 if args.n_trials is None else int(args.n_trials)
    print_main_effectiveness_protocol(args.dataset, n_trials, args.seed)

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction=direction,
        load_if_exists=bool(args.storage),
        sampler=optuna.samplers.TPESampler(
            seed=42 if is_main_effectiveness_fixed_split(args.dataset) else args.seed,
            n_startup_trials=min(10, max(2, n_trials // 3)),
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10, interval_steps=5
        ),
    )
    optimize_to_total(
        study,
        lambda trial: objective(trial, base_cfg, args, optimize_metric),
        n_trials,
    )

    print("\nBest trial:")
    print(f"  number: {study.best_trial.number}")
    print(f"  {optimize_metric}: {study.best_value:.6f}")
    print("  params:", json.dumps(study.best_trial.params, sort_keys=True))
    print("  val:", json.dumps(study.best_trial.user_attrs.get('best_val_perf', {}), sort_keys=True))
    print("  test: not evaluated during tuning")

    if is_main_effectiveness_fixed_split(args.dataset):
        return run_final_evaluation(
            study, suggest_cfg, base_cfg, args, run_i2gnn,
            'I2GNN', optimize_metric, final_seeds=(42, 43, 44),
        )
    if is_tu_dataset(args.dataset):
        return run_tu_cv_final_evaluation(
            study, suggest_cfg, base_cfg, args, run_i2gnn,
            'I2GNN', optimize_metric,
        )


if __name__ == '__main__':
    main()
