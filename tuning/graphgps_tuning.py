import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import optuna
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import init_cfg
from datasets import good_model_dataset_type, is_good_dataset_name
from example.graphgps import (
    ensure_graphgps_cfg,
    run_graphgps,
    validate_graphgps_cfg,
)
from tuning.main_baselines_tuning import (
    OGB_BINARY,
    OGB_MULTI_LABEL,
    LRGB_MULTI_LABEL,
    LRGB_REGRESSION,
    PYG_MOL_REGRESSION,
    GCARE_REGRESSION,
    TU_DATASETS,
    compact_stats,
    default_val_metric,
    metric_direction,
    normalize_metric,
)
from tuning.gnce_protocol import (
    apply_trial_protocol,
    is_fixed_split_dataset as protocol_is_fixed_split_dataset,
    make_pruner,
    optimize_to_total,
    print_protocol,
    resolve_n_trials as resolve_protocol_n_trials,
    resolve_tuning_seed,
    run_final_evaluation,
)
from tuning.main_effectiveness_protocol import (
    apply_main_effectiveness_trial_protocol,
    main_effectiveness_default_metric,
    print_main_effectiveness_protocol,
)
from tuning.evaluation import is_tu_dataset, run_tu_cv_final_evaluation


SEARCH_SPACE_VERSION = 'gps_v3_count_main30_r2'

COUNT_REFERENCE_CONFIGS = {
    'Count-4-Clique': {
        'hidden_size': 64,
        'num_layers': 10,
        'dropout': 0.25,
        'attn_dropout': 0.5,
        'lr': 0.000376300131436296,
        'weight_decay': 0.000979896058891537,
        'num_warmup_epochs': 19,
    },
    'Count-Tri-Rect': {
        'hidden_size': 64,
        'num_layers': 8,
        'dropout': 0.4,
        'attn_dropout': 0.3,
        'lr': 0.00160017204402897,
        'weight_decay': 0.0000641720062349284,
        'num_warmup_epochs': 7,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ogbg-molhiv')
    parser.add_argument('--val_metric', default='auto')
    parser.add_argument('--n_trials', type=int, default=None)
    parser.add_argument('--train_epochs', type=int, default=None)
    parser.add_argument('--train_patience', type=int, default=None)
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
    return f"graphgps_{SEARCH_SPACE_VERSION}_{dataset}_{optimize_metric}"


def dataset_type(dataset):
    dataset = str(dataset)
    dataset_lower = dataset.lower()
    if dataset.startswith('ogbg-'):
        return 'ogb'
    if dataset in TU_DATASETS:
        return 'tu'
    if dataset_lower in LRGB_MULTI_LABEL or dataset_lower in LRGB_REGRESSION:
        return 'lrgb'
    if dataset_lower in PYG_MOL_REGRESSION:
        return 'pyg_mol'
    if dataset_lower in GCARE_REGRESSION or dataset_lower.startswith('gnce-'):
        return 'tu'
    if is_good_dataset_name(dataset):
        return good_model_dataset_type(dataset)
    if dataset.startswith('Count-'):
        return 'counting'
    raise ValueError(f"Dataset {dataset} is not supported by GraphGPS tuning.")


def is_gnce_dataset(dataset):
    return dataset.lower().startswith('gnce-')


def is_card_query_dataset(dataset):
    return dataset.lower().startswith('gnce-')


def resolve_n_trials(args):
    return resolve_protocol_n_trials(
        args.dataset,
        args.n_trials,
        non_gnce_default=200,
    )


def supports_lappe(dataset):
    return (
        dataset_type(dataset) in {'ogb', 'pyg_mol', 'good_mol'}
        or is_good_dataset_name(dataset)
    )


def local_gnn_choices(dataset):
    dtype = dataset_type(dataset)
    if dtype == 'pyg_mol':
        return ['GINE', 'CustomGatedGCN']
    if dtype in {'ogb', 'good_mol'}:
        return ['CustomGatedGCN', 'GINE', 'PNA']
    return ['GINE', 'PNA']


def hidden_choices(dataset):
    if dataset.startswith('Count-'):
        return [64]
    if str(dataset) in {'DD', 'COLLAB', 'REDDIT-BINARY'}:
        return [64, 128]
    return [64, 128, 256, 384]


def layer_choices(dataset):
    if dataset == 'Count-4-Clique':
        return [8, 10, 12]
    if dataset == 'Count-Tri-Rect':
        return [6, 8, 10]
    if dataset.startswith('Count-'):
        return [6, 8, 10]
    if str(dataset) in {'DD', 'COLLAB', 'REDDIT-BINARY'}:
        return [3, 5]
    return [3, 5, 8, 10]


def compatible_heads(hidden_size):
    return [heads for heads in [2, 4, 8] if hidden_size % heads == 0]


def compatible_lappe_dims(hidden_size):
    return [dim for dim in [4, 8, 16, 32] if dim < hidden_size]


def compatible_lappe_heads(lappe_dim):
    return [heads for heads in [2, 4, 8] if lappe_dim % heads == 0]


def count_search_choices(dataset):
    if dataset == 'Count-4-Clique':
        return {
            'dropout': [0.15, 0.25, 0.35],
            'attn_dropout': [0.3, 0.4, 0.5],
            'lr': [
                1.5e-4, 0.000376300131436296, 5e-4, 8e-4, 1.2e-3,
            ],
            'weight_decay': [
                1e-4, 3e-4, 6e-4, 0.000979896058891537, 2e-3,
            ],
            'warmup': [10, 19, 25],
        }
    if dataset == 'Count-Tri-Rect':
        return {
            'dropout': [0.3, 0.4, 0.5],
            'attn_dropout': [0.2, 0.3, 0.4],
            'lr': [
                5e-4, 1e-3, 0.00160017204402897, 2.5e-3, 4e-3,
            ],
            'weight_decay': [
                1e-5, 3e-5, 0.0000641720062349284, 1e-4, 3e-4,
            ],
            'warmup': [3, 7, 12],
        }
    return {
        'dropout': [0.2, 0.3, 0.4],
        'attn_dropout': [0.2, 0.3, 0.4],
        'lr': [3e-4, 5e-4, 1e-3, 2e-3],
        'weight_decay': [1e-5, 1e-4, 5e-4, 1e-3],
        'warmup': [5, 10, 20],
    }


def suggest_pe_cfg(trial, cfg, dataset):
    if dataset.startswith('Count-'):
        # Reference GraphGPS counting recipe: steps 1..20, raw BatchNorm,
        # Linear projection to a separately reserved 20-dimensional RWSE.
        cfg.graphgps.use_lappe = False
        cfg.graphgps.use_rwpe = True
        cfg.graphgps.rwpe_dim = 20
        cfg.graphgps.rwpe_encoder_type = 'graphgps'
        cfg.graphgps.rwpe_embed_dim = 20
        cfg.graphgps.rwpe_model = 'linear'
        cfg.graphgps.rwpe_layers = 1
        cfg.graphgps.rwpe_raw_norm_type = 'BatchNorm'
        return

    pe_choices = ['none', 'rwse']
    if supports_lappe(dataset):
        pe_choices += ['lappe', 'lappe_rwse']
    pe_mode = trial.suggest_categorical('pe_mode', pe_choices)

    cfg.graphgps.use_lappe = pe_mode in {'lappe', 'lappe_rwse'}
    cfg.graphgps.use_rwpe = pe_mode in {'rwse', 'lappe_rwse'}

    cfg.graphgps.lappe_eigvec_norm = 'L2'
    cfg.graphgps.lappe_laplacian_norm = 'none'

    if cfg.graphgps.use_lappe:
        cfg.graphgps.lappe_dim = trial.suggest_categorical(
            'lappe_dim',
            compatible_lappe_dims(cfg.graphgps.hidden_size),
        )
        cfg.graphgps.lappe_max_freqs = trial.suggest_categorical(
            'lappe_max_freqs',
            [4, 8, 10, 16],
        )
        cfg.graphgps.lappe_model = trial.suggest_categorical(
            'lappe_model',
            ['DeepSet', 'Transformer'],
        )
        cfg.graphgps.lappe_layers = trial.suggest_categorical(
            'lappe_layers',
            [1, 2, 3],
        )
        cfg.graphgps.lappe_raw_norm_type = trial.suggest_categorical(
            'lappe_raw_norm_type',
            ['none', 'BatchNorm'],
        )
        if cfg.graphgps.lappe_model == 'Transformer':
            lappe_heads_param = f'lappe_even_heads_dim{cfg.graphgps.lappe_dim}'
            cfg.graphgps.lappe_heads = trial.suggest_categorical(
                lappe_heads_param,
                compatible_lappe_heads(cfg.graphgps.lappe_dim),
            )
        else:
            cfg.graphgps.lappe_heads = 4
    else:
        cfg.graphgps.lappe_dim = 8
        cfg.graphgps.lappe_max_freqs = 8
        cfg.graphgps.lappe_model = 'DeepSet'
        cfg.graphgps.lappe_layers = 2
        cfg.graphgps.lappe_heads = 4
        cfg.graphgps.lappe_raw_norm_type = 'none'

    if cfg.graphgps.use_rwpe:
        cfg.graphgps.rwpe_dim = trial.suggest_categorical(
            'rwpe_dim',
            [8, 16, 20, 32],
        )
    else:
        cfg.graphgps.rwpe_dim = 20


def suggest_cfg(trial, base_cfg, args):
    cfg = copy.deepcopy(base_cfg)
    cfg.dataset = args.dataset
    cfg.device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    cfg.seed = args.seed
    cfg.cv.fold_k = args.fold_k
    cfg.cv.seed = args.fold_seed
    cfg.deterministic = args.deterministic
    cfg.deterministic_warn_only = True
    cfg.load_best_params = False
    cfg.use_augm_gca = False
    cfg.model.use_fingerprints = False

    cfg.graphgps.hidden_size = trial.suggest_categorical(
        'hidden_size',
        hidden_choices(args.dataset),
    )
    is_counting = args.dataset.startswith('Count-')
    cfg.graphgps.num_heads = trial.suggest_categorical(
        'num_heads', compatible_heads(cfg.graphgps.hidden_size)
    )
    cfg.graphgps.num_layers = trial.suggest_int('num_layers', 2, 10)
    cfg.graphgps.local_gnn_type = trial.suggest_categorical(
        'local_gnn_type', local_gnn_choices(args.dataset)
    )
    cfg.graphgps.global_model_type = trial.suggest_categorical(
        'global_model_type', ['Transformer', 'Performer']
    )
    cfg.graphgps.dropout = trial.suggest_float('dropout', 0.0, 0.5)
    cfg.graphgps.attn_dropout = trial.suggest_float('attn_dropout', 0.0, 0.5)
    cfg.graphgps.act_type = trial.suggest_categorical(
        'act_type', ['relu', 'gelu', 'swish']
    )
    norm_type = trial.suggest_categorical(
        'norm_type', ['none', 'batch_norm', 'layer_norm']
    )
    cfg.graphgps.batch_norm = norm_type == 'batch_norm'
    cfg.graphgps.layer_norm = norm_type == 'layer_norm'
    cfg.graphgps.readout_type = trial.suggest_categorical(
        'readout_type', ['mean', 'add', 'max', 'attn']
    )
    cfg.graphgps.head_layers = trial.suggest_int('head_layers', 1, 3)
    suggest_pe_cfg(trial, cfg, args.dataset)

    cfg.train.lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    cfg.train.wd = trial.suggest_categorical(
        'weight_decay', [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3]
    )
    cfg.train.scheduler = trial.suggest_categorical(
        'scheduler', [None, 'StepLR', 'ReduceLR', 'CosRestart', 'cosine_with_warmup']
    )
    if str(cfg.train.scheduler).lower() == 'cosine_with_warmup':
        warmup_epochs = trial.suggest_categorical(
            'num_warmup_epochs_choice', [1, 5, 20, 50]
        )
        resolved_epochs = args.train_epochs or (2000 if is_counting else 500)
        cfg.train.num_warmup_epochs = min(
            warmup_epochs, max(1, resolved_epochs - 1)
        )

    cfg.train.epochs = args.train_epochs or (2000 if is_counting else 500)
    cfg.train.patience = args.train_patience or (500 if is_counting else 50)
    cfg.train.clip_grad_norm = 5.0 if is_counting else 0.0
    cfg.train.runs = args.train_runs
    cfg.train.batch_size = args.batch_size

    validate_graphgps_cfg(cfg)
    cfg = apply_trial_protocol(cfg, args)
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
    config_string = f"tuning_trial_{trial.number:04d}_{cfg.dataset}_GraphGPS"

    def prune_callback(value, step):
        trial.report(value, step=step)
        # Strong 4-Clique runs can emerge only after hundreds of epochs.
        # Short-warmup median pruning systematically removes them.
        if not args.dataset.startswith('Count-') and trial.should_prune():
            raise optuna.TrialPruned()

    try:
        result = run_graphgps(
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
    except (FloatingPointError, ImportError) as exc:
        close_trial_logger(config_string)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise optuna.TrialPruned(f"Unstable or unavailable trial: {exc}") from exc
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
    trial.set_user_attr('best_val_perf', compact_stats(val_stats))
    trial.set_user_attr('efficiency', compact_stats(efficiency))

    print(f"\n[Trial {trial.number}] optimize {optimize_metric}={value:.6f}")
    print("  Val:", json.dumps(compact_stats(val_stats), sort_keys=True))
    print("  Test: skipped (validation-only tuning trial)")
    print("  Efficiency:", json.dumps(compact_stats(efficiency), sort_keys=True))

    return value


def main():
    args = parse_args()

    base_cfg = ensure_graphgps_cfg(init_cfg())
    base_cfg.dataset = args.dataset

    optimize_metric = normalize_metric(args.val_metric, args.dataset)
    if args.val_metric == 'auto':
        optimize_metric = (
            main_effectiveness_default_metric(args.dataset)
            if args.dataset.startswith('Count-')
            else default_val_metric(args.dataset)
        )
    direction = metric_direction(optimize_metric)

    study_name = args.study_name or default_study_name(args, optimize_metric)

    print(f"Optimizing {optimize_metric} ({direction})")
    n_trials = resolve_n_trials(args)
    sampler_seed = (
        42 if args.dataset.startswith('Count-')
        else resolve_tuning_seed(args.dataset, args.seed)
    )
    print_protocol(args.dataset, n_trials, args.seed)
    print_main_effectiveness_protocol(args.dataset, n_trials, args.seed)
    if args.dataset.startswith('Count-'):
        print(
            "[GraphGPS Count protocol] official fixed split; "
            f"n_trials={n_trials}, one seed (42) per trial, "
            "validation-R2 selection, no test evaluation during tuning."
        )
        print(
            "[GraphGPS Count protocol] selected configuration is then "
            f"reinitialized for seeds="
            "[42, 43, 44] and tested once."
        )

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction=direction,
        load_if_exists=bool(args.storage),
        sampler=optuna.samplers.TPESampler(
            seed=sampler_seed,
            n_startup_trials=min(10, max(2, n_trials // 3)),
        ),
        pruner=make_pruner(),
    )
    if args.dataset in COUNT_REFERENCE_CONFIGS and len(study.trials) == 0:
        study.enqueue_trial(COUNT_REFERENCE_CONFIGS[args.dataset])
        print(
            "[GraphGPS Count protocol] enqueued a reference configuration "
            "as the first trial."
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

    if (
        protocol_is_fixed_split_dataset(args.dataset)
        or args.dataset.startswith('Count-')
    ):
        run_final_evaluation(
            study, suggest_cfg, base_cfg, args, run_graphgps,
            "GraphGPS", optimize_metric,
            final_seeds=(42, 43, 44)
            if args.dataset.startswith('Count-') else None,
        )
    elif is_tu_dataset(args.dataset):
        run_tu_cv_final_evaluation(
            study, suggest_cfg, base_cfg, args, run_graphgps,
            'GraphGPS', optimize_metric,
        )


if __name__ == '__main__':
    main()
