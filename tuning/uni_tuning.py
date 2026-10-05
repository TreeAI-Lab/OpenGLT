import argparse
import copy
import json
import logging
import math

import optuna
import torch

from configs import init_cfg
from tuning.main_baselines_tuning import (
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


SEARCH_SPACE_VERSION = 'v2_main30'
OGB_DATASETS = {'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molpcba', 'ogbg-moltox21'}
TU_DATASETS = {
    'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB',
    'MUTAG', 'NCI1', 'DD', 'ENZYMES', 'PROTEINS',
}
MOLECULAR_DATASETS = {'zinc', 'aqsol', 'peptides-func', 'peptides-struct'}
LARGE_DATASETS = {
    'REDDIT-BINARY', 'COLLAB', 'DD',
    'ogbg-molpcba', 'peptides-func', 'peptides-struct',
}


def parse_common_args(default_dataset='ogbg-molbace'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=default_dataset)
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
    parser.add_argument(
        '--tuning_fold_limit', type=int, default=0,
        help='Validation-only TU trials use at most this many folds; 0 uses all.',
    )
    return parser.parse_args()


def prepare_base_cfg(base_cfg, args):
    cfg = copy.deepcopy(base_cfg)
    cfg.dataset = args.dataset
    cfg.device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    cfg.seed = args.seed
    cfg.cv.fold_k = args.fold_k
    cfg.cv.seed = args.fold_seed
    cfg.cv.tuning_fold_limit = max(
        0, int(getattr(args, 'tuning_fold_limit', 0))
    )
    cfg.deterministic = args.deterministic
    cfg.deterministic_warn_only = True
    cfg.load_best_params = False

    cfg.model.use_fingerprints = False
    cfg.model.use_rwpe = False
    cfg.use_augm_gca = False

    cfg.train.epochs = args.train_epochs
    cfg.train.patience = args.train_patience
    cfg.train.runs = args.train_runs
    cfg.train.batch_size = args.batch_size
    return apply_main_effectiveness_trial_protocol(cfg, args)


def suggest_training_cfg(trial, cfg, args=None):
    dataset = cfg.dataset if args is None else args.dataset
    family = dataset_family(dataset)
    cfg.train.lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    cfg.train.wd = trial.suggest_categorical(
        'weight_decay', [0.0, 1e-5, 1e-4, 5e-4, 1e-3]
    )
    cfg.train.scheduler = trial.suggest_categorical(
        'scheduler', [None, 'StepLR', 'ReduceLR', 'CosRestart']
    )


def dataset_family(dataset):
    dataset_lower = dataset.lower()
    if dataset.startswith('Count-'):
        return 'counting'
    if dataset in OGB_DATASETS:
        return 'ogb'
    if dataset in TU_DATASETS:
        return 'tu'
    if dataset_lower in MOLECULAR_DATASETS:
        return 'molecular_regression'
    if dataset_lower in {'gcare-aids', 'gcare-yago'} or dataset_lower.startswith('gnce-'):
        return 'query_regression'
    return 'generic'


def parse_width_layers(value):
    hidden_size, num_layers = value.split('x')
    return int(hidden_size), int(num_layers)


def parse_profile(value, expected_parts):
    parts = value.split('x')
    if len(parts) != expected_parts:
        raise ValueError(f"Expected {expected_parts} profile fields, got: {value}")
    return tuple(int(part) for part in parts)


def resolve_n_trials(args, default=200, card_default=200):
    if args.n_trials is not None:
        return args.n_trials
    if is_main_effectiveness_fixed_split(args.dataset):
        return default
    return card_default if args.dataset.lower().startswith('gnce-') else default


def close_trial_logger(name):
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        handler.flush()
        logger.removeHandler(handler)
        if isinstance(handler, logging.FileHandler):
            handler.close()


def parse_width_heads(value):
    hidden_size, num_heads = value.split('x')
    return int(hidden_size), int(num_heads)


def predictor_multiplier_from_profile(profile, hidden_size):
    if profile == 'official':
        return int(hidden_size)
    if profile == 'medium':
        return max(1, int(hidden_size) // 2)
    if profile == 'small':
        return max(1, int(hidden_size) // 8)
    raise ValueError(f"Unknown predictor profile: {profile}")


def default_study_name(model_name, args, optimize_metric):
    dataset = args.dataset.replace('/', '_')
    return f"{model_name.lower()}_uni_{SEARCH_SPACE_VERSION}_{dataset}_{optimize_metric}"


def objective(trial, base_cfg, args, optimize_metric, model_name, run_fn, suggest_cfg_fn,
              config_string_fn=None, prune_exceptions=()):
    cfg = suggest_cfg_fn(trial, base_cfg, args)
    if config_string_fn is None:
        config_string = f"tuning_trial_{trial.number:04d}_{cfg.dataset}_{model_name}"
    else:
        config_string = config_string_fn(trial, cfg, args, model_name)

    def prune_callback(value, step):
        trial.report(value, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    try:
        result = run_fn(
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
    except (FloatingPointError,) + tuple(prune_exceptions) as exc:
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
    if not math.isfinite(float(value)):
        raise optuna.TrialPruned(f"Non-finite objective value: {value}")

    trial.set_user_attr('best_val_perf', compact_stats(val_stats))
    trial.set_user_attr('efficiency', compact_stats(efficiency))

    print(f"\n[Trial {trial.number}] optimize {optimize_metric}={value:.6f}")
    print("  Val:", json.dumps(compact_stats(val_stats), sort_keys=True))
    print("  Test: skipped (validation-only tuning trial)")
    print("  Efficiency:", json.dumps(compact_stats(efficiency), sort_keys=True))
    return value


def run_tuning(model_name, run_fn, suggest_cfg_fn, args=None, parse_args_fn=None,
               base_cfg_fn=None, study_name_fn=None, n_trials_fn=None,
               config_string_fn=None, pre_run_validate_fn=None,
               prune_exceptions=(), enqueue_trials=None):
    if args is None and parse_args_fn is not None:
        args = parse_args_fn()
    elif args is None:
        args = parse_common_args()
    if pre_run_validate_fn is not None:
        pre_run_validate_fn(args)
    base_cfg = init_cfg()
    if base_cfg_fn is not None:
        base_cfg = base_cfg_fn(args, base_cfg)

    optimize_metric = normalize_metric(args.val_metric, args.dataset)
    if args.val_metric == 'auto':
        optimize_metric = (
            main_effectiveness_default_metric(args.dataset)
            if is_main_effectiveness_fixed_split(args.dataset)
            else default_val_metric(args.dataset)
        )
    direction = metric_direction(optimize_metric)
    name_fn = study_name_fn or default_study_name
    study_name = args.study_name or name_fn(model_name, args, optimize_metric)
    n_trials = n_trials_fn(args) if n_trials_fn is not None else resolve_n_trials(args)
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
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5,
        ),
    )
    if enqueue_trials and len(study.trials) == 0:
        for params in enqueue_trials:
            study.enqueue_trial(params)
        print(
            f"[Optuna] enqueued {len(enqueue_trials)} reference "
            "configuration(s) before sampled trials."
        )
    optimize_to_total(
        study,
        lambda trial: objective(
            trial, base_cfg, args, optimize_metric, model_name, run_fn, suggest_cfg_fn,
            config_string_fn=config_string_fn,
            prune_exceptions=prune_exceptions,
        ),
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
            study,
            suggest_cfg_fn,
            base_cfg,
            args,
            run_fn,
            model_name,
            optimize_metric,
            final_seeds=(42, 43, 44),
        )
    if is_tu_dataset(args.dataset):
        return run_tu_cv_final_evaluation(
            study,
            suggest_cfg_fn,
            base_cfg,
            args,
            run_fn,
            model_name,
            optimize_metric,
        )
