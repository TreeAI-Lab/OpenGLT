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
from datasets import is_good_dataset_name
from example.gnnak import run_gnnak, supported_dataset_type
from tuning.main_baselines_tuning import (
    GCARE_REGRESSION,
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
    is_main_effectiveness_fixed_split,
    main_effectiveness_default_metric,
    print_main_effectiveness_protocol,
)
from tuning.evaluation import is_tu_dataset, run_tu_cv_final_evaluation


SEARCH_SPACE_VERSION = 'gnnak_v2_main30'


def is_good_zinc_dataset(dataset):
    return dataset.lower().replace('_', '-').startswith('good-zinc-')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ogbg-molbace')
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
    if is_good_zinc_dataset(args.dataset):
        return f"gnnak_good_zinc_targeted10_{dataset}_{optimize_metric}"
    return f"gnnak_{SEARCH_SPACE_VERSION}_{dataset}_{optimize_metric}"


def dataset_type(dataset):
    return supported_dataset_type(dataset)


def is_gnce_dataset(dataset):
    return dataset.lower().startswith('gnce-')


def is_card_query_dataset(dataset):
    return dataset.lower().startswith('gnce-')


def resolve_n_trials(args):
    return resolve_protocol_n_trials(
        args.dataset, args.n_trials, non_gnce_default=200
    )


def size_bucket(dataset):
    dataset = str(dataset)
    dataset_lower = dataset.lower()
    if dataset_lower.startswith('gnce-'):
        return 'card_query'
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


def outer_layer_choices(dataset):
    return [3, 4, 5]


def mini_layer_choices(dataset):
    return [1, 2, 3]


def dropout_choices():
    return [0.0, 0.1, 0.3, 0.5]


def pool_choices(dataset):
    return ['mean', 'max', 'sum']


def hop_dim_choices(dataset):
    return [8, 16, 32]


def embs_profile_choices(dataset):
    if size_bucket(dataset) == 'counting':
        return ['0,2|add', '1|add']
    return ['0,1,2|concat', '0,1,2|add', '0,2|add']


def subgraph_pattern_choices(dataset):
    if size_bucket(dataset) == 'counting':
        return ['ego1']
    if size_bucket(dataset) in {'large', 'large_tu'}:
        return ['ego1']
    return ['ego1', 'ego2']


def sampling_profile_choices(dataset):
    if size_bucket(dataset) == 'counting':
        return ['shortest_r2_s5_b4']
    if size_bucket(dataset) == 'large_tu':
        return ['shortest_r2_s3_b2']
    return ['none', 'shortest_r2_s3_b2', 'shortest_r2_s5_b4']


def parse_embs_profile(profile):
    embs, combine = profile.split('|')
    return [int(x) for x in embs.split(',') if x], combine


def apply_subgraph_pattern(cfg, profile):
    if profile.startswith('ego'):
        cfg.gnnak.subgraph_hops = int(profile.replace('ego', ''))
        cfg.gnnak.walk_length = 0
        cfg.gnnak.walk_repeat = 5
        cfg.gnnak.walk_p = 1.0
        cfg.gnnak.walk_q = 1.0
        cfg.gnnak.use_normal_gnn = False
        return

    if profile == 'rw8_r4':
        cfg.gnnak.subgraph_hops = 0
        cfg.gnnak.walk_length = 8
        cfg.gnnak.walk_repeat = 4
    elif profile == 'rw10_r5':
        cfg.gnnak.subgraph_hops = 0
        cfg.gnnak.walk_length = 10
        cfg.gnnak.walk_repeat = 5
    else:
        raise ValueError(f"Unknown GNNAK subgraph pattern: {profile}")
    cfg.gnnak.walk_p = 1.0
    cfg.gnnak.walk_q = 1.0
    # Official code recommends adding the normal GNN path when RW subgraphs may
    # miss the full first-hop neighborhood.
    cfg.gnnak.use_normal_gnn = True


def apply_sampling_profile(cfg, profile):
    if profile == 'none':
        cfg.gnnak.sampling_mode = 'none'
        cfg.gnnak.sampling_redundancy = 0
        cfg.gnnak.sampling_stride = 2
        cfg.gnnak.sampling_random_rate = 0.5
        cfg.gnnak.sampling_batch_factor = 1
        return

    parts = profile.split('_')
    mode = 'shortest_path' if parts[0] == 'shortest' else parts[0]
    cfg.gnnak.sampling_mode = mode
    cfg.gnnak.sampling_redundancy = 0
    cfg.gnnak.sampling_stride = 2
    cfg.gnnak.sampling_random_rate = 0.5
    cfg.gnnak.sampling_batch_factor = 1
    for part in parts[1:]:
        if part.startswith('r') and part[1:].isdigit():
            cfg.gnnak.sampling_redundancy = int(part[1:])
        elif part.startswith('s') and part[1:].isdigit():
            cfg.gnnak.sampling_stride = int(part[1:])
        elif part.startswith('b') and part[1:].isdigit():
            cfg.gnnak.sampling_batch_factor = int(part[1:])
        elif part.startswith('rate'):
            cfg.gnnak.sampling_random_rate = float(f"0.{part.replace('rate', '')}")
        else:
            raise ValueError(f"Unknown GNNAK sampling token: {part}")


def set_fixed_defaults(cfg):
    cfg.load_best_params = False
    cfg.use_augm_gca = False
    cfg.model.backbone = 'GIN'
    cfg.model.pooling_method = 'none'
    cfg.model.use_fingerprints = False
    cfg.model.use_rwpe = False
    cfg.gnnak.gnn_type = 'GINEConv'
    cfg.gnnak.force_reload = False
    cfg.gnnak.online = True


def suggest_arch_cfg(trial, cfg, args):
    cfg.gnnak.hidden_size = trial.suggest_categorical(
        'gnnak_hidden',
        hidden_choices(args.dataset),
    )
    cfg.gnnak.num_layers = trial.suggest_categorical(
        'gnnak_outer_layers',
        outer_layer_choices(args.dataset),
    )
    cfg.gnnak.mini_layers = trial.suggest_categorical(
        'gnnak_mini_layers',
        mini_layer_choices(args.dataset),
    )
    cfg.gnnak.dropout = trial.suggest_categorical(
        'gnnak_dropout',
        dropout_choices(),
    )
    cfg.gnnak.pool = trial.suggest_categorical(
        'gnnak_pool',
        pool_choices(args.dataset),
    )
    cfg.gnnak.residual = trial.suggest_categorical(
        'gnnak_residual',
        [True, False],
    )
    cfg.gnnak.virtual_node = trial.suggest_categorical('gnnak_virtual_node', [False, True])
    cfg.gnnak.hops_dim = trial.suggest_categorical(
        'gnnak_hops_dim',
        hop_dim_choices(args.dataset),
    )
    cfg.gnnak.mlp_layers = trial.suggest_categorical(
        'gnnak_mlp_layers',
        [1],
    )

    embs_profile = trial.suggest_categorical(
        'gnnak_embs_profile',
        embs_profile_choices(args.dataset),
    )
    cfg.gnnak.embs, cfg.gnnak.embs_combine_mode = parse_embs_profile(embs_profile)


def suggest_subgraph_cfg(trial, cfg, args):
    pattern = trial.suggest_categorical(
        'gnnak_subgraph_pattern',
        subgraph_pattern_choices(args.dataset),
    )
    apply_subgraph_pattern(cfg, pattern)

    sampling = trial.suggest_categorical(
        'gnnak_sampling_profile',
        sampling_profile_choices(args.dataset),
    )
    apply_sampling_profile(cfg, sampling)


def suggest_training_cfg(trial, cfg, args):
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


def validate_cfg(cfg):
    if int(cfg.gnnak.mini_layers) <= 0:
        raise ValueError("GNNAK+ tuning requires gnnak.mini_layers > 0.")
    if int(cfg.gnnak.hops_dim) <= 0:
        raise ValueError("GNNAK+ tuning requires gnnak.hops_dim > 0.")
    if cfg.gnnak.embs_combine_mode not in {'add', 'concat'}:
        raise ValueError("gnnak.embs_combine_mode must be add or concat.")
    if not cfg.gnnak.embs:
        raise ValueError("gnnak.embs must contain at least one embedding type.")
    if any(e not in {0, 1, 2} for e in cfg.gnnak.embs):
        raise ValueError("gnnak.embs values must be 0, 1, or 2.")
    if str(cfg.gnnak.sampling_mode).lower() not in {'none', 'shortest_path', 'random', 'min_set_cover'}:
        raise ValueError(f"Unknown GNNAK sampling mode: {cfg.gnnak.sampling_mode}")
    if cfg.gnnak.walk_length > 0 and not cfg.gnnak.use_normal_gnn:
        raise ValueError("Random-walk GNNAK subgraphs should enable use_normal_gnn.")


def suggest_cfg(trial, base_cfg, args):
    cfg = copy.deepcopy(base_cfg)
    cfg.dataset = args.dataset
    cfg.device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    cfg.seed = args.seed
    cfg.cv.fold_k = args.fold_k
    cfg.cv.seed = args.fold_seed
    cfg.deterministic = args.deterministic
    cfg.deterministic_warn_only = True

    set_fixed_defaults(cfg)
    if is_good_zinc_dataset(args.dataset):
        # Broader but bounded space, centred on the main-ZINC validation-best
        # configuration.  Important AK+ architecture and optimization choices
        # are sampled independently by Optuna rather than bundled as presets.
        cfg.gnnak.hidden_size = trial.suggest_categorical(
            'gnnak_hidden', [128, 256]
        )
        cfg.gnnak.num_layers = trial.suggest_categorical(
            'gnnak_outer_layers', [3, 4, 5]
        )
        cfg.gnnak.mini_layers = trial.suggest_categorical(
            'gnnak_mini_layers', [1, 2, 3]
        )
        cfg.gnnak.dropout = trial.suggest_categorical(
            'gnnak_dropout', [0.0, 0.1, 0.3]
        )
        cfg.gnnak.pool = trial.suggest_categorical(
            'gnnak_pool', ['mean', 'add', 'max']
        )
        cfg.gnnak.residual = trial.suggest_categorical(
            'gnnak_residual', [True, False]
        )
        cfg.gnnak.virtual_node = trial.suggest_categorical(
            'gnnak_virtual_node', [True, False]
        )
        cfg.gnnak.hops_dim = 8
        cfg.gnnak.mlp_layers = 1
        embs_profile = trial.suggest_categorical(
            'gnnak_embs_profile',
            ['0,1,2|concat', '0,1,2|add', '0,2|add'],
        )
        cfg.gnnak.embs, cfg.gnnak.embs_combine_mode = parse_embs_profile(
            embs_profile
        )
        pattern = trial.suggest_categorical(
            'gnnak_subgraph_pattern', ['ego1', 'ego2']
        )
        apply_subgraph_pattern(cfg, pattern)
        sampling = trial.suggest_categorical(
            'gnnak_sampling_profile', ['none', 'shortest_r2_s3_b2']
        )
        apply_sampling_profile(cfg, sampling)

        cfg.train.lr = trial.suggest_categorical(
            'lr', [1e-4, 3e-4, 5e-4, 1e-3]
        )
        cfg.train.wd = trial.suggest_categorical(
            'weight_decay', [0.0, 1e-5, 1e-4, 1e-3]
        )
        cfg.train.scheduler = trial.suggest_categorical(
            'scheduler', ['StepLR', 'ReduceLR', 'CosRestart']
        )
        cfg.train.epochs = args.train_epochs
        cfg.train.patience = args.train_patience
        cfg.train.runs = args.train_runs
        cfg.train.batch_size = args.batch_size
    else:
        suggest_arch_cfg(trial, cfg, args)
        suggest_subgraph_cfg(trial, cfg, args)
        suggest_training_cfg(trial, cfg, args)
    validate_cfg(cfg)
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
    config_string = f"tuning_trial_{trial.number:04d}_{cfg.dataset}_GNNAK"

    def prune_callback(value, step):
        trial.report(value, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    try:
        result = run_gnnak(
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
    trial.set_user_attr('best_val_perf', compact_stats(val_stats))
    trial.set_user_attr('efficiency', compact_stats(efficiency))

    print(f"\n[Trial {trial.number}] optimize {optimize_metric}={value:.6f}")
    print("  Val:", json.dumps(compact_stats(val_stats), sort_keys=True))
    print("  Test: skipped (validation-only tuning trial)")
    print("  Efficiency:", json.dumps(compact_stats(efficiency), sort_keys=True))
    return value


def main():
    args = parse_args()
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

    # Every tuning trial is validation-only; test evaluation starts only after
    # Optuna selects the best configuration.
    study_name = args.study_name or default_study_name(args, optimize_metric)
    n_trials = resolve_n_trials(args)
    sampler_seed = (
        42 if is_main_effectiveness_fixed_split(args.dataset)
        else resolve_tuning_seed(args.dataset, args.seed)
    )
    print_protocol(args.dataset, n_trials, args.seed)
    print_main_effectiveness_protocol(args.dataset, n_trials, args.seed)

    sampler = optuna.samplers.TPESampler(
        seed=sampler_seed,
        n_startup_trials=(
            3 if is_good_zinc_dataset(args.dataset)
            else min(10, max(2, n_trials // 3))
        ),
        multivariate=True,
    )
    if is_good_zinc_dataset(args.dataset):
        print(
            "[GOOD AK Optuna space] independently sampling bounded "
            "architecture, subgraph, readout, and optimizer choices."
        )

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction=direction,
        load_if_exists=bool(args.storage),
        sampler=sampler,
        pruner=(
            optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=10, interval_steps=5
            )
            if is_main_effectiveness_fixed_split(args.dataset)
            else make_pruner()
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

    if (
        protocol_is_fixed_split_dataset(args.dataset)
        or is_main_effectiveness_fixed_split(args.dataset)
    ):
        run_final_evaluation(
            study, suggest_cfg, base_cfg, args, run_gnnak,
            "GNNAK+", optimize_metric,
            final_seeds=(42, 43, 44)
            if is_main_effectiveness_fixed_split(args.dataset) else None,
        )
    elif is_tu_dataset(args.dataset):
        run_tu_cv_final_evaluation(
            study, suggest_cfg, base_cfg, args, run_gnnak,
            'GNNAK+', optimize_metric,
        )


if __name__ == '__main__':
    main()
