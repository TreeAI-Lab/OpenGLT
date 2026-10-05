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
from datasets import is_good_dataset_name, parse_good_dataset_name
from example.mosgsl import run_mosgsl, supported_dataset_type
from tuning.main_baselines_tuning import (
    GCARE_REGRESSION,
    OGB_BINARY,
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
    is_main_effectiveness_fixed_split,
    main_effectiveness_default_metric,
    print_main_effectiveness_protocol,
)
from tuning.evaluation import is_tu_dataset, run_tu_cv_final_evaluation


SEARCH_SPACE_VERSION = 'mosgsl_v2_gncefix'
SINGLE_LABEL_DATASETS = OGB_BINARY | TU_DATASETS


class _ManifestChoiceCompat:
    """Replay saved categorical values while preserving current search choices."""

    def __init__(self, trial):
        self._trial = trial
        self._fixed_params = getattr(trial, '_params', {})

    def __getattr__(self, name):
        return getattr(self._trial, name)

    def suggest_categorical(self, name, choices):
        choices = list(choices)
        fixed = self._fixed_params.get(name)
        if name in self._fixed_params and fixed not in choices:
            choices.insert(0, fixed)
        return self._trial.suggest_categorical(name, choices)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ogbg-molbace')
    parser.add_argument('--val_metric', default='auto')
    parser.add_argument('--n_trials', type=int, default=None)
    parser.add_argument('--train_epochs', type=int, default=500)
    parser.add_argument('--train_patience', type=int, default=50)
    parser.add_argument('--train_runs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--pretrain_epochs', type=int, default=200)
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
    return f"mosgsl_{SEARCH_SPACE_VERSION}_{dataset}_{optimize_metric}"


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


def make_mosgsl_sampler(n_trials, seed):
    """Give small, expensive MOSGSL studies some actual TPE-guided trials."""
    if n_trials <= 20:
        # Optuna's default is 10 random startup trials, which turns a
        # 10-trial budget into pure random search.
        startup_trials = min(3, max(1, n_trials // 3))
        return optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=startup_trials
        )
    return optuna.samplers.TPESampler(seed=seed)


def make_mosgsl_pruner(n_trials):
    """Use pruning thresholds that remain active for a 10-trial budget."""
    if n_trials <= 20:
        # The shared GNCE pruner waits for 20 completed trials, so it can
        # never prune a 10-trial MOSGSL run.  MOSGSL is unusually expensive;
        # wait 30 main-training epochs, then compare every 10 epochs.
        return optuna.pruners.MedianPruner(
            n_startup_trials=min(2, max(1, n_trials // 4)),
            n_warmup_steps=30,
            interval_steps=10,
        )
    return make_pruner()


def size_bucket(dataset):
    dataset = str(dataset)
    dataset_lower = dataset.lower()
    if dataset_lower.startswith('gnce-'):
        return 'card_query'
    if dataset in {'ogbg-molpcba'} or dataset_lower in {'peptides-func', 'peptides-struct'}:
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


def is_single_label(dataset):
    dataset = str(dataset)
    if is_good_dataset_name(dataset):
        spec, _, _, _ = parse_good_dataset_name(dataset)
        return spec.task in {'b', 'mc'}
    return dataset in SINGLE_LABEL_DATASETS


def hidden_choices(dataset):
    return [64, 128, 256, 384]


def layer_choices(dataset):
    return [2, 3, 4, 5]


def backbone_choices(dataset):
    return ['gcn', 'gin', 'sage']


def parsing_choices(dataset):
    # The official model uses parse_bfs. Keeping original out of the default
    # space avoids huge full-edge subgraphs on large graph datasets.
    return ['parse_bfs']


def max_n_sub_choices(dataset):
    return [1, 2, 3]


def max_sub_size_choices(dataset):
    return [5, 8, 12, 15]


def use_select_choices(dataset):
    if is_gnce_dataset(dataset):
        # Every triple pattern is semantically part of a cardinality query;
        # dropping subgraphs is not a label-preserving augmentation.
        return [False]
    return [False, True]


def gsl_one_by_one_choices(dataset):
    if is_gnce_dataset(dataset):
        # Keep learned links local to each parsed query subgraph and avoid
        # cross-subgraph dense similarity construction.
        return [True]
    return [False, True]


def metric_heads_choices(hidden_size):
    return [h for h in [4, 8, 16] if h <= hidden_size]


def pretrain_choices(max_pretrain):
    if max_pretrain <= 0:
        return [0]
    base = [20, 50, 100, 200]
    choices = [x for x in base if x <= max_pretrain]
    if max_pretrain not in choices:
        choices.append(max_pretrain)
    return sorted(set(choices))


def gnce_pretrain_choices(max_pretrain):
    return pretrain_choices(max_pretrain)


def update_epoch_choices(train_epochs):
    choices = [5, 10, 20, 50]
    return [x for x in choices if x < max(train_epochs, 2)] or [1]


def set_fixed_defaults(cfg):
    cfg.load_best_params = False
    cfg.use_augm_gca = False
    cfg.model.use_fingerprints = False
    cfg.model.use_rwpe = False
    cfg.model.pooling_method = 'none'


def suggest_mosgsl_arch(trial, cfg, args):
    gnce = is_gnce_dataset(args.dataset)
    cfg.mosgsl.hidden_size = trial.suggest_categorical(
        'mos_hidden',
        hidden_choices(args.dataset),
    )
    cfg.mosgsl.num_layers = trial.suggest_categorical(
        'mos_layers',
        layer_choices(args.dataset),
    )
    cfg.mosgsl.backbone = trial.suggest_categorical(
        'mos_backbone',
        backbone_choices(args.dataset),
    )
    cfg.mosgsl.dropout = trial.suggest_categorical(
        'mos_dropout',
        [0.1, 0.3] if gnce else [0.0, 0.1, 0.2, 0.3, 0.5],
    )
    cfg.mosgsl.norm = trial.suggest_categorical(
        'mos_norm',
        ['BatchNorm', 'LayerNorm'] if gnce else ['BatchNorm', 'LayerNorm', 'none'],
    )
    cfg.mosgsl.pool = trial.suggest_categorical('mos_pool', ['mean', 'add', 'max'])
    cfg.mosgsl.act = trial.suggest_categorical(
        'mos_act', ['relu', 'gelu'] if gnce else ['relu', 'gelu', 'swish']
    )
    cfg.mosgsl.output_layers = trial.suggest_categorical(
        'mos_output_layers',
        [1, 2, 3],
    )


def suggest_parsing_and_gsl(trial, cfg, args):
    gnce = is_gnce_dataset(args.dataset)
    cfg.mosgsl.parsing = trial.suggest_categorical(
        'mos_parsing',
        parsing_choices(args.dataset),
    )
    cfg.mosgsl.max_n_sub = trial.suggest_categorical(
        'mos_max_n_sub',
        max_n_sub_choices(args.dataset),
    )
    cfg.mosgsl.max_sub_size = trial.suggest_categorical(
        'mos_max_sub_size',
        max_sub_size_choices(args.dataset),
    )
    cfg.mosgsl.force_reload = False

    cfg.mosgsl.use_gsl = True
    cfg.mosgsl.gsl_epsilon = trial.suggest_categorical(
        'mos_gsl_epsilon', [0.5, 0.7, 0.8, 0.9]
    )
    cfg.mosgsl.gsl_lambda = trial.suggest_categorical(
        'mos_gsl_lambda', [0.1, 0.2, 0.5, 0.8]
    )
    cfg.mosgsl.gsl_one_by_one = trial.suggest_categorical(
        'mos_gsl_one_by_one',
        gsl_one_by_one_choices(args.dataset),
    )
    cfg.mosgsl.metric_heads = trial.suggest_categorical(
        f'mos_metric_heads_hidden{cfg.mosgsl.hidden_size}',
        metric_heads_choices(cfg.mosgsl.hidden_size),
    )
    cfg.mosgsl.metric_weighted = trial.suggest_categorical(
        'mos_metric_weighted',
        [True] if gnce else [True, False],
    )

    cfg.mosgsl.use_select = trial.suggest_categorical(
        'mos_use_select',
        use_select_choices(args.dataset),
    )
    if cfg.mosgsl.use_select:
        cfg.mosgsl.select_topk = trial.suggest_categorical(
            'mos_select_topk',
            [0.3, 0.5, 0.8],
        )
        cfg.mosgsl.select_estimator = trial.suggest_categorical(
            'mos_select_estimator',
            ['projection', 'mlp'],
        )
    else:
        cfg.mosgsl.select_topk = 1.0
        cfg.mosgsl.select_estimator = 'projection'
    cfg.mosgsl.select_min_score = -1.0


def suggest_pretrain_and_motif(trial, cfg, args):
    if cfg.mosgsl.use_gsl:
        cfg.mosgsl.pretrain_epochs = trial.suggest_categorical(
            f'mos_pretrain_epochs_max{args.pretrain_epochs}',
            (
                [epoch for epoch in [20, 50, 100, 200] if epoch <= args.pretrain_epochs]
                or [min(20, args.pretrain_epochs)]
            ) if is_gnce_dataset(args.dataset) else pretrain_choices(args.pretrain_epochs),
        )
        cfg.mosgsl.pretrain_backbone_only = trial.suggest_categorical(
            'mos_pretrain_backbone_only',
            [True] if is_gnce_dataset(args.dataset) else [False, True],
        )
    else:
        cfg.mosgsl.pretrain_epochs = 0
        cfg.mosgsl.pretrain_backbone_only = False

    if is_single_label(args.dataset) and cfg.mosgsl.use_gsl:
        cfg.mosgsl.use_motif = trial.suggest_categorical(
            f'mos_use_motif_{dataset_type(args.dataset)}',
            [True, False],
        )
    else:
        cfg.mosgsl.use_motif = False

    if cfg.mosgsl.use_motif:
        cfg.mosgsl.use_separate_encoder = trial.suggest_categorical(
            'mos_use_separate_encoder',
            [True, False],
        )
        cfg.mosgsl.motif_per_class = trial.suggest_categorical(
            'mos_motif_per_class',
            [1, 2, 3],
        )
        cfg.mosgsl.motif_sim_type = trial.suggest_categorical(
            'mos_motif_sim_type',
            ['cosine', 'euclidean_1', 'euclidean_2'],
        )
        cfg.mosgsl.motif_temperature = trial.suggest_categorical(
            'mos_motif_temperature',
            [0.2, 0.5, 0.7, 1.0],
        )
        cfg.mosgsl.motif_update_type = 'kmeans'
        cfg.mosgsl.motif_k1 = trial.suggest_categorical(
            'mos_motif_k1',
            [0.4, 0.6, 0.8, 1.0],
        )
        cfg.mosgsl.motif_k2 = 0.2
        cfg.mosgsl.motif_tau = trial.suggest_categorical(
            'mos_motif_tau',
            [0.5, 0.7, 0.9, 0.99],
        )
        cfg.mosgsl.motif_weighted_mean = False
        cfg.mosgsl.motif_update_min_epoch = trial.suggest_categorical(
            f'mos_motif_update_min_epoch_e{args.train_epochs}',
            update_epoch_choices(args.train_epochs),
        )
        cfg.mosgsl.motif_update_each = trial.suggest_categorical(
            'mos_motif_update_each',
            [5, 10, 20],
        )
        cfg.mosgsl.motif_loss_weight = trial.suggest_categorical(
            'mos_motif_loss_weight',
            [0.01, 0.05, 0.1, 0.2, 0.5],
        )
    else:
        cfg.mosgsl.use_separate_encoder = False
        cfg.mosgsl.motif_per_class = 2
        cfg.mosgsl.motif_sim_type = 'cosine'
        cfg.mosgsl.motif_temperature = 0.7
        cfg.mosgsl.motif_update_type = 'kmeans'
        cfg.mosgsl.motif_k1 = 0.6
        cfg.mosgsl.motif_k2 = 0.2
        cfg.mosgsl.motif_tau = 0.7
        cfg.mosgsl.motif_weighted_mean = False
        cfg.mosgsl.motif_update_min_epoch = 10
        cfg.mosgsl.motif_update_each = 5
        cfg.mosgsl.motif_loss_weight = 0.1


def suggest_training_cfg(trial, cfg, args):
    gnce = is_gnce_dataset(args.dataset)
    cfg.train.lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    cfg.train.wd = trial.suggest_categorical(
        'weight_decay',
        [0.0, 1e-5, 1e-4, 5e-4]
        if gnce else [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3],
    )
    cfg.train.scheduler = trial.suggest_categorical(
        'scheduler',
        [None, 'ReduceLR']
        if gnce else [None, 'StepLR', 'ReduceLR', 'CosRestart'],
    )
    cfg.train.epochs = args.train_epochs
    cfg.train.patience = args.train_patience
    cfg.train.runs = args.train_runs
    cfg.train.batch_size = args.batch_size


def validate_cfg(cfg, args, allow_manifest_replay=False):
    if cfg.mosgsl.use_motif and not is_single_label(args.dataset):
        raise ValueError("MOSGSL motif is only compatible with single-label graph classification.")
    if cfg.mosgsl.use_motif and not cfg.mosgsl.use_gsl:
        raise ValueError("MOSGSL motif alignment requires SGSL enabled.")
    if cfg.mosgsl.max_n_sub < 1:
        raise ValueError("mosgsl.max_n_sub must be positive.")
    if cfg.mosgsl.max_sub_size < 2:
        raise ValueError("mosgsl.max_sub_size must be at least 2.")
    if not 0.0 <= cfg.mosgsl.gsl_lambda <= 1.0:
        raise ValueError("mosgsl.gsl_lambda must be in [0, 1].")
    if cfg.mosgsl.motif_update_each <= 0:
        raise ValueError("mosgsl.motif_update_each must be positive.")
    if is_gnce_dataset(args.dataset) and not allow_manifest_replay:
        if cfg.mosgsl.use_motif:
            raise ValueError("GNCE regression must keep class-specific motifs disabled.")
        if cfg.mosgsl.use_select:
            raise ValueError("GNCE must not discard semantic query subgraphs.")
        if not cfg.mosgsl.gsl_one_by_one:
            raise ValueError("GNCE uses subgraph-local structure learning.")
        if cfg.mosgsl.pool != 'mean':
            raise ValueError("GNCE uses size-stable mean pooling.")


def suggest_cfg(trial, base_cfg, args):
    manifest_replay = (
        isinstance(trial, optuna.trial.FixedTrial)
        or bool(getattr(trial, 'is_manifest_replay', False))
    )
    if manifest_replay:
        trial = _ManifestChoiceCompat(trial)
    cfg = copy.deepcopy(base_cfg)

    cfg.dataset = args.dataset
    cfg.device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    cfg.seed = args.seed
    cfg.cv.fold_k = args.fold_k
    cfg.cv.seed = args.fold_seed
    cfg.deterministic = args.deterministic
    cfg.deterministic_warn_only = True
    set_fixed_defaults(cfg)

    suggest_mosgsl_arch(trial, cfg, args)
    suggest_parsing_and_gsl(trial, cfg, args)
    suggest_pretrain_and_motif(trial, cfg, args)
    suggest_training_cfg(trial, cfg, args)
    validate_cfg(cfg, args, allow_manifest_replay=manifest_replay)
    return apply_main_effectiveness_trial_protocol(
        apply_trial_protocol(cfg, args), args
    )






def close_trial_logger(name):
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        handler.flush()
        logger.removeHandler(handler)
        if isinstance(handler, logging.FileHandler):
            handler.close()


def objective(trial, base_cfg, args, optimize_metric):
    cfg = suggest_cfg(trial, base_cfg, args)
    config_string = f"tuning_trial_{trial.number:04d}_{cfg.dataset}_MOSGSL"

    def prune_callback(value, step):
        trial.report(value, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    try:
        result = run_mosgsl(
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
        # MEDQ is a useful reported cardinality statistic, but a single median
        # cannot control MOSGSL's rare exponential tail errors.  log_mae ranks
        # the entire validation distribution in the space actually trained.
        optimize_metric = (
            'log_mae' if is_gnce_dataset(args.dataset)
            else main_effectiveness_default_metric(args.dataset)
            if is_main_effectiveness_fixed_split(args.dataset)
            else default_val_metric(args.dataset)
        )
    direction = metric_direction(optimize_metric)

    study_name = args.study_name or default_study_name(args, optimize_metric)
    n_trials = resolve_n_trials(args)
    sampler_seed = resolve_tuning_seed(args.dataset, args.seed)
    print_protocol(args.dataset, n_trials, args.seed)
    print_main_effectiveness_protocol(args.dataset, n_trials, args.seed)
    if is_gnce_dataset(args.dataset):
        print(
            "[MOSGSL GNCE adaptation] validation/checkpoint metric=log_mae; "
            "class motifs=off; edge-to-node predicate/direction context=on; "
            "subgraph selection=off; local conservative SGSL search space."
        )

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction=direction,
        load_if_exists=bool(args.storage),
        sampler=make_mosgsl_sampler(n_trials, sampler_seed),
        pruner=make_mosgsl_pruner(n_trials),
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
            study, suggest_cfg, base_cfg, args, run_mosgsl,
            "MOSGSL", optimize_metric,
            final_seeds=(42, 43, 44)
            if is_main_effectiveness_fixed_split(args.dataset) else None,
        )
    elif is_tu_dataset(args.dataset):
        run_tu_cv_final_evaluation(
            study, suggest_cfg, base_cfg, args, run_mosgsl,
            'MOSGSL', optimize_metric,
        )


if __name__ == '__main__':
    main()
