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
from example.main_baselines import run_baselines
from models import TABULAR_BASELINE_BACKBONES, gnn_type_supports_edgepool
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
    print_main_effectiveness_protocol,
)
from tuning.evaluation import is_tu_dataset, run_tu_cv_final_evaluation


OGB_BINARY = {'ogbg-molhiv', 'ogbg-molbace'}
OGB_MULTI_LABEL = {'ogbg-molpcba', 'ogbg-moltox21'}
LRGB_MULTI_LABEL = {'peptides-func'}
LRGB_REGRESSION = {'peptides-struct'}
PYG_MOL_REGRESSION = {'zinc', 'aqsol'}
GCARE_REGRESSION = {'gcare-aids', 'gcare-yago'}
GNCE_LOG_MAE_DATASETS = {'gnce-lubm-path', 'gnce-lubm-star'}
TU_DATASETS = {
    'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB', 'MUTAG', 'NCI1', 'DD', 'ENZYMES', 'PROTEINS'
}
EDGE_ENHANCED_BACKBONES = {'GCNplus', 'GINE', 'SAGEplus', 'PNA'}
SEARCH_SPACE_VERSION = 'release_v1'


def is_good_hiv_gmt(args):
    dataset = args.dataset.lower().replace('_', '-')
    return (
        dataset.startswith('good-hiv-')
        and args.backbone == 'GINE'
        and args.pooling_method == 'gmt'
    )


def is_ogb_hiv_gmt(args):
    """The main-table GMT baseline uses a GIN backbone on OGB-HIV."""
    return (
        args.dataset.lower().replace('_', '-') == 'ogbg-molhiv'
        and args.backbone == 'GIN'
        and args.pooling_method == 'gmt'
    )


def is_targeted_hiv_gmt(args):
    return is_good_hiv_gmt(args) or is_ogb_hiv_gmt(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ogbg-molbace')
    parser.add_argument('--backbone', default='GCN')
    parser.add_argument('--pooling_method', default='none', choices=['none', 'topk', 'edgepool', 'gmt'])
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


def default_val_metric(dataset):
    dataset = str(dataset)
    dataset_lower = dataset.lower()
    from datasets import good_key_metric, is_good_dataset_name
    if is_good_dataset_name(dataset):
        return good_key_metric(dataset)
    if dataset in OGB_BINARY:
        return 'auc'
    if dataset == 'ogbg-molpcba':
        return 'ap_macro'
    if dataset == 'ogbg-moltox21':
        return 'auc_macro'
    if dataset_lower in LRGB_MULTI_LABEL:
        return 'ap_macro'
    if dataset_lower in LRGB_REGRESSION:
        return 'mae'
    if dataset_lower in PYG_MOL_REGRESSION:
        return 'mae'
    # MEDQ is the primary GNCE metric in general, but it is saturated at 1.0
    # on the two easy LUBM workloads and consequently cannot reliably rank
    # configurations.  Use log-space MAE for validation/checkpoint selection
    # there; all GNCE effectiveness metrics are still reported at test time.
    if dataset_lower in GNCE_LOG_MAE_DATASETS:
        return 'log_mae'
    if dataset_lower in GCARE_REGRESSION or dataset_lower.startswith('gnce-'):
        return 'medq'
    if dataset in TU_DATASETS:
        return 'acc'
    if dataset.startswith('Count-'):
        return 'r2'
    return 'acc'


def normalize_metric(metric, dataset):
    metric = default_val_metric(dataset) if metric == 'auto' else metric
    metric = metric.lower().replace('-', '_')
    dataset_lower = dataset.lower()
    if dataset == 'ogbg-molpcba' and metric in {'ap', 'average_precision'}:
        return 'ap_macro'
    if dataset_lower in LRGB_MULTI_LABEL and metric in {'ap', 'average_precision'}:
        return 'ap_macro'
    from datasets import is_good_dataset_name, parse_good_dataset_name
    if is_good_dataset_name(dataset):
        spec, _, _, _ = parse_good_dataset_name(dataset)
        if spec.key == 'pcba' and metric in {'ap', 'average_precision'}:
            return 'ap_macro'
    if dataset in OGB_MULTI_LABEL and metric in {'auc', 'auc_roc', 'roc_auc'}:
        return 'auc_macro'
    if dataset_lower in LRGB_MULTI_LABEL and metric in {'auc', 'auc_roc', 'roc_auc'}:
        return 'auc_macro'
    if dataset in OGB_BINARY and metric in {'auc_roc', 'roc_auc'}:
        return 'auc'
    return metric


def metric_direction(metric):
    return 'minimize' if metric in {'mae', 'mre', 'mse', 'rmse', 'medq', 'p90q', 'p95q', 'p99q', 'meanq', 'maxq', 'log_mae'} else 'maximize'


def suggest_num_pool_layers(trial, num_gnn_layers, param_name):
    sampled = trial.suggest_categorical(param_name, [1, 2, 3])
    return min(sampled, int(num_gnn_layers))


def default_study_name(args, optimize_metric):
    dataset = args.dataset.replace('/', '_')
    return f"main_baselines_{SEARCH_SPACE_VERSION}_{dataset}_{args.backbone}_{args.pooling_method}_{optimize_metric}"


def is_gnce_dataset(dataset):
    return dataset.lower().startswith('gnce-')


def is_gcare_dataset(dataset):
    return dataset.lower() in GCARE_REGRESSION


def is_card_query_dataset(dataset):
    return dataset.lower().startswith('gnce-')


def is_query_gmt_dataset(dataset, pooling_method):
    return pooling_method == 'gmt' and (is_gnce_dataset(dataset) or is_gcare_dataset(dataset))


def is_tabular_baseline(backbone):
    return backbone in TABULAR_BASELINE_BACKBONES


def resolve_n_trials(args):
    if protocol_is_fixed_split_dataset(args.dataset):
        return resolve_protocol_n_trials(args.dataset, args.n_trials)
    if args.n_trials is not None:
        return args.n_trials
    return 200








def suggest_cfg(trial, base_cfg, args):
    cfg = copy.deepcopy(base_cfg)
    is_query_card = is_card_query_dataset(args.dataset)
    is_query_gmt = is_query_gmt_dataset(args.dataset, args.pooling_method)
    is_gnce_gmt = is_query_gmt and is_gnce_dataset(args.dataset)
    is_gcare_gmt = is_query_gmt and is_gcare_dataset(args.dataset)

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

    cfg.model.backbone = args.backbone
    if is_tabular_baseline(args.backbone):
        cfg.model.pooling_method = 'none'
        cfg.model.use_virtual_node = False
        cfg.model.use_fingerprints = False
        cfg.model.use_rwpe = False
        cfg.train.epochs = args.train_epochs
        cfg.train.patience = args.train_patience
        cfg.train.runs = args.train_runs
        cfg.train.batch_size = args.batch_size

        if args.backbone in {'MLPFeature', 'MLP'}:
            cfg.baseline.feature_mode = 'feature'
        elif args.backbone in {'MLPFeatureStruct', 'MLPStruct'}:
            cfg.baseline.feature_mode = 'feature_struct'
        elif args.backbone == 'LightGBM':
            cfg.baseline.feature_mode = 'feature_struct'
        else:
            cfg.baseline.feature_mode = trial.suggest_categorical(
                'baseline_feature_mode',
                ['feature', 'feature_struct'],
            )

        if args.backbone == 'LightGBM':
            cfg.baseline.lightgbm_n_estimators = trial.suggest_int(
                'lgbm_n_estimators', 200, 1200,
            )
            cfg.baseline.lightgbm_learning_rate = trial.suggest_float(
                'lgbm_learning_rate', 0.01, 0.2, log=True,
            )
            cfg.baseline.lightgbm_num_leaves = trial.suggest_int(
                'lgbm_num_leaves', 15, 127,
            )
            cfg.baseline.lightgbm_max_depth = trial.suggest_categorical(
                'lgbm_max_depth',
                [-1, 5, 7, 10, 15],
            )
            cfg.baseline.lightgbm_min_child_samples = trial.suggest_int(
                'lgbm_min_child_samples', 10, 100,
            )
            cfg.baseline.lightgbm_subsample = trial.suggest_float(
                'lgbm_subsample', 0.6, 1.0,
            )
            cfg.baseline.lightgbm_colsample_bytree = trial.suggest_float(
                'lgbm_colsample_bytree', 0.6, 1.0,
            )
            cfg.baseline.lightgbm_reg_alpha = trial.suggest_categorical(
                'lgbm_reg_alpha',
                [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0],
            )
            cfg.baseline.lightgbm_reg_lambda = trial.suggest_categorical(
                'lgbm_reg_lambda',
                [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0],
            )
            return apply_main_effectiveness_trial_protocol(
                apply_trial_protocol(cfg, args), args
            )

        cfg.baseline.hidden_size = trial.suggest_categorical(
            'mlp_hidden',
            [64, 128, 256, 512],
        )
        cfg.baseline.num_layers = trial.suggest_int('mlp_layers', 2, 5)
        cfg.baseline.dropout = trial.suggest_float('mlp_dropout', 0.0, 0.5)
        cfg.baseline.norm_type = trial.suggest_categorical('mlp_norm_type', ['none', 'batch_norm', 'layer_norm'])
        cfg.baseline.act_type = trial.suggest_categorical('mlp_act_type', ['relu', 'prelu', 'swish'])
        cfg.train.lr = trial.suggest_float('tabular_lr', 1e-4, 1e-2, log=True)
        cfg.train.wd = trial.suggest_categorical('tabular_weight_decay', [0.0, 1e-6, 1e-5, 1e-4])
        cfg.train.scheduler = trial.suggest_categorical('tabular_scheduler', [None, 'ReduceLR'])
        return apply_main_effectiveness_trial_protocol(
            apply_trial_protocol(cfg, args), args
        )

    cfg.model.pooling_method = args.pooling_method
    cfg.model.hidden_size = trial.suggest_categorical(
        'nhid',
        [64, 128, 256, 512],
    )
    cfg.model.num_layers = trial.suggest_int('n_layer_basicgnn', 2, 10)
    cfg.model.dropout = trial.suggest_float('dropout', 0.0, 0.5)
    cfg.model.norm_type = trial.suggest_categorical(
        'norm_type',
        ['none', 'batch_norm', 'layer_norm'],
    )
    cfg.model.act_type = trial.suggest_categorical('act_type', ['relu', 'prelu', 'swish'])
    cfg.model.res = trial.suggest_categorical('res', [False, True])
    if args.pooling_method == 'none':
        cfg.model.readout_type = trial.suggest_categorical(
            'readout_type',
            ['mean', 'sum', 'max', 'attn'],
        )
        cfg.model.use_virtual_node = trial.suggest_categorical('use_virtual_node', [False, True])
    else:
        cfg.model.readout_type = 'mean'
        cfg.model.use_virtual_node = False

    cfg.train.lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    cfg.train.wd = trial.suggest_categorical(
        'weight_decay',
        [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3],
    )
    cfg.train.scheduler = trial.suggest_categorical(
        'scheduler',
        [None, 'StepLR', 'ReduceLR', 'CosRestart'],
    )

    cfg.train.epochs = args.train_epochs
    cfg.train.patience = args.train_patience
    cfg.train.runs = args.train_runs
    cfg.train.batch_size = args.batch_size

    if args.backbone in EDGE_ENHANCED_BACKBONES:
        cfg.model.use_ffn = trial.suggest_categorical('use_ffn', [False, True])
        if cfg.model.use_ffn:
            cfg.model.ffn_expansion = trial.suggest_categorical('ffn_expansion', [1, 2, 4])
        cfg.model.use_rwpe = trial.suggest_categorical('use_rwpe', [False, True])
        if cfg.model.use_rwpe:
            cfg.model.rwpe_dim = trial.suggest_categorical('rwpe_dim', [16, 20, 32])
    else:
        cfg.model.use_ffn = False
        cfg.model.use_rwpe = False

    if args.backbone in {'GIN', 'GINE'}:
        cfg.model.train_eps_gin = trial.suggest_categorical('train_eps_gin', [False, True])
    if args.backbone in {'SAGE', 'SAGEplus'}:
        cfg.model.aggr_sage = trial.suggest_categorical('aggr_sage', ['mean', 'sum', 'max'])
    if args.backbone == 'PNA':
        cfg.model.towers_pna = trial.suggest_categorical('towers_pna', [1, 2, 4])
        cfg.model.pre_layers_pna = trial.suggest_categorical('pre_layers_pna', [1, 2])
        cfg.model.post_layers_pna = trial.suggest_categorical('post_layers_pna', [1, 2])
        cfg.model.divide_input_pna = trial.suggest_categorical('divide_input_pna', [False, True])

    if args.pooling_method == 'topk':
        cfg.model.pool_ratio = trial.suggest_categorical('pool_ratio', [0.3, 0.5, 0.8, 0.9])
        cfg.model.num_pool_layers = suggest_num_pool_layers(trial, cfg.model.num_layers, 'topk_num_pool_layers')
        cfg.model.pool_readout = trial.suggest_categorical('topk_pool_readout', ['mean', 'mean_max'])
        cfg.model.use_virtual_node = False
    elif args.pooling_method == 'edgepool':
        cfg.model.edgepool_dropout = trial.suggest_float('edgepool_dropout', 0.0, 0.3)
        cfg.model.num_pool_layers = suggest_num_pool_layers(trial, cfg.model.num_layers, 'edgepool_num_pool_layers')
        cfg.model.pool_readout = trial.suggest_categorical('edgepool_pool_readout', ['mean', 'mean_max'])
        cfg.model.use_virtual_node = False
    elif args.pooling_method == 'gmt':
        cfg.model.gmt_k = trial.suggest_categorical('gmt_k', [4, 8, 16, 32])
        cfg.model.gmt_heads = trial.suggest_categorical('gmt_heads', [1, 2, 4, 8])
        cfg.model.gmt_blocks = trial.suggest_int('gmt_blocks', 1, 3)
        cfg.model.gmt_layer_norm = trial.suggest_categorical('gmt_layer_norm', [False, True])

    return apply_main_effectiveness_trial_protocol(
        apply_trial_protocol(cfg, args), args
    )


def compact_stats(stats):
    return {k: round(v['mean'], 6) for k, v in stats.items()}


def close_trial_logger(name):
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        handler.flush()
        logger.removeHandler(handler)
        if isinstance(handler, logging.FileHandler):
            handler.close()


def objective(trial, base_cfg, args, optimize_metric):
    cfg = suggest_cfg(trial, base_cfg, args)
    config_string = f"tuning_trial_{trial.number:04d}_{cfg.dataset}_{cfg.model.backbone}_{cfg.model.pooling_method}"

    def prune_callback(value, step):
        trial.report(value, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    try:
        result = run_baselines(
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


def run_gcare_final_evaluation(study, base_cfg, args, optimize_metric):
    """Evaluate the validation-selected configuration on all five test folds.

    G-CARE uses deterministic outer test folds rather than a single fixed test
    split.  Tuning already aggregates validation performance over all five
    folds, with ``evaluate_test=False``.  Once tuning finishes, reconstruct the
    best conditional configuration and run the same five folds once more with
    test evaluation enabled.  ``run_baselines`` independently initializes and
    trains a fresh model for every fold.
    """
    fixed_trial = optuna.trial.FixedTrial(study.best_trial.params)
    final_cfg = suggest_cfg(fixed_trial, base_cfg, args)
    final_cfg.seed = int(args.seed)
    # The runner ignores train.runs for G-CARE and uses the five folds, but set
    # this explicitly so the printed configuration cannot imply three repeats.
    final_cfg.train.runs = 1

    safe_model = str(args.backbone).replace('/', '_').replace(' ', '_')
    config_string = (
        f"final_5fold_{final_cfg.dataset}_{safe_model}_"
        f"seed_{final_cfg.seed}"
    )
    print("\nG-CARE final five-fold evaluation:")
    print(f"  model: {args.backbone}")
    print(f"  dataset: {final_cfg.dataset}")
    print(f"  model initialization seed in every fold: {final_cfg.seed}")
    print(
        "  selected params:",
        json.dumps(study.best_trial.params, sort_keys=True),
    )
    print(
        "  protocol: five independently initialized/train/val/test folds; "
        "test was not evaluated in Optuna trials"
    )

    result = run_baselines(
        cfg=final_cfg,
        key_metric_override=optimize_metric,
        config_string=config_string,
        prune_callback=None,
        evaluate_test=True,
    )
    print("Validation metrics (5 folds, mean +/- std):")
    for metric in sorted(result['best_val_perf']):
        value = result['best_val_perf'][metric]
        print(
            f"  {metric}: {float(value['mean']):.6f} "
            f"+/- {float(value['std']):.6f}"
        )
    print("Test metrics (5 folds, mean +/- std):")
    for metric in sorted(result['test_perf']):
        value = result['test_perf'][metric]
        print(
            f"  {metric}: {float(value['mean']):.6f} "
            f"+/- {float(value['std']):.6f}"
        )
    return result


def main():
    args = parse_args()
    if args.pooling_method == 'edgepool' and not is_tabular_baseline(args.backbone) and args.backbone not in ('GCN', 'GIN', 'SAGE'):
        raise ValueError("pooling_method='edgepool' only supports GCN / GIN / SAGE / GraphSAGE.")

    base_cfg = init_cfg()
    optimize_metric = normalize_metric(args.val_metric, args.dataset)
    direction = metric_direction(optimize_metric)
    study_name = args.study_name or default_study_name(args, optimize_metric)
    n_trials = resolve_n_trials(args)
    sampler_seed = 42 if is_ogb_hiv_gmt(args) else resolve_tuning_seed(args.dataset, args.seed)
    print_protocol(args.dataset, n_trials, args.seed)
    if is_main_effectiveness_fixed_split(args.dataset):
        print_main_effectiveness_protocol(args.dataset, n_trials, args.seed)
    sampler = optuna.samplers.TPESampler(
        seed=sampler_seed,
        n_startup_trials=min(20, max(5, n_trials // 5)),
        multivariate=True,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction=direction,
        load_if_exists=bool(args.storage),
        sampler=sampler,
        pruner=make_pruner(),
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
        or is_ogb_hiv_gmt(args)
        or is_main_effectiveness_fixed_split(args.dataset)
    ):
        model_name = (
            args.backbone
            if args.pooling_method == 'none'
            else f"{args.backbone}+{args.pooling_method.upper()}"
        )
        run_final_evaluation(
            study, suggest_cfg, base_cfg, args, run_baselines,
            model_name, optimize_metric,
            final_seeds=(42, 43, 44)
            if is_main_effectiveness_fixed_split(args.dataset) else None,
        )
    elif is_tu_dataset(args.dataset):
        model_name = (
            args.backbone if args.pooling_method == 'none'
            else f"{args.backbone}+{args.pooling_method.upper()}"
        )
        run_tu_cv_final_evaluation(
            study, suggest_cfg, base_cfg, args, run_baselines,
            model_name, optimize_metric,
        )
    elif is_gcare_dataset(args.dataset):
        run_gcare_final_evaluation(
            study, base_cfg, args, optimize_metric,
        )


if __name__ == '__main__':
    main()
