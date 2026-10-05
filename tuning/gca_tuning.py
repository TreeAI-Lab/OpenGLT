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
from example.gca import run_gca
from models import gnn_type_supports_edgepool
from tuning.main_baselines_tuning import (
    EDGE_ENHANCED_BACKBONES,
    GCARE_REGRESSION,
    compact_stats,
    metric_direction,
    normalize_metric,
    suggest_num_pool_layers,
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


SEARCH_SPACE_VERSION = 'gca_v1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ogbg-molbace')
    parser.add_argument('--backbone', default='GINE')
    parser.add_argument('--pooling_method', default='none', choices=['none', 'topk', 'edgepool', 'gmt'])
    parser.add_argument('--val_metric', default='auto')
    parser.add_argument('--n_trials', type=int, default=None)
    parser.add_argument('--train_epochs', type=int, default=500)
    parser.add_argument('--train_patience', type=int, default=50)
    parser.add_argument('--train_runs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--pretrain_epochs', type=int, default=100)
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
    return f"gca_{SEARCH_SPACE_VERSION}_{dataset}_{args.backbone}_{args.pooling_method}_{optimize_metric}"


def is_gnce_dataset(dataset):
    return dataset.lower().startswith('gnce-')


def is_card_query_dataset(dataset):
    return dataset.lower().startswith('gnce-')


def resolve_n_trials(args):
    return resolve_protocol_n_trials(
        args.dataset, args.n_trials, non_gnce_default=200
    )


def pretrain_epoch_choices(max_pretrain_epochs):
    max_pretrain_epochs = max(0, int(max_pretrain_epochs))
    choices = [
        value for value in [0, 20, 50, 100, 150, 200]
        if value <= max_pretrain_epochs
    ]
    if max_pretrain_epochs not in choices:
        choices.append(max_pretrain_epochs)
    return sorted(set(choices))


def edge_drop_pair_choices():
    return [
        '0.00,0.10',
        '0.05,0.15',
        '0.00,0.20',
        '0.10,0.20',
        '0.10,0.30',
        '0.20,0.30',
        '0.20,0.40',
        '0.30,0.40',
    ]


def feature_drop_pair_choices():
    return [
        '0.00,0.10',
        '0.05,0.15',
        '0.10,0.20',
        '0.10,0.30',
        '0.20,0.30',
        '0.20,0.40',
    ]


def centrality_choices():
    return ['degree', 'pr', 'evc']


def parse_drop_pair(pair):
    left, right = pair.split(',')
    return float(left), float(right)


def suggest_cfg(trial, base_cfg, args):
    cfg = copy.deepcopy(base_cfg)
    is_query_card = is_card_query_dataset(args.dataset)

    cfg.dataset = args.dataset
    cfg.device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    cfg.seed = args.seed
    cfg.cv.fold_k = args.fold_k
    cfg.cv.seed = args.fold_seed
    cfg.deterministic = args.deterministic
    cfg.deterministic_warn_only = True
    cfg.load_best_params = False
    cfg.use_augm_gca = False

    cfg.model.backbone = args.backbone
    cfg.model.pooling_method = args.pooling_method
    cfg.model.hidden_size = trial.suggest_categorical('nhid', [64, 128, 256, 512])
    cfg.model.num_layers = trial.suggest_int('n_layer_basicgnn', 2, 10)
    cfg.model.dropout = trial.suggest_float('dropout', 0.0, 0.5)
    cfg.model.norm_type = trial.suggest_categorical(
        'norm_type',
        ['none', 'batch_norm', 'layer_norm'],
    )
    cfg.model.act_type = trial.suggest_categorical('act_type', ['relu', 'prelu', 'swish'])
    cfg.model.res = trial.suggest_categorical('res', [False, True])
    cfg.model.use_rwpe = False

    cfg.model.use_fingerprints = False
    cfg.model.fingerprint_dim = 0
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
    else:
        cfg.model.use_ffn = False

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

    pretrain_epochs_choice = trial.suggest_categorical(
        'gca_pretrain_epochs_choice',
        pretrain_epoch_choices(args.pretrain_epochs),
    )
    cfg.gca.pretrain_epochs = pretrain_epochs_choice
    cfg.gca.pretrain_lr = trial.suggest_float(
        'gca_pretrain_lr', 1e-4, 3e-3, log=True
    )
    cfg.gca.pretrain_wd = trial.suggest_categorical(
        'gca_pretrain_weight_decay',
        [0.0, 1e-6, 1e-5, 1e-4, 5e-4],
    )
    cfg.gca.temperature = trial.suggest_float(
        'gca_temperature', 0.1, 1.0, log=True
    )
    cfg.gca.drop_scheme = trial.suggest_categorical(
        'gca_drop_scheme',
        centrality_choices(),
    )
    edge_drop_pair = trial.suggest_categorical(
        'gca_edge_drop_pair',
        edge_drop_pair_choices(),
    )
    cfg.gca.pe1, cfg.gca.pe2 = parse_drop_pair(edge_drop_pair)
    feature_drop_pair = trial.suggest_categorical(
        'gca_feature_drop_pair',
        feature_drop_pair_choices(),
    )
    cfg.gca.pf1, cfg.gca.pf2 = parse_drop_pair(feature_drop_pair)
    cfg.gca.p_tau = trial.suggest_categorical('gca_p_tau', [0.5, 0.7, 0.9])
    # cfg.gca.p_corr = trial.suggest_categorical('gca_p_corr', ['min', 'mean'])
    cfg.gca.p_corr = 'mean'
    cfg.gca.reset_prediction_head = True

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
    config_string = f"tuning_trial_{trial.number:04d}_{cfg.dataset}_GCA_{cfg.model.backbone}_{cfg.model.pooling_method}"

    def prune_callback(value, step):
        trial.report(value, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    try:
        result = run_gca(
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
    if args.pooling_method == 'edgepool' and not gnn_type_supports_edgepool(args.backbone):
        raise ValueError("pooling_method='edgepool' only supports GCN / GIN / SAGE / GraphSAGE.")

    base_cfg = init_cfg()
    optimize_metric = normalize_metric(args.val_metric, args.dataset)
    if args.val_metric == 'auto' and is_main_effectiveness_fixed_split(args.dataset):
        optimize_metric = main_effectiveness_default_metric(args.dataset)
    direction = metric_direction(optimize_metric)

    study_name = args.study_name or default_study_name(args, optimize_metric)
    n_trials = resolve_n_trials(args)
    sampler_seed = resolve_tuning_seed(args.dataset, args.seed)
    print_protocol(args.dataset, n_trials, args.seed)
    print_main_effectiveness_protocol(args.dataset, n_trials, args.seed)
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
            study, suggest_cfg, base_cfg, args, run_gca,
            "GCA", optimize_metric,
            final_seeds=(42, 43, 44)
            if is_main_effectiveness_fixed_split(args.dataset) else None,
        )
    elif is_tu_dataset(args.dataset):
        run_tu_cv_final_evaluation(
            study, suggest_cfg, base_cfg, args, run_gca,
            'GCA', optimize_metric,
        )


if __name__ == '__main__':
    main()
