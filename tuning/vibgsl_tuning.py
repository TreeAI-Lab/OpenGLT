import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from example.vibgsl import run_vibgsl
from tuning.uni_tuning import (
    dataset_family,
    parse_common_args,
    parse_profile,
    prepare_base_cfg,
    run_tuning,
    suggest_training_cfg,
)


MODEL_NAME = 'VIBGSL'

def parse_width_profile(value):
    return parse_profile(value, 3)


def suggest_width_profile(trial, family):
    choices = (
        ['64x32x16', '128x32x32']
        if family == 'counting'
        else ['64x16x16', '64x32x16', '128x32x32']
    )
    return parse_width_profile(trial.suggest_categorical('vibgsl_width_profile', choices))


def suggest_graph_profile(trial, family):
    choices = (
        ['knn_mlp_k10', 'knn_attention_k10', 'knn_weighted_cosine_k10']
        if family == 'counting'
        else ['knn_mlp_k5', 'knn_mlp_k10', 'knn_attention_k10', 'eps_attention_e03']
    )
    profile = trial.suggest_categorical('vibgsl_graph_profile', choices)
    parts = profile.split('_')
    graph_kind = parts[0]

    if graph_kind == 'prob':
        graph_type = 'prob'
        metric = '_'.join(parts[1:])
        top_k = 10
        epsilon = 0.3
    elif graph_kind == 'knn':
        graph_type = 'KNN'
        metric = '_'.join(parts[1:-1])
        top_k = int(parts[-1][1:])
        epsilon = 0.3
    elif graph_kind == 'eps':
        graph_type = 'epsilonNN'
        metric = '_'.join(parts[1:-1])
        top_k = 10
        epsilon = int(parts[-1][1:]) / 10.0
    else:
        raise ValueError(f'Unknown graph profile: {profile}')

    return graph_type, metric, top_k, epsilon


def suggest_backbone(trial, family):
    choices = ['GAT'] if family == 'counting' else ['GIN', 'GAT']
    return trial.suggest_categorical('vibgsl_backbone', choices)


def suggest_num_per(trial, graph_metric_type):
    if graph_metric_type in {'attention', 'weighted_cosine', 'multi_mlp'}:
        return trial.suggest_categorical('vibgsl_num_per', [4, 8, 16])
    return 1


def suggest_feature_denoise(trial, family):
    choices = [False, True] if family == 'counting' else [False]
    return trial.suggest_categorical('vibgsl_feature_denoise', choices)


def suggest_skip_conn(trial):
    return trial.suggest_float('vibgsl_graph_skip_conn', 0.0, 0.5)


def suggest_beta(trial):
    return trial.suggest_float('vibgsl_beta', 1e-6, 1e-3, log=True)


def suggest_dropout(trial):
    return trial.suggest_categorical('vibgsl_dropout', [0.0, 0.3, 0.5])


def suggest_cfg(trial, base_cfg, args):
    cfg = prepare_base_cfg(base_cfg, args)
    family = dataset_family(args.dataset)

    nhid, hidden_dim, ib_size = suggest_width_profile(trial, family)
    cfg.vibgsl.nhid = nhid
    cfg.vibgsl.hidden_dim = hidden_dim
    cfg.vibgsl.ib_size = ib_size

    cfg.vibgsl.backbone = suggest_backbone(trial, family)
    graph_type, graph_metric_type, top_k, epsilon = suggest_graph_profile(trial, family)
    cfg.vibgsl.graph_type = graph_type
    cfg.vibgsl.graph_metric_type = graph_metric_type
    cfg.vibgsl.top_k = top_k
    cfg.vibgsl.epsilon = epsilon
    cfg.vibgsl.num_per = suggest_num_per(trial, graph_metric_type)

    cfg.vibgsl.repar = trial.suggest_categorical(
        'vibgsl_repar', [False] if family == 'counting' else [True, False]
    )
    cfg.vibgsl.feature_denoise = suggest_feature_denoise(trial, family)
    cfg.vibgsl.graph_skip_conn = suggest_skip_conn(trial)
    cfg.vibgsl.graph_include_self = (
        True if cfg.vibgsl.graph_skip_conn > 0 else
        trial.suggest_categorical('vibgsl_graph_include_self', [True, False])
    )
    cfg.vibgsl.beta = suggest_beta(trial)
    cfg.vibgsl.dropout = suggest_dropout(trial)

    suggest_training_cfg(trial, cfg, args)
    return cfg


def main():
    args = parse_common_args()
    run_tuning(MODEL_NAME, run_vibgsl, suggest_cfg, args=args)


if __name__ == '__main__':
    main()
