import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from example.ecs import run_ecs
from tuning.uni_tuning import dataset_family, parse_common_args, parse_width_layers, prepare_base_cfg, run_tuning, suggest_training_cfg


MODEL_NAME = 'ECS'

def suggest_width_layers(trial, family, is_large):
    choices = ['64x3', '64x5', '128x3', '128x5', '256x3', '256x5']
    return parse_width_layers(trial.suggest_categorical('ecs_width_layers', choices))


def suggest_h(trial, family):
    return trial.suggest_categorical('ecs_h', [2, 3])


def suggest_dropout(trial):
    return trial.suggest_float('ecs_dropout', 0.0, 0.5)


def suggest_pooling(trial):
    return trial.suggest_categorical('ecs_pooling', ['mean', 'sum', 'max'])


def suggest_max_nodes_per_hop(trial, family):
    return trial.suggest_categorical(
        'ecs_max_nodes_per_hop', [16, 32, 64, 0]
    )


def suggest_cfg(trial, base_cfg, args):
    cfg = prepare_base_cfg(base_cfg, args)
    family = dataset_family(args.dataset)
    is_large = args.dataset.lower() in {
        'ogbg-molpcba', 'peptides-func', 'peptides-struct',
        'dd', 'collab', 'reddit-binary',
    }
    hidden_size, num_layers = suggest_width_layers(trial, family, is_large)

    cfg.ecs.hidden_size = hidden_size
    cfg.ecs.num_layers = num_layers
    cfg.ecs.dropout = suggest_dropout(trial)
    cfg.ecs.pooling = suggest_pooling(trial)
    cfg.ecs.residual = trial.suggest_categorical('ecs_residual', [False, True])
    cfg.ecs.virtual_node = trial.suggest_categorical('ecs_virtual_node', [False, True])
    cfg.ecs.h = suggest_h(trial, family)
    cfg.ecs.node_label = 'hop'
    cfg.ecs.use_rd = trial.suggest_categorical('ecs_use_rd', [False, True])
    cfg.ecs.self_loop = trial.suggest_categorical('ecs_self_loop', [False, True])
    cfg.ecs.max_nodes_per_hop = suggest_max_nodes_per_hop(trial, family)
    suggest_training_cfg(trial, cfg, args)
    return cfg


def main():
    args = parse_common_args()
    run_tuning(MODEL_NAME, run_ecs, suggest_cfg, args=args)


if __name__ == '__main__':
    main()
