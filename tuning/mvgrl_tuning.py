import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from example.mvgrl import run_mvgrl
from tuning.uni_tuning import (
    dataset_family,
    parse_common_args,
    parse_width_layers,
    prepare_base_cfg,
    run_tuning,
    suggest_training_cfg,
)


MODEL_NAME = 'MVGRL'

def suggest_width_layers(trial, family, is_large):
    if family == 'counting':
        choices = ['64x2', '64x3']
    elif is_large:
        choices = ['64x2', '128x2']
    else:
        choices = ['128x2', '128x3', '256x2']
    return parse_width_layers(trial.suggest_categorical('mvgrl_width_layers', choices))


def suggest_dropout(trial):
    return trial.suggest_categorical('mvgrl_dropout', [0.0, 0.1, 0.2])


def suggest_alpha(trial):
    return trial.suggest_float('mvgrl_alpha', 0.05, 0.30)


def suggest_cfg(trial, base_cfg, args):
    cfg = prepare_base_cfg(base_cfg, args)
    family = dataset_family(args.dataset)
    is_large = args.dataset.lower() in {
        'ogbg-molpcba', 'peptides-func', 'peptides-struct',
        'dd', 'collab', 'reddit-binary',
    }

    hidden_size, num_layers = suggest_width_layers(trial, family, is_large)
    cfg.mvgrl.hidden_size = hidden_size
    cfg.mvgrl.num_layers = num_layers
    cfg.mvgrl.dropout = suggest_dropout(trial)
    cfg.mvgrl.alpha = suggest_alpha(trial)

    suggest_training_cfg(trial, cfg, args)
    return cfg


def main():
    args = parse_common_args()
    run_tuning(MODEL_NAME, run_mvgrl, suggest_cfg, args=args)


if __name__ == '__main__':
    main()
