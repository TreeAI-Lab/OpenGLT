import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from example.hymn import run_hymn
from tuning.uni_tuning import (
    dataset_family,
    parse_common_args,
    predictor_multiplier_from_profile,
    prepare_base_cfg,
    run_tuning,
    suggest_training_cfg,
)


MODEL_NAME = 'HyMN'


# First trial for Count workloads: the architecture/training configuration in
# HyMN-old/counting_substructures/wandb_sweep/counting.yaml.  Enqueuing it
# guarantees that a 30-trial budget actually evaluates the reference regime.
COUNT_REFERENCE_ANCHOR = {
    'hymn_hidden_size': 32,
    'hymn_num_layers': 6,
    'hymn_num_samples': 60,
    'hymn_num_colours': 1,
    'hymn_dropout': 0.0,
    'hymn_pooling': 'mean',
    'hymn_residual': False,
    'hymn_inject_colours': False,
    'hymn_conv_multiplier': 1,
    'hymn_batch_norm': True,
    'hymn_colouring': 'individualize',
    'hymn_counting_input': 'constant',
    'hymn_predictor': 'linear',
    'lr': 1e-3,
    'weight_decay': 0.0,
    'scheduler': None,
}


def suggest_sample_colour(trial, family):
    choices = (
        ['2x2', '3x2', '4x2', '5x2']
        if family == 'counting'
        else ['1x2', '2x2', '3x2', '2x3', '2x4']
    )
    value = trial.suggest_categorical('hymn_samples_colours', choices)
    samples, colours = value.split('x')
    return int(samples), int(colours)


def suggest_cfg(trial, base_cfg, args):
    cfg = prepare_base_cfg(base_cfg, args)
    family = dataset_family(args.dataset)

    if family == 'counting':
        # Focus the small budget around the old high-performing mechanism.
        # The architectural switches unrelated to the question are fixed; the
        # search spends its budget on width, depth and optimization.
        cfg.hymn.hidden_size = trial.suggest_categorical(
            'hymn_hidden_size', [32, 64]
        )
        cfg.hymn.num_layers = trial.suggest_categorical(
            'hymn_num_layers', [4, 6]
        )
        cfg.hymn.num_samples = trial.suggest_categorical(
            'hymn_num_samples', [60]
        )
        cfg.hymn.num_colours = trial.suggest_categorical(
            'hymn_num_colours', [1]
        )
        cfg.hymn.dropout = trial.suggest_categorical('hymn_dropout', [0.0])
        cfg.hymn.pooling = trial.suggest_categorical('hymn_pooling', ['mean'])
        cfg.hymn.residual = trial.suggest_categorical('hymn_residual', [False])
        cfg.hymn.inject_colours = trial.suggest_categorical(
            'hymn_inject_colours', [False]
        )
        cfg.hymn.multiplier = trial.suggest_categorical(
            'hymn_conv_multiplier', [1]
        )
        cfg.hymn.bn = trial.suggest_categorical('hymn_batch_norm', [True])
        cfg.hymn.colouring = trial.suggest_categorical(
            'hymn_colouring', ['individualize']
        )
        cfg.hymn.counting_input = trial.suggest_categorical(
            'hymn_counting_input', ['constant']
        )
        cfg.hymn.predictor = trial.suggest_categorical(
            'hymn_predictor', ['linear']
        )
        # AdamW with weight_decay=0 is equivalent here to the old Adam setup.
        cfg.train.lr = trial.suggest_categorical(
            'lr', [5e-4, 1e-3, 3e-3]
        )
        cfg.train.wd = trial.suggest_categorical(
            'weight_decay', [0.0, 1e-5]
        )
        cfg.train.scheduler = trial.suggest_categorical(
            'scheduler', [None, 'StepLR']
        )
        cfg.train.lr_patience = 50
        cfg.train.lr_decay = 0.5
        return cfg

    hidden_size = trial.suggest_categorical(
        'hymn_hidden_size',
        [64, 128] if args.dataset.lower() in {'ogbg-molpcba', 'peptides-func', 'peptides-struct'} else [64, 128, 256],
    )
    num_samples, num_colours = suggest_sample_colour(trial, family)
    predictor_profile = trial.suggest_categorical(
        'hymn_predictor_profile', ['small', 'medium', 'official']
    )

    cfg.hymn.hidden_size = hidden_size
    cfg.hymn.num_layers = trial.suggest_categorical(
        'hymn_num_layers', [3, 4, 5, 6]
    )
    cfg.hymn.num_samples = num_samples
    cfg.hymn.num_colours = num_colours
    cfg.hymn.dropout = trial.suggest_categorical(
        'hymn_dropout', [0.0, 0.1, 0.2, 0.3]
    )
    cfg.hymn.pooling = trial.suggest_categorical(
        'hymn_pooling', ['sum', 'mean', 'max']
    )
    cfg.hymn.residual = trial.suggest_categorical(
        'hymn_residual', [False, True]
    )
    cfg.hymn.inject_colours = trial.suggest_categorical('hymn_inject_colours', [False, True])
    cfg.hymn.multiplier = trial.suggest_categorical(
        'hymn_conv_multiplier', [1, 2, 4]
    )
    cfg.hymn.predictor_multiplier = predictor_multiplier_from_profile(
        predictor_profile, hidden_size
    )
    cfg.hymn.bn = trial.suggest_categorical('hymn_batch_norm', [False, True])
    cfg.hymn.colouring = 'uniform'
    cfg.hymn.counting_input = 'degree'
    cfg.hymn.predictor = 'mlp'

    suggest_training_cfg(trial, cfg, args)
    return cfg


def main():
    args = parse_common_args()
    anchors = [COUNT_REFERENCE_ANCHOR] if dataset_family(args.dataset) == 'counting' else None
    run_tuning(
        MODEL_NAME,
        run_hymn,
        suggest_cfg,
        args=args,
        enqueue_trials=anchors,
    )


if __name__ == '__main__':
    main()
