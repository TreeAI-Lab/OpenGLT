import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from example.nagphormer import run_nagphormer
from tuning.uni_tuning import (
    dataset_family,
    parse_common_args,
    parse_width_heads,
    prepare_base_cfg,
    run_tuning,
    suggest_training_cfg,
)


MODEL_NAME = 'NAGphormer'


def suggest_hops(trial):
    return trial.suggest_categorical('nag_hops', [3, 4, 5])


def suggest_cfg(trial, base_cfg, args):
    cfg = prepare_base_cfg(base_cfg, args)
    width_heads = trial.suggest_categorical(
        'nag_width_heads',
        ['128x4', '256x4', '256x8'],
    )
    hidden_size, num_heads = parse_width_heads(width_heads)
    ffn_mult = trial.suggest_categorical('nag_ffn_mult', [3, 4])

    cfg.nagphormer.hidden_size = hidden_size
    cfg.nagphormer.num_heads = num_heads
    cfg.nagphormer.num_layers = trial.suggest_int('nag_num_layers', 1, 4)
    cfg.nagphormer.hops = suggest_hops(trial)
    cfg.nagphormer.ffn_dim = hidden_size * ffn_mult
    cfg.nagphormer.dropout = trial.suggest_float('nag_dropout', 0.0, 0.5)
    cfg.nagphormer.attention_dropout = trial.suggest_categorical(
        'nag_attn_dropout', [0.05, 0.1, 0.2]
    )
    cfg.nagphormer.pooling = trial.suggest_categorical(
        'nag_pooling', ['mean', 'sum']
    )

    suggest_training_cfg(trial, cfg, args)
    return cfg


def main():
    args = parse_common_args()
    run_tuning(MODEL_NAME, run_nagphormer, suggest_cfg, args=args)


if __name__ == '__main__':
    main()
