import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from example.hubgt import run_hubgt
from tuning.uni_tuning import (
    dataset_family,
    parse_common_args,
    parse_width_heads,
    prepare_base_cfg,
    run_tuning,
    suggest_training_cfg,
)


MODEL_NAME = 'HubGT'


def suggest_cfg(trial, base_cfg, args):
    cfg = prepare_base_cfg(base_cfg, args)
    width_heads = trial.suggest_categorical(
        'hubgt_width_heads',
        ['128x4', '128x8', '256x4', '256x8'],
    )
    hidden_size, num_heads = parse_width_heads(width_heads)
    ffn_mult = trial.suggest_categorical('hubgt_ffn_mult', [2, 3, 4])

    cfg.hubgt.hidden_size = hidden_size
    cfg.hubgt.num_heads = num_heads
    cfg.hubgt.num_layers = trial.suggest_int('hubgt_num_layers', 2, 5)
    cfg.hubgt.ffn_dim = hidden_size * ffn_mult
    cfg.hubgt.dropout = trial.suggest_float('hubgt_dropout', 0.0, 0.5)
    cfg.hubgt.dp_attn = trial.suggest_float('hubgt_attn_dropout', 0.0, 0.3)
    cfg.hubgt.dp_bias = trial.suggest_float('hubgt_bias_dropout', 0.0, 0.2)
    cfg.hubgt.pooling = trial.suggest_categorical(
        'hubgt_pooling', ['mean', 'sum']
    )

    suggest_training_cfg(trial, cfg, args)
    return cfg


def main():
    args = parse_common_args()
    run_tuning(MODEL_NAME, run_hubgt, suggest_cfg, args=args)


if __name__ == '__main__':
    main()
