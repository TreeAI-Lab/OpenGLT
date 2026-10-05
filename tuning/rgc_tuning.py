import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from example.rgc import run_rgc
from tuning.uni_tuning import (
    dataset_family,
    parse_profile,
    parse_common_args,
    prepare_base_cfg,
    run_tuning,
    suggest_training_cfg,
)


MODEL_NAME = 'RGC'

def parse_width_profile(value):
    return parse_profile(value, 4)


def suggest_width_profile(trial, family):
    choices = (
        ['128x32x16x16', '128x64x32x16', '256x64x64x32']
        if family == 'counting'
        else ['128x32x16x16', '128x64x32x16', '256x64x32x16', '256x64x64x32']
    )
    return parse_width_profile(trial.suggest_categorical('rgc_width_profile', choices))


def suggest_backbone(trial, family):
    return trial.suggest_categorical('rgc_backbone', ['gcn', 'sage'])


def suggest_num_layers(trial):
    return trial.suggest_categorical('rgc_num_layers', [2, 3, 4])


def suggest_drop_profile(trial):
    choices = ['0.0x0.0', '0.1x0.1', '0.2x0.1']
    drop_edge, drop_node = trial.suggest_categorical('rgc_drop_profile', choices).split('x')
    return float(drop_edge), float(drop_node)


def suggest_aux_weight(trial):
    return trial.suggest_categorical('rgc_aux_weight', [0.0, 0.05, 0.1])


def suggest_pooling(trial):
    return trial.suggest_categorical('rgc_pooling', ['mean', 'max'])


def suggest_cfg(trial, base_cfg, args):
    cfg = prepare_base_cfg(base_cfg, args)
    family = dataset_family(args.dataset)

    nhid, hidden_features, embed_features, struct_dim = suggest_width_profile(trial, family)
    cfg.rgc.nhid = nhid
    cfg.rgc.hidden_features = hidden_features
    cfg.rgc.embed_features = embed_features
    cfg.rgc.struct_dim = struct_dim

    cfg.rgc.backbone = suggest_backbone(trial, family)
    cfg.rgc.num_layers = suggest_num_layers(trial)
    cfg.rgc.n_heads = (
        trial.suggest_categorical('rgc_n_heads', [2, 4, 8])
        if cfg.rgc.backbone == 'gat'
        else 1
    )
    cfg.rgc.drop_edge, cfg.rgc.drop_node = suggest_drop_profile(trial)
    cfg.rgc.temperature = trial.suggest_categorical('rgc_temperature', [0.1, 0.2, 0.5])
    cfg.rgc.pooling = suggest_pooling(trial)
    cfg.rgc.aux_weight = suggest_aux_weight(trial)

    suggest_training_cfg(trial, cfg, args)
    return cfg


def main():
    args = parse_common_args()
    run_tuning(MODEL_NAME, run_rgc, suggest_cfg, args=args)


if __name__ == '__main__':
    main()
