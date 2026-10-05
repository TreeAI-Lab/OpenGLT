import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from example.hgpsl import run_hgpsl
from tuning.uni_tuning import (
    dataset_family,
    parse_common_args,
    prepare_base_cfg,
    run_tuning,
    suggest_training_cfg,
)


MODEL_NAME = 'HGPSL'

def suggest_hidden_size(trial, family, is_large):
    return trial.suggest_categorical('hgpsl_hidden_size', [64, 128, 256])


def suggest_pooling_ratio(trial, family):
    return trial.suggest_categorical('hgpsl_pooling_ratio', [0.3, 0.5, 0.8])


def suggest_dropout(trial):
    return trial.suggest_categorical('hgpsl_dropout', [0.0, 0.1, 0.2])


def suggest_structure_profile(trial, family, is_large):
    if family == 'counting' or is_large:
        choices = ['sample_sparse_sl', 'sample_softmax_sl', 'sample_no_sl']
    else:
        choices = ['sample_sparse_sl', 'dense_softmax_sl', 'sample_no_sl']
    profile = trial.suggest_categorical('hgpsl_structure_profile', choices)
    sample_neighbor = profile.startswith('sample')
    structure_learning = not profile.endswith('no_sl')
    sparse_attention = 'sparse' in profile and structure_learning
    return sample_neighbor, sparse_attention, structure_learning


def suggest_lambda(trial, structure_learning):
    if not structure_learning:
        return 1.0
    return trial.suggest_categorical('hgpsl_lamb', [0.1, 0.5, 1.0, 2.0])


def suggest_cfg(trial, base_cfg, args):
    cfg = prepare_base_cfg(base_cfg, args)
    family = dataset_family(args.dataset)
    is_large = args.dataset.lower() in {
        'ogbg-molpcba', 'peptides-func', 'peptides-struct',
        'dd', 'collab', 'reddit-binary',
    }

    cfg.hgpsl.hidden_size = suggest_hidden_size(trial, family, is_large)
    cfg.hgpsl.pooling_ratio = suggest_pooling_ratio(trial, family)
    cfg.hgpsl.dropout = suggest_dropout(trial)

    sample_neighbor, sparse_attention, structure_learning = suggest_structure_profile(
        trial, family, is_large
    )
    cfg.hgpsl.sample_neighbor = sample_neighbor
    cfg.hgpsl.sparse_attention = sparse_attention
    cfg.hgpsl.structure_learning = structure_learning
    cfg.hgpsl.lamb = suggest_lambda(trial, structure_learning)

    suggest_training_cfg(trial, cfg, args)
    return cfg


def main():
    args = parse_common_args()
    run_tuning(MODEL_NAME, run_hgpsl, suggest_cfg, args=args)


if __name__ == '__main__':
    main()
