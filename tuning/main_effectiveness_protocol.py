"""Protocol helpers for non-CV main-effectiveness datasets.

OGB, LRGB, PyG molecules, and Count-* use fixed train/validation/test splits.
Hyperparameters are selected from one seed using validation only, then the
chosen configuration is retrained from scratch with three seeds. TU datasets
are excluded because their uncertainty comes from 10-fold CV.
"""


MAIN_EFFECTIVENESS_CLASSIFICATION = {
    'ogbg-molbace',
    'ogbg-molhiv',
    'ogbg-moltox21',
    'ogbg-molpcba',
    'peptides-func',
}

MAIN_EFFECTIVENESS_REGRESSION = {
    'aqsol',
    'zinc',
    'peptides-struct',
}


def is_main_effectiveness_fixed_split(dataset):
    name = str(dataset)
    lower = name.lower()
    return (
        lower in MAIN_EFFECTIVENESS_CLASSIFICATION
        or lower in MAIN_EFFECTIVENESS_REGRESSION
        or name.startswith('Count-')
    )


def main_effectiveness_default_metric(dataset):
    """Return the primary validation metric for the dataset."""
    name = str(dataset)
    lower = name.lower()
    if lower in MAIN_EFFECTIVENESS_CLASSIFICATION:
        return 'acc'
    if lower in MAIN_EFFECTIVENESS_REGRESSION or name.startswith('Count-'):
        return 'r2'
    raise ValueError(f"Not a fixed-split main-effectiveness dataset: {dataset}")


def apply_main_effectiveness_trial_protocol(cfg, args):
    """Use one seed per Optuna trial for fixed-split main datasets."""
    if is_main_effectiveness_fixed_split(args.dataset):
        cfg.seed = 42
        cfg.train.runs = 1
    return cfg


def print_main_effectiveness_protocol(dataset, n_trials, requested_seed):
    if not is_main_effectiveness_fixed_split(dataset):
        return
    if int(requested_seed) != 42:
        print(
            f"[Main effectiveness protocol] ignoring --seed={requested_seed} "
            "during tuning; using seed=42."
        )
    print(
        f"[Main effectiveness protocol] dataset={dataset}, fixed split, "
        f"Optuna target_total={int(n_trials)}, one seed (42) per trial, "
        "validation-only selection."
    )
    print(
        "[Main effectiveness protocol] after tuning: retrain from scratch "
        "with seeds=[42, 43, 44], then evaluate validation and test."
    )
