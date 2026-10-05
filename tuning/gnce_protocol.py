"""Shared fixed-split tuning and final-evaluation protocol.

GNCE and GOOD use fixed data splits. Hyperparameters are selected with one
training run (seed 42) and validation metrics only. The selected configuration
is then reinitialized and trained independently with seeds 42, 43, and 44;
only this final stage evaluates the fixed test split.
"""

import json

import optuna


GNCE_TUNING_SEED = 42
GNCE_DEFAULT_N_TRIALS = 200
GNCE_FINAL_SEEDS = (42, 43, 44)
GOOD_TUNING_SEED = 42
GOOD_DEFAULT_N_TRIALS = 200
GOOD_FINAL_SEEDS = (42, 43, 44)


def is_gnce_dataset(dataset):
    return str(dataset).lower().startswith('gnce-')


def is_good_dataset(dataset):
    return str(dataset).lower().replace('_', '-').startswith('good-')


def is_fixed_split_dataset(dataset):
    return is_gnce_dataset(dataset) or is_good_dataset(dataset)


def _protocol_label(dataset):
    if is_gnce_dataset(dataset):
        return 'GNCE'
    if is_good_dataset(dataset):
        return 'GOOD'
    return 'fixed-split'


def _tuning_seed(dataset):
    if is_good_dataset(dataset):
        return GOOD_TUNING_SEED
    return GNCE_TUNING_SEED


def _final_seeds(dataset):
    if is_good_dataset(dataset):
        return GOOD_FINAL_SEEDS
    return GNCE_FINAL_SEEDS


def resolve_tuning_seed(dataset, requested_seed):
    if is_fixed_split_dataset(dataset):
        tuning_seed = _tuning_seed(dataset)
        if int(requested_seed) != tuning_seed:
            print(
                f"[{_protocol_label(dataset)} protocol] ignoring "
                f"--seed={requested_seed}; Optuna and tuning training use "
                f"seed={tuning_seed}."
            )
        return tuning_seed
    return int(requested_seed)


def resolve_n_trials(dataset, requested_trials, non_gnce_default=200):
    if requested_trials is not None:
        requested_trials = int(requested_trials)
        expected = (
            GOOD_DEFAULT_N_TRIALS
            if is_good_dataset(dataset) else GNCE_DEFAULT_N_TRIALS
        )
        if is_fixed_split_dataset(dataset) and requested_trials != expected:
            print(
                f"[{_protocol_label(dataset)} protocol override] "
                f"n_trials={requested_trials}; the benchmark protocol uses "
                f"n_trials={expected}."
            )
        return requested_trials
    if is_good_dataset(dataset):
        return GOOD_DEFAULT_N_TRIALS
    if is_gnce_dataset(dataset):
        return GNCE_DEFAULT_N_TRIALS
    return int(non_gnce_default)


def apply_trial_protocol(cfg, args):
    """Use exactly one run per trial and never average seeds during tuning."""
    if is_fixed_split_dataset(args.dataset):
        cfg.seed = _tuning_seed(args.dataset)
        cfg.train.runs = 1
    else:
        cfg.seed = int(args.seed)
        cfg.train.runs = int(args.train_runs)
    return cfg


def make_pruner():
    """A conservative median pruner suitable for 200 epoch-wise trials."""
    return optuna.pruners.MedianPruner(
        n_startup_trials=20,
        n_warmup_steps=10,
        interval_steps=5,
    )


def print_protocol(dataset, n_trials, requested_seed):
    if not is_fixed_split_dataset(dataset):
        return
    tuning_seed = resolve_tuning_seed(dataset, requested_seed)
    label = _protocol_label(dataset)
    final_seeds = _final_seeds(dataset)
    print(
        f"[{label} protocol] fixed official split; "
        f"Optuna n_trials={n_trials}, sampler/training seed={tuning_seed}, "
        "one run per trial, validation-only selection."
    )
    print(
        f"[{label} protocol] after tuning: independently "
        "reinitialize/train with "
        f"seeds={list(final_seeds)}, then evaluate validation and test."
    )


def optimize_to_total(study, objective, target_n_trials):
    """Resume a study up to a total budget instead of adding a new budget."""
    from optuna.trial import TrialState

    target_n_trials = int(target_n_trials)
    terminal_states = {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}
    finished = sum(trial.state in terminal_states for trial in study.trials)
    queued = sum(trial.state == TrialState.WAITING for trial in study.trials)
    running = sum(trial.state == TrialState.RUNNING for trial in study.trials)
    remaining = max(0, target_n_trials - finished)
    print(
        f"[Optuna] study={study.study_name}, finished_trials={finished}, "
        f"waiting_trials={queued}, running_trials={running}, "
        f"target_total={target_n_trials}, optimize_calls_to_run={remaining}"
    )
    if remaining:
        study.optimize(objective, n_trials=remaining)
    return remaining


def build_final_cfg(study, suggest_cfg_fn, base_cfg, args, final_seeds=None):
    """Reconstruct the best conditional search-space configuration."""
    fixed_trial = optuna.trial.FixedTrial(study.best_trial.params)
    cfg = suggest_cfg_fn(fixed_trial, base_cfg, args)
    if final_seeds is not None:
        seeds = tuple(int(seed) for seed in final_seeds)
        if not seeds:
            raise ValueError("final_seeds must contain at least one seed.")
        cfg.seed = seeds[0]
        cfg.train.runs = len(seeds)
    elif is_fixed_split_dataset(args.dataset):
        seeds = _final_seeds(args.dataset)
        cfg.seed = seeds[0]
        cfg.train.runs = len(seeds)
    else:
        cfg.seed = int(args.seed)
        cfg.train.runs = int(args.train_runs)
    return cfg


def _print_metric_block(title, stats):
    print(title)
    if not stats:
        print("  <none>")
        return
    for metric in sorted(stats):
        value = stats[metric]
        print(
            f"  {metric}: {float(value['mean']):.6f} "
            f"+/- {float(value['std']):.6f}"
        )


def run_final_evaluation(
        study, suggest_cfg_fn, base_cfg, args, run_fn, model_name,
        optimize_metric, final_seeds=None):
    """Run the selected configuration and print all validation/test metrics."""
    final_cfg = build_final_cfg(
        study, suggest_cfg_fn, base_cfg, args, final_seeds=final_seeds
    )
    if final_seeds is not None:
        seeds = tuple(int(seed) for seed in final_seeds)
    elif is_fixed_split_dataset(args.dataset):
        seeds = _final_seeds(args.dataset)
    else:
        seeds = tuple(final_cfg.seed + i for i in range(final_cfg.train.runs))

    safe_model = str(model_name).replace('/', '_').replace(' ', '_')
    config_string = (
        f"final_{len(seeds)}runs_{final_cfg.dataset}_{safe_model}_"
        f"seeds_{'-'.join(str(seed) for seed in seeds)}"
    )
    print("\nFinal evaluation:")
    print(f"  model: {model_name}")
    print(f"  dataset: {final_cfg.dataset}")
    print(f"  independent run seeds: {list(seeds)}")
    print(
        "  selected params:",
        json.dumps(study.best_trial.params, sort_keys=True),
    )

    result = run_fn(
        cfg=final_cfg,
        key_metric_override=optimize_metric,
        config_string=config_string,
        prune_callback=None,
        evaluate_test=True,
    )
    _print_metric_block(
        f"Validation metrics ({len(seeds)} runs, mean +/- std):",
        result['best_val_perf'],
    )
    _print_metric_block(
        f"Test metrics ({len(seeds)} runs, mean +/- std):",
        result['test_perf'],
    )
    return result
