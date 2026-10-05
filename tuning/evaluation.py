"""Final evaluation helpers shared by the public Optuna tuners."""

import json

import optuna


def is_tu_dataset(dataset):
    from datasets.loader import TU_DATASETS
    return str(dataset) in TU_DATASETS


def _print_stats(title, stats):
    print(title)
    for metric in sorted(stats):
        item = stats[metric]
        print(f"  {metric}: {item['mean']:.6f} +/- {item['std']:.6f}")


def run_tu_cv_final_evaluation(
        study, suggest_cfg_fn, base_cfg, args, run_fn, model_name,
        optimize_metric):
    """Evaluate one selected configuration across the fixed ten TU folds."""
    if not is_tu_dataset(args.dataset):
        raise ValueError("TU CV final evaluation requires a TU dataset.")
    cfg = suggest_cfg_fn(optuna.trial.FixedTrial(study.best_trial.params), base_cfg, args)
    cfg.seed = 42
    cfg.train.runs = 1
    print("\nTU 10-fold final evaluation:")
    print(f"  model: {model_name}")
    print(f"  dataset: {args.dataset}")
    print("  split seed: 2024; model seed: 42")
    print("  selected params:", json.dumps(study.best_trial.params, sort_keys=True))
    safe_model = str(model_name).replace("/", "_").replace(" ", "_")
    result = run_fn(
        cfg=cfg,
        key_metric_override=optimize_metric,
        config_string=f"final_10fold_{cfg.dataset}_{safe_model}_seed42",
        prune_callback=None,
        evaluate_test=True,
    )
    if int(result.get("num_sessions", -1)) != 10:
        raise RuntimeError("TU final evaluation did not return exactly ten folds.")
    _print_stats("Validation metrics (10 folds, mean +/- std):", result["best_val_perf"])
    _print_stats("Test metrics (10 folds, mean +/- std):", result["test_perf"])
    return result


__all__ = ["is_tu_dataset", "run_tu_cv_final_evaluation"]
