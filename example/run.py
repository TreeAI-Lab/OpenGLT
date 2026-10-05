"""Unified fixed-configuration runner for OpenGLT datasets except CardBench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import optuna

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import init_cfg
from configs.best import canonical_model, load_best_entry, merge_cfg
from tuning.main_baselines_tuning import default_val_metric


MODELS = (
    "GCN", "GCNPlus", "GIN", "GINE", "SAGE", "SAGEPlus", "PNA",
    "GraphGPS", "HGT", "NAG", "EdgePool", "TopkPool", "GMT", "HMN",
    "AK", "ECS", "I2GNN", "MO", "HGP", "VIB", "RGC", "MVG", "GCA",
    "MLP-F", "MLP-S", "LGBM",
)


class ReplayFixedTrial:
    """Replay a saved Optuna parameter dictionary."""

    is_manifest_replay = True

    def __init__(self, params):
        self.params = dict(params)
        self.trial = optuna.trial.FixedTrial(self.params)

    def suggest_categorical(self, name, choices):
        choices = list(choices)
        aliases = {"num_warmup_epochs_choice": "num_warmup_epochs"}
        source = name if name in self.params else aliases.get(name)
        return self.params[source] if source in self.params else choices[0]

    def suggest_int(self, name, low, high, **kwargs):
        return int(self.params[name]) if name in self.params else int(low)

    def suggest_float(self, name, low, high, **kwargs):
        return float(self.params[name]) if name in self.params else float(low)

    def __getattr__(self, name):
        return getattr(self.trial, name)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--use_best_hyper_params", action="store_true")
    parser.add_argument("--params", default="{}", help="Optuna parameter JSON when best parameters are disabled.")
    parser.add_argument("--gpu", type=int, default=0, help="Use -1 for CPU.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_metric", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args()


def model_adapter(model):
    baseline = {
        "GCN": ("GCN", "none"), "GCNPlus": ("GCNplus", "none"),
        "GIN": ("GIN", "none"), "GINE": ("GINE", "none"),
        "SAGE": ("SAGE", "none"), "SAGEPlus": ("SAGEplus", "none"),
        "PNA": ("PNA", "none"), "EdgePool": ("GIN", "edgepool"),
        "TopkPool": ("GIN", "topk"), "GMT": ("GIN", "gmt"),
        "MLP-F": ("MLPFeature", "none"),
        "MLP-S": ("MLPFeatureStruct", "none"), "LGBM": ("LightGBM", "none"),
    }
    if model in baseline:
        from example.main_baselines import run_baselines
        from tuning.main_baselines_tuning import suggest_cfg
        return suggest_cfg, run_baselines, init_cfg, baseline[model]
    modules = {
        "GraphGPS": ("graphgps_tuning", "graphgps", "run_graphgps"),
        "HGT": ("hubgt_tuning", "hubgt", "run_hubgt"),
        "NAG": ("nagphormer_tuning", "nagphormer", "run_nagphormer"),
        "HMN": ("hymn_tuning", "hymn", "run_hymn"),
        "AK": ("gnnak_tuning", "gnnak", "run_gnnak"),
        "ECS": ("ecs_tuning", "ecs", "run_ecs"),
        "I2GNN": ("i2gnn_tuning", "i2gnn", "run_i2gnn"),
        "MO": ("mosgsl_tuning", "mosgsl", "run_mosgsl"),
        "HGP": ("hgpsl_tuning", "hgpsl", "run_hgpsl"),
        "VIB": ("vibgsl_tuning", "vibgsl", "run_vibgsl"),
        "RGC": ("rgc_tuning", "rgc", "run_rgc"),
        "MVG": ("mvgrl_tuning", "mvgrl", "run_mvgrl"),
        "GCA": ("gca_tuning", "gca", "run_gca"),
    }
    tuning_name, example_name, run_name = modules[model]
    tuning_module = __import__(f"tuning.{tuning_name}", fromlist=["suggest_cfg"])
    example_module = __import__(f"example.{example_name}", fromlist=[run_name])
    base = init_cfg
    if model == "GraphGPS":
        from example.graphgps import ensure_graphgps_cfg
        base = lambda: ensure_graphgps_cfg(init_cfg())
    return tuning_module.suggest_cfg, getattr(example_module, run_name), base, (None, None)


def tuning_args(args, backbone, pooling):
    counting = args.dataset.startswith("Count-")
    family = __import__("configs.best", fromlist=["dataset_family"]).dataset_family(args.dataset)
    epochs = args.epochs or (2000 if counting else 500)
    patience = args.patience or 50
    return SimpleNamespace(
        dataset=args.dataset, val_metric=args.val_metric, n_trials=200,
        train_epochs=epochs, train_patience=patience,
        train_runs=args.runs if args.runs is not None else (1 if family in {"tu", "gcare"} else 3),
        batch_size=args.batch_size or 128,
        gpu=args.gpu, seed=args.seed, fold_k=0, fold_seed=2024,
        deterministic=args.deterministic, study_name=None, storage=None,
        backbone=backbone or "GINE", pooling_method=pooling or "none",
        pretrain_epochs=200,
    )


def print_stats(title, stats):
    print(title)
    for metric in sorted(stats):
        item = stats[metric]
        print(f"  {metric}: {item['mean']:.6f} +/- {item['std']:.6f}")


def main():
    args = parse_args()
    model = canonical_model(args.model)
    suggest_cfg, run_fn, base_cfg_fn, (backbone, pooling) = model_adapter(model)
    run_args = tuning_args(args, backbone, pooling)
    if args.use_best_hyper_params:
        try:
            entry = load_best_entry(args.dataset, model)
        except KeyError as exc:
            raise SystemExit(str(exc)) from None
    else:
        entry = {"source": "command_line", "params": json.loads(args.params)}

    if "config" in entry:
        cfg = merge_cfg(base_cfg_fn(), entry["config"])
        cfg.dataset = args.dataset
        cfg.device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    else:
        cfg = suggest_cfg(ReplayFixedTrial(entry.get("params", {})), base_cfg_fn(), run_args)

    # A best-parameter record may also pin training controls that materially
    # affect reproducibility (for example CosineAnnealingWarmRestarts T_0).
    # Explicit command-line values below always take precedence.
    for key, value in entry.get("train", {}).items():
        setattr(cfg.train, key, value)

    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.patience is not None:
        cfg.train.patience = args.patience
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.runs is not None:
        cfg.train.runs = args.runs
    cfg.seed = args.seed
    cfg.deterministic = args.deterministic

    metric = default_val_metric(args.dataset) if args.val_metric == "auto" else args.val_metric
    print(f"[fixed configuration] {args.dataset} / {model}; source={entry.get('source')}")
    if args.validate_only:
        print("Configuration reconstruction passed.")
        return cfg
    result = run_fn(
        cfg=cfg, key_metric_override=metric,
        config_string=f"release_{args.dataset}_{model}_{args.seed}",
        prune_callback=None, evaluate_test=True,
    )
    print_stats("Validation:", result["best_val_perf"])
    print_stats("Test:", result["test_perf"])
    return result


if __name__ == "__main__":
    main()
