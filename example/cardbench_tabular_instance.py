from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from example import cardbench_tabular as tab


cb = tab.cb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CardBench instance-based LGBM / MLP-Feat / MLP-Struct baselines."
    )
    parser.add_argument("--npz_path", type=str, required=True, help="One .npz file, a directory, or a glob.")
    parser.add_argument("--model_name", choices=tab.TABULAR_MODELS, required=True)
    parser.add_argument("--out_dir", type=str, default="cardbench_tabular_instance_results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU for MLPs; use -1 for CPU. LGBM is always CPU-only.")
    parser.add_argument("--max_graphs", type=int, default=None)
    parser.add_argument("--holdout_test_size", type=int, default=500)
    parser.add_argument("--cap_trainval", type=int, default=4500)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--first_seed", type=int, default=2024)
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--use_best_hyper_params", action="store_true")
    parser.add_argument("--debug", action="store_true")
    tab.add_tabular_cli_args(parser)
    return parser.parse_args()


def choose_device(args: argparse.Namespace) -> torch.device:
    if args.model_name == "lgbm" or args.gpu < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise SystemExit("MLP baselines require CUDA unless --gpu -1 is supplied.")
    if args.gpu >= torch.cuda.device_count():
        raise SystemExit(f"Requested --gpu={args.gpu}, but only {torch.cuda.device_count()} GPU(s) are visible.")
    torch.cuda.set_device(args.gpu)
    return torch.device(f"cuda:{args.gpu}")


def main() -> None:
    args = parse_args()
    args.model_name = tab.normalize_model_name(args.model_name)
    if args.cap_trainval <= 0:
        args.cap_trainval = None
    if args.n_runs <= 0 or args.n_trials < 0:
        raise SystemExit("--n_runs must be positive and --n_trials must be non-negative.")
    device = choose_device(args)
    npz_files = cb.list_cardbench_npz_files(args.npz_path)
    if not npz_files:
        raise SystemExit(f"No CardBench .npz files resolved from: {args.npz_path}")
    os.makedirs(args.out_dir, exist_ok=True)
    print(
        f"[CardBench tabular instance] model={args.model_name} | datasets={len(npz_files)} | "
        f"n_trials={args.n_trials} | final_seeds={[args.first_seed + i for i in range(args.n_runs)]} | device={device}"
    )
    print("[protocol] HPO uses validation median q-error only; no test inference is executed during trials.")

    all_results: List[Any] = []
    all_run_rows: List[Dict[str, Any]] = []
    include_structure = tab.uses_structure(args.model_name)

    for dataset_idx, npz_path in enumerate(npz_files, start=1):
        print("\n" + "=" * 96)
        print(f"[{dataset_idx}/{len(npz_files)}] {npz_path}")
        raw_dataset = cb.load_raw_graph_dataset(npz_path, max_graphs=args.max_graphs)
        usable = [g for g in raw_dataset.graphs if cb.is_usable_query_graph(g)]
        if len(usable) <= args.holdout_test_size + 2:
            print("Skipping: insufficient usable graphs.")
            continue
        canonical_seed = int(args.first_seed)
        canonical = cb.prepare_split(
            raw_dataset,
            canonical_seed,
            args.holdout_test_size,
            args.cap_trainval,
            debug=args.debug,
        )
        train_raw = tab.extract_arrays(canonical.train_graphs, canonical.num_edge_types, include_structure)
        val_raw = tab.extract_arrays(canonical.val_graphs, canonical.num_edge_types, include_structure)
        hpo_scaler = tab.FeatureStandardizer.fit(train_raw.x)
        train_a = hpo_scaler.transform(train_raw)
        val_a = hpo_scaler.transform(val_raw)
        print(
            f"split train/val/test={len(train_a.y_norm)}/{len(val_a.y_norm)}/{len(canonical.test_graphs)} | "
            f"tabular_dim={train_a.x.shape[1]} | structure={include_structure}"
        )
        if args.use_best_hyper_params:
            from configs.best import load_best_entry
            model_map = {"mlp_feat": "MLP-F", "mlp_struct": "MLP-S", "lgbm": "LGBM"}
            best_hparams = dict(load_best_entry(
                "CardBench", model_map[args.model_name], setting=raw_dataset.name
            )["params"])
        elif args.n_trials > 0:
            best_hparams = tab.tune_hparams(
                args.model_name,
                train_a,
                val_a,
                canonical.feature_processor,
                args,
                device,
            )
        else:
            best_hparams = tab.default_hparams(args, args.model_name)
        print("Best / selected hyperparameters:")
        for key, value in best_hparams.items():
            print(f"  {key}: {value}")

        result_dir = os.path.join(
            args.out_dir,
            raw_dataset.workload_type,
            raw_dataset.name,
            args.model_name,
        )
        os.makedirs(result_dir, exist_ok=True)
        cb.write_json(os.path.join(result_dir, "best_hparams.json"), best_hparams)

        for run_idx in range(args.n_runs):
            seed = int(args.first_seed + run_idx)
            prepared = canonical if seed == canonical_seed else cb.prepare_split(
                raw_dataset,
                seed,
                args.holdout_test_size,
                args.cap_trainval,
                debug=args.debug,
            )
            train_a, val_a, test_a, _ = tab.prepare_array_splits(
                prepared.train_graphs,
                prepared.val_graphs,
                prepared.test_graphs,
                prepared.num_edge_types,
                include_structure,
            )
            print(f"\nRun {run_idx + 1}/{args.n_runs} | seed={seed}")
            outcome = tab.run_stage(
                args.model_name,
                train_a,
                val_a,
                test_a,
                prepared.feature_processor,
                best_hparams,
                device,
                seed,
            )
            print(f"Best epoch/iteration: {outcome.best_epoch}")
            cb.print_metrics("Validation metrics", outcome.best_val_metrics)
            cb.print_metrics("Test metrics", outcome.test_metrics)
            result = cb.RunResult(
                dataset_name=raw_dataset.name,
                base_name=raw_dataset.base_name,
                workload_type=raw_dataset.workload_type,
                npz_path=npz_path,
                model_name=args.model_name,
                seed=seed,
                best_epoch=outcome.best_epoch,
                hparams=dict(best_hparams),
                best_val_metrics=dict(outcome.best_val_metrics),
                test_metrics=dict(outcome.test_metrics),
                efficiency_metrics=dict(outcome.efficiency_metrics),
                y_true_raw=np.asarray(outcome.y_true_raw),
                y_pred_raw=np.asarray(outcome.y_pred_raw),
            )
            all_results.append(result)
            row = {
                "dataset_name": raw_dataset.name,
                "base_name": raw_dataset.base_name,
                "workload_type": raw_dataset.workload_type,
                "npz_path": npz_path,
                "model_name": args.model_name,
                "feature_set": "feature+structure" if include_structure else "feature",
                "seed": seed,
                "best_epoch": outcome.best_epoch,
                **{f"hparam_{k}": v for k, v in best_hparams.items()},
                **cb.metrics_to_row("val", outcome.best_val_metrics),
                **cb.metrics_to_row("test", outcome.test_metrics),
                **cb.metrics_to_row("eff", outcome.efficiency_metrics),
            }
            all_run_rows.append(row)

        dataset_results = [
            r for r in all_results
            if r.dataset_name == raw_dataset.name and r.model_name == args.model_name
        ]
        cb.write_csv(os.path.join(result_dir, "run_results.csv"), [
            r for r in all_run_rows
            if r["dataset_name"] == raw_dataset.name and r["model_name"] == args.model_name
        ])
        cb.write_csv(os.path.join(result_dir, "dataset_summary.csv"), cb.aggregate_runs_by_dataset(dataset_results))
        cb.write_json(os.path.join(result_dir, "dataset_summary.json"), cb.aggregate_runs_by_dataset(dataset_results))

    if not all_results:
        raise SystemExit("No CardBench tabular instance experiments were completed.")
    cb.write_csv(os.path.join(args.out_dir, f"{args.model_name}_all_run_results.csv"), all_run_rows)
    dataset_rows = cb.aggregate_runs_by_dataset(all_results)
    cb.write_csv(os.path.join(args.out_dir, f"{args.model_name}_dataset_summary.csv"), dataset_rows)
    family_summary = {
        "model_name": args.model_name,
        "feature_set": "feature+structure" if include_structure else "feature",
        "num_datasets": len(dataset_rows),
        "num_runs": len(all_results),
        **cb.aggregate_family_macro(dataset_rows),
        **cb.aggregate_family_micro(all_results),
    }
    cb.write_json(os.path.join(args.out_dir, f"{args.model_name}_family_summary.json"), family_summary)
    print(f"\nSaved instance results under: {args.out_dir}")


if __name__ == "__main__":
    main()
