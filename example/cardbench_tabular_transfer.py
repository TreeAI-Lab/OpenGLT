from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from example import cardbench_tabular as tab


cb = tab.cb


def load_shot_module():
    name = "cardbench_gnn_shot_protocol"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(os.path.dirname(__file__), "cardbench_transfer.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load shared CardBench transfer protocol: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


shot = load_shot_module()


@dataclass
class TransferRecord:
    target_dataset_name: str
    target_npz_path: str
    workload_type: str
    model_name: str
    seed: int
    setting: str
    shot_size: int
    hparams: Dict[str, Any]
    outcome: tab.StageOutcome


def parse_shot_sizes(text: str) -> List[int]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values or min(values) < 2:
        raise ValueError("--fewshot_sizes must contain integers >= 2.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CardBench BJ leave-one-dataset-out transfer for LGBM / MLP-Feat / MLP-Struct."
    )
    parser.add_argument("--test_npz_path", type=str, required=True)
    parser.add_argument("--model_name", choices=tab.TABULAR_MODELS, required=True)
    parser.add_argument("--out_dir", type=str, default="cardbench_tabular_transfer_results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU for MLPs; -1 selects CPU. LGBM is CPU-only.")
    parser.add_argument("--max_graphs", type=int, default=None)
    parser.add_argument(
        "--expected_num_datasets",
        type=int,
        default=20,
        help="Fail if the workload directory does not contain exactly this many .npz datasets; use 0 to disable.",
    )
    parser.add_argument("--holdout_test_size", type=int, default=500)
    parser.add_argument("--cap_trainval", type=int, default=4500)
    parser.add_argument("--fewshot_sizes", type=str, default="1000")
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--first_seed", type=int, default=2024)
    parser.add_argument("--n_trials", type=int, default=200, help="Source-validation-only HPO budget; use 0 to select the fixed model-wide configuration.")
    parser.add_argument("--debug", action="store_true")
    tab.add_tabular_cli_args(parser)
    return parser.parse_args()


def choose_device(args: argparse.Namespace) -> torch.device:
    if args.model_name == "lgbm" or args.gpu < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise SystemExit("MLP transfer requires CUDA unless --gpu -1 is supplied.")
    if args.gpu >= torch.cuda.device_count():
        raise SystemExit(f"Requested --gpu={args.gpu}, but only {torch.cuda.device_count()} GPU(s) are visible.")
    torch.cuda.set_device(args.gpu)
    return torch.device(f"cuda:{args.gpu}")


def stable_hash(obj: Dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


def save_checkpoint(path: str, record: TransferRecord) -> None:
    mode = "zero_shot_pretrain" if record.setting == "zero_shot" else record.setting
    payload = {
        "best_epoch": int(record.outcome.best_epoch),
        "best_val_metrics": dict(record.outcome.best_val_metrics),
        "test_metrics": dict(record.outcome.test_metrics),
        "efficiency_metrics": dict(record.outcome.efficiency_metrics),
        "y_true_raw": np.asarray(record.outcome.y_true_raw),
        "y_pred_raw": np.asarray(record.outcome.y_pred_raw),
        "best_state": record.outcome.best_state,
        "hparams": dict(record.hparams),
        "meta": {
            "mode": mode,
            "target_npz_path": record.target_npz_path,
            "target_dataset_name": record.target_dataset_name,
            "seed": int(record.seed),
            "shot_size": int(record.shot_size),
            "model_name": record.model_name,
            "feature_set": "feature+structure" if tab.uses_structure(record.model_name) else "feature",
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def record_to_row(record: TransferRecord) -> Dict[str, Any]:
    return {
        "target_dataset_name": record.target_dataset_name,
        "target_npz_path": record.target_npz_path,
        "workload_type": record.workload_type,
        "model_name": record.model_name,
        "feature_set": "feature+structure" if tab.uses_structure(record.model_name) else "feature",
        "seed": record.seed,
        "setting": record.setting,
        "shot_size": record.shot_size,
        "best_epoch": record.outcome.best_epoch,
        **{f"hparam_{k}": v for k, v in record.hparams.items()},
        **cb.metrics_to_row("val", record.outcome.best_val_metrics),
        **cb.metrics_to_row("test", record.outcome.test_metrics),
        **cb.metrics_to_row("eff", record.outcome.efficiency_metrics),
    }


def aggregate_records(records: Sequence[TransferRecord]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[TransferRecord]] = {}
    for record in records:
        grouped.setdefault((record.setting, record.shot_size), []).append(record)
    rows: List[Dict[str, Any]] = []
    for (setting, shot_size), bucket in sorted(grouped.items()):
        row: Dict[str, Any] = {
            "target_dataset_name": bucket[0].target_dataset_name,
            "workload_type": bucket[0].workload_type,
            "model_name": bucket[0].model_name,
            "setting": setting,
            "shot_size": shot_size,
            "n_runs": len(bucket),
        }
        for prefix, attr in (("test", "test_metrics"), ("val", "best_val_metrics"), ("eff", "efficiency_metrics")):
            metric_names = sorted(getattr(bucket[0].outcome, attr).keys())
            for metric in metric_names:
                values = [float(getattr(item.outcome, attr)[metric]) for item in bucket]
                row[f"{prefix}_{metric}_mean"] = float(mean(values))
                row[f"{prefix}_{metric}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    args.model_name = tab.normalize_model_name(args.model_name)
    if args.cap_trainval <= 0:
        args.cap_trainval = None
    if args.n_runs <= 0 or args.n_trials < 0:
        raise SystemExit("--n_runs must be positive and --n_trials must be non-negative.")
    if not os.path.isfile(args.test_npz_path):
        raise SystemExit(f"Missing target .npz: {args.test_npz_path}")
    shot_sizes = parse_shot_sizes(args.fewshot_sizes)
    device = choose_device(args)
    os.makedirs(args.out_dir, exist_ok=True)
    _, target_path, workload_type, source_paths = shot.resolve_workload_files(args.test_npz_path)
    actual_num_datasets = len(source_paths) + 1
    if args.expected_num_datasets > 0 and actual_num_datasets != args.expected_num_datasets:
        raise SystemExit(
            f"Expected {args.expected_num_datasets} CardBench datasets beside {target_path}, "
            f"but found {actual_num_datasets}. Refusing to change the leave-one-dataset-out protocol."
        )
    target_name, _, _ = cb.dataset_identity_from_path(target_path)
    print(
        f"[CardBench tabular transfer] target={target_name} | sources={len(source_paths)} | "
        f"model={args.model_name} | shots={shot_sizes} | n_trials={args.n_trials} | device={device}"
    )
    print("[protocol] source HPO is validation-only; target test is evaluated only after model selection/training.")
    include_structure = tab.uses_structure(args.model_name)
    result_dir = os.path.join(args.out_dir, workload_type, target_name, args.model_name)
    os.makedirs(result_dir, exist_ok=True)

    canonical_bundle = shot.get_or_create_shot_bundle(args, int(args.first_seed))
    shot.validate_nested_fewshot_subsets(canonical_bundle, shot_sizes)
    # Source-only HPO: do not materialize target-test feature vectors here.
    source_train_raw = tab.extract_arrays(
        canonical_bundle.source_train_graphs, canonical_bundle.num_edge_types, include_structure
    )
    source_val_raw = tab.extract_arrays(
        canonical_bundle.source_val_graphs, canonical_bundle.num_edge_types, include_structure
    )
    hpo_scaler = tab.FeatureStandardizer.fit(source_train_raw.x)
    source_train = hpo_scaler.transform(source_train_raw)
    source_val = hpo_scaler.transform(source_val_raw)
    if args.n_trials > 0:
        best_hparams = tab.tune_hparams(
            args.model_name,
            source_train,
            source_val,
            canonical_bundle.feature_processor,
            args,
            device,
        )
    else:
        best_hparams = tab.default_hparams(args, args.model_name)
    cb.write_json(os.path.join(result_dir, f"best_hparams_seed{args.first_seed}.json"), best_hparams)
    print("Best / selected hyperparameters:")
    for key, value in best_hparams.items():
        print(f"  {key}: {value}")

    records: List[TransferRecord] = []
    for run_idx in range(args.n_runs):
        seed = int(args.first_seed + run_idx)
        print("\n" + "-" * 96)
        print(f"Run {run_idx + 1}/{args.n_runs} | seed={seed}")
        bundle = canonical_bundle if seed == args.first_seed else shot.get_or_create_shot_bundle(args, seed)
        source_train_raw = tab.extract_arrays(bundle.source_train_graphs, bundle.num_edge_types, include_structure)
        source_val_raw = tab.extract_arrays(bundle.source_val_graphs, bundle.num_edge_types, include_structure)
        source_scaler = tab.FeatureStandardizer.fit(source_train_raw.x)
        source_train = source_scaler.transform(source_train_raw)
        source_val = source_scaler.transform(source_val_raw)
        target_test = source_scaler.transform(
            tab.extract_arrays(bundle.target_test_graphs, bundle.num_edge_types, include_structure)
        )

        zero_outcome = tab.run_stage(
            args.model_name,
            source_train,
            source_val,
            target_test,
            bundle.feature_processor,
            best_hparams,
            device,
            seed,
        )
        zero_record = TransferRecord(
            target_name, target_path, workload_type, args.model_name, seed,
            "zero_shot", 0, dict(best_hparams), zero_outcome,
        )
        records.append(zero_record)
        print("[zero-shot]")
        cb.print_metrics("Validation metrics", zero_outcome.best_val_metrics)
        cb.print_metrics("Test metrics", zero_outcome.test_metrics)
        zero_path = os.path.join(
            result_dir,
            f"zeroshotpretrain_{args.model_name}_exclude-{target_name}_seed{seed}_hp{stable_hash(best_hparams)}.pt",
        )
        save_checkpoint(zero_path, zero_record)

        for shot_size in shot_sizes:
            subset = shot.build_fewshot_subset(bundle, shot_size)
            target_train = source_scaler.transform(
                tab.extract_arrays(subset.train_graphs, bundle.num_edge_types, include_structure)
            )
            target_val = source_scaler.transform(
                tab.extract_arrays(subset.val_graphs, bundle.num_edge_types, include_structure)
            )
            fine_outcome = tab.run_stage(
                args.model_name,
                target_train,
                target_val,
                target_test,
                bundle.feature_processor,
                best_hparams,
                device,
                seed,
                init_state=zero_outcome.best_state,
            )
            fine_record = TransferRecord(
                target_name, target_path, workload_type, args.model_name, seed,
                "finetune", shot_size, dict(best_hparams), fine_outcome,
            )
            records.append(fine_record)
            print(f"[finetune-{shot_size}]")
            cb.print_metrics("Validation metrics", fine_outcome.best_val_metrics)
            cb.print_metrics("Test metrics", fine_outcome.test_metrics)
            save_checkpoint(
                os.path.join(result_dir, f"finetune_{args.model_name}_target-{target_name}_seed{seed}_shot{shot_size}_hp{stable_hash(best_hparams)}.pt"),
                fine_record,
            )

            scratch_outcome = tab.run_stage(
                args.model_name,
                target_train,
                target_val,
                target_test,
                bundle.feature_processor,
                best_hparams,
                device,
                seed,
                init_state=None,
            )
            scratch_record = TransferRecord(
                target_name, target_path, workload_type, args.model_name, seed,
                "scratch", shot_size, dict(best_hparams), scratch_outcome,
            )
            records.append(scratch_record)
            print(f"[scratch-{shot_size}]")
            cb.print_metrics("Validation metrics", scratch_outcome.best_val_metrics)
            cb.print_metrics("Test metrics", scratch_outcome.test_metrics)
            save_checkpoint(
                os.path.join(result_dir, f"scratch_{args.model_name}_target-{target_name}_seed{seed}_shot{shot_size}_hp{stable_hash(best_hparams)}.pt"),
                scratch_record,
            )

    cb.write_csv(os.path.join(result_dir, "shot_run_results.csv"), [record_to_row(r) for r in records])
    summaries = aggregate_records(records)
    cb.write_csv(os.path.join(result_dir, "shot_summary.csv"), summaries)
    cb.write_json(os.path.join(result_dir, "shot_summary.json"), summaries)
    print(f"\nSaved transfer artifacts under: {result_dir}")


if __name__ == "__main__":
    main()
