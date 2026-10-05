from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import pickle
import sys
from pathlib import Path
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BASE_MODULE_NAME = "cardbench_instance_base_v7_1"
BASE_FILENAME = "cardbench.py"


def load_base_module(base_path: str):
    if BASE_MODULE_NAME in sys.modules:
        return sys.modules[BASE_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(BASE_MODULE_NAME, base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base module from: {base_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[BASE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


cb = load_base_module(os.path.join(os.path.dirname(__file__), BASE_FILENAME))


@dataclass
class ShotBundle:
    workload_type: str
    workload_dir: str
    target_npz_path: str
    target_dataset_name: str
    source_npz_paths: List[str]
    seed: int
    holdout_test_size: int
    cap_trainval: Optional[int]
    feature_processor: Any
    num_edge_types: int
    protected_edge_type_ids: Tuple[int, ...]
    source_train_graphs: List[Any]
    source_val_graphs: List[Any]
    target_test_graphs: List[Any]
    target_trainval_graphs: List[Any]
    global_edge_type_to_id: Dict[str, int]


@dataclass
class ShotRunResult:
    target_dataset_name: str
    target_npz_path: str
    workload_type: str
    model_name: str
    seed: int
    setting: str
    shot_size: int
    best_epoch: int
    hparams: Dict[str, Any]
    best_val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    efficiency_metrics: Dict[str, float]
    y_true_raw: np.ndarray
    y_pred_raw: np.ndarray


@dataclass
class FineTuneSubset:
    shot_size: int
    train_graphs: List[Any]
    val_graphs: List[Any]


@dataclass
class TrainOutcome:
    best_epoch: int
    best_val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    efficiency_metrics: Dict[str, float]
    y_true_raw: np.ndarray
    y_pred_raw: np.ndarray
    best_state: Dict[str, torch.Tensor]


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    import csv

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_metric_values(values: Sequence[float]) -> Dict[str, float]:
    values = [float(v) for v in values]
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": float(mean(values)), "std": float(pstdev(values))}


def canonicalize_for_hash(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: canonicalize_for_hash(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, (list, tuple)):
        return [canonicalize_for_hash(v) for v in obj]
    return obj


def stable_hash_dict(obj: Dict[str, Any]) -> str:
    payload = json.dumps(canonicalize_for_hash(obj), sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


def format_metrics_for_row(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{k}": float(v) for k, v in metrics.items()}


def validate_args(args: argparse.Namespace) -> None:
    if args.gpu < 0:
        raise SystemExit("--gpu must be a non-negative integer.")
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for this script, but no GPU is available.")
    if args.gpu >= torch.cuda.device_count():
        raise SystemExit(f"Requested --gpu={args.gpu}, but only {torch.cuda.device_count()} CUDA device(s) are available.")
    if not os.path.isfile(args.test_npz_path):
        raise SystemExit(f"--test_npz_path must point to one .npz file, got: {args.test_npz_path}")
    if args.cap_trainval is not None and args.cap_trainval <= 0:
        args.cap_trainval = None
    if args.n_runs <= 0:
        raise SystemExit("--n_runs must be >= 1.")
    if args.holdout_test_size <= 0:
        raise SystemExit("--holdout_test_size must be positive.")
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be positive.")
    try:
        args.use_aug, args.use_flag = cb.validate_cardbench_aug_args(
            args.model_name,
            args.use_aug,
            args.use_flag,
            args.flag_steps,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))


def parse_shot_sizes(text: str) -> List[int]:
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"few-shot size must be positive, got {value}.")
        values.append(value)
    if not values:
        raise ValueError("--fewshot_sizes must contain at least one positive integer.")
    return sorted(set(values))


def validate_nested_fewshot_subsets(bundle: ShotBundle, shot_sizes: Sequence[int]) -> None:
    """Check the deterministic nested 85/15 target subsets used by all models."""
    previous_size = 0
    for shot_size in sorted(shot_sizes):
        subset = build_fewshot_subset(bundle, shot_size)
        if len(subset.train_graphs) + len(subset.val_graphs) != shot_size:
            raise RuntimeError(f"Invalid few-shot split for shot_size={shot_size}.")
        if previous_size and not all(
            left is right
            for left, right in zip(
                bundle.target_trainval_graphs[:previous_size],
                bundle.target_trainval_graphs[:shot_size][:previous_size],
            )
        ):
            raise RuntimeError("Few-shot subsets are not nested as required.")
        print(
            f"[protocol] shot={shot_size}: train={len(subset.train_graphs)}, "
            f"val={len(subset.val_graphs)}; deterministic prefix of seed-specific target train/val pool."
        )
        previous_size = shot_size


def resolve_workload_files(test_npz_path: str) -> Tuple[str, str, str, List[str]]:
    workload_dir = os.path.dirname(os.path.abspath(test_npz_path))
    all_npz = sorted(glob.glob(os.path.join(workload_dir, "*.npz")))
    if not all_npz:
        raise RuntimeError(f"No .npz files found under workload directory: {workload_dir}")
    target_abs = os.path.abspath(test_npz_path)
    if target_abs not in {os.path.abspath(p) for p in all_npz}:
        raise RuntimeError(f"Target npz is not inside workload directory enumeration: {test_npz_path}")
    target_name, _, workload_type = cb.dataset_identity_from_path(target_abs)
    if workload_type != "binary_join":
        raise ValueError("CardBench transfer is defined only for binary-join workloads.")
    source_npz = [p for p in all_npz if os.path.abspath(p) != target_abs]
    if not source_npz:
        raise RuntimeError("Need at least one source dataset for zero-shot pretraining.")
    return workload_dir, target_abs, workload_type, source_npz


def collect_global_edge_type_map(raw_datasets: Sequence[Any]) -> Dict[str, int]:
    edge_names = sorted({edge_name for ds in raw_datasets for edge_name in ds.schema.keys()})
    return {edge_name: idx for idx, edge_name in enumerate(edge_names)}


def remap_edge_types(data: Any, schema: Dict[str, Tuple[str, str]], global_edge_type_to_id: Dict[str, int]) -> Any:
    if getattr(data, "edge_type", None) is None or data.edge_type.numel() == 0:
        return data
    local_edge_names = sorted(schema.keys())
    local_to_global = torch.tensor(
        [global_edge_type_to_id[name] for name in local_edge_names],
        dtype=torch.long,
        device=data.edge_type.device,
    )
    data.edge_type = local_to_global[data.edge_type]
    return data


def transform_graph_with_global_edge_map(processor: Any, graph: Any, schema: Dict[str, Tuple[str, str]], global_edge_type_to_id: Dict[str, int]) -> Optional[Any]:
    data = processor.transform_graph(graph, schema)
    if data is None:
        return None
    return remap_edge_types(data, schema, global_edge_type_to_id)


def make_bundle_cache_path(args: argparse.Namespace, workload_type: str, target_dataset_name: str, seed: int) -> str:
    cache_dir = os.path.join(args.out_dir, "cache_shots", workload_type)
    os.makedirs(cache_dir, exist_ok=True)
    cap_token = "all" if args.cap_trainval is None else str(args.cap_trainval)
    filename = f"bundle_{target_dataset_name}_seed{seed}_holdout{args.holdout_test_size}_cap{cap_token}.pkl"
    return os.path.join(cache_dir, filename)


def get_or_create_shot_bundle(args: argparse.Namespace, seed: int) -> ShotBundle:
    workload_dir, target_npz_path, workload_type, source_npz_paths = resolve_workload_files(args.test_npz_path)
    target_dataset_name, _, _ = cb.dataset_identity_from_path(target_npz_path)
    cache_path = make_bundle_cache_path(args, workload_type, target_dataset_name, seed)
    if os.path.exists(cache_path):
        print(f"[cache] Loading processed shot bundle: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    all_npz_paths = source_npz_paths + [target_npz_path]
    raw_datasets = [cb.load_raw_graph_dataset(p, max_graphs=args.max_graphs) for p in sorted(all_npz_paths)]
    raw_by_path = {os.path.abspath(ds.npz_path): ds for ds in raw_datasets}
    global_edge_type_to_id = collect_global_edge_type_map(raw_datasets)
    protected_edge_type_ids = tuple(
        global_edge_type_to_id[name]
        for name in sorted(global_edge_type_to_id.keys())
        if name in {"op_to_op", "pred_to_op", "attr_to_op"}
    )

    source_train_raw: List[Any] = []
    source_val_raw: List[Any] = []
    target_test_raw: List[Any] = []
    target_trainval_raw: List[Any] = []

    print("\n[shot] Source datasets for pretraining:")
    for src in sorted(source_npz_paths):
        print(f"  - {os.path.basename(src)}")
    print(f"[shot] Target dataset for downstream evaluation: {os.path.basename(target_npz_path)}")

    for npz_path in sorted(all_npz_paths):
        raw_dataset = raw_by_path[os.path.abspath(npz_path)]
        usable_graphs = [g for g in raw_dataset.graphs if cb.is_usable_query_graph(g)]
        split = cb.official_instance_split(
            usable_graphs,
            seed=seed,
            holdout_test_size=args.holdout_test_size,
            cap_trainval=args.cap_trainval,
        )
        if os.path.abspath(npz_path) == os.path.abspath(target_npz_path):
            target_test_raw = list(split.test)
            target_trainval_raw = list(split.train) + list(split.val)
        else:
            source_train_raw.extend(split.train)
            source_val_raw.extend(split.val)

    processor = cb.CardBenchFeatureProcessor(debug=args.debug)
    processor.fit(source_train_raw)

    source_train_graphs: List[Any] = []
    source_val_graphs: List[Any] = []
    target_test_graphs: List[Any] = []
    target_trainval_graphs: List[Any] = []

    for npz_path in sorted(all_npz_paths):
        raw_dataset = raw_by_path[os.path.abspath(npz_path)]
        usable_graphs = [g for g in raw_dataset.graphs if cb.is_usable_query_graph(g)]
        split = cb.official_instance_split(
            usable_graphs,
            seed=seed,
            holdout_test_size=args.holdout_test_size,
            cap_trainval=args.cap_trainval,
        )
        if os.path.abspath(npz_path) == os.path.abspath(target_npz_path):
            for g in split.test:
                data = transform_graph_with_global_edge_map(processor, g, raw_dataset.schema, global_edge_type_to_id)
                if data is not None:
                    target_test_graphs.append(data)
            for g in list(split.train) + list(split.val):
                data = transform_graph_with_global_edge_map(processor, g, raw_dataset.schema, global_edge_type_to_id)
                if data is not None:
                    target_trainval_graphs.append(data)
        else:
            for g in split.train:
                data = transform_graph_with_global_edge_map(processor, g, raw_dataset.schema, global_edge_type_to_id)
                if data is not None:
                    source_train_graphs.append(data)
            for g in split.val:
                data = transform_graph_with_global_edge_map(processor, g, raw_dataset.schema, global_edge_type_to_id)
                if data is not None:
                    source_val_graphs.append(data)

    if not source_train_graphs or not source_val_graphs or not target_test_graphs or not target_trainval_graphs:
        raise RuntimeError("Shot bundle is empty after transformation.")

    bundle = ShotBundle(
        workload_type=workload_type,
        workload_dir=workload_dir,
        target_npz_path=target_npz_path,
        target_dataset_name=target_dataset_name,
        source_npz_paths=sorted(source_npz_paths),
        seed=seed,
        holdout_test_size=args.holdout_test_size,
        cap_trainval=args.cap_trainval,
        feature_processor=processor,
        num_edge_types=len(global_edge_type_to_id),
        protected_edge_type_ids=protected_edge_type_ids,
        source_train_graphs=source_train_graphs,
        source_val_graphs=source_val_graphs,
        target_test_graphs=target_test_graphs,
        target_trainval_graphs=target_trainval_graphs,
        global_edge_type_to_id=global_edge_type_to_id,
    )
    with open(cache_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[cache] Saved processed shot bundle: {cache_path}")
    return bundle


def build_prepared_split_from_graphs(bundle: ShotBundle, train_graphs: Sequence[Any], val_graphs: Sequence[Any], test_graphs: Sequence[Any]):
    return cb.PreparedSplit(
        raw_split=cb.SplitRawGraphs(train=[], val=[], test=[]),
        feature_processor=bundle.feature_processor,
        train_graphs=list(train_graphs),
        val_graphs=list(val_graphs),
        test_graphs=list(test_graphs),
        pna_deg=cb.compute_pna_degree_histogram(list(train_graphs)),
        num_edge_types=bundle.num_edge_types,
        protected_edge_type_ids=bundle.protected_edge_type_ids,
    )


def build_zero_shot_prepared(bundle: ShotBundle):
    return build_prepared_split_from_graphs(bundle, bundle.source_train_graphs, bundle.source_val_graphs, bundle.target_test_graphs)


def build_fewshot_subset(bundle: ShotBundle, shot_size: int) -> FineTuneSubset:
    if shot_size < 2:
        raise ValueError("shot_size must be >= 2 so the downstream split can contain both train and validation graphs.")
    if shot_size > len(bundle.target_trainval_graphs):
        raise ValueError(f"Requested shot_size={shot_size}, but only {len(bundle.target_trainval_graphs)} target trainval graphs are available.")
    subset = list(bundle.target_trainval_graphs[:shot_size])
    n_train = max(1, int(round(shot_size * 0.85)))
    n_train = min(n_train, shot_size - 1)
    train_graphs = subset[:n_train]
    val_graphs = subset[n_train:]
    return FineTuneSubset(shot_size=shot_size, train_graphs=train_graphs, val_graphs=val_graphs)


def train_with_init_state(
    prepared: Any,
    model_name: str,
    hparams: Dict[str, Any],
    train_config: Any,
    device: torch.device,
    verbose: bool,
    seed: int,
    init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> TrainOutcome:
    cb.seed_everything(seed)
    if (train_config.use_aug or train_config.use_flag) and not cb.supports_cardbench_root_augmentation(model_name):
        raise ValueError(
            "CardBench augmentation is currently supported only for gin_root/gine_root root-readout models."
        )
    if not prepared.train_graphs:
        raise ValueError("Training split is empty.")
    if not prepared.val_graphs:
        raise ValueError("Validation split is empty.")
    if not prepared.test_graphs:
        raise ValueError("Test split is empty.")

    model = cb.build_model(
        model_name,
        prepared.train_graphs[0].x.size(1),
        hparams,
        prepared.pna_deg,
        prepared.num_edge_types,
    ).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(hparams.get("lr", train_config.lr)),
        weight_decay=float(hparams.get("weight_decay", train_config.weight_decay)),
    )

    batch_size = int(hparams.get("batch_size", train_config.batch_size))
    cb.assert_no_singleton_training_last_batch(
        dataset_name="shot_exp",
        split_name="train",
        num_graphs=len(prepared.train_graphs),
        batch_sizes=(batch_size,),
    )
    train_loader = cb.make_dataloader(prepared.train_graphs, batch_size, shuffle=True, seed=seed)
    val_loader = cb.make_dataloader(prepared.val_graphs, batch_size, shuffle=False, seed=seed)
    test_loader = cb.make_dataloader(prepared.test_graphs, batch_size, shuffle=False, seed=seed)

    optimizer, pre_time, pre_alloc, pre_reserved, pre_epochs = cb.maybe_run_mo_backbone_pretrain(
        model=model,
        hparams=hparams,
        train_loader=train_loader,
        val_loader=val_loader,
        feature_processor=prepared.feature_processor,
        train_config=train_config,
        protected_edge_type_ids=prepared.protected_edge_type_ids,
        device=device,
        optimizer=optimizer,
        verbose=verbose,
        # Zero-shot source training and scratch get MO warm-up.  Fine-tuning
        # inherits a fully trained structure learner from the source model.
        enabled=init_state_dict is None,
    )

    best_val_score = float("inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_metrics: Optional[Dict[str, float]] = None
    bad_epochs = 0
    total_train_time = pre_time
    max_train_alloc_mb = pre_alloc
    max_train_reserved_mb = pre_reserved
    num_epochs_trained = pre_epochs

    for epoch in range(1, train_config.max_epochs + 1):
        train_loss, train_epoch_time, train_alloc_mb, train_reserved_mb = cb.train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train_config,
            prepared.protected_edge_type_ids,
        )
        total_train_time += train_epoch_time
        max_train_alloc_mb = max(max_train_alloc_mb, train_alloc_mb)
        max_train_reserved_mb = max(max_train_reserved_mb, train_reserved_mb)
        num_epochs_trained += 1

        val_eval = cb.evaluate(model, val_loader, device, prepared.feature_processor)
        val_metrics = val_eval.metrics
        val_score = float(val_metrics["median_q_error"])
        if val_score < best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = dict(val_metrics)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose and (epoch == 1 or epoch % train_config.verbose_every == 0):
            print(
                f"Epoch {epoch:03d} | train_norm_mse={train_loss:.6f} | "
                f"val_median_q={val_metrics['median_q_error']:.6f} | "
                f"val_p95_q={val_metrics['p95_q_error']:.6f} | "
                f"val_norm_mse={val_metrics['norm_mse']:.6f} | "
                f"t_train_epoch={train_epoch_time:.4f}s | "
                f"train_mem_alloc={train_alloc_mb:.2f}MB | train_mem_reserved={train_reserved_mb:.2f}MB"
            )
        if bad_epochs >= train_config.early_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None or best_val_metrics is None:
        raise RuntimeError("Training failed to produce a valid checkpoint.")

    model.load_state_dict(best_state)
    test_eval = cb.evaluate(model, test_loader, device, prepared.feature_processor)
    efficiency_metrics = {
        "num_epochs": float(num_epochs_trained),
        "t_train_total": float(total_train_time),
        "t_train_epoch_avg": float(total_train_time / max(num_epochs_trained, 1)),
        "train_max_memory_allocated_mb": float(max_train_alloc_mb),
        "train_max_memory_reserved_mb": float(max_train_reserved_mb),
        "t_infer": float(test_eval.infer_time_seconds),
        "infer_max_memory_allocated_mb": float(test_eval.infer_max_memory_allocated_mb),
        "infer_max_memory_reserved_mb": float(test_eval.infer_max_memory_reserved_mb),
    }

    del model, optimizer, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()

    return TrainOutcome(
        best_epoch=best_epoch,
        best_val_metrics=best_val_metrics,
        test_metrics=test_eval.metrics,
        efficiency_metrics=efficiency_metrics,
        y_true_raw=test_eval.y_true_raw,
        y_pred_raw=test_eval.y_pred_raw,
        best_state=best_state,
    )


def make_pretrain_result_dir(args: argparse.Namespace, bundle: ShotBundle) -> str:
    result_dir = os.path.join(args.out_dir, bundle.workload_type, bundle.target_dataset_name, args.model_name)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def make_pretrain_checkpoint_path(result_dir: str, model_name: str, target_name: str, seed: int, max_epochs: int, early_stop: int, hparams: Dict[str, Any]) -> str:
    hp_hash = stable_hash_dict(hparams)
    filename = f"zeroshotpretrain_{model_name}_exclude-{target_name}_seed{seed}_ep{max_epochs}_es{early_stop}_hp{hp_hash}.pt"
    return os.path.join(result_dir, filename)


def make_downstream_checkpoint_path(result_dir: str, tag: str, model_name: str, target_name: str, seed: int, shot_size: int, max_epochs: int, early_stop: int, hparams: Dict[str, Any]) -> str:
    hp_hash = stable_hash_dict(hparams)
    filename = f"{tag}_{model_name}_target-{target_name}_seed{seed}_shot{shot_size}_ep{max_epochs}_es{early_stop}_hp{hp_hash}.pt"
    return os.path.join(result_dir, filename)


def save_training_checkpoint(path: str, outcome: TrainOutcome, hparams: Dict[str, Any], meta: Dict[str, Any]) -> None:
    payload = {
        "meta": meta,
        "hparams": hparams,
        "best_epoch": outcome.best_epoch,
        "best_val_metrics": outcome.best_val_metrics,
        "test_metrics": outcome.test_metrics,
        "efficiency_metrics": outcome.efficiency_metrics,
        "y_true_raw": outcome.y_true_raw,
        "y_pred_raw": outcome.y_pred_raw,
        "best_state": outcome.best_state,
    }
    torch.save(payload, path)


def load_training_checkpoint(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def build_train_config_from_args(args: argparse.Namespace, hparams: Dict[str, Any], mode: str) -> Any:
    if mode == "pretrain":
        max_epochs = args.zs_max_epochs
        early_stop = args.zs_early_stop
    else:
        max_epochs = args.max_epochs
        early_stop = args.early_stop
    return cb.TrainConfig(
        max_epochs=max_epochs,
        early_stop=early_stop,
        batch_size=int(hparams.get("batch_size", args.batch_size)),
        lr=float(hparams.get("lr", args.lr)),
        weight_decay=float(hparams.get("weight_decay", args.weight_decay)),
        verbose_every=args.verbose_every,
        use_aug=args.use_aug,
        aug_edge_drop_p1=args.aug_edge_drop_p1,
        aug_feat_mask_p1=args.aug_feat_mask_p1,
        aug_edge_drop_p2=args.aug_edge_drop_p2,
        aug_feat_mask_p2=args.aug_feat_mask_p2,
        aug_prob_cap=args.aug_prob_cap,
        contrast_weight=args.contrast_weight,
        contrast_temperature=args.contrast_temperature,
        use_flag=args.use_flag,
        flag_steps=args.flag_steps,
        flag_step_size=args.flag_step_size,
        flag_ops_scale=args.flag_ops_scale,
        flag_root_scale=args.flag_root_scale,
    )


def maybe_load_or_tune_hparams(args: argparse.Namespace, zero_shot_prepared: Any, device: torch.device, result_dir: str) -> Dict[str, Any]:
    hparams_path = os.path.join(result_dir, f"best_hparams_seed{args.first_seed}.json")
    if os.path.exists(hparams_path):
        print(f"[cache] Loading best hyperparameters: {hparams_path}")
        with open(hparams_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if args.n_trials > 0:
        print(f"\nTuning hyperparameters on zero-shot pretraining task (n_trials={args.n_trials}) ...")
        best_hparams = cb.tune_hparams(zero_shot_prepared, args.model_name, args, device)
    else:
        best_hparams = cb.default_hparams_from_args(args, args.model_name)
    write_json(hparams_path, best_hparams)
    return best_hparams


def run_zero_shot_pretrain(args: argparse.Namespace, bundle: ShotBundle, hparams: Dict[str, Any], device: torch.device, result_dir: str) -> Dict[str, Any]:
    zero_shot_prepared = build_zero_shot_prepared(bundle)
    config = build_train_config_from_args(args, hparams, mode="pretrain")
    ckpt_path = make_pretrain_checkpoint_path(
        result_dir,
        args.model_name,
        bundle.target_dataset_name,
        bundle.seed,
        config.max_epochs,
        config.early_stop,
        hparams,
    )
    if os.path.exists(ckpt_path):
        print(f"[cache] Loading zero-shot pretraining checkpoint: {ckpt_path}")
        return load_training_checkpoint(ckpt_path)

    print("\n[zero-shot pretraining] building checkpoint from source datasets ...")
    print(f"  source_train_graphs: {len(bundle.source_train_graphs)}")
    print(f"  source_val_graphs:   {len(bundle.source_val_graphs)}")
    print(f"  target_test_graphs:  {len(bundle.target_test_graphs)}")
    outcome = train_with_init_state(
        prepared=zero_shot_prepared,
        model_name=args.model_name,
        hparams=hparams,
        train_config=config,
        device=device,
        verbose=True,
        seed=bundle.seed,
        init_state_dict=None,
    )
    payload_meta = {
        "mode": "zero_shot_pretrain",
        "target_npz_path": bundle.target_npz_path,
        "exclude_npz_name": bundle.target_dataset_name,
        "source_npz_paths": bundle.source_npz_paths,
        "seed": bundle.seed,
        "max_epochs": config.max_epochs,
        "early_stop": config.early_stop,
    }
    save_training_checkpoint(ckpt_path, outcome, hparams, payload_meta)
    return load_training_checkpoint(ckpt_path)


def run_downstream_stage(
    args: argparse.Namespace,
    bundle: ShotBundle,
    hparams: Dict[str, Any],
    device: torch.device,
    result_dir: str,
    shot_size: int,
    tag: str,
    init_state_dict: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, Any]:
    subset = build_fewshot_subset(bundle, shot_size)
    prepared = build_prepared_split_from_graphs(bundle, subset.train_graphs, subset.val_graphs, bundle.target_test_graphs)
    config = build_train_config_from_args(args, hparams, mode="downstream")
    ckpt_path = make_downstream_checkpoint_path(
        result_dir,
        tag,
        args.model_name,
        bundle.target_dataset_name,
        bundle.seed,
        shot_size,
        config.max_epochs,
        config.early_stop,
        hparams,
    )
    if os.path.exists(ckpt_path):
        print(f"[cache] Loading downstream checkpoint: {ckpt_path}")
        return load_training_checkpoint(ckpt_path)

    print(f"\n[{tag}] shot_size={shot_size} | train={len(subset.train_graphs)} | val={len(subset.val_graphs)} | test={len(bundle.target_test_graphs)}")
    outcome = train_with_init_state(
        prepared=prepared,
        model_name=args.model_name,
        hparams=hparams,
        train_config=config,
        device=device,
        verbose=True,
        seed=bundle.seed,
        init_state_dict=init_state_dict,
    )
    payload_meta = {
        "mode": tag,
        "target_npz_path": bundle.target_npz_path,
        "target_dataset_name": bundle.target_dataset_name,
        "seed": bundle.seed,
        "shot_size": shot_size,
        "fewshot_train": len(subset.train_graphs),
        "fewshot_val": len(subset.val_graphs),
        "max_epochs": config.max_epochs,
        "early_stop": config.early_stop,
    }
    save_training_checkpoint(ckpt_path, outcome, hparams, payload_meta)
    return load_training_checkpoint(ckpt_path)


def aggregate_shot_runs(results: Sequence[ShotRunResult]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[ShotRunResult]] = {}
    for result in results:
        key = (result.setting, result.shot_size)
        grouped.setdefault(key, []).append(result)
    rows: List[Dict[str, Any]] = []
    for (setting, shot_size), runs in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        row: Dict[str, Any] = {
            "target_dataset_name": runs[0].target_dataset_name,
            "target_npz_path": runs[0].target_npz_path,
            "workload_type": runs[0].workload_type,
            "model_name": runs[0].model_name,
            "setting": setting,
            "shot_size": shot_size,
            "n_runs": len(runs),
        }
        test_metric_names = list(runs[0].test_metrics.keys())
        eff_metric_names = list(runs[0].efficiency_metrics.keys())
        for metric_name in test_metric_names:
            stats = summarize_metric_values([r.test_metrics[metric_name] for r in runs])
            row[f"test_{metric_name}_mean"] = stats["mean"]
            row[f"test_{metric_name}_std"] = stats["std"]
        for metric_name in eff_metric_names:
            stats = summarize_metric_values([r.efficiency_metrics[metric_name] for r in runs])
            row[f"eff_{metric_name}_mean"] = stats["mean"]
            row[f"eff_{metric_name}_std"] = stats["std"]
        pooled = cb.compute_metrics(
            y_true_raw=np.concatenate([r.y_true_raw for r in runs], axis=0),
            y_pred_raw=np.concatenate([r.y_pred_raw for r in runs], axis=0),
            norm_mse=float(np.mean([r.test_metrics["norm_mse"] for r in runs])),
            elapsed_seconds=float(np.sum([r.efficiency_metrics["t_infer"] for r in runs])),
        )
        for metric_name, value in pooled.items():
            row[f"pooled_{metric_name}"] = float(value)
        rows.append(row)
    return rows


def shot_result_to_row(result: ShotRunResult) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "target_dataset_name": result.target_dataset_name,
        "target_npz_path": result.target_npz_path,
        "workload_type": result.workload_type,
        "model_name": result.model_name,
        "seed": result.seed,
        "setting": result.setting,
        "shot_size": result.shot_size,
        "best_epoch": result.best_epoch,
    }
    row.update(format_metrics_for_row("val", result.best_val_metrics))
    row.update(format_metrics_for_row("test", result.test_metrics))
    row.update(format_metrics_for_row("eff", result.efficiency_metrics))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CardBench transfer benchmark for GINE/PNA/GMT/GraphGPS/AK/MO/GCA.")
    parser.add_argument("--test_npz_path", type=str, required=True, help="Target dataset .npz file. The other .npz files in the same folder are used for zero-shot pretraining.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gine_root",
        choices=[
            "gin_root_gca",
            "gine_root",
            "gine_root_gca",
            "pna_root",
            "gine_gmt",
            "graphgps_vn",
            "gnnak_root",
            "mo",
        ],
    )
    parser.add_argument("--max_graphs", type=int, default=None)
    parser.add_argument(
        "--expected_num_datasets",
        type=int,
        default=20,
        help="Fail if the workload directory does not contain exactly this many .npz datasets; use 0 to disable.",
    )
    parser.add_argument("--holdout_test_size", type=int, default=500)
    parser.add_argument("--cap_trainval", type=int, default=4500, help="Use -1 to disable capping and keep all remaining train/val graphs.")
    parser.add_argument("--fewshot_sizes", type=str, default="250,500,1000")
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--first_seed", type=int, default=2024)
    parser.add_argument("--n_trials", type=int, default=200, help="Optuna trials on the source-validation task; set to 0 for explicit CLI parameters.")
    parser.add_argument("--zs_max_epochs", type=int, default=20)
    parser.add_argument("--zs_early_stop", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--hpo_max_epochs", type=int, default=20)
    parser.add_argument("--hpo_early_stop", type=int, default=5)
    parser.add_argument("--verbose_every", type=int, default=5)

    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--pna_towers", type=int, default=1)
    parser.add_argument("--gmt_heads", type=int, default=4)
    parser.add_argument("--gmt_k", type=int, default=8)
    parser.add_argument("--gps_heads", type=int, default=4)
    parser.add_argument("--mo_gsl_epsilon", type=float, default=0.85)
    parser.add_argument("--mo_gsl_lambda", type=float, default=0.35)
    parser.add_argument("--mo_metric_heads", type=int, default=8)
    parser.add_argument("--mo_pretrain_epochs", type=int, default=10)
    parser.add_argument("--hgp_pool_ratio", type=float, default=0.8)
    parser.add_argument("--hgp_topk", type=int, default=4)
    parser.add_argument("--hgp_lambda", type=float, default=1.0)
    parser.add_argument("--hgp_score_bias_ops", type=float, default=0.5)
    parser.add_argument("--hgp_score_bias_predicates", type=float, default=0.25)
    parser.add_argument("--gnnak_hops", type=int, default=3)
    parser.add_argument("--gnnak_inner_layers", type=int, default=1)
    parser.add_argument("--gnnak_hops_dim", type=int, default=16)
    parser.add_argument("--gnnak_embs", type=str, default="0,1,2")
    parser.add_argument("--gnnak_embs_combine_mode", type=str, default="concat", choices=["add", "concat"])

    parser.add_argument("--use_aug", action="store_true")
    parser.add_argument("--aug_edge_drop_p1", type=float, default=0.10)
    parser.add_argument("--aug_feat_mask_p1", type=float, default=0.05)
    parser.add_argument("--aug_edge_drop_p2", type=float, default=0.05)
    parser.add_argument("--aug_feat_mask_p2", type=float, default=0.10)
    parser.add_argument("--aug_prob_cap", type=float, default=0.35)
    parser.add_argument("--contrast_weight", type=float, default=0.10)
    parser.add_argument("--contrast_temperature", type=float, default=0.50)
    parser.add_argument("--use_flag", action="store_true")
    parser.add_argument("--flag_steps", type=int, default=3)
    parser.add_argument("--flag_step_size", type=float, default=0.002)
    parser.add_argument("--flag_ops_scale", type=float, default=0.25)
    parser.add_argument("--flag_root_scale", type=float, default=0.0)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--out_dir", type=str, default="cardbench_shot_results")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    shot_sizes = parse_shot_sizes(args.fewshot_sizes)

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    os.makedirs(args.out_dir, exist_ok=True)

    workload_dir, target_npz_path, workload_type, source_npz_paths = resolve_workload_files(args.test_npz_path)
    actual_num_datasets = len(source_npz_paths) + 1
    if args.expected_num_datasets > 0 and actual_num_datasets != args.expected_num_datasets:
        raise SystemExit(
            f"Expected {args.expected_num_datasets} CardBench datasets in {workload_dir}, "
            f"but found {actual_num_datasets}. Refusing to change the leave-one-dataset-out protocol."
        )
    target_dataset_name, _, _ = cb.dataset_identity_from_path(target_npz_path)

    print("=" * 88)
    print(f"Shot experiment target: {target_npz_path}")
    print(f"Workload directory: {workload_dir}")
    print(f"Workload type: {workload_type}")
    print(f"Number of source datasets: {len(source_npz_paths)}")
    print(f"Few-shot sizes: {shot_sizes}")
    print(f"Running on device: {device}")

    result_dir = make_pretrain_result_dir(args, ShotBundle(
        workload_type=workload_type,
        workload_dir=workload_dir,
        target_npz_path=target_npz_path,
        target_dataset_name=target_dataset_name,
        source_npz_paths=source_npz_paths,
        seed=args.first_seed,
        holdout_test_size=args.holdout_test_size,
        cap_trainval=args.cap_trainval,
        feature_processor=None,
        num_edge_types=0,
        protected_edge_type_ids=tuple(),
        source_train_graphs=[],
        source_val_graphs=[],
        target_test_graphs=[],
        target_trainval_graphs=[],
        global_edge_type_to_id={},
    ))

    all_results: List[ShotRunResult] = []
    seeds = [args.first_seed + i for i in range(args.n_runs)]
    canonical_seed = seeds[0]

    bundle_canonical = get_or_create_shot_bundle(args, canonical_seed)
    validate_nested_fewshot_subsets(bundle_canonical, shot_sizes)
    zero_shot_prepared_canonical = build_zero_shot_prepared(bundle_canonical)
    best_hparams = maybe_load_or_tune_hparams(args, zero_shot_prepared_canonical, device, result_dir)

    print("\nBest / selected hyperparameters")
    for key, value in best_hparams.items():
        print(f"  {key}: {value}")

    for run_idx, seed in enumerate(seeds, start=1):
        print("\n" + "-" * 88)
        print(f"Run {run_idx}/{len(seeds)} | seed={seed}")
        bundle = bundle_canonical if seed == canonical_seed else get_or_create_shot_bundle(args, seed)

        zero_payload = run_zero_shot_pretrain(args, bundle, best_hparams, device, result_dir)
        zero_result = ShotRunResult(
            target_dataset_name=bundle.target_dataset_name,
            target_npz_path=bundle.target_npz_path,
            workload_type=bundle.workload_type,
            model_name=args.model_name,
            seed=seed,
            setting="zero_shot",
            shot_size=0,
            best_epoch=int(zero_payload["best_epoch"]),
            hparams=dict(best_hparams),
            best_val_metrics=dict(zero_payload["best_val_metrics"]),
            test_metrics=dict(zero_payload["test_metrics"]),
            efficiency_metrics=dict(zero_payload["efficiency_metrics"]),
            y_true_raw=np.asarray(zero_payload["y_true_raw"]),
            y_pred_raw=np.asarray(zero_payload["y_pred_raw"]),
        )
        all_results.append(zero_result)
        cb.print_metrics("\n[zero-shot] best validation metrics", zero_result.best_val_metrics)
        cb.print_metrics("\n[zero-shot] test metrics", zero_result.test_metrics)

        init_state = zero_payload["best_state"]
        for shot_size in shot_sizes:
            ft_payload = run_downstream_stage(
                args,
                bundle,
                best_hparams,
                device,
                result_dir,
                shot_size=shot_size,
                tag="finetune",
                init_state_dict=init_state,
            )
            ft_result = ShotRunResult(
                target_dataset_name=bundle.target_dataset_name,
                target_npz_path=bundle.target_npz_path,
                workload_type=bundle.workload_type,
                model_name=args.model_name,
                seed=seed,
                setting="finetune",
                shot_size=shot_size,
                best_epoch=int(ft_payload["best_epoch"]),
                hparams=dict(best_hparams),
                best_val_metrics=dict(ft_payload["best_val_metrics"]),
                test_metrics=dict(ft_payload["test_metrics"]),
                efficiency_metrics=dict(ft_payload["efficiency_metrics"]),
                y_true_raw=np.asarray(ft_payload["y_true_raw"]),
                y_pred_raw=np.asarray(ft_payload["y_pred_raw"]),
            )
            all_results.append(ft_result)
            cb.print_metrics(f"\n[finetune-{shot_size}] best validation metrics", ft_result.best_val_metrics)
            cb.print_metrics(f"\n[finetune-{shot_size}] test metrics", ft_result.test_metrics)

            scratch_payload = run_downstream_stage(
                args,
                bundle,
                best_hparams,
                device,
                result_dir,
                shot_size=shot_size,
                tag="scratch",
                init_state_dict=None,
            )
            scratch_result = ShotRunResult(
                target_dataset_name=bundle.target_dataset_name,
                target_npz_path=bundle.target_npz_path,
                workload_type=bundle.workload_type,
                model_name=args.model_name,
                seed=seed,
                setting="scratch",
                shot_size=shot_size,
                best_epoch=int(scratch_payload["best_epoch"]),
                hparams=dict(best_hparams),
                best_val_metrics=dict(scratch_payload["best_val_metrics"]),
                test_metrics=dict(scratch_payload["test_metrics"]),
                efficiency_metrics=dict(scratch_payload["efficiency_metrics"]),
                y_true_raw=np.asarray(scratch_payload["y_true_raw"]),
                y_pred_raw=np.asarray(scratch_payload["y_pred_raw"]),
            )
            all_results.append(scratch_result)
            cb.print_metrics(f"\n[scratch-{shot_size}] best validation metrics", scratch_result.best_val_metrics)
            cb.print_metrics(f"\n[scratch-{shot_size}] test metrics", scratch_result.test_metrics)

    run_rows = [shot_result_to_row(r) for r in all_results]
    summary_rows = aggregate_shot_runs(all_results)
    write_csv(os.path.join(result_dir, "shot_run_results.csv"), run_rows)
    write_csv(os.path.join(result_dir, "shot_summary.csv"), summary_rows)
    write_json(os.path.join(result_dir, "shot_summary.json"), summary_rows)

    print("\n" + "=" * 88)
    print("Saved artifacts:")
    print(f"  {os.path.join(result_dir, 'shot_run_results.csv')}")
    print(f"  {os.path.join(result_dir, 'shot_summary.csv')}")
    print(f"  {os.path.join(result_dir, 'shot_summary.json')}")


if __name__ == "__main__":
    main()
