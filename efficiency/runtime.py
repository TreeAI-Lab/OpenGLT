from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from torch_geometric.data import Batch


GraphTransform = Callable[[Any], Any]
Collate = Callable[[Any], Any]
Forward = Callable[[torch.nn.Module, Any], Any]


_DATASET_METADATA_FIELDS = (
    "output_dim", "task", "key_metric", "dim_node_deg", "dim_node_label",
    "dim_node_attr", "num_edge_classes", "num_tasks", "good_name",
)


def dataset_metadata(dataset: Any) -> dict[str, Any]:
    """Extract runner metadata without persisting a stale temporary root."""
    return {
        name: getattr(dataset, name)
        for name in _DATASET_METADATA_FIELDS
        if hasattr(dataset, name)
    }


def restore_dataset_metadata(metadata: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**metadata)


def prepared_input_path(cfg) -> Path:
    root = str(getattr(cfg.efficiency, "cache_root", ""))
    if not root:
        raise RuntimeError("Efficiency prepared input requires cfg.efficiency.cache_root")
    key = str(getattr(cfg.efficiency, "prepared_cache_key", "default"))
    if not key or any(token in key for token in ("/", "\\", "..")):
        raise ValueError(f"Unsafe prepared_cache_key={key!r}")
    return Path(root).resolve() / "prepared_inputs" / f"{key}.pt"


def save_prepared_input(cfg, payload: dict[str, Any]) -> Path:
    """Atomically serialize the final stage-one training input, untimed."""
    if not bool(getattr(cfg.efficiency, "require_prepared_input", False)):
        raise RuntimeError("This runner is not configured to persist a prepared input")
    path = prepared_input_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = {
        "schema_version": 1,
        "entry_id": str(cfg.efficiency.entry_id),
        "prepared_cache_key": str(cfg.efficiency.prepared_cache_key),
        "payload": payload,
    }
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(wrapped, temporary)
    os.replace(temporary, path)
    print(f"[efficiency prepared input] saved untimed artifact: {path}", flush=True)
    return path


def load_prepared_input(cfg) -> dict[str, Any] | None:
    """Load the finalized repeat-2 artifact during strict GPU measurement."""
    if (
        not efficiency_enabled(cfg, "strict")
        or not bool(getattr(cfg.efficiency, "require_prepared_input", False))
    ):
        return None
    path = prepared_input_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing finalized stage-one prepared input: {path}. "
            "Rerun stage one for this setting."
        )
    try:
        wrapped = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        wrapped = torch.load(path, map_location="cpu")
    if (
        wrapped.get("schema_version") != 1
        or wrapped.get("entry_id") != str(cfg.efficiency.entry_id)
        or wrapped.get("prepared_cache_key")
        != str(cfg.efficiency.prepared_cache_key)
    ):
        raise RuntimeError(f"Stale or mismatched prepared input: {path}")
    print(f"[efficiency prepared input] loaded stage-one artifact: {path}", flush=True)
    return wrapped["payload"]


def efficiency_enabled(cfg, phase: str | None = None) -> bool:
    node = getattr(cfg, "efficiency", None)
    if node is None or not bool(getattr(node, "enabled", False)):
        return False
    return phase is None or str(getattr(node, "phase", "none")).lower() == phase


def _sync(device: str | torch.device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device: str | torch.device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak(device: str | torch.device) -> dict[str, float]:
    device = torch.device(device)
    if device.type != "cuda":
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}
    return {
        "allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        "reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20,
    }


def default_collate(data: Any) -> Batch:
    return Batch.from_data_list([data])


def identity_transform(data: Any) -> Any:
    return data.clone() if hasattr(data, "clone") else data


def _unwrap_output(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        return next(iter(output.values()))
    return output


def _summary(per_round: Sequence[Sequence[float]]) -> dict[str, Any]:
    arrays = [np.asarray(values, dtype=np.float64) for values in per_round]
    pooled = np.concatenate(arrays) if arrays else np.asarray([], dtype=np.float64)
    medians = np.asarray([np.median(values) for values in arrays], dtype=np.float64)
    p95s = np.asarray([np.percentile(values, 95) for values in arrays], dtype=np.float64)
    return {
        "n_per_round": int(arrays[0].size) if arrays else 0,
        "rounds": len(arrays),
        "median_ms": float(np.median(pooled)) if pooled.size else None,
        "median_round_std_ms": float(medians.std(ddof=0)) if medians.size else None,
        "p95_ms": float(np.percentile(pooled, 95)) if pooled.size else None,
        "p95_round_std_ms": float(p95s.std(ddof=0)) if p95s.size else None,
        "round_median_ms": medians.tolist(),
        "round_p95_ms": p95s.tolist(),
    }


def summarize_latency(per_round: Sequence[Sequence[float]]) -> dict[str, Any]:
    return _summary(per_round)


def deterministic_indices(length: int, count: int, seed: int) -> list[int]:
    if length <= 0:
        raise ValueError("Cannot benchmark an empty test dataset.")
    count = min(int(count), length)
    rng = random.Random(int(seed))
    indices = list(range(length))
    rng.shuffle(indices)
    return indices[:count]


def benchmark_inference(
    model: torch.nn.Module,
    processed_dataset: Sequence[Any],
    raw_dataset: Sequence[Any] | None,
    device: str | torch.device,
    forward_fn: Forward,
    *,
    transform_fn: GraphTransform | None = None,
    collate_fn: Collate = default_collate,
    sample_count: int = 200,
    sample_seed: int = 2024,
    warmup_count: int = 20,
    repeats: int = 3,
) -> dict[str, Any]:
    """Measure synchronized batch=1 forward and end-to-end latency."""
    if repeats < 1:
        raise ValueError("latency repeats must be positive")
    transform_fn = transform_fn or identity_transform
    raw_dataset = raw_dataset if raw_dataset is not None else processed_dataset
    if len(raw_dataset) != len(processed_dataset):
        raise ValueError(
            "raw and processed test datasets must preserve one-to-one graph membership "
            f"({len(raw_dataset)} != {len(processed_dataset)})"
        )
    indices = deterministic_indices(len(processed_dataset), sample_count, sample_seed)
    device = torch.device(device)
    model.eval()

    warm_indices = indices[: min(warmup_count, len(indices))]
    with torch.inference_mode():
        for idx in warm_indices:
            batch = collate_fn(processed_dataset[idx]).to(device)
            _unwrap_output(forward_fn(model, batch))
        _sync(device)

    forward_rounds: list[list[float]] = []
    _reset_peak(device)
    with torch.inference_mode():
        for _ in range(repeats):
            values: list[float] = []
            for idx in indices:
                # Collation and transfer occur outside Infer-fwd.
                batch = collate_fn(processed_dataset[idx]).to(device)
                _sync(device)
                t0 = time.perf_counter()
                _unwrap_output(forward_fn(model, batch))
                _sync(device)
                values.append((time.perf_counter() - t0) * 1000.0)
            forward_rounds.append(values)
    forward_mem = _peak(device)

    # Warm up the complete path separately; transform results are discarded.
    with torch.inference_mode():
        for idx in warm_indices:
            batch = collate_fn(transform_fn(raw_dataset[idx])).to(device)
            _unwrap_output(forward_fn(model, batch))
        _sync(device)

    e2e_rounds: list[list[float]] = []
    _reset_peak(device)
    with torch.inference_mode():
        for _ in range(repeats):
            values = []
            for idx in indices:
                _sync(device)
                t0 = time.perf_counter()
                transformed = transform_fn(raw_dataset[idx])
                batch = collate_fn(transformed).to(device)
                _unwrap_output(forward_fn(model, batch))
                _sync(device)
                values.append((time.perf_counter() - t0) * 1000.0)
            e2e_rounds.append(values)
    e2e_mem = _peak(device)

    return {
        "sample_indices": indices,
        "sample_seed": int(sample_seed),
        "warmup_count": len(warm_indices),
        "forward": _summary(forward_rounds),
        "end_to_end": _summary(e2e_rounds),
        "batch1_memory": {
            "forward": forward_mem,
            "end_to_end": e2e_mem,
        },
    }


def benchmark_throughput(model: torch.nn.Module, loader: Iterable[Any],
                         device: str | torch.device, forward_fn: Forward) -> dict[str, float]:
    device = torch.device(device)
    model.eval()
    graphs = 0
    _reset_peak(device)
    _sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device)
            _unwrap_output(forward_fn(model, batch))
            if hasattr(batch, "num_graphs"):
                graphs += int(batch.num_graphs)
            elif torch.is_tensor(batch) and batch.dim() > 0:
                graphs += int(batch.size(0))
            else:
                graphs += 1
    _sync(device)
    elapsed = time.perf_counter() - start
    result = {
        "graphs": graphs,
        "seconds": elapsed,
        "graphs_per_second": graphs / elapsed if elapsed > 0 else math.inf,
    }
    result.update({f"memory_{key}": value for key, value in _peak(device).items()})
    return result


def serialized_state_dict_size_mb(model: torch.nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        path = Path(handle.name)
    try:
        torch.save(model.state_dict(), path)
        return path.stat().st_size / 2**20
    finally:
        path.unlink(missing_ok=True)


def cache_miss_guard(cache_paths: Sequence[str | os.PathLike[str]], required: bool) -> None:
    existing = []
    for raw_path in cache_paths:
        path = Path(raw_path)
        if path.is_file() or (path.is_dir() and next(path.iterdir(), None) is not None):
            existing.append(str(path))
    if not existing:
        print("[efficiency cache check] PASS: generated preprocessing cache is absent.", flush=True)
        return
    message = (
        "[efficiency cache check] WARNING: generated cache exists; preprocessing "
        "timing would measure cache loading, not cache-miss transformation:\n  - "
        + "\n  - ".join(existing)
    )
    print(message, flush=True)
    if required:
        raise RuntimeError(
            "Strict preprocessing requested with a populated cache. Run the "
            "provided cache-reset script, which only removes the isolated "
            "efficiency cache root."
        )


def _fold_graphs(folds: Sequence[Any]) -> list[Any]:
    graphs: list[Any] = []
    for fold in folds:
        for split in fold[:3]:
            if split is not None:
                graphs.extend(split[i] for i in range(len(split)))
    return graphs


def finish_preprocessing(
    cfg,
    started_at: float,
    processed_folds: Sequence[Any],
    raw_folds: Sequence[Any] | None = None,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    elapsed = time.perf_counter() - started_at
    processed = _fold_graphs(processed_folds)
    raw = _fold_graphs(raw_folds) if raw_folds is not None else processed
    processed_nodes = sum(int(getattr(graph, "num_nodes", 0) or 0) for graph in processed)
    raw_nodes = sum(int(getattr(graph, "num_nodes", 0) or 0) for graph in raw)
    payload = {
        "schema_version": 1,
        "entry_id": str(cfg.efficiency.entry_id),
        "repeat": int(cfg.efficiency.prep_repeat),
        "environment": environment_snapshot(),
        "prep_total_s": elapsed,
        "graph_count": len(processed),
        "prep_per_graph_ms": elapsed * 1000.0 / max(len(processed), 1),
        "raw_total_nodes": raw_nodes,
        "processed_total_nodes": processed_nodes,
        "node_inflation_factor": processed_nodes / max(raw_nodes, 1),
    }
    if extra:
        payload.update(extra)
    safe_id = str(cfg.efficiency.entry_id).replace("/", "_").replace("::", "__")
    path = (
        Path(str(cfg.efficiency.output_dir)) / "prep" /
        f"{safe_id}__repeat{int(cfg.efficiency.prep_repeat)}.json"
    )
    write_result(path, payload)
    print(f"[efficiency prep] wrote {path}", flush=True)
    return payload


def environment_snapshot() -> dict[str, Any]:
    try:
        import torch_geometric
        pyg_version = torch_geometric.__version__
    except Exception:
        pyg_version = None
    try:
        import lightgbm
        lightgbm_version = lightgbm.__version__
    except Exception:
        lightgbm_version = None
    payload: dict[str, Any] = {
        "pid": os.getpid(),
        "torch": torch.__version__,
        "torch_geometric": pyg_version,
        "lightgbm": lightgbm_version,
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "efficiency_parallel_gpu_isolation": os.environ.get(
            "EFFICIENCY_PARALLEL_GPU_ISOLATION", "0"
        ),
    }
    if hasattr(os, "sched_getaffinity"):
        payload["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        payload.update({
            "gpu_name": props.name,
            "gpu_total_memory_mb": props.total_memory / 2**20,
        })
        try:
            visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
            query = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,clocks.current.graphics,clocks.current.memory,power.draw",
                    "--format=csv,noheader,nounits",
                    "-i", visible_gpu,
                ],
                text=True,
                timeout=5,
            ).strip()
            temperature, graphics_clock, memory_clock, power = [part.strip() for part in query.split(",")]
            payload["gpu_telemetry"] = {
                "temperature_c": float(temperature),
                "graphics_clock_mhz": float(graphics_clock),
                "memory_clock_mhz": float(memory_clock),
                "power_w": float(power),
            }
        except Exception as exc:
            payload["gpu_telemetry_error"] = repr(exc)
    return payload


def write_result(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def benchmark_session_if_enabled(
    cfg,
    model: torch.nn.Module,
    processed_test: Sequence[Any],
    raw_test: Sequence[Any] | None,
    forward_fn: Forward,
    *,
    session_id: int,
    transform_fn: GraphTransform | None = None,
    collate_fn: Collate = default_collate,
    throughput_loader: Iterable[Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not efficiency_enabled(cfg, "strict"):
        return None
    # Training-cycle statistics are collected for every seed/fold, whereas
    # deployment latency/throughput is intentionally benchmarked from the
    # first checkpoint only.  The outer efficiency driver sets this flag.
    if not bool(getattr(cfg.efficiency, "measure_inference", True)):
        return None
    node = cfg.efficiency
    result = {
        "schema_version": 1,
        "entry_id": str(node.entry_id),
        "session_id": int(session_id),
        "environment": environment_snapshot(),
        "model": {
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "state_dict_size_mb": serialized_state_dict_size_mb(model),
        },
        "inference": benchmark_inference(
            model,
            processed_test,
            raw_test,
            cfg.device,
            forward_fn,
            transform_fn=transform_fn,
            collate_fn=collate_fn,
            sample_count=int(node.sample_count),
            sample_seed=int(node.sample_seed),
            warmup_count=int(node.warmup_count),
            repeats=int(node.latency_repeats),
        ),
    }
    if throughput_loader is not None:
        result["throughput"] = benchmark_throughput(
            model, throughput_loader, cfg.device, forward_fn
        )
    result["environment_after"] = environment_snapshot()
    if extra:
        result.update(extra)
    safe_id = str(node.entry_id).replace("/", "_").replace("::", "__")
    path = Path(str(node.output_dir)) / "strict" / f"{safe_id}__session{session_id}.json"
    write_result(path, result)
    print(f"[efficiency strict] wrote {path}", flush=True)
    return result


def save_efficiency_checkpoint_if_enabled(
    cfg, model: torch.nn.Module, *, session_id: int, best_epoch: int,
) -> str | None:
    """Persist the selected short-run state outside every timed region."""
    if not efficiency_enabled(cfg) or not bool(getattr(cfg.efficiency, "save_checkpoint", False)):
        return None
    output_session_id = int(getattr(cfg.efficiency, "output_session_id", session_id))
    safe_id = str(cfg.efficiency.entry_id).replace("/", "_").replace("::", "__")
    path = (
        Path(str(cfg.efficiency.output_dir)) / "checkpoints" /
        str(cfg.efficiency.phase) / f"{safe_id}__session{output_session_id}.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({
        "entry_id": str(cfg.efficiency.entry_id),
        "session_id": output_session_id,
        "seed": int(cfg.seed),
        "best_epoch_within_measured_window": int(best_epoch),
        "state_dict": state,
    }, path)
    print(f"[efficiency checkpoint] wrote {path}", flush=True)
    return str(path)
