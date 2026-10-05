from __future__ import annotations

import importlib.util
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_MODULE_NAME = "cardbench_instance_base_v7_1"
BASE_FILENAME = "cardbench.py"


def load_base_module():
    if BASE_MODULE_NAME in sys.modules:
        return sys.modules[BASE_MODULE_NAME]
    base_path = os.path.join(os.path.dirname(__file__), BASE_FILENAME)
    spec = importlib.util.spec_from_file_location(BASE_MODULE_NAME, base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load CardBench base module: {base_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[BASE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


cb = load_base_module()

TABULAR_MODELS = ("mlp_feat", "mlp_struct", "lgbm")
STRUCTURE_DIM = 24


def normalize_model_name(name: str) -> str:
    aliases = {
        "mlp-feat": "mlp_feat",
        "mlpfeature": "mlp_feat",
        "mlp-struct": "mlp_struct",
        "mlpfeaturestruct": "mlp_struct",
        "lightgbm": "lgbm",
    }
    normalized = aliases.get(str(name).strip().lower(), str(name).strip().lower())
    if normalized not in TABULAR_MODELS:
        raise ValueError(f"Unknown tabular baseline {name!r}; choose from {TABULAR_MODELS}.")
    return normalized


def uses_structure(model_name: str) -> bool:
    return normalize_model_name(model_name) in {"mlp_struct", "lgbm"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stats(matrix: np.ndarray, feature_dim: int) -> np.ndarray:
    """Return mean/sum/std/max/min, matching OpenGLT's GNCE baselines."""
    if matrix.size == 0 or matrix.shape[0] == 0:
        return np.zeros(5 * feature_dim, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32).reshape(-1, feature_dim)
    return np.concatenate(
        [
            matrix.mean(axis=0),
            matrix.sum(axis=0),
            matrix.std(axis=0),
            matrix.max(axis=0),
            matrix.min(axis=0),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _structure_features(data: Any) -> np.ndarray:
    """Generic graph statistics plus CardBench-specific DAG depth summaries."""
    n = int(data.num_nodes)
    edge_index = data.edge_index.detach().cpu().numpy()
    m = int(edge_index.shape[1])
    if m:
        src, dst = edge_index
        out_deg = np.bincount(src, minlength=n).astype(np.float32)
        in_deg = np.bincount(dst, minlength=n).astype(np.float32)
        self_loops = float(np.sum(src == dst))
    else:
        out_deg = np.zeros(n, dtype=np.float32)
        in_deg = np.zeros(n, dtype=np.float32)
        self_loops = 0.0
    total_deg = in_deg + out_deg
    density = float(m) / max(float(n * max(n - 1, 1)), 1.0)
    depth = np.asarray(getattr(data, "depth_to_root", torch.zeros(n)), dtype=np.float32).reshape(-1)
    topo = np.asarray(getattr(data, "topological_level", torch.zeros(n)), dtype=np.float32).reshape(-1)
    if depth.size != n:
        depth = np.zeros(n, dtype=np.float32)
    if topo.size != n:
        topo = np.zeros(n, dtype=np.float32)
    return np.asarray(
        [
            np.log1p(n),
            np.log1p(m),
            density,
            float(total_deg.mean()) if n else 0.0,
            float(total_deg.std()) if n else 0.0,
            float(total_deg.max()) if n else 0.0,
            float(total_deg.min()) if n else 0.0,
            float(in_deg.mean()) if n else 0.0,
            float(in_deg.std()) if n else 0.0,
            float(in_deg.max()) if n else 0.0,
            float(out_deg.mean()) if n else 0.0,
            float(out_deg.std()) if n else 0.0,
            float(out_deg.max()) if n else 0.0,
            float(np.mean(total_deg == 0)) if n else 0.0,
            float(np.mean(in_deg == 0)) if n else 0.0,
            float(np.mean(out_deg == 0)) if n else 0.0,
            self_loops / max(float(m), 1.0),
            float(m) / max(float(n), 1.0),
            float(depth.mean()) if n else 0.0,
            float(depth.std()) if n else 0.0,
            float(depth.max()) if n else 0.0,
            float(topo.mean()) if n else 0.0,
            float(topo.std()) if n else 0.0,
            float(topo.max()) if n else 0.0,
        ],
        dtype=np.float32,
    )


def graph_feature_vector(data: Any, num_edge_types: int, include_structure: bool) -> np.ndarray:
    x = data.x.detach().cpu().numpy().astype(np.float32, copy=False)
    node_part = _stats(x, x.shape[1])
    num_edge_types = max(int(num_edge_types), 1)
    edge_type = data.edge_type.detach().cpu().numpy().astype(np.int64, copy=False)
    edge_one_hot = np.zeros((edge_type.size, num_edge_types), dtype=np.float32)
    if edge_type.size:
        if int(edge_type.min()) < 0 or int(edge_type.max()) >= num_edge_types:
            raise ValueError(
                f"edge_type outside [0, {num_edge_types}): min={edge_type.min()}, max={edge_type.max()}"
            )
        edge_one_hot[np.arange(edge_type.size), edge_type] = 1.0
    edge_part = _stats(edge_one_hot, num_edge_types)
    pieces = [node_part, edge_part]
    if include_structure:
        pieces.append(_structure_features(data))
    return np.concatenate(pieces, axis=0).astype(np.float32, copy=False)


@dataclass
class TabularArrays:
    x: np.ndarray
    y_norm: np.ndarray
    y_raw: np.ndarray


def extract_arrays(
    graphs: Sequence[Any],
    num_edge_types: int,
    include_structure: bool,
) -> TabularArrays:
    if not graphs:
        raise ValueError("Cannot extract tabular features from an empty graph list.")
    features = [graph_feature_vector(g, num_edge_types, include_structure) for g in graphs]
    x = np.stack(features, axis=0).astype(np.float32, copy=False)
    y_norm = np.asarray([float(g.y.view(-1)[0].item()) for g in graphs], dtype=np.float32)
    y_raw = np.asarray([float(g.y_raw.view(-1)[0].item()) for g in graphs], dtype=np.float64)
    return TabularArrays(x=x, y_norm=y_norm, y_raw=y_raw)


@dataclass
class FeatureStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "FeatureStandardizer":
        mean = np.asarray(x, dtype=np.float64).mean(axis=0)
        std = np.asarray(x, dtype=np.float64).std(axis=0)
        std[~np.isfinite(std) | (std < 1e-8)] = 1.0
        mean[~np.isfinite(mean)] = 0.0
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, arrays: TabularArrays) -> TabularArrays:
        x = (arrays.x - self.mean) / self.std
        x = np.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0)
        x = np.clip(x, -20.0, 20.0).astype(np.float32, copy=False)
        return TabularArrays(x=x, y_norm=arrays.y_norm, y_raw=arrays.y_raw)

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def prepare_array_splits(
    train_graphs: Sequence[Any],
    val_graphs: Sequence[Any],
    test_graphs: Sequence[Any],
    num_edge_types: int,
    include_structure: bool,
) -> Tuple[TabularArrays, TabularArrays, TabularArrays, FeatureStandardizer]:
    train_raw = extract_arrays(train_graphs, num_edge_types, include_structure)
    val_raw = extract_arrays(val_graphs, num_edge_types, include_structure)
    test_raw = extract_arrays(test_graphs, num_edge_types, include_structure)
    scaler = FeatureStandardizer.fit(train_raw.x)
    return scaler.transform(train_raw), scaler.transform(val_raw), scaler.transform(test_raw), scaler


def transform_target_splits(
    scaler: FeatureStandardizer,
    train_graphs: Sequence[Any],
    val_graphs: Sequence[Any],
    test_graphs: Sequence[Any],
    num_edge_types: int,
    include_structure: bool,
) -> Tuple[TabularArrays, TabularArrays, TabularArrays]:
    return (
        scaler.transform(extract_arrays(train_graphs, num_edge_types, include_structure)),
        scaler.transform(extract_arrays(val_graphs, num_edge_types, include_structure)),
        scaler.transform(extract_arrays(test_graphs, num_edge_types, include_structure)),
    )


def _inverse_label(pred_norm: np.ndarray, feature_processor: Any) -> np.ndarray:
    pred_log = np.asarray(pred_norm, dtype=np.float64) * float(feature_processor.label_std)
    pred_log += float(feature_processor.label_mean)
    pred_log = np.clip(pred_log, -20.0, 20.0)
    return np.power(10.0, pred_log)


def evaluate_predictions(
    pred_norm: np.ndarray,
    arrays: TabularArrays,
    feature_processor: Any,
    elapsed_seconds: float,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    pred_norm = np.asarray(pred_norm, dtype=np.float64).reshape(-1)
    if pred_norm.shape[0] != arrays.y_norm.shape[0]:
        raise ValueError(f"Prediction size {pred_norm.shape[0]} != label size {arrays.y_norm.shape[0]}")
    norm_mse = float(np.mean(np.square(pred_norm - arrays.y_norm)))
    pred_raw = _inverse_label(pred_norm, feature_processor)
    metrics = cb.compute_metrics(
        y_true_raw=arrays.y_raw,
        y_pred_raw=pred_raw,
        norm_mse=norm_mse,
        elapsed_seconds=float(elapsed_seconds),
    )
    return metrics, arrays.y_raw.copy(), pred_raw


class GraphFeatureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        num_layers = max(int(num_layers), 1)
        layers = []
        last_dim = int(input_dim)
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(last_dim, int(hidden_dim)),
                    nn.LayerNorm(int(hidden_dim)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


@dataclass
class StageOutcome:
    best_epoch: int
    best_val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    efficiency_metrics: Dict[str, float]
    y_true_raw: np.ndarray
    y_pred_raw: np.ndarray
    best_state: Any


def _rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))
    except Exception:
        return 0.0


@torch.no_grad()
def _predict_mlp(model: nn.Module, arrays: TabularArrays, batch_size: int, device: torch.device) -> Tuple[np.ndarray, float]:
    loader = DataLoader(TensorDataset(torch.from_numpy(arrays.x)), batch_size=int(batch_size), shuffle=False)
    model.eval()
    preds = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for (x_batch,) in loader:
        preds.append(model(x_batch.to(device)).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    return torch.cat(preds).numpy(), float(elapsed)


def train_mlp(
    train: TabularArrays,
    val: TabularArrays,
    test: Optional[TabularArrays],
    feature_processor: Any,
    hparams: Dict[str, Any],
    device: torch.device,
    seed: int,
    init_state: Optional[Dict[str, torch.Tensor]] = None,
    trial: Optional[Any] = None,
) -> StageOutcome:
    seed_everything(seed)
    model = GraphFeatureMLP(
        input_dim=train.x.shape[1],
        hidden_dim=int(hparams["hidden_dim"]),
        num_layers=int(hparams["mlp_layers"]),
        dropout=float(hparams["dropout"]),
    ).to(device)
    if init_state is not None:
        model.load_state_dict(init_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train.x), torch.from_numpy(train.y_norm)),
        batch_size=int(hparams["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    best_score = float("inf")
    best_epoch = -1
    best_state = None
    best_val_metrics = None
    bad_epochs = 0
    epochs_trained = 0
    total_train_time = 0.0
    peak_train_rss = _rss_mb()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    max_epochs = int(hparams.get("max_epochs", 500))
    early_stop = int(hparams.get("early_stop", 50))
    for epoch in range(1, max_epochs + 1):
        epochs_trained = epoch
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_train_time += time.perf_counter() - t0
        peak_train_rss = max(peak_train_rss, _rss_mb())
        val_pred, val_time = _predict_mlp(model, val, int(hparams["batch_size"]), device)
        val_metrics, _, _ = evaluate_predictions(val_pred, val, feature_processor, val_time)
        score = float(val_metrics["median_q_error"])
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if trial is not None:
            trial.report(score, step=epoch)
            if trial.should_prune():
                raise cb.optuna.TrialPruned()
        if bad_epochs >= early_stop:
            break
    if best_state is None or best_val_metrics is None:
        raise RuntimeError("MLP training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    test_metrics: Dict[str, float] = {}
    y_true = np.empty((0,), dtype=np.float64)
    y_pred = np.empty((0,), dtype=np.float64)
    infer_time = 0.0
    peak_infer_rss = _rss_mb()
    if device.type == "cuda":
        train_alloc = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        train_reserved = float(torch.cuda.max_memory_reserved(device) / (1024 ** 2))
    else:
        train_alloc = train_reserved = 0.0
    if test is not None:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        pred_norm, infer_time = _predict_mlp(model, test, int(hparams["batch_size"]), device)
        peak_infer_rss = max(peak_infer_rss, _rss_mb())
        test_metrics, y_true, y_pred = evaluate_predictions(pred_norm, test, feature_processor, infer_time)
        if device.type == "cuda":
            infer_alloc = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            infer_reserved = float(torch.cuda.max_memory_reserved(device) / (1024 ** 2))
        else:
            infer_alloc = infer_reserved = 0.0
    else:
        infer_alloc = infer_reserved = 0.0
    efficiency = {
        "num_epochs": float(epochs_trained),
        "t_train_total": float(total_train_time),
        "t_train_epoch_avg": float(total_train_time / max(epochs_trained, 1)),
        "train_max_memory_allocated_mb": train_alloc,
        "train_max_memory_reserved_mb": train_reserved,
        "train_peak_rss_mb": float(peak_train_rss),
        "t_infer": float(infer_time),
        "infer_max_memory_allocated_mb": infer_alloc,
        "infer_max_memory_reserved_mb": infer_reserved,
        "infer_peak_rss_mb": float(peak_infer_rss),
    }
    return StageOutcome(best_epoch, best_val_metrics, test_metrics, efficiency, y_true, y_pred, best_state)


def _import_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("The lgbm baseline requires `pip install lightgbm>=4.0`.") from exc
    return lgb


def _lgbm_params(hparams: Dict[str, Any], seed: int) -> Dict[str, Any]:
    return {
        "objective": "regression",
        # Disable the implicit L2 metric so validation early stopping is driven
        # solely by the custom median q-error metric below.
        "metric": "None",
        "n_estimators": int(hparams["lgbm_n_estimators"]),
        "learning_rate": float(hparams["lgbm_learning_rate"]),
        "num_leaves": int(hparams["lgbm_num_leaves"]),
        "max_depth": int(hparams["lgbm_max_depth"]),
        "min_child_samples": int(hparams["lgbm_min_child_samples"]),
        "subsample": float(hparams["lgbm_subsample"]),
        "subsample_freq": 1,
        "colsample_bytree": float(hparams["lgbm_colsample_bytree"]),
        "reg_alpha": float(hparams["lgbm_reg_alpha"]),
        "reg_lambda": float(hparams["lgbm_reg_lambda"]),
        "random_state": int(seed),
        "n_jobs": int(hparams.get("lgbm_n_jobs", -1)),
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def train_lgbm(
    train: TabularArrays,
    val: TabularArrays,
    test: Optional[TabularArrays],
    feature_processor: Any,
    hparams: Dict[str, Any],
    seed: int,
) -> StageOutcome:
    lgb = _import_lightgbm()
    model = lgb.LGBMRegressor(**_lgbm_params(hparams, seed))

    def medq_eval(y_true_norm: np.ndarray, y_pred_norm: np.ndarray):
        pred_raw = _inverse_label(y_pred_norm, feature_processor)
        true_raw = _inverse_label(y_true_norm, feature_processor)
        q = np.maximum(pred_raw / np.clip(true_raw, 1e-9, None), true_raw / np.clip(pred_raw, 1e-9, None))
        return "median_q_error", float(np.median(q)), False

    start_rss = _rss_mb()
    t0 = time.perf_counter()
    callbacks = [lgb.log_evaluation(period=0)]
    early_stop = int(hparams.get("lgbm_early_stop", 20))
    if early_stop > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stop, first_metric_only=True, verbose=False))
    model.fit(
        train.x,
        train.y_norm,
        eval_set=[(val.x, val.y_norm)],
        eval_metric=medq_eval,
        callbacks=callbacks,
    )
    train_time = time.perf_counter() - t0
    peak_train_rss = max(start_rss, _rss_mb())
    t_val = time.perf_counter()
    val_pred = model.predict(val.x)
    val_elapsed = time.perf_counter() - t_val
    val_metrics, _, _ = evaluate_predictions(val_pred, val, feature_processor, val_elapsed)
    test_metrics: Dict[str, float] = {}
    y_true = np.empty((0,), dtype=np.float64)
    y_pred = np.empty((0,), dtype=np.float64)
    infer_time = 0.0
    peak_infer_rss = _rss_mb()
    if test is not None:
        t_test = time.perf_counter()
        test_pred = model.predict(test.x)
        infer_time = time.perf_counter() - t_test
        peak_infer_rss = max(peak_infer_rss, _rss_mb())
        test_metrics, y_true, y_pred = evaluate_predictions(test_pred, test, feature_processor, infer_time)
    best_iteration = int(getattr(model, "best_iteration_", 0) or hparams["lgbm_n_estimators"])
    efficiency = {
        "num_epochs": float(best_iteration),
        "t_train_total": float(train_time),
        "t_train_epoch_avg": float(train_time / max(best_iteration, 1)),
        "train_max_memory_allocated_mb": 0.0,
        "train_max_memory_reserved_mb": 0.0,
        "train_peak_rss_mb": float(peak_train_rss),
        "t_infer": float(infer_time),
        "infer_max_memory_allocated_mb": 0.0,
        "infer_max_memory_reserved_mb": 0.0,
        "infer_peak_rss_mb": float(peak_infer_rss),
    }
    return StageOutcome(
        best_iteration,
        val_metrics,
        test_metrics,
        efficiency,
        y_true,
        y_pred,
        # Keep only the validation-selected boosting rounds.  This makes the
        # serialized zero-shot model identical to the model used for its test
        # predictions and prevents refitting early-stopping tail trees.
        model.booster_.model_to_string(num_iteration=best_iteration),
    )


def refit_lgbm(
    source_model_string: str,
    train: TabularArrays,
    val: TabularArrays,
    test: TabularArrays,
    feature_processor: Any,
    hparams: Dict[str, Any],
) -> StageOutcome:
    """Refit existing leaves on target data without adding trees/capacity."""
    lgb = _import_lightgbm()
    source = lgb.Booster(model_str=source_model_string)
    if not hasattr(source, "refit"):
        raise RuntimeError("LightGBM transfer requires Booster.refit (install lightgbm>=4.0).")
    decay = float(hparams.get("lgbm_refit_decay_rate", 0.5))
    start_rss = _rss_mb()
    t0 = time.perf_counter()
    adapted = source.refit(train.x, train.y_norm, decay_rate=decay)
    train_time = time.perf_counter() - t0
    peak_train_rss = max(start_rss, _rss_mb())
    t_val = time.perf_counter()
    val_pred = adapted.predict(val.x)
    val_elapsed = time.perf_counter() - t_val
    val_metrics, _, _ = evaluate_predictions(val_pred, val, feature_processor, val_elapsed)
    t_test = time.perf_counter()
    infer_start_rss = _rss_mb()
    test_pred = adapted.predict(test.x)
    infer_time = time.perf_counter() - t_test
    test_metrics, y_true, y_pred = evaluate_predictions(test_pred, test, feature_processor, infer_time)
    efficiency = {
        "num_epochs": 1.0,
        "t_train_total": float(train_time),
        "t_train_epoch_avg": float(train_time),
        "train_max_memory_allocated_mb": 0.0,
        "train_max_memory_reserved_mb": 0.0,
        "train_peak_rss_mb": float(peak_train_rss),
        "t_infer": float(infer_time),
        "infer_max_memory_allocated_mb": 0.0,
        "infer_max_memory_reserved_mb": 0.0,
        "infer_peak_rss_mb": float(max(_rss_mb(), infer_start_rss)),
    }
    return StageOutcome(1, val_metrics, test_metrics, efficiency, y_true, y_pred, adapted.model_to_string())


def run_stage(
    model_name: str,
    train: TabularArrays,
    val: TabularArrays,
    test: Optional[TabularArrays],
    feature_processor: Any,
    hparams: Dict[str, Any],
    device: torch.device,
    seed: int,
    init_state: Optional[Any] = None,
    trial: Optional[Any] = None,
) -> StageOutcome:
    model_name = normalize_model_name(model_name)
    if model_name == "lgbm":
        if init_state is not None:
            if test is None:
                raise ValueError("LightGBM refit requires a test array for the final transfer stage.")
            return refit_lgbm(init_state, train, val, test, feature_processor, hparams)
        return train_lgbm(train, val, test, feature_processor, hparams, seed)
    return train_mlp(train, val, test, feature_processor, hparams, device, seed, init_state, trial)


def default_hparams(args: Any, model_name: str) -> Dict[str, Any]:
    model_name = normalize_model_name(model_name)
    if model_name == "lgbm":
        return {
            "lgbm_n_estimators": int(args.lgbm_n_estimators),
            "lgbm_learning_rate": float(args.lgbm_learning_rate),
            "lgbm_num_leaves": int(args.lgbm_num_leaves),
            "lgbm_max_depth": int(args.lgbm_max_depth),
            "lgbm_min_child_samples": int(args.lgbm_min_child_samples),
            "lgbm_subsample": float(args.lgbm_subsample),
            "lgbm_colsample_bytree": float(args.lgbm_colsample_bytree),
            "lgbm_reg_alpha": float(args.lgbm_reg_alpha),
            "lgbm_reg_lambda": float(args.lgbm_reg_lambda),
            "lgbm_early_stop": int(args.lgbm_early_stop),
            "lgbm_refit_decay_rate": float(args.lgbm_refit_decay_rate),
            "lgbm_n_jobs": int(args.lgbm_n_jobs),
        }
    return {
        "hidden_dim": int(args.hidden_dim),
        "mlp_layers": int(args.mlp_layers),
        "dropout": float(args.dropout),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "max_epochs": int(args.max_epochs),
        "early_stop": int(args.early_stop),
    }


def sample_hparams(trial: Any, args: Any, model_name: str) -> Dict[str, Any]:
    model_name = normalize_model_name(model_name)
    compact_hpo = bool(getattr(args, "compact_hpo", False))
    if model_name == "lgbm":
        hp = default_hparams(args, model_name)
        if compact_hpo:
            # The completed 1000-shot transfer runs concentrated on these
            # choices.  Keep weakly influential controls fixed so ten trials
            # can cover the tree-size/regularization trade-off.
            hp.update(
                {
                    "lgbm_n_estimators": trial.suggest_categorical("lgbm_n_estimators", [500, 800]),
                    "lgbm_learning_rate": trial.suggest_categorical("lgbm_learning_rate", [0.03, 0.05, 0.1]),
                    "lgbm_num_leaves": trial.suggest_categorical("lgbm_num_leaves", [31, 63]),
                    "lgbm_max_depth": -1,
                    "lgbm_min_child_samples": trial.suggest_categorical("lgbm_min_child_samples", [20, 50]),
                    "lgbm_subsample": 1.0,
                    "lgbm_colsample_bytree": 0.8,
                    "lgbm_reg_alpha": trial.suggest_categorical("lgbm_reg_alpha", [0.0, 1e-3, 0.1]),
                    "lgbm_reg_lambda": 1e-3,
                }
            )
            return hp
        hp.update(
            {
                "lgbm_n_estimators": trial.suggest_categorical("lgbm_n_estimators", [200, 500, 800]),
                "lgbm_learning_rate": trial.suggest_categorical("lgbm_learning_rate", [0.01, 0.03, 0.05, 0.1]),
                "lgbm_num_leaves": trial.suggest_categorical("lgbm_num_leaves", [15, 31, 63]),
                "lgbm_max_depth": trial.suggest_categorical("lgbm_max_depth", [-1, 5, 7, 10]),
                "lgbm_min_child_samples": trial.suggest_categorical("lgbm_min_child_samples", [10, 20, 50]),
                "lgbm_subsample": trial.suggest_categorical("lgbm_subsample", [0.8, 1.0]),
                "lgbm_colsample_bytree": trial.suggest_categorical("lgbm_colsample_bytree", [0.8, 1.0]),
                "lgbm_reg_alpha": trial.suggest_categorical("lgbm_reg_alpha", [0.0, 1e-3, 1e-2, 1e-1]),
                "lgbm_reg_lambda": trial.suggest_categorical("lgbm_reg_lambda", [0.0, 1e-3, 1e-2, 1e-1]),
            }
        )
        return hp
    if compact_hpo:
        hp = default_hparams(args, model_name)
        hp.update(
            {
                "hidden_dim": 256,
                "mlp_layers": trial.suggest_categorical("mlp_layers", [3, 4]),
                "dropout": trial.suggest_categorical("dropout", [0.0, 0.05, 0.1]),
                "lr": trial.suggest_categorical("lr", [3e-4, 5e-4, 7e-4, 1e-3, 1.3e-3]),
                "weight_decay": trial.suggest_categorical("weight_decay", [1e-6, 1e-5, 1e-4]),
                "batch_size": 64,
                "max_epochs": int(args.hpo_max_epochs),
                "early_stop": int(args.hpo_early_stop),
            }
        )
        return hp
    hp = default_hparams(args, model_name)
    hp.update(
        {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
            "mlp_layers": trial.suggest_categorical("mlp_layers", [2, 3, 4]),
            "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        }
    )
    hp["max_epochs"] = int(args.hpo_max_epochs)
    hp["early_stop"] = int(args.hpo_early_stop)
    return hp


def tune_hparams(
    model_name: str,
    train: TabularArrays,
    val: TabularArrays,
    feature_processor: Any,
    args: Any,
    device: torch.device,
) -> Dict[str, Any]:
    if cb.optuna is None:
        raise RuntimeError("Optuna is required for --n_trials > 0 (`pip install optuna`).")

    def objective(trial: Any) -> float:
        hp = sample_hparams(trial, args, model_name)
        # Deliberately pass test=None: HPO never constructs test predictions.
        outcome = run_stage(
            model_name,
            train,
            val,
            None,
            feature_processor,
            hp,
            device,
            int(args.first_seed),
            trial=trial,
        )
        trial.set_user_attr("best_val_metrics", outcome.best_val_metrics)
        return float(outcome.best_val_metrics["median_q_error"])

    sampler = cb.optuna.samplers.TPESampler(seed=int(args.first_seed))
    pruner = cb.optuna.pruners.MedianPruner(
        n_startup_trials=min(5, max(1, int(args.n_trials))),
        n_warmup_steps=10,
    )
    study = cb.optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    if bool(getattr(args, "compact_hpo", False)):
        if model_name == "lgbm":
            study.enqueue_trial(
                {
                    "lgbm_n_estimators": 500,
                    "lgbm_learning_rate": 0.05,
                    "lgbm_num_leaves": 63,
                    "lgbm_min_child_samples": 50,
                    "lgbm_reg_alpha": 1e-3,
                }
            )
        else:
            study.enqueue_trial(
                {
                    "mlp_layers": 4,
                    "dropout": 0.0,
                    "lr": 7e-4,
                    "weight_decay": 1e-5,
                }
            )
    study.optimize(objective, n_trials=int(args.n_trials), show_progress_bar=False)
    best = sample_hparams(cb.optuna.trial.FixedTrial(study.best_trial.params), args, model_name)
    # Final runs use the full training budget, not the shorter HPO budget.
    if model_name != "lgbm":
        best["max_epochs"] = int(args.max_epochs)
        best["early_stop"] = int(args.early_stop)
    return best


def add_tabular_cli_args(parser: Any) -> None:
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--mlp_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--early_stop", type=int, default=50)
    parser.add_argument("--hpo_max_epochs", type=int, default=60)
    parser.add_argument("--hpo_early_stop", type=int, default=15)
    parser.add_argument("--lgbm_n_estimators", type=int, default=500)
    parser.add_argument("--lgbm_learning_rate", type=float, default=0.05)
    parser.add_argument("--lgbm_num_leaves", type=int, default=31)
    parser.add_argument("--lgbm_max_depth", type=int, default=-1)
    parser.add_argument("--lgbm_min_child_samples", type=int, default=20)
    parser.add_argument("--lgbm_subsample", type=float, default=0.8)
    parser.add_argument("--lgbm_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--lgbm_reg_alpha", type=float, default=0.0)
    parser.add_argument("--lgbm_reg_lambda", type=float, default=0.01)
    parser.add_argument("--lgbm_early_stop", type=int, default=20)
    parser.add_argument("--lgbm_refit_decay_rate", type=float, default=0.5)
    parser.add_argument("--lgbm_n_jobs", type=int, default=-1)
