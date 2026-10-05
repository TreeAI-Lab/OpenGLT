from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from sparse_deferred.structs import graph_struct
except ImportError as exc:
    raise SystemExit(
        "Missing dependency `sparse_deferred`. Install it with:\n"
        "  pip install sparse-deferred"
    ) from exc

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, GINConv, GINEConv, PNAConv, global_add_pool, global_mean_pool
    try:
        from torch_geometric.nn import GraphMultisetTransformer
    except Exception:
        from torch_geometric.nn.aggr import GraphMultisetTransformer
    from torch_geometric.utils import coalesce, degree, to_dense_batch
    from torch_geometric.utils import k_hop_subgraph as pyg_k_hop_subgraph
except ImportError as exc:
    raise SystemExit(
        "Missing dependency `torch_geometric`. Install it with:\n"
        "  pip install torch-geometric"
    ) from exc

try:
    import optuna
except Exception:
    optuna = None


GRAPH_NODE_TYPE = "g"
NODE_TYPES = ["tables", "attributes", "predicates", "ops", "correlations"]
DEFAULT_SCHEMA = {
    "table_to_attr": ("tables", "attributes"),
    "attr_to_pred": ("attributes", "predicates"),
    "pred_to_pred": ("predicates", "predicates"),
    "attr_to_op": ("attributes", "ops"),
    "op_to_op": ("ops", "ops"),
    "pred_to_op": ("predicates", "ops"),
    "attr_to_corr": ("attributes", "correlations"),
    "corr_to_pred": ("correlations", "predicates"),
}

EXCLUDED_INPUT_KEYS = {
    GRAPH_NODE_TYPE: {"query", "query_id", "database_query_id", "query_run_id", "workload_id", "exec_time", "cardinality", "query_string"},
    "tables": {"name"},
    "attributes": {"name"},
    "predicates": set(),
    "ops": set(),
    "correlations": set(),
}

TEXT_SUMMARY_KEYS = {
    "predicates": {"constant"},
    "attributes": {"percentiles_100_string", "min_string", "max_string"},
}

CATEGORICAL_KEYS = {
    "attributes": {"data_type"},
    "predicates": {"predicate_operator"},
    "ops": {"operator"},
    "correlations": {"validity", "type"},
}

MINUS_ONE_MISSING_KEYS = {
    "percentiles_100_numeric",
    "min_numeric",
    "max_numeric",
    "estimated_selectivity",
    "offset",
    "encoded_constant",
}

NO_LOG_KEYS = {
    "null_frac",
    "estimated_selectivity",
    "correlation",
    "offset",
}

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def safe_decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def parse_int_tuple_arg(value: Any) -> Tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    text = str(value).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    if not text:
        return tuple()
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


MODEL_NAME_ALIASES = {
    "gin_root_gca": "gin_root",
    "gine_root_gca": "gine_root",
}


def normalize_model_name(model_name: str) -> str:
    name = str(model_name).strip().lower()
    return MODEL_NAME_ALIASES.get(name, name)


def is_gca_model_variant(model_name: str) -> bool:
    return str(model_name).strip().lower() in MODEL_NAME_ALIASES


def supports_cardbench_root_augmentation(model_name: str) -> bool:
    return normalize_model_name(model_name) in {"gin_root", "gine_root"}


def resolve_requested_aug_flags(model_name: str, use_aug: bool, use_flag: bool) -> Tuple[bool, bool]:
    return is_gca_model_variant(model_name) or bool(use_aug), bool(use_flag)


def validate_cardbench_aug_args(
    model_name: str,
    use_aug: bool,
    use_flag: bool,
    flag_steps: int,
) -> Tuple[bool, bool]:
    resolved_use_aug, resolved_use_flag = resolve_requested_aug_flags(model_name, use_aug, use_flag)
    if (resolved_use_aug or resolved_use_flag) and not supports_cardbench_root_augmentation(model_name):
        raise ValueError(
            "CardBench augmentation is currently supported only for gin_root/gine_root root-readout models."
        )
    if resolved_use_flag and int(flag_steps) <= 0:
        raise ValueError("--flag_steps must be >= 1 when --use_flag is enabled.")
    return resolved_use_aug, resolved_use_flag


def stable_hash_to_unit_interval(text: str) -> float:
    if not text:
        return 0.0
    value = 2166136261
    for ch in text:
        value ^= ord(ch)
        value = (value * 16777619) & 0xFFFFFFFF
    return float(value) / float(0xFFFFFFFF)


def flatten_to_numpy(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    return np.reshape(arr, (-1,))


def is_numeric_dtype(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.number)


def is_object_or_bytes_dtype(arr: np.ndarray) -> bool:
    return arr.dtype == object or arr.dtype.kind in {"S", "U"}


def signed_log1p_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class NumericFieldSpec:
    node_type: str
    key: str
    dim: int
    use_log: bool
    minus_one_missing: bool
    mean: np.ndarray
    std: np.ndarray

    @property
    def out_dim(self) -> int:
        return self.dim * 2


@dataclass
class CategoricalFieldSpec:
    node_type: str
    key: str
    vocab: List[str]
    stoi: Dict[str, int]

    @property
    def out_dim(self) -> int:
        return len(self.vocab) + 1


@dataclass
class TextSummaryFieldSpec:
    node_type: str
    key: str

    @property
    def out_dim(self) -> int:
        return 5


FieldSpec = NumericFieldSpec | CategoricalFieldSpec | TextSummaryFieldSpec


class CardBenchFeatureProcessor:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.field_specs: List[FieldSpec] = []
        self.feature_dim: int = 0
        self.label_mean: float = 0.0
        self.label_std: float = 1.0
        self.fitted: bool = False

    def fit(self, train_graphs: Sequence[Any]) -> None:
        if not train_graphs:
            raise ValueError("No training graphs provided to feature processor.")

        field_specs: List[FieldSpec] = []
        for node_type in NODE_TYPES:
            key_to_values: Dict[str, List[np.ndarray]] = defaultdict(list)
            for graph in train_graphs:
                node_dict = graph.nodes.get(node_type, {})
                for key, values in node_dict.items():
                    if key in EXCLUDED_INPUT_KEYS.get(node_type, set()):
                        continue
                    if len(values) == 0:
                        continue
                    for item in values:
                        key_to_values[key].append(np.asarray(item))

            if self.debug:
                print(f"[fit] node_type={node_type} discovered_keys={sorted(key_to_values.keys())}")

            for key in sorted(key_to_values.keys()):
                arrays = key_to_values[key]
                sample = arrays[0]
                if key in TEXT_SUMMARY_KEYS.get(node_type, set()):
                    field_specs.append(TextSummaryFieldSpec(node_type=node_type, key=key))
                    continue
                if key in CATEGORICAL_KEYS.get(node_type, set()):
                    vocab = self._build_vocab(arrays)
                    field_specs.append(
                        CategoricalFieldSpec(
                            node_type=node_type,
                            key=key,
                            vocab=vocab,
                            stoi={token: i for i, token in enumerate(vocab)},
                        )
                    )
                    continue
                if is_numeric_dtype(sample):
                    dim = int(flatten_to_numpy(sample).shape[0])
                    use_log = key not in NO_LOG_KEYS
                    minus_one_missing = key in MINUS_ONE_MISSING_KEYS or "percentiles_100_numeric" in key
                    mean_arr, std_arr = self._fit_numeric_stats(arrays, use_log, minus_one_missing)
                    field_specs.append(
                        NumericFieldSpec(
                            node_type=node_type,
                            key=key,
                            dim=dim,
                            use_log=use_log,
                            minus_one_missing=minus_one_missing,
                            mean=mean_arr,
                            std=std_arr,
                        )
                    )
                    continue
                if is_object_or_bytes_dtype(sample):
                    vocab = self._build_vocab(arrays)
                    if 0 < len(vocab) <= 128:
                        field_specs.append(
                            CategoricalFieldSpec(
                                node_type=node_type,
                                key=key,
                                vocab=vocab,
                                stoi={token: i for i, token in enumerate(vocab)},
                            )
                        )
                    else:
                        field_specs.append(TextSummaryFieldSpec(node_type=node_type, key=key))
                    continue
                if self.debug:
                    print(f"[fit] skip unsupported field {node_type}.{key} dtype={sample.dtype}")

        self.field_specs = field_specs
        self.feature_dim = len(NODE_TYPES) + sum(spec.out_dim for spec in self.field_specs)

        labels = np.asarray(
            [float(graph.nodes[GRAPH_NODE_TYPE]["cardinality"][0]) for graph in train_graphs],
            dtype=np.float64,
        )
        labels = labels[labels > 0]
        label_log = np.log10(labels)
        self.label_mean = float(np.mean(label_log))
        self.label_std = float(np.std(label_log))
        if self.label_std < 1e-8:
            self.label_std = 1.0
        self.fitted = True

        if self.debug:
            print(f"[fit] feature_dim={self.feature_dim}")
            print(f"[fit] num_field_specs={len(self.field_specs)}")
            print(f"[fit] label_mean={self.label_mean:.6f} label_std={self.label_std:.6f}")

    def transform_graph(self, graph: Any, schema: Dict[str, Tuple[str, str]]) -> Optional[Data]:
        if not self.fitted:
            raise RuntimeError("Call fit() before transform_graph().")
        if not is_usable_query_graph(graph):
            return None

        cardinality = float(graph.nodes[GRAPH_NODE_TYPE]["cardinality"][0])
        node_offsets: Dict[str, int] = {}
        rows: List[np.ndarray] = []
        offset = 0
        for node_type in NODE_TYPES:
            node_offsets[node_type] = offset
            count = self._get_num_nodes(graph, node_type)
            for idx in range(count):
                rows.append(self._encode_node(graph, node_type, idx))
            offset += count

        if not rows:
            return None

        x = torch.tensor(np.stack(rows, axis=0), dtype=torch.float32)
        edge_index, edge_type = self._build_edge_index(graph, schema, node_offsets, num_nodes=x.size(0))
        root_index = self._find_root_index(graph, node_offsets)
        depth_to_root = self._compute_depth_to_root(edge_index, root_index, x.size(0))
        topological_level = self._compute_topological_level(edge_index, x.size(0))
        node_type_id = self._compute_node_type_id(x.size(0), graph, node_offsets)

        y_raw = torch.tensor([cardinality], dtype=torch.float32)
        y_log = math.log10(cardinality)
        y = torch.tensor([(y_log - self.label_mean) / self.label_std], dtype=torch.float32)
        query_id = self._extract_query_id(graph)

        return Data(
            x=x,
            edge_index=edge_index,
            y=y,
            y_raw=y_raw,
            query_id=torch.tensor([query_id], dtype=torch.long),
            root_index=torch.tensor([root_index], dtype=torch.long),
            edge_type=edge_type,
            depth_to_root=torch.tensor(depth_to_root, dtype=torch.long),
            topological_level=torch.tensor(topological_level, dtype=torch.long),
            node_type_id=torch.tensor(node_type_id, dtype=torch.long),
            num_nodes=x.size(0),
        )

    def inverse_label_transform(self, y_pred_norm: Tensor) -> Tensor:
        y_log = y_pred_norm * self.label_std + self.label_mean
        return torch.pow(torch.tensor(10.0, device=y_pred_norm.device), y_log)

    def _fit_numeric_stats(
        self,
        arrays: Sequence[np.ndarray],
        use_log: bool,
        minus_one_missing: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        values = [flatten_to_numpy(v).astype(np.float32) for v in arrays]
        dim = values[0].shape[0]
        stacked = np.stack(values, axis=0)
        mask = np.isfinite(stacked)
        if minus_one_missing:
            mask &= stacked != -1
        work = stacked.copy()
        if use_log:
            work = signed_log1p_array(work)
        means = np.zeros(dim, dtype=np.float32)
        stds = np.ones(dim, dtype=np.float32)
        for j in range(dim):
            valid = mask[:, j]
            if np.any(valid):
                col = work[valid, j]
                means[j] = float(np.mean(col))
                std = float(np.std(col))
                stds[j] = std if std >= 1e-6 else 1.0
            else:
                means[j] = 0.0
                stds[j] = 1.0
        return means, stds

    def _build_vocab(self, arrays: Sequence[np.ndarray]) -> List[str]:
        tokens: set[str] = set()
        for value in arrays:
            arr = flatten_to_numpy(value)
            for item in arr:
                token = safe_decode(item).strip().lower()
                if token:
                    tokens.add(token)
        return sorted(tokens)

    def _get_num_nodes(self, graph: Any, node_type: str) -> int:
        if node_type not in graph.nodes:
            return 0
        node_dict = graph.nodes[node_type]
        if not node_dict:
            return 0
        first_key = next(iter(node_dict))
        return int(len(node_dict[first_key]))

    def _encode_node(self, graph: Any, node_type: str, idx: int) -> np.ndarray:
        out = np.zeros(self.feature_dim, dtype=np.float32)
        out[NODE_TYPES.index(node_type)] = 1.0
        cursor = len(NODE_TYPES)
        for spec in self.field_specs:
            if spec.node_type != node_type:
                cursor += spec.out_dim
                continue
            node_dict = graph.nodes.get(node_type, {})
            has_key = spec.key in node_dict and len(node_dict[spec.key]) > idx
            raw_value = node_dict[spec.key][idx] if has_key else None
            if isinstance(spec, NumericFieldSpec):
                enc = self._encode_numeric(raw_value, spec)
            elif isinstance(spec, CategoricalFieldSpec):
                enc = self._encode_categorical(raw_value, spec)
            else:
                enc = self._encode_text_summary(raw_value)
            out[cursor: cursor + spec.out_dim] = enc
            cursor += spec.out_dim
        return out

    def _encode_numeric(self, raw_value: Any, spec: NumericFieldSpec) -> np.ndarray:
        values = np.zeros(spec.dim, dtype=np.float32)
        mask = np.ones(spec.dim, dtype=np.float32)
        if raw_value is not None:
            arr = flatten_to_numpy(raw_value).astype(np.float32)
            if arr.shape[0] != spec.dim:
                padded = np.zeros(spec.dim, dtype=np.float32)
                take = min(spec.dim, arr.shape[0])
                padded[:take] = arr[:take]
                arr = padded
            valid = np.isfinite(arr)
            if spec.minus_one_missing:
                valid &= arr != -1
            work = arr.copy()
            if spec.use_log:
                work = signed_log1p_array(work)
            if np.any(valid):
                values[valid] = (work[valid] - spec.mean[valid]) / spec.std[valid]
                mask[valid] = 0.0
        return np.concatenate([values, mask], axis=0)

    def _encode_categorical(self, raw_value: Any, spec: CategoricalFieldSpec) -> np.ndarray:
        out = np.zeros(spec.out_dim, dtype=np.float32)
        token = ""
        if raw_value is not None:
            token = safe_decode(np.asarray(raw_value).reshape(-1)[0]).strip().lower()
        index = spec.stoi.get(token, len(spec.vocab))
        out[index] = 1.0
        return out

    def _encode_text_summary(self, raw_value: Any) -> np.ndarray:
        if raw_value is None:
            return np.zeros(5, dtype=np.float32)
        arr = flatten_to_numpy(raw_value)
        tokens = [safe_decode(v) for v in arr]
        tokens = [t for t in tokens if t not in {"", "-1", "none", "nan"}]
        if not tokens:
            return np.zeros(5, dtype=np.float32)
        lengths = np.asarray([len(t) for t in tokens], dtype=np.float32)
        hashes = np.asarray([stable_hash_to_unit_interval(t) for t in tokens], dtype=np.float32)
        return np.asarray(
            [1.0, float(np.mean(lengths)), float(np.min(lengths)), float(np.max(lengths)), float(np.mean(hashes))],
            dtype=np.float32,
        )

    def _build_edge_index(
        self,
        graph: Any,
        schema: Dict[str, Tuple[str, str]],
        node_offsets: Dict[str, int],
        num_nodes: int,
    ) -> Tuple[Tensor, Tensor]:
        pieces: List[np.ndarray] = []
        edge_type_pieces: List[np.ndarray] = []
        edge_type_to_id = {edge_name: idx for idx, edge_name in enumerate(sorted(schema.keys()))}
        for edge_name, edge_value in graph.edges.items():
            if edge_name not in schema:
                continue
            source_type, target_type = schema[edge_name]
            src = np.asarray(edge_value[0][0], dtype=np.int64)
            dst = np.asarray(edge_value[0][1], dtype=np.int64)
            if src.size == 0:
                continue
            src = src + node_offsets[source_type]
            dst = dst + node_offsets[target_type]
            pieces.append(np.stack([src, dst], axis=0))
            edge_type_pieces.append(np.full(src.shape[0], edge_type_to_id[edge_name], dtype=np.int64))
        if not pieces:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
        edge_index = np.concatenate(pieces, axis=1)
        edge_type = np.concatenate(edge_type_pieces, axis=0)
        if np.any(edge_index < 0) or np.any(edge_index >= num_nodes):
            raise ValueError("edge_index contains out-of-range node ids.")
        return torch.tensor(edge_index, dtype=torch.long), torch.tensor(edge_type, dtype=torch.long)

    def _find_root_index(self, graph: Any, node_offsets: Dict[str, int]) -> int:
        num_ops = self._get_num_nodes(graph, "ops")
        if num_ops == 0:
            return 0

        ops = graph.nodes.get("ops", {})
        op_names = []
        if "operator" in ops:
            op_names = [safe_decode(v).strip().lower() for v in ops["operator"]]

        if "op_to_op" in graph.edges:
            src = np.asarray(graph.edges["op_to_op"][0][0], dtype=np.int64)
            dst = np.asarray(graph.edges["op_to_op"][0][1], dtype=np.int64)
            if src.size > 0:
                out_deg = np.zeros(num_ops, dtype=np.int64)
                in_deg = np.zeros(num_ops, dtype=np.int64)
                np.add.at(out_deg, src, 1)
                np.add.at(in_deg, dst, 1)
                sink_candidates = np.where(out_deg == 0)[0]
                if sink_candidates.size == 1:
                    return int(node_offsets["ops"] + sink_candidates[0])
                if sink_candidates.size > 1:
                    if op_names:
                        join_sinks = [idx for idx in sink_candidates.tolist() if op_names[idx] == "join"]
                        if len(join_sinks) == 1:
                            return int(node_offsets["ops"] + join_sinks[0])
                    sink_scores = in_deg[sink_candidates]
                    best_local = int(sink_candidates[int(np.argmax(sink_scores))])
                    return int(node_offsets["ops"] + best_local)

        if op_names:
            join_nodes = [i for i, name in enumerate(op_names) if name == "join"]
            if len(join_nodes) == 1:
                return int(node_offsets["ops"] + join_nodes[0])

        incoming_score = np.zeros(num_ops, dtype=np.int64)
        for edge_name in ("pred_to_op", "attr_to_op", "corr_to_op"):
            if edge_name in graph.edges:
                dst = np.asarray(graph.edges[edge_name][0][1], dtype=np.int64)
                if dst.size > 0:
                    np.add.at(incoming_score, dst, 1)
        if np.any(incoming_score > 0):
            best_local = int(np.argmax(incoming_score))
            return int(node_offsets["ops"] + best_local)

        return int(node_offsets["ops"] + num_ops - 1)

    def _compute_depth_to_root(self, edge_index: Tensor, root_index: int, num_nodes: int) -> List[int]:
        rev_adj: List[List[int]] = [[] for _ in range(num_nodes)]
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        for s, d in zip(src.tolist(), dst.tolist()):
            rev_adj[d].append(s)
        dist = [-1] * num_nodes
        dist[root_index] = 0
        queue = [root_index]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            for prev in rev_adj[cur]:
                if dist[prev] == -1:
                    dist[prev] = dist[cur] + 1
                    queue.append(prev)
        max_seen = max([d for d in dist if d >= 0], default=0)
        return [d if d >= 0 else max_seen + 1 for d in dist]

    def _compute_topological_level(self, edge_index: Tensor, num_nodes: int) -> List[int]:
        if num_nodes == 0:
            return []
        if edge_index.numel() == 0:
            return [0] * num_nodes

        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        adj: List[List[int]] = [[] for _ in range(num_nodes)]
        indegree = np.zeros(num_nodes, dtype=np.int64)
        for s, d in zip(src.tolist(), dst.tolist()):
            adj[s].append(d)
            indegree[d] += 1

        level = [0] * num_nodes
        work_indegree = indegree.copy()
        queue = [idx for idx in range(num_nodes) if work_indegree[idx] == 0]
        if not queue:
            return level

        head = 0
        seen = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            seen += 1
            for nxt in adj[cur]:
                level[nxt] = max(level[nxt], level[cur] + 1)
                work_indegree[nxt] -= 1
                if work_indegree[nxt] == 0:
                    queue.append(nxt)

        if seen < num_nodes:
            cycle_level = max(level, default=0) + 1
            for idx in range(num_nodes):
                if work_indegree[idx] > 0:
                    level[idx] = cycle_level
        return level

    def _compute_node_type_id(self, num_nodes: int, graph: Any, node_offsets: Dict[str, int]) -> List[int]:
        ids = [0] * num_nodes
        for node_type_idx, node_type in enumerate(NODE_TYPES):
            start = node_offsets[node_type]
            count = self._get_num_nodes(graph, node_type)
            for local_idx in range(count):
                ids[start + local_idx] = node_type_idx
        return ids

    def _extract_query_id(self, graph: Any) -> int:
        g = graph.nodes.get(GRAPH_NODE_TYPE, {})
        for key in ("query_id", "database_query_id"):
            if key in g:
                value = np.asarray(g[key]).reshape(-1)[0]
                try:
                    return int(value)
                except Exception:
                    return -1
        return -1


def load_inmemory_db(npz_path: str):
    if any(ch in npz_path for ch in "*?[]"):
        filenames = sorted(glob.glob(npz_path))
        if not filenames:
            raise FileNotFoundError(f"No files matched pattern: {npz_path}")
        if len(filenames) == 1:
            with open(filenames[0], "rb") as f:
                return graph_struct.InMemoryDB.from_bytes(f.read())
        entire_db = graph_struct.InMemoryDB()
        for filename in filenames:
            with open(filename, "rb") as f:
                db = graph_struct.InMemoryDB.from_bytes(f.read())
            for i in range(db.size):
                entire_db.add(db.get_item(i))
        entire_db.finalize(np.array)
        return entire_db
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File does not exist: {npz_path}")
    with open(npz_path, "rb") as f:
        return graph_struct.InMemoryDB.from_bytes(f.read())


def get_graph_schema(db) -> Dict[str, Tuple[str, str]]:
    schema = getattr(db, "schema", None)
    if not schema:
        return DEFAULT_SCHEMA.copy()
    normalized = {}
    for edge_name, value in schema.items():
        normalized[edge_name] = tuple(value)
    return normalized


@dataclass
class RawGraphDataset:
    name: str
    base_name: str
    workload_type: str
    npz_path: str
    schema: Dict[str, Tuple[str, str]]
    graphs: List[Any]


CARDBENCH_BASE_DATASETS = {
    "accidents", "airline", "consumer", "employee", "movielens",
    "sample_cms_synthetic_patient_data_omop",
    "sample_covid19_weathersource_com", "sample_crypto_bitcoin_cash",
    "sample_ethereum_blockchain", "sample_geo_openstreetmap",
    "sample_github_repos", "sample_human_variant_annotation",
    "sample_idc_v10", "sample_open_targets_genetics", "sample_samples",
    "sample_stackoverflow", "sample_usfs_fia", "sample_uspto_oce_claims",
    "sample_wikipedia", "tpch_10G",
}


def dataset_identity_from_path(npz_path: str) -> Tuple[str, str, str]:
    base = os.path.basename(npz_path)
    stem = os.path.splitext(base)[0]
    for suffix, workload_type in [
        ("_single_table", "single_table"),
        ("_binary_join", "binary_join"),
    ]:
        if stem.endswith(suffix):
            base_name = stem[: -len(suffix)]
            if base_name not in CARDBENCH_BASE_DATASETS:
                raise ValueError(f"Unsupported CardBench dataset: {stem}")
            return stem, base_name, workload_type
    raise ValueError(
        f"Unsupported CardBench workload {stem!r}; expected one of the released "
        "single-table or binary-join datasets."
    )


def load_raw_graph_dataset(npz_path: str, max_graphs: Optional[int] = None) -> RawGraphDataset:
    db = load_inmemory_db(npz_path)
    schema = get_graph_schema(db)
    limit = db.size if max_graphs is None else min(db.size, max_graphs)
    graphs = [db.get_item(i) for i in range(limit)]
    dataset_name, base_name, workload_type = dataset_identity_from_path(npz_path)
    return RawGraphDataset(
        name=dataset_name,
        base_name=base_name,
        workload_type=workload_type,
        npz_path=npz_path,
        schema=schema,
        graphs=graphs,
    )


def list_cardbench_npz_files(task_dir_or_glob: str) -> List[str]:
    if os.path.isdir(task_dir_or_glob):
        return sorted(glob.glob(os.path.join(task_dir_or_glob, "*.npz")))
    matched = sorted(glob.glob(task_dir_or_glob))
    if matched:
        return matched
    if os.path.isfile(task_dir_or_glob):
        return [task_dir_or_glob]
    raise FileNotFoundError(f"Could not resolve path/glob: {task_dir_or_glob}")


def is_usable_query_graph(graph: Any) -> bool:
    g = graph.nodes.get(GRAPH_NODE_TYPE, {})
    if "cardinality" not in g:
        return False
    try:
        if float(np.asarray(g["cardinality"]).reshape(-1)[0]) <= 0:
            return False
    except Exception:
        return False
    pred_nodes = graph.nodes.get("predicates", {})
    if pred_nodes:
        try:
            first_key = next(iter(pred_nodes))
            if len(pred_nodes[first_key]) == 0:
                return False
        except StopIteration:
            return False
    else:
        return False
    return True


@dataclass
class SplitRawGraphs:
    train: List[Any]
    val: List[Any]
    test: List[Any]


def official_instance_split(
    graphs: Sequence[Any],
    seed: int,
    holdout_test_size: int = 500,
    train_val_ratio: Tuple[float, float] = (0.85, 0.15),
    cap_trainval: Optional[int] = 4500,
) -> SplitRawGraphs:
    """CardBench-style protocol split.

    This implements the protocol described in the CardBench paper: a random
    holdout test set of size 500 and, optionally, a cap of 4500 train+val
    queries for fair per-dataset comparison. It does not load an external split
    file released by CardBench.
    """
    if len(graphs) <= holdout_test_size + 2:
        raise ValueError(f"Need more than {holdout_test_size + 2} graphs; got {len(graphs)}.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(graphs)).tolist()
    test_idx = perm[:holdout_test_size]
    remaining_idx = perm[holdout_test_size:]
    if cap_trainval is not None and cap_trainval > 0:
        remaining_idx = remaining_idx[:cap_trainval]
    remaining = [graphs[i] for i in remaining_idx]
    train_ratio, val_ratio = train_val_ratio
    n_rem = len(remaining)
    n_train = max(1, int(round(n_rem * train_ratio / (train_ratio + val_ratio))))
    n_train = min(n_train, n_rem - 1)
    train = remaining[:n_train]
    val = remaining[n_train:]
    test = [graphs[i] for i in test_idx]
    return SplitRawGraphs(train=train, val=val, test=test)


class ListDataset(Dataset):
    def __init__(self, items: Sequence[Data]):
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Data:
        return self.items[idx]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0, num_layers: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def make_pyg_gine_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float = 0.0,
    num_layers: int = 2,
) -> nn.Sequential:
    if num_layers < 2:
        raise ValueError(f"GINE MLP requires num_layers >= 2, got {num_layers}.")
    layers: List[nn.Module] = []
    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class RootReadoutGNN(nn.Module):
    def __init__(
        self,
        model_name: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        pna_deg: Optional[Tensor] = None,
        pna_towers: int = 1,
        num_edge_types: int = 0,
    ):
        super().__init__()
        self.model_name = model_name
        self.dropout = dropout
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.edge_encoder = (
            nn.Embedding(max(int(num_edge_types), 1), hidden_channels)
            if self.model_name.lower() == "gine_root"
            else None
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(self._build_conv(hidden_channels, pna_deg, pna_towers))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        self.readout = MLP(hidden_channels, hidden_channels, 1, dropout=dropout, num_layers=2)

    def _build_conv(self, channels: int, pna_deg: Optional[Tensor], pna_towers: int):
        name = self.model_name.lower()
        if name == "gcn_root":
            return GCNConv(channels, channels)
        if name == "gin_root":
            return GINConv(MLP(channels, channels, channels, dropout=self.dropout, num_layers=2))
        if name == "gine_root":
            return GINEConv(
                make_pyg_gine_mlp(channels, channels, channels, dropout=self.dropout, num_layers=2),
                edge_dim=channels,
            )
        if name == "pna_root":
            if pna_deg is None:
                raise ValueError("PNA requires degree histogram computed from the training split.")
            return PNAConv(
                in_channels=channels,
                out_channels=channels,
                aggregators=["mean", "min", "max", "std"],
                scalers=["identity", "amplification", "attenuation"],
                deg=pna_deg,
                towers=pna_towers,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        raise ValueError(f"Unsupported root-readout model: {self.model_name}")

    def _encode_edge_attr(self, data: Data, x: Tensor, edge_type: Optional[Tensor] = None) -> Optional[Tensor]:
        if self.edge_encoder is None:
            return None
        edge_type = getattr(data, "edge_type", None) if edge_type is None else edge_type
        if edge_type is None:
            raise ValueError(f"{self.model_name} requires data.edge_type for edge encoding.")
        if edge_type.numel() == 0:
            return x.new_zeros((0, x.size(1)))
        return self.edge_encoder(edge_type)

    def project_inputs(self, x: Tensor) -> Tensor:
        return self.input_proj(x)

    def forward_projected(
        self,
        data: Data,
        projected_x: Tensor,
        edge_index: Optional[Tensor] = None,
        edge_type: Optional[Tensor] = None,
        return_graph_embedding: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        edge_index = data.edge_index if edge_index is None else edge_index
        x = projected_x
        edge_attr = self._encode_edge_attr(data, x, edge_type=edge_type)
        prev = x
        for conv, norm in zip(self.convs, self.norms):
            if edge_attr is None:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + prev
            prev = x
        root_x = x[data.root_index.view(-1)]
        pred = self.readout(root_x).view(-1)
        if not return_graph_embedding:
            return pred
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        pooled_x = global_mean_pool(x, batch)
        graph_emb = torch.cat([root_x, pooled_x], dim=-1)
        return pred, graph_emb

    def forward(self, data: Data) -> Tensor:
        x = self.project_inputs(data.x)
        return self.forward_projected(data, x, edge_index=data.edge_index, return_graph_embedding=False)


class CardBenchHGPSLPool(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        pool_ratio: float,
        sl_topk: int,
        sl_lambda: float,
        score_bias_ops: float,
        score_bias_predicates: float,
    ):
        super().__init__()
        if not (0.0 < pool_ratio <= 1.0):
            raise ValueError(f"hgp_pool_ratio must be in (0, 1], got {pool_ratio}.")
        if sl_topk < 0:
            raise ValueError(f"hgp_topk must be >= 0, got {sl_topk}.")
        if sl_lambda < 0:
            raise ValueError(f"hgp_lambda must be >= 0, got {sl_lambda}.")
        self.hidden_channels = hidden_channels
        self.pool_ratio = float(pool_ratio)
        self.sl_topk = int(sl_topk)
        self.sl_lambda = float(sl_lambda)
        self.score_bias_ops = float(score_bias_ops)
        self.score_bias_predicates = float(score_bias_predicates)
        self.score_mlp = MLP(hidden_channels, hidden_channels, 1, dropout=0.0, num_layers=2)
        self.structure_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        root_index: Tensor,
        node_type_id: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.numel() == 0:
            raise ValueError("CardBenchHGPSLPool received an empty node representation tensor.")

        score = self.score_mlp(x).view(-1)
        score = score.clone()
        ops_mask = node_type_id == NODE_TYPES.index("ops")
        pred_mask = node_type_id == NODE_TYPES.index("predicates")
        score[ops_mask] += self.score_bias_ops
        score[pred_mask] += self.score_bias_predicates
        score[root_index.view(-1)] = torch.finfo(score.dtype).max

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        perm_parts: List[Tensor] = []
        for graph_idx in range(num_graphs):
            graph_nodes = torch.nonzero(batch == graph_idx, as_tuple=False).view(-1)
            if graph_nodes.numel() == 0:
                continue
            keep_count = max(1, int(math.ceil(graph_nodes.numel() * self.pool_ratio)))
            keep_count = min(keep_count, graph_nodes.numel())
            local_scores = score[graph_nodes]
            top_local = torch.topk(local_scores, k=keep_count, largest=True, sorted=False).indices
            selected = graph_nodes[top_local]
            selected = selected[torch.argsort(selected)]
            perm_parts.append(selected)

        if not perm_parts:
            raise RuntimeError("CardBenchHGPSLPool failed to keep any nodes.")

        perm = torch.cat(perm_parts, dim=0)
        pooled_x = x[perm]
        pooled_batch = batch[perm]

        original_to_pooled = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
        original_to_pooled[perm] = torch.arange(perm.numel(), device=x.device, dtype=torch.long)
        pooled_root_index = original_to_pooled[root_index.view(-1)]
        if torch.any(pooled_root_index < 0):
            raise RuntimeError("Root node was unexpectedly removed during HGP pooling.")

        induced_edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
        if edge_index.numel() > 0:
            mapped_src = original_to_pooled[edge_index[0]]
            mapped_dst = original_to_pooled[edge_index[1]]
            induced_mask = (mapped_src >= 0) & (mapped_dst >= 0)
            if torch.any(induced_mask):
                induced_edge_index = torch.stack([mapped_src[induced_mask], mapped_dst[induced_mask]], dim=0)

        learned_edge_parts: List[Tensor] = []
        scale = math.sqrt(float(self.hidden_channels))
        for graph_idx in range(num_graphs):
            pooled_nodes = torch.nonzero(pooled_batch == graph_idx, as_tuple=False).view(-1)
            num_nodes_graph = int(pooled_nodes.numel())
            if num_nodes_graph == 0:
                continue

            local_lookup = torch.full((pooled_x.size(0),), -1, dtype=torch.long, device=x.device)
            local_lookup[pooled_nodes] = torch.arange(num_nodes_graph, device=x.device, dtype=torch.long)
            induced_local = torch.zeros((num_nodes_graph, num_nodes_graph), dtype=torch.bool, device=x.device)
            if induced_edge_index.numel() > 0:
                graph_edge_mask = (pooled_batch[induced_edge_index[0]] == graph_idx) & (pooled_batch[induced_edge_index[1]] == graph_idx)
                if torch.any(graph_edge_mask):
                    local_src = local_lookup[induced_edge_index[0, graph_edge_mask]]
                    local_dst = local_lookup[induced_edge_index[1, graph_edge_mask]]
                    induced_local[local_src, local_dst] = True

            final_local = induced_local
            if self.sl_topk > 0 and num_nodes_graph > 1:
                x_graph = pooled_x[pooled_nodes]
                learned_scores = torch.matmul(self.structure_proj(x_graph), x_graph.t()) / scale
                learned_scores = learned_scores + self.sl_lambda * induced_local.float()
                learned_scores.fill_diagonal_(-float("inf"))
                topk = min(self.sl_topk, num_nodes_graph - 1)
                learned_local = torch.zeros((num_nodes_graph, num_nodes_graph), dtype=torch.bool, device=x.device)
                if topk > 0:
                    neighbor_idx = torch.topk(learned_scores, k=topk, dim=1, largest=True).indices
                    src_idx = torch.arange(num_nodes_graph, device=x.device, dtype=torch.long).unsqueeze(1).expand(-1, topk)
                    learned_local[src_idx, neighbor_idx] = True
                final_local = induced_local | learned_local

            local_edges = torch.nonzero(final_local, as_tuple=False)
            if local_edges.numel() == 0:
                continue
            global_src = pooled_nodes[local_edges[:, 0]]
            global_dst = pooled_nodes[local_edges[:, 1]]
            learned_edge_parts.append(torch.stack([global_src, global_dst], dim=0))

        pooled_edge_index = (
            torch.cat(learned_edge_parts, dim=1)
            if learned_edge_parts
            else torch.empty((2, 0), dtype=torch.long, device=x.device)
        )
        return pooled_x, pooled_edge_index, pooled_batch, pooled_root_index


class HGPGINRootRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        pool_ratio: float,
        sl_topk: int,
        sl_lambda: float,
        score_bias_ops: float,
        score_bias_predicates: float,
    ):
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}.")
        if int(num_layers) < 2:
            raise ValueError(f"hgp requires num_layers >= 2, got {num_layers}.")
        self.dropout = dropout
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.pre_conv = GINConv(MLP(hidden_channels, hidden_channels, hidden_channels, dropout=dropout, num_layers=2))
        self.pre_norm = nn.BatchNorm1d(hidden_channels)
        post_layers = max(1, int(num_layers) - 1)
        self.post_convs = nn.ModuleList(
            [GINConv(MLP(hidden_channels, hidden_channels, hidden_channels, dropout=dropout, num_layers=2)) for _ in range(post_layers)]
        )
        self.post_norms = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(post_layers)])
        self.pool = CardBenchHGPSLPool(
            hidden_channels=hidden_channels,
            pool_ratio=pool_ratio,
            sl_topk=sl_topk,
            sl_lambda=sl_lambda,
            score_bias_ops=score_bias_ops,
            score_bias_predicates=score_bias_predicates,
        )
        self.readout = MLP(hidden_channels * 2, hidden_channels, 1, dropout=dropout, num_layers=2)

    def _apply_gin_block(self, x: Tensor, edge_index: Tensor, conv: GINConv, norm: nn.BatchNorm1d) -> Tensor:
        residual = x
        x = conv(x, edge_index)
        x = norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x + residual

    def forward(self, data: Data) -> Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        x = self.input_proj(data.x)
        x = self._apply_gin_block(x, data.edge_index, self.pre_conv, self.pre_norm)
        x, edge_index, batch, pooled_root_index = self.pool(
            x,
            data.edge_index,
            batch,
            data.root_index.view(-1),
            data.node_type_id,
        )
        for conv, norm in zip(self.post_convs, self.post_norms):
            x = self._apply_gin_block(x, edge_index, conv, norm)

        root_x = x[pooled_root_index]
        pooled_x = global_mean_pool(x, batch)
        graph_repr = torch.cat([root_x, pooled_x], dim=-1)
        return self.readout(graph_repr).view(-1)


class GNNAKGINEStack(nn.Module):
    def __init__(self, hidden_channels: int, inner_layers: int, edge_dim: int, hops_dim: int, dropout: float):
        super().__init__()
        if inner_layers <= 0:
            raise ValueError("GNNAK requires gnnak_inner_layers >= 1.")
        self.hidden_channels = hidden_channels
        self.hops_dim = hops_dim
        self.dropout = dropout

        first_in = hidden_channels + hops_dim
        self.first_conv = GINEConv(
            nn.Sequential(
                nn.Linear(first_in, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            edge_dim=edge_dim,
        )
        self.first_norm = nn.BatchNorm1d(hidden_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(inner_layers - 1):
            self.convs.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, hidden_channels),
                    ),
                    edge_dim=edge_dim,
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        h = self.first_conv(x, edge_index, edge_attr)
        h = self.first_norm(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_res
        return h


class GNNAKLayer(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        inner_layers: int,
        num_hops: int,
        hops_dim: int,
        dropout: float,
        num_edge_types: int,
        embs: Tuple[int, ...],
        embs_combine_mode: str,
    ):
        super().__init__()
        if not embs:
            raise ValueError("gnnak_embs must not be empty.")
        if any(idx not in {0, 1, 2} for idx in embs):
            raise ValueError(f"Unsupported gnnak_embs={embs}. Expected values from {{0,1,2}}.")
        if embs_combine_mode not in {"add", "concat"}:
            raise ValueError(f"Unsupported gnnak_embs_combine_mode={embs_combine_mode}.")

        if num_hops < 1:
            raise ValueError("GNNAK requires gnnak_hops >= 1.")
        if hops_dim < 0:
            raise ValueError("GNNAK requires gnnak_hops_dim >= 0.")
        self.hidden_channels = hidden_channels
        self.num_hops = num_hops
        self.hops_dim = hops_dim
        self.use_hops = hops_dim > 0
        self.dropout = dropout
        self.embs = tuple(embs)
        self.embs_combine_mode = embs_combine_mode

        self.edge_encoder = nn.Embedding(max(int(num_edge_types), 1), hidden_channels)
        self.hop_embedder = nn.Embedding(num_hops + 2, hops_dim) if self.use_hops else None
        self.gnn = GNNAKGINEStack(
            hidden_channels=hidden_channels,
            inner_layers=inner_layers,
            edge_dim=hidden_channels,
            hops_dim=hops_dim,
            dropout=dropout,
        )
        self.subgraph_transform = MLP(hidden_channels, hidden_channels, hidden_channels, dropout=dropout, num_layers=2)
        self.context_transform = MLP(hidden_channels, hidden_channels, hidden_channels, dropout=dropout, num_layers=2)
        combined_in = hidden_channels * len(self.embs)
        self.combine_proj = (
            MLP(combined_in, hidden_channels, hidden_channels, dropout=dropout, num_layers=2)
            if self.embs_combine_mode == "concat"
            else None
        )

    @staticmethod
    def _compute_hop_indicator(edge_index: Tensor, center_local: int, num_nodes: int) -> Tensor:
        if num_nodes == 0:
            return torch.empty((0,), dtype=torch.long, device=edge_index.device)
        if edge_index.numel() == 0:
            out = torch.full((num_nodes,), 1, dtype=torch.long, device=edge_index.device)
            out[center_local] = 0
            return out

        src = edge_index[0].detach().cpu().tolist()
        dst = edge_index[1].detach().cpu().tolist()
        adj: List[List[int]] = [[] for _ in range(num_nodes)]
        for s, d in zip(src, dst):
            adj[s].append(d)

        dist = [-1] * num_nodes
        dist[center_local] = 0
        queue = [center_local]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            for nxt in adj[cur]:
                if dist[nxt] == -1:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)

        max_seen = max((d for d in dist if d >= 0), default=0)
        dist = [d if d >= 0 else max_seen + 1 for d in dist]
        return torch.tensor(dist, dtype=torch.long, device=edge_index.device)

    def build_egonet_batch(self, data: Data) -> Dict[str, Tensor]:
        edge_index = data.edge_index
        edge_type = data.edge_type
        num_nodes = int(data.num_nodes)
        device = edge_index.device

        if num_nodes == 0:
            raise ValueError("GNNAK received an empty graph.")

        if edge_index.numel() == 0:
            return {
                "nodes_mapper": torch.arange(num_nodes, device=device, dtype=torch.long),
                "subgraph_batch": torch.arange(num_nodes, device=device, dtype=torch.long),
                "center_indices": torch.arange(num_nodes, device=device, dtype=torch.long),
                "hop_indicator": torch.zeros(num_nodes, device=device, dtype=torch.long),
                "combined_edge_index": torch.empty((2, 0), device=device, dtype=torch.long),
                "combined_edge_type": torch.empty((0,), device=device, dtype=torch.long),
            }

        undirected_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        node_mapper_parts: List[Tensor] = []
        subgraph_batch_parts: List[Tensor] = []
        center_indices: List[int] = []
        hop_parts: List[Tensor] = []
        edge_index_parts: List[Tensor] = []
        edge_type_parts: List[Tensor] = []
        offset = 0

        for center in range(num_nodes):
            subset, undirected_subgraph_edge_index, mapping, _ = pyg_k_hop_subgraph(
                center,
                self.num_hops,
                undirected_edge_index,
                relabel_nodes=True,
                num_nodes=num_nodes,
            )
            node_mask = torch.zeros(num_nodes, device=device, dtype=torch.bool)
            node_mask[subset] = True
            directed_edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            relabel = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
            relabel[subset] = torch.arange(subset.size(0), device=device, dtype=torch.long)
            directed_subgraph_edge_index = relabel[edge_index[:, directed_edge_mask]]

            node_mapper_parts.append(subset)
            subgraph_batch_parts.append(torch.full((subset.size(0),), center, device=device, dtype=torch.long))
            hop_parts.append(self._compute_hop_indicator(undirected_subgraph_edge_index, int(mapping.item()), int(subset.size(0))))
            center_indices.append(offset + int(mapping.item()))
            if directed_subgraph_edge_index.numel() > 0:
                edge_index_parts.append(directed_subgraph_edge_index + offset)
                edge_type_parts.append(edge_type[directed_edge_mask])
            offset += int(subset.size(0))

        return {
            "nodes_mapper": torch.cat(node_mapper_parts, dim=0),
            "subgraph_batch": torch.cat(subgraph_batch_parts, dim=0),
            "center_indices": torch.tensor(center_indices, device=device, dtype=torch.long),
            "hop_indicator": torch.cat(hop_parts, dim=0),
            "combined_edge_index": torch.cat(edge_index_parts, dim=1)
            if edge_index_parts
            else torch.empty((2, 0), device=device, dtype=torch.long),
            "combined_edge_type": torch.cat(edge_type_parts, dim=0)
            if edge_type_parts
            else torch.empty((0,), device=device, dtype=torch.long),
        }

    def forward(self, x: Tensor, egonet_batch: Dict[str, Tensor]) -> Tensor:
        lifted_x = x[egonet_batch["nodes_mapper"]]
        if self.use_hops:
            hop_input = torch.clamp(egonet_batch["hop_indicator"] + 1, min=0, max=self.num_hops + 1)
            hop_emb = self.hop_embedder(hop_input)
            lifted_x = torch.cat([lifted_x, hop_emb], dim=-1)
        edge_attr = (
            self.edge_encoder(egonet_batch["combined_edge_type"])
            if egonet_batch["combined_edge_type"].numel() > 0
            else x.new_zeros((0, self.hidden_channels))
        )

        lifted_x = self.gnn(lifted_x, egonet_batch["combined_edge_index"], edge_attr)

        centroid_x = lifted_x[egonet_batch["center_indices"]]
        subgraph_x = global_add_pool(
            self.subgraph_transform(F.dropout(lifted_x, p=self.dropout, training=self.training)),
            egonet_batch["subgraph_batch"],
        )
        context_x = global_add_pool(
            self.context_transform(F.dropout(lifted_x, p=self.dropout, training=self.training)),
            egonet_batch["nodes_mapper"],
        )

        parts = [centroid_x, subgraph_x, context_x]
        selected = [parts[idx] for idx in self.embs]
        if self.embs_combine_mode == "add":
            out = selected[0]
            for part in selected[1:]:
                out = out + part
            return out
        return self.combine_proj(torch.cat(selected, dim=-1))


class GNNAKRootReadoutGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        num_edge_types: int,
        num_hops: int,
        inner_layers: int,
        hops_dim: int,
        embs: Tuple[int, ...],
        embs_combine_mode: str,
    ):
        super().__init__()
        self.dropout = dropout
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList(
            [
                GNNAKLayer(
                    hidden_channels=hidden_channels,
                    inner_layers=inner_layers,
                    num_hops=num_hops,
                    hops_dim=hops_dim,
                    dropout=dropout,
                    num_edge_types=num_edge_types,
                    embs=embs,
                    embs_combine_mode=embs_combine_mode,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        self.readout = MLP(hidden_channels, hidden_channels, 1, dropout=dropout, num_layers=2)

    def forward(self, data: Data) -> Tensor:
        x = self.input_proj(data.x)
        egonet_batch = self.layers[0].build_egonet_batch(data)
        prev = x
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, egonet_batch)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + prev
            prev = x
        root_x = x[data.root_index.view(-1)]
        return self.readout(root_x).view(-1)


class GCNGMTRegressor(nn.Module):
    def __init__(
        self,
        model_name: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        num_edge_types: int = 0,
        gmt_heads: int = 4,
        gmt_k: int = 8,
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.dropout = dropout
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.edge_encoder = (
            nn.Embedding(max(int(num_edge_types), 1), hidden_channels)
            if self.model_name == "gine_gmt"
            else None
        )
        self.convs = nn.ModuleList([self._build_conv(hidden_channels) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        self.gmt = GraphMultisetTransformer(
            channels=hidden_channels,
            k=gmt_k,
            heads=gmt_heads,
            layer_norm=True,
            dropout=dropout,
        )
        self.readout = MLP(hidden_channels, hidden_channels, 1, dropout=dropout, num_layers=2)

    def _build_conv(self, channels: int):
        if self.model_name == "gcn_gmt":
            return GCNConv(channels, channels)
        if self.model_name == "gine_gmt":
            return GINEConv(
                make_pyg_gine_mlp(channels, channels, channels, dropout=self.dropout, num_layers=2),
                edge_dim=channels,
            )
        raise ValueError(f"Unsupported GMT model: {self.model_name}")

    def forward(self, data: Data) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_proj(x)
        edge_attr: Optional[Tensor] = None
        if self.edge_encoder is not None:
            if data.edge_type.numel() == 0:
                edge_attr = x.new_zeros((0, x.size(1)))
            else:
                edge_attr = self.edge_encoder(data.edge_type)
        prev = x
        for conv, norm in zip(self.convs, self.norms):
            if edge_attr is None:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + prev
            prev = x
        pooled = self.gmt(x, batch)
        return self.readout(pooled).view(-1)


class SimpleGraphGPSLayer(nn.Module):
    def __init__(self, hidden_channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.local_conv = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ),
            edge_dim=hidden_channels,
        )
        self.global_attn = nn.MultiheadAttention(
            hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_local = nn.BatchNorm1d(hidden_channels)
        self.norm_attn = nn.BatchNorm1d(hidden_channels)
        self.norm_ff = nn.BatchNorm1d(hidden_channels)
        self.ff = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch_index: Tensor) -> Tensor:
        h_in = x
        h_local = self.local_conv(x, edge_index, edge_attr)
        h_local = self.norm_local(h_in + F.dropout(h_local, p=self.dropout, training=self.training))

        h_dense, mask = to_dense_batch(x, batch_index)
        mask = mask.to(torch.bool)
        h_attn_dense, _ = self.global_attn(
            h_dense,
            h_dense,
            h_dense,
            key_padding_mask=~mask,
            need_weights=False,
        )
        h_attn = h_attn_dense[mask]
        h_attn = self.norm_attn(h_in + F.dropout(h_attn, p=self.dropout, training=self.training))

        h = h_local + h_attn
        h = self.norm_ff(h + self.ff(h))
        return h


class GraphGPSVirtualNodeRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        num_heads: int,
        num_edge_types: int,
    ):
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by gps_heads ({num_heads})."
            )

        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.struct_encoder = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.edge_encoder = nn.Embedding(max(num_edge_types, 1), hidden_channels)
        self.virtual_node = nn.Embedding(1, hidden_channels)
        self.layers = nn.ModuleList(
            [SimpleGraphGPSLayer(hidden_channels, num_heads, dropout) for _ in range(num_layers)]
        )
        self.virtual_mlps = nn.ModuleList(
            [MLP(hidden_channels, hidden_channels, hidden_channels, dropout=dropout, num_layers=2) for _ in range(num_layers)]
        )
        self.readout = MLP(hidden_channels, hidden_channels, 1, dropout=dropout, num_layers=2)

    def forward(self, data: Data) -> Tensor:
        x = self.input_proj(data.x)
        struct_features = torch.stack(
            [
                torch.log1p(data.depth_to_root.float()),
                torch.log1p(data.topological_level.float()),
            ],
            dim=-1,
        )
        x = x + self.struct_encoder(struct_features)

        batch_index = data.batch
        num_graphs = int(data.num_graphs)
        virtual_node = self.virtual_node.weight[0].unsqueeze(0).expand(num_graphs, -1)
        edge_attr = self.edge_encoder(data.edge_type) if data.edge_type.numel() > 0 else x.new_zeros((0, x.size(1)))

        for layer, virtual_mlp in zip(self.layers, self.virtual_mlps):
            x = x + virtual_node[batch_index]
            x = layer(x, data.edge_index, edge_attr, batch_index)
            virtual_update = global_add_pool(x, batch_index)
            virtual_node = virtual_node + virtual_mlp(virtual_update)

        if not hasattr(data, "root_index"):
            raise ValueError("GraphGPSVirtualNodeRegressor requires root_index in the batch.")
        root_x = x[data.root_index.view(-1)]
        graph_repr = virtual_node + root_x
        return self.readout(graph_repr).view(-1)


class MOSWeightedGINBlock(nn.Module):
    """GIN block that consumes the scalar edge weights learned by MO."""

    def __init__(self, hidden_channels: int, dropout: float):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = float(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        aggregated = x.new_zeros(x.shape)
        if edge_index.numel() > 0:
            src, dst = edge_index
            messages = x.index_select(0, src) * edge_weight.view(-1, 1).to(x)
            aggregated.index_add_(0, dst, messages)
        out = self.mlp((1.0 + self.eps) * x + aggregated)
        out = self.norm(out)
        out = F.gelu(out)
        return F.dropout(out, p=self.dropout, training=self.training)


class CardBenchMOSGSLRegressor(nn.Module):
    """CardBench adaptation of MO/MOSGSL for cardinality regression.

    CardBench query graphs are small semantic units, so each complete query is
    treated as the local subgraph on which MO learns adjacency.  Edge-relation
    IDs are first lifted into incident-node context, because learned edges only
    carry scalar weights.  Class-specific motif learning is intentionally
    disabled: CardBench is continuous regression, not graph classification.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        num_edge_types: int,
        gsl_epsilon: float,
        gsl_lambda: float,
        metric_heads: int,
    ):
        super().__init__()
        if not 0.0 <= float(gsl_epsilon) <= 1.0:
            raise ValueError(f"mo_gsl_epsilon must be in [0, 1], got {gsl_epsilon}.")
        if not 0.0 <= float(gsl_lambda) <= 1.0:
            raise ValueError(f"mo_gsl_lambda must be in [0, 1], got {gsl_lambda}.")
        if int(metric_heads) <= 0:
            raise ValueError("mo_metric_heads must be positive.")

        self.hidden_channels = int(hidden_channels)
        self.gsl_epsilon = float(gsl_epsilon)
        self.gsl_lambda = float(gsl_lambda)
        self.gsl_enabled = True
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.edge_encoder = nn.Embedding(max(int(num_edge_types), 1), hidden_channels)
        self.relation_context = nn.Linear(2 * hidden_channels, hidden_channels)
        self.metric_encoder = MOSWeightedGINBlock(hidden_channels, dropout)
        self.metric_weight = nn.Parameter(torch.empty(int(metric_heads), hidden_channels))
        self.layers = nn.ModuleList(
            [MOSWeightedGINBlock(hidden_channels, dropout) for _ in range(int(num_layers))]
        )
        # The operation root is task-specific; the mean term retains the
        # whole-query information used by the original graph-level MO readout.
        self.readout = MLP(2 * hidden_channels, hidden_channels, 1, dropout=dropout, num_layers=2)
        nn.init.xavier_uniform_(self.metric_weight)

    def set_gsl_enabled(self, enabled: bool) -> None:
        self.gsl_enabled = bool(enabled)

    def _relation_aware_input(self, data: Data) -> Tensor:
        x = F.gelu(self.input_proj(data.x))
        if data.edge_index.numel() == 0:
            return x
        edge_type = getattr(data, "edge_type", None)
        if edge_type is None:
            raise ValueError("CardBench MO requires edge_type for relation-aware features.")
        relation = self.edge_encoder(edge_type)
        src, dst = data.edge_index
        incoming = x.new_zeros(x.shape)
        outgoing = x.new_zeros(x.shape)
        incoming.index_add_(0, dst, relation)
        outgoing.index_add_(0, src, relation)
        ones = x.new_ones(relation.size(0))
        in_count = x.new_zeros(x.size(0))
        out_count = x.new_zeros(x.size(0))
        in_count.index_add_(0, dst, ones)
        out_count.index_add_(0, src, ones)
        incoming = incoming / in_count.clamp_min(1.0).unsqueeze(1)
        outgoing = outgoing / out_count.clamp_min(1.0).unsqueeze(1)
        return x + self.relation_context(torch.cat([incoming, outgoing], dim=-1))

    def _learned_adjacency(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if not self.gsl_enabled:
            return edge_index, x.new_ones(edge_index.size(1))

        observed_weight = x.new_ones(edge_index.size(1))
        metric_x = x + self.metric_encoder(x, edge_index, observed_weight)
        dense_x, node_mask = to_dense_batch(metric_x, batch)
        # Multiple diagonal metric perspectives, matching MOSGSL's weighted
        # cosine learner while evaluating all graphs in the mini-batch at once.
        weighted = dense_x.unsqueeze(1) * self.metric_weight.unsqueeze(0).unsqueeze(2)
        weighted = F.normalize(weighted, p=2, dim=-1)
        similarity = torch.matmul(weighted, weighted.transpose(-1, -2)).mean(dim=1)
        similarity = similarity.clamp_min(0.0)
        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        diagonal = torch.eye(similarity.size(-1), dtype=torch.bool, device=x.device).unsqueeze(0)
        learned_mask = pair_mask & ~diagonal & (similarity > self.gsl_epsilon)
        learned_pos = torch.nonzero(learned_mask, as_tuple=False)

        if learned_pos.numel() > 0:
            graph_id, local_src, local_dst = learned_pos.unbind(dim=1)
            counts = torch.bincount(batch, minlength=dense_x.size(0))
            ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
            learned_src = ptr[graph_id] + local_src
            learned_dst = ptr[graph_id] + local_dst
            learned_edge_index = torch.stack([learned_src, learned_dst], dim=0)
            learned_weight = similarity[graph_id, local_src, local_dst] * self.gsl_lambda
        else:
            learned_edge_index = edge_index.new_empty((2, 0))
            learned_weight = x.new_empty(0)

        original_weight = x.new_full((edge_index.size(1),), 1.0 - self.gsl_lambda)
        combined_edge_index = torch.cat([edge_index, learned_edge_index], dim=1)
        combined_weight = torch.cat([original_weight, learned_weight], dim=0)
        return coalesce(
            combined_edge_index,
            combined_weight,
            num_nodes=x.size(0),
            reduce="sum",
        )

    def forward(self, data: Data) -> Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        x = self._relation_aware_input(data)
        edge_index, edge_weight = self._learned_adjacency(x, data.edge_index, batch)
        for layer in self.layers:
            x = x + layer(x, edge_index, edge_weight)
        root_x = x[data.root_index.view(-1)]
        global_x = global_mean_pool(x, batch)
        return self.readout(torch.cat([root_x, global_x], dim=-1)).view(-1)


def build_model(
    model_name: str,
    in_channels: int,
    hparams: Dict[str, Any],
    pna_deg: Optional[Tensor],
    num_edge_types: int,
) -> nn.Module:
    name = normalize_model_name(model_name)
    if name in {"gcn_root", "gin_root", "gine_root", "pna_root"}:
        return RootReadoutGNN(
            model_name=name,
            in_channels=in_channels,
            hidden_channels=int(hparams["hidden_channels"]),
            num_layers=int(hparams["num_layers"]),
            dropout=float(hparams["dropout"]),
            pna_deg=pna_deg,
            pna_towers=int(hparams.get("pna_towers", 1)),
            num_edge_types=num_edge_types,
        )
    if name == "hgp":
        return HGPGINRootRegressor(
            in_channels=in_channels,
            hidden_channels=int(hparams["hidden_channels"]),
            num_layers=int(hparams["num_layers"]),
            dropout=float(hparams["dropout"]),
            pool_ratio=float(hparams["hgp_pool_ratio"]),
            sl_topk=int(hparams["hgp_topk"]),
            sl_lambda=float(hparams["hgp_lambda"]),
            score_bias_ops=float(hparams["hgp_score_bias_ops"]),
            score_bias_predicates=float(hparams["hgp_score_bias_predicates"]),
        )
    if name == "gnnak_root":
        return GNNAKRootReadoutGNN(
            in_channels=in_channels,
            hidden_channels=int(hparams["hidden_channels"]),
            num_layers=int(hparams["num_layers"]),
            dropout=float(hparams["dropout"]),
            num_edge_types=num_edge_types,
            num_hops=int(hparams["gnnak_hops"]),
            inner_layers=int(hparams["gnnak_inner_layers"]),
            hops_dim=int(hparams["gnnak_hops_dim"]),
            embs=tuple(int(v) for v in hparams["gnnak_embs"]),
            embs_combine_mode=str(hparams["gnnak_embs_combine_mode"]),
        )
    if name in {"gcn_gmt", "gine_gmt"}:
        return GCNGMTRegressor(
            model_name=name,
            in_channels=in_channels,
            hidden_channels=int(hparams["hidden_channels"]),
            num_layers=int(hparams["num_layers"]),
            dropout=float(hparams["dropout"]),
            num_edge_types=num_edge_types,
            gmt_heads=int(hparams.get("gmt_heads", 4)),
            gmt_k=int(hparams.get("gmt_k", 8)),
        )
    if name == "graphgps_vn":
        return GraphGPSVirtualNodeRegressor(
            in_channels=in_channels,
            hidden_channels=int(hparams["hidden_channels"]),
            num_layers=int(hparams["num_layers"]),
            dropout=float(hparams["dropout"]),
            num_heads=int(hparams.get("gps_heads", 4)),
            num_edge_types=num_edge_types,
        )
    if name == "mo":
        return CardBenchMOSGSLRegressor(
            in_channels=in_channels,
            hidden_channels=int(hparams["hidden_channels"]),
            num_layers=int(hparams["num_layers"]),
            dropout=float(hparams["dropout"]),
            num_edge_types=num_edge_types,
            gsl_epsilon=float(hparams["mo_gsl_epsilon"]),
            gsl_lambda=float(hparams["mo_gsl_lambda"]),
            metric_heads=int(hparams["mo_metric_heads"]),
        )
    raise ValueError(f"Unknown model_name: {model_name}")


@dataclass
class PreparedSplit:
    raw_split: SplitRawGraphs
    feature_processor: CardBenchFeatureProcessor
    train_graphs: List[Data]
    val_graphs: List[Data]
    test_graphs: List[Data]
    pna_deg: Optional[Tensor]
    num_edge_types: int
    protected_edge_type_ids: Tuple[int, ...]


@dataclass
class TrainConfig:
    max_epochs: int
    early_stop: int
    batch_size: int
    lr: float
    weight_decay: float
    verbose_every: int
    use_aug: bool = False
    aug_edge_drop_p1: float = 0.0
    aug_feat_mask_p1: float = 0.0
    aug_edge_drop_p2: float = 0.0
    aug_feat_mask_p2: float = 0.0
    aug_prob_cap: float = 0.0
    contrast_weight: float = 0.0
    contrast_temperature: float = 0.5
    use_flag: bool = False
    flag_steps: int = 0
    flag_step_size: float = 0.0
    flag_ops_scale: float = 0.25
    flag_root_scale: float = 0.0


@dataclass
class RunResult:
    dataset_name: str
    base_name: str
    workload_type: str
    npz_path: str
    model_name: str
    seed: int
    best_epoch: int
    hparams: Dict[str, Any]
    best_val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    efficiency_metrics: Dict[str, float]
    y_true_raw: np.ndarray
    y_pred_raw: np.ndarray


@dataclass
class EvalOutput:
    metrics: Dict[str, float]
    y_true_raw: np.ndarray
    y_pred_raw: np.ndarray
    infer_time_seconds: float
    infer_max_memory_allocated_mb: float
    infer_max_memory_reserved_mb: float


def _cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _get_peak_memory_mb(device: torch.device) -> Tuple[float, float]:
    if device.type == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        return float(allocated), float(reserved)
    return 0.0, 0.0


def _reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def get_cardbench_protected_edge_type_ids(schema: Dict[str, Tuple[str, str]]) -> Tuple[int, ...]:
    protected_names = {"op_to_op", "pred_to_op", "attr_to_op"}
    edge_names = sorted(schema.keys())
    return tuple(idx for idx, edge_name in enumerate(edge_names) if edge_name in protected_names)


def _expand_group_reference_scores(
    scores: Tensor,
    group_index: Tensor,
    num_groups: int,
    base_mode: str,
) -> Tuple[Tensor, Tensor]:
    if base_mode not in {"mean", "min"}:
        raise ValueError(f"Unsupported base_mode: {base_mode}")
    device = scores.device
    max_scores = torch.zeros(num_groups, device=device, dtype=scores.dtype)
    base_scores = torch.zeros(num_groups, device=device, dtype=scores.dtype)
    for group_id in range(num_groups):
        mask = group_index == group_id
        if not torch.any(mask):
            continue
        current = scores[mask]
        max_scores[group_id] = torch.max(current)
        base_scores[group_id] = torch.mean(current) if base_mode == "mean" else torch.min(current)
    return max_scores[group_index], base_scores[group_index]


def adaptive_aug_cardbench(
    data: Data,
    x_emb: Tensor,
    p_e: float,
    p_f: float,
    p_tau: float,
    protected_edge_type_ids: Tuple[int, ...],
    p_corr: str = "min",
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    if p_corr not in {"mean", "min"}:
        raise ValueError(f"Unsupported p_corr: {p_corr}")
    device = x_emb.device
    num_nodes = int(data.num_nodes)
    node_batch = getattr(data, "batch", None)
    if node_batch is None:
        node_batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if data.edge_index.numel() > 0:
        degree_index = torch.cat([data.edge_index[0], data.edge_index[1]], dim=0)
        node_scores = torch.log(degree(degree_index, num_nodes=num_nodes, dtype=x_emb.dtype) + 1e-6)
    else:
        node_scores = torch.zeros(num_nodes, dtype=x_emb.dtype, device=device)

    new_edge_index = data.edge_index
    new_edge_type = getattr(data, "edge_type", None)
    if p_e > 0 and data.edge_index.numel() > 0:
        edge_scores = 0.5 * (node_scores[data.edge_index[0]] + node_scores[data.edge_index[1]])
        edge_batch = node_batch[data.edge_index[0]]
        edge_max, edge_base = _expand_group_reference_scores(
            edge_scores,
            edge_batch,
            int(node_batch.max().item()) + 1,
            p_corr,
        )
        edge_denom = torch.where(
            torch.abs(edge_max - edge_base) < 1e-6,
            torch.ones_like(edge_max),
            edge_max - edge_base,
        )
        prob_drop = ((edge_max - edge_scores) / edge_denom * p_e).clamp(0.0, p_tau)
        if protected_edge_type_ids:
            protected_mask = torch.zeros_like(prob_drop, dtype=torch.bool)
            for edge_type_id in protected_edge_type_ids:
                protected_mask |= data.edge_type == edge_type_id
            prob_drop = torch.where(protected_mask, torch.zeros_like(prob_drop), prob_drop)
        keep_mask = ~torch.bernoulli(prob_drop).to(torch.bool)
        if not torch.any(keep_mask):
            keep_mask = torch.ones_like(keep_mask, dtype=torch.bool)
        new_edge_index = data.edge_index[:, keep_mask]
        if new_edge_type is not None:
            new_edge_type = new_edge_type[keep_mask]

    new_x_emb = x_emb
    if p_f > 0 and num_nodes > 0:
        node_max, node_base = _expand_group_reference_scores(
            node_scores,
            node_batch,
            int(node_batch.max().item()) + 1,
            p_corr,
        )
        node_denom = torch.where(
            torch.abs(node_max - node_base) < 1e-6,
            torch.ones_like(node_max),
            node_max - node_base,
        )
        prob_mask = ((node_max - node_scores) / node_denom * p_f).clamp(0.0, p_tau)
        prob_mask[data.root_index.view(-1)] = 0.0
        mask_indices = torch.bernoulli(prob_mask).to(torch.bool)
        new_x_emb = x_emb.clone()
        new_x_emb[mask_indices] = 0.0

    return new_edge_index, new_edge_type, new_x_emb


def compute_nt_xent_loss(z1: Tensor, z2: Tensor, temperature: float = 0.5) -> Tensor:
    batch_size = z1.size(0)
    if batch_size == 0:
        return z1.new_tensor(0.0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    out = torch.cat([z1, z2], dim=0)
    logits = torch.mm(out, out.t()) / temperature
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=out.device)
    logits = logits.masked_fill(mask, -1e9)
    labels = torch.cat(
        [
            torch.arange(batch_size, device=out.device) + batch_size,
            torch.arange(batch_size, device=out.device),
        ],
        dim=0,
    )
    return F.cross_entropy(logits, labels)


def build_cardbench_flag_scale(
    data: Data,
    reference: Tensor,
    ops_scale: float,
    root_scale: float,
) -> Tensor:
    scale = torch.ones((reference.size(0), 1), dtype=reference.dtype, device=reference.device)
    ops_mask = data.node_type_id == NODE_TYPES.index("ops")
    scale[ops_mask] = float(ops_scale)
    scale[data.root_index.view(-1)] = float(root_scale)
    return scale


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    feature_processor: CardBenchFeatureProcessor,
) -> EvalOutput:
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    y_true_raw_list: List[Tensor] = []
    y_pred_raw_list: List[Tensor] = []
    t_forward_sum = 0.0

    _reset_peak_memory_stats(device)
    for batch in loader:
        batch = batch.to(device)
        _cuda_synchronize(device)
        t0 = time.perf_counter()
        pred_norm = model(batch)
        _cuda_synchronize(device)
        t_forward_sum += time.perf_counter() - t0

        target_norm = batch.y.view(-1)
        loss = F.mse_loss(pred_norm, target_norm, reduction="sum")
        total_loss += float(loss.item())
        total_graphs += batch.num_graphs
        y_pred_raw = feature_processor.inverse_label_transform(pred_norm)
        y_true_raw_list.append(batch.y_raw.view(-1).cpu())
        y_pred_raw_list.append(y_pred_raw.cpu())

    infer_alloc_mb, infer_reserved_mb = _get_peak_memory_mb(device)
    y_true_raw = torch.cat(y_true_raw_list, dim=0).numpy() if y_true_raw_list else np.empty((0,), dtype=np.float32)
    y_pred_raw = torch.cat(y_pred_raw_list, dim=0).numpy() if y_pred_raw_list else np.empty((0,), dtype=np.float32)
    metrics = compute_metrics(
        y_true_raw=y_true_raw,
        y_pred_raw=y_pred_raw,
        norm_mse=total_loss / max(total_graphs, 1),
        elapsed_seconds=t_forward_sum,
    )
    return EvalOutput(
        metrics=metrics,
        y_true_raw=y_true_raw,
        y_pred_raw=y_pred_raw,
        infer_time_seconds=t_forward_sum,
        infer_max_memory_allocated_mb=infer_alloc_mb,
        infer_max_memory_reserved_mb=infer_reserved_mb,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_config: TrainConfig,
    protected_edge_type_ids: Tuple[int, ...],
) -> Tuple[float, float, float, float]:
    model.train()
    total_loss = 0.0
    total_graphs = 0
    train_time_sum = 0.0

    _reset_peak_memory_stats(device)
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        _cuda_synchronize(device)
        t0 = time.perf_counter()
        if (train_config.use_aug or train_config.use_flag) and (
            not isinstance(model, RootReadoutGNN) or not supports_cardbench_root_augmentation(model.model_name)
        ):
            raise ValueError(
                "CardBench augmentation currently supports only gin_root/gine_root root-readout backbones."
            )
        pred = model(batch)
        target = batch.y.view(-1)
        loss = F.mse_loss(pred, target)
        loss.backward()
        batch_loss_value = float(loss.item()) * batch.num_graphs
        if train_config.use_flag:
            with torch.no_grad():
                projected_x_flag = model.project_inputs(batch.x)
            flag_base = projected_x_flag.detach().clone()
            perturb_scale = build_cardbench_flag_scale(
                batch,
                flag_base,
                ops_scale=train_config.flag_ops_scale,
                root_scale=train_config.flag_root_scale,
            )
            perturb = torch.empty_like(flag_base).uniform_(-train_config.flag_step_size, train_config.flag_step_size)
            perturb = (perturb * perturb_scale).detach()
            perturb.requires_grad_(True)
            for _ in range(train_config.flag_steps):
                pred_flag_temp, _ = model.forward_projected(
                    batch,
                    flag_base + perturb,
                    edge_index=batch.edge_index,
                    return_graph_embedding=True,
                )
                loss_flag_temp = F.mse_loss(pred_flag_temp, target)
                grad_p = torch.autograd.grad(loss_flag_temp, perturb)[0]
                perturb = (perturb + train_config.flag_step_size * torch.sign(grad_p)).detach()
                perturb = perturb * perturb_scale
                perturb.requires_grad_(True)
            pred_flag, _ = model.forward_projected(
                batch,
                flag_base + perturb,
                edge_index=batch.edge_index,
                return_graph_embedding=True,
            )
            loss_flag = F.mse_loss(pred_flag, target)
            loss_flag.backward()
            batch_loss_value += float(loss_flag.item()) * batch.num_graphs
        if train_config.use_aug:
            projected_x_aug = model.project_inputs(batch.x)
            edge_index_1, edge_type_1, x_aug_1 = adaptive_aug_cardbench(
                batch,
                projected_x_aug,
                train_config.aug_edge_drop_p1,
                train_config.aug_feat_mask_p1,
                train_config.aug_prob_cap,
                protected_edge_type_ids,
            )
            edge_index_2, edge_type_2, x_aug_2 = adaptive_aug_cardbench(
                batch,
                projected_x_aug,
                train_config.aug_edge_drop_p2,
                train_config.aug_feat_mask_p2,
                train_config.aug_prob_cap,
                protected_edge_type_ids,
            )
            _, graph_emb_1 = model.forward_projected(
                batch,
                x_aug_1,
                edge_index=edge_index_1,
                edge_type=edge_type_1,
                return_graph_embedding=True,
            )
            _, graph_emb_2 = model.forward_projected(
                batch,
                x_aug_2,
                edge_index=edge_index_2,
                edge_type=edge_type_2,
                return_graph_embedding=True,
            )
            loss_cl = compute_nt_xent_loss(
                graph_emb_1,
                graph_emb_2,
                temperature=train_config.contrast_temperature,
            ) * train_config.contrast_weight
            loss_cl.backward()
            batch_loss_value += float(loss_cl.item()) * batch.num_graphs
        optimizer.step()
        _cuda_synchronize(device)
        train_time_sum += time.perf_counter() - t0

        total_loss += batch_loss_value
        total_graphs += batch.num_graphs

    peak_alloc_mb, peak_reserved_mb = _get_peak_memory_mb(device)
    return total_loss / max(total_graphs, 1), train_time_sum, peak_alloc_mb, peak_reserved_mb


def compute_metrics(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    norm_mse: float,
    elapsed_seconds: float,
) -> Dict[str, float]:
    eps = 1e-9
    if y_true_raw.size == 0:
        return {
            "norm_mse": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "mean_q_error": float("nan"),
            "median_q_error": float("nan"),
            "p90_q_error": float("nan"),
            "p95_q_error": float("nan"),
            "max_q_error": float("nan"),
            "avg_inference_ms_per_graph": float("nan"),
        }
    abs_error = np.abs(y_pred_raw - y_true_raw)
    sq_error = np.square(y_pred_raw - y_true_raw)
    q_error = np.maximum(
        y_pred_raw / np.clip(y_true_raw, eps, None),
        y_true_raw / np.clip(y_pred_raw, eps, None),
    )
    denom = float(np.sum(np.square(y_true_raw - np.mean(y_true_raw))))
    r2 = float("nan") if denom <= 0 else 1.0 - float(np.sum(sq_error) / denom)
    return {
        "norm_mse": float(norm_mse),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(sq_error))),
        "r2": float(r2),
        "mean_q_error": float(np.mean(q_error)),
        "median_q_error": float(np.median(q_error)),
        "p90_q_error": float(np.quantile(q_error, 0.90)),
        "p95_q_error": float(np.quantile(q_error, 0.95)),
        "max_q_error": float(np.max(q_error)),
        "avg_inference_ms_per_graph": float(1000.0 * elapsed_seconds / max(len(y_true_raw), 1)),
    }


def prepare_split(
    raw_dataset: RawGraphDataset,
    seed: int,
    holdout_test_size: int,
    cap_trainval: Optional[int],
    debug: bool = False,
) -> PreparedSplit:
    usable_graphs = [g for g in raw_dataset.graphs if is_usable_query_graph(g)]
    raw_split = official_instance_split(
        usable_graphs,
        seed=seed,
        holdout_test_size=holdout_test_size,
        cap_trainval=cap_trainval,
    )
    processor = CardBenchFeatureProcessor(debug=debug)
    processor.fit(raw_split.train)

    train_graphs = [processor.transform_graph(g, raw_dataset.schema) for g in raw_split.train]
    val_graphs = [processor.transform_graph(g, raw_dataset.schema) for g in raw_split.val]
    test_graphs = [processor.transform_graph(g, raw_dataset.schema) for g in raw_split.test]
    train_graphs = [g for g in train_graphs if g is not None]
    val_graphs = [g for g in val_graphs if g is not None]
    test_graphs = [g for g in test_graphs if g is not None]
    if not train_graphs or not val_graphs or not test_graphs:
        raise RuntimeError("Prepared split is empty after transformation.")

    pna_deg = compute_pna_degree_histogram(train_graphs)
    return PreparedSplit(
        raw_split=raw_split,
        feature_processor=processor,
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        test_graphs=test_graphs,
        pna_deg=pna_deg,
        num_edge_types=len(raw_dataset.schema),
        protected_edge_type_ids=get_cardbench_protected_edge_type_ids(raw_dataset.schema),
    )


def compute_pna_degree_histogram(train_graphs: Sequence[Data]) -> Optional[Tensor]:
    max_degree = 0
    total_nodes = 0
    for graph in train_graphs:
        total_nodes += int(graph.num_nodes)
        if graph.edge_index.numel() == 0:
            continue
        deg = degree(graph.edge_index[1], num_nodes=graph.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(deg.max().item()))
    if max_degree == 0:
        return torch.tensor([total_nodes], dtype=torch.long)
    deg_hist = torch.zeros(max_degree + 1, dtype=torch.long)
    for graph in train_graphs:
        if graph.edge_index.numel() == 0:
            deg_hist[0] += int(graph.num_nodes)
            continue
        deg = degree(graph.edge_index[1], num_nodes=graph.num_nodes, dtype=torch.long)
        deg_hist += torch.bincount(deg, minlength=deg_hist.numel())
    return deg_hist


def candidate_batch_sizes_for_model(model_name: str, args: argparse.Namespace) -> Tuple[int, ...]:
    return (int(args.batch_size),)


def assert_no_singleton_training_last_batch(
    dataset_name: str,
    split_name: str,
    num_graphs: int,
    batch_sizes: Sequence[int],
) -> None:
    bad_batch_sizes = sorted({int(batch_size) for batch_size in batch_sizes if int(batch_size) > 0 and num_graphs % int(batch_size) == 1})
    if not bad_batch_sizes:
        return
    raise ValueError(
        f"{dataset_name} {split_name} split has {num_graphs} graphs; batch_size(s) {bad_batch_sizes} would create a last batch "
        "of size 1. This codebase uses BatchNorm1d in training-mode model components, so singleton training batches are rejected."
    )


def make_dataloader(graphs: Sequence[Data], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ListDataset(graphs), batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)


def maybe_run_mo_backbone_pretrain(
    model: nn.Module,
    hparams: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    feature_processor: CardBenchFeatureProcessor,
    train_config: TrainConfig,
    protected_edge_type_ids: Tuple[int, ...],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    verbose: bool,
    enabled: bool = True,
    return_epoch_times: bool = False,
) -> Tuple[torch.optim.Optimizer, float, float, float, int]:
    """Warm up MO on the observed graph before enabling learned adjacency.

    This is the CardBench equivalent of MOSGSL's backbone-only pretraining.
    Validation chooses the warm-up checkpoint; test data are never touched.
    Fine-tuning from a zero-shot MO checkpoint sets ``enabled=False`` because
    both its backbone and structure learner are already initialized.
    """
    if not isinstance(model, CardBenchMOSGSLRegressor):
        result = (optimizer, 0.0, 0.0, 0.0, 0)
        return (*result, []) if return_epoch_times else result
    pretrain_epochs = int(hparams.get("mo_pretrain_epochs", 0)) if enabled else 0
    if pretrain_epochs <= 0:
        model.set_gsl_enabled(True)
        result = (optimizer, 0.0, 0.0, 0.0, 0)
        return (*result, []) if return_epoch_times else result

    model.set_gsl_enabled(False)
    best_score = float("inf")
    best_state: Optional[Dict[str, Tensor]] = None
    total_time = 0.0
    max_alloc = 0.0
    max_reserved = 0.0
    cycle_times: List[float] = []
    for epoch in range(1, pretrain_epochs + 1):
        train_loss, epoch_time, alloc_mb, reserved_mb = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train_config,
            protected_edge_type_ids,
        )
        total_time += epoch_time
        max_alloc = max(max_alloc, alloc_mb)
        max_reserved = max(max_reserved, reserved_mb)
        val_output = evaluate(model, val_loader, device, feature_processor)
        val_metrics = val_output.metrics
        cycle_times.append(float(epoch_time + val_output.infer_time_seconds))
        score = float(val_metrics["median_q_error"])
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch == 1 or epoch == pretrain_epochs or epoch % 5 == 0):
            print(
                f"MO backbone pretrain {epoch:03d}/{pretrain_epochs} | "
                f"train_norm_mse={train_loss:.6f} | val_median_q={score:.6f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("MO backbone pretraining did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    model.set_gsl_enabled(True)
    # MOSGSL starts its learned-structure stage with fresh optimizer moments.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(hparams.get("lr", train_config.lr)),
        weight_decay=float(hparams.get("weight_decay", train_config.weight_decay)),
    )
    result = (optimizer, total_time, max_alloc, max_reserved, pretrain_epochs)
    return (*result, cycle_times) if return_epoch_times else result


def train_with_early_stopping(
    prepared: PreparedSplit,
    model_name: str,
    hparams: Dict[str, Any],
    train_config: TrainConfig,
    device: torch.device,
    verbose: bool,
    seed: int,
    trial: Optional["optuna.Trial"] = None,
    evaluate_test: bool = True,
) -> Tuple[int, Dict[str, float], Dict[str, float], Dict[str, float], np.ndarray, np.ndarray]:
    seed_everything(seed)
    if (train_config.use_aug or train_config.use_flag) and not supports_cardbench_root_augmentation(model_name):
        raise ValueError(
            "CardBench augmentation is currently supported only for gin_root/gine_root root-readout models."
        )
    model = build_model(
        model_name,
        prepared.train_graphs[0].x.size(1),
        hparams,
        prepared.pna_deg,
        prepared.num_edge_types,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(hparams.get("lr", train_config.lr)),
        weight_decay=float(hparams.get("weight_decay", train_config.weight_decay)),
    )

    batch_size = int(hparams.get("batch_size", train_config.batch_size))
    train_loader = make_dataloader(prepared.train_graphs, batch_size, shuffle=True, seed=seed)
    val_loader = make_dataloader(prepared.val_graphs, batch_size, shuffle=False, seed=seed)
    test_loader = (
        make_dataloader(prepared.test_graphs, batch_size, shuffle=False, seed=seed)
        if evaluate_test
        else None
    )

    optimizer, pre_time, pre_alloc, pre_reserved, pre_epochs = maybe_run_mo_backbone_pretrain(
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
        enabled=True,
    )

    best_val_score = float("inf")
    best_epoch = -1
    best_state: Optional[Dict[str, Tensor]] = None
    best_val_metrics: Optional[Dict[str, float]] = None
    bad_epochs = 0

    total_train_time = pre_time
    max_train_alloc_mb = pre_alloc
    max_train_reserved_mb = pre_reserved
    num_epochs_trained = pre_epochs

    for epoch in range(1, train_config.max_epochs + 1):
        train_loss, train_epoch_time, train_alloc_mb, train_reserved_mb = train_one_epoch(
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

        val_eval = evaluate(model, val_loader, device, prepared.feature_processor)
        val_metrics = val_eval.metrics
        val_score = val_metrics["median_q_error"]
        if val_score < best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = dict(val_metrics)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if trial is not None:
            trial.report(float(val_score), step=epoch)
            if trial.should_prune():
                del model, optimizer, train_loader, val_loader
                if test_loader is not None:
                    del test_loader
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()

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
    test_eval = (
        evaluate(model, test_loader, device, prepared.feature_processor)
        if test_loader is not None
        else None
    )
    efficiency_metrics = {
        "num_epochs": float(num_epochs_trained),
        "t_train_total": float(total_train_time),
        "t_train_epoch_avg": float(total_train_time / max(num_epochs_trained, 1)),
        "train_max_memory_allocated_mb": float(max_train_alloc_mb),
        "train_max_memory_reserved_mb": float(max_train_reserved_mb),
        "t_infer": float(test_eval.infer_time_seconds) if test_eval is not None else 0.0,
        "infer_max_memory_allocated_mb": float(test_eval.infer_max_memory_allocated_mb) if test_eval is not None else 0.0,
        "infer_max_memory_reserved_mb": float(test_eval.infer_max_memory_reserved_mb) if test_eval is not None else 0.0,
    }

    del model, optimizer, train_loader, val_loader
    if test_loader is not None:
        del test_loader
    torch.cuda.empty_cache()

    if test_eval is None:
        return (
            best_epoch,
            best_val_metrics,
            {},
            efficiency_metrics,
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return best_epoch, best_val_metrics, test_eval.metrics, efficiency_metrics, test_eval.y_true_raw, test_eval.y_pred_raw


def default_hparams_from_args(args: argparse.Namespace, model_name: str) -> Dict[str, Any]:
    name = normalize_model_name(model_name)
    resolved_use_aug, resolved_use_flag = resolve_requested_aug_flags(model_name, args.use_aug, args.use_flag)
    hp = {
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
    }
    if name == "pna_root":
        hp["pna_towers"] = args.pna_towers
    if name in {"gcn_gmt", "gine_gmt"}:
        hp["gmt_heads"] = args.gmt_heads
        hp["gmt_k"] = args.gmt_k
    if name == "graphgps_vn":
        hp["gps_heads"] = args.gps_heads
    if name == "mo":
        hp["mo_gsl_epsilon"] = args.mo_gsl_epsilon
        hp["mo_gsl_lambda"] = args.mo_gsl_lambda
        hp["mo_metric_heads"] = args.mo_metric_heads
        hp["mo_pretrain_epochs"] = args.mo_pretrain_epochs
    if name == "hgp":
        hp["hgp_pool_ratio"] = args.hgp_pool_ratio
        hp["hgp_topk"] = args.hgp_topk
        hp["hgp_lambda"] = args.hgp_lambda
        hp["hgp_score_bias_ops"] = args.hgp_score_bias_ops
        hp["hgp_score_bias_predicates"] = args.hgp_score_bias_predicates
    if name == "gnnak_root":
        hp["gnnak_hops"] = args.gnnak_hops
        hp["gnnak_inner_layers"] = args.gnnak_inner_layers
        hp["gnnak_hops_dim"] = args.gnnak_hops_dim
        hp["gnnak_embs"] = parse_int_tuple_arg(args.gnnak_embs)
        hp["gnnak_embs_combine_mode"] = args.gnnak_embs_combine_mode
    if supports_cardbench_root_augmentation(model_name):
        hp["use_aug"] = resolved_use_aug
        hp["aug_edge_drop_p1"] = args.aug_edge_drop_p1
        hp["aug_feat_mask_p1"] = args.aug_feat_mask_p1
        hp["aug_edge_drop_p2"] = args.aug_edge_drop_p2
        hp["aug_feat_mask_p2"] = args.aug_feat_mask_p2
        hp["aug_prob_cap"] = args.aug_prob_cap
        hp["contrast_weight"] = args.contrast_weight
        hp["contrast_temperature"] = args.contrast_temperature
        hp["use_flag"] = resolved_use_flag
        hp["flag_steps"] = args.flag_steps
        hp["flag_step_size"] = args.flag_step_size
        hp["flag_ops_scale"] = args.flag_ops_scale
        hp["flag_root_scale"] = args.flag_root_scale
    return hp


def sample_hparams(trial: "optuna.Trial", model_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    name = normalize_model_name(model_name)
    resolved_use_aug, resolved_use_flag = resolve_requested_aug_flags(model_name, args.use_aug, args.use_flag)
    if name == "mo":
        # Five MO trials are too few for a broad architecture search.  Fix the
        # robust CardBench architecture and tune only optimization plus the
        # learned/original adjacency trade-off.
        return {
            "hidden_channels": int(args.hidden_channels),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "lr": trial.suggest_categorical("lr", [3e-4, 5e-4, 1e-3]),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "mo_gsl_epsilon": trial.suggest_categorical(
                "mo_gsl_epsilon", [0.75, 0.85, 0.90]
            ),
            "mo_gsl_lambda": trial.suggest_categorical(
                "mo_gsl_lambda", [0.20, 0.35, 0.50]
            ),
            "mo_metric_heads": int(args.mo_metric_heads),
            "mo_pretrain_epochs": int(args.mo_pretrain_epochs),
        }
    if is_gca_model_variant(model_name) and bool(getattr(args, "compact_hpo", False)):
        # CardBench instance/transfer runs consistently favored a moderately
        # deep, wide GIN backbone.  With only ten trials, keep the augmentation
        # recipe fixed and spend the budget on the architecture/optimizer
        # choices that materially changed source-validation MedQ.
        hp = {
            "hidden_channels": trial.suggest_categorical("hidden_channels", [128, 256]),
            "num_layers": trial.suggest_categorical("num_layers", [3, 4]),
            "dropout": trial.suggest_categorical("dropout", [0.0, 0.05, 0.1, 0.2]),
            "lr": trial.suggest_categorical("lr", [1e-4, 2e-4, 3e-4, 5e-4, 1e-3]),
            "weight_decay": trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 3e-4]),
            "batch_size": int(args.batch_size),
        }
        hp.update(
            {
                "use_aug": resolved_use_aug,
                "aug_edge_drop_p1": args.aug_edge_drop_p1,
                "aug_feat_mask_p1": args.aug_feat_mask_p1,
                "aug_edge_drop_p2": args.aug_edge_drop_p2,
                "aug_feat_mask_p2": args.aug_feat_mask_p2,
                "aug_prob_cap": args.aug_prob_cap,
                "contrast_weight": args.contrast_weight,
                "contrast_temperature": args.contrast_temperature,
                "use_flag": resolved_use_flag,
                "flag_steps": args.flag_steps,
                "flag_step_size": args.flag_step_size,
                "flag_ops_scale": args.flag_ops_scale,
                "flag_root_scale": args.flag_root_scale,
            }
        )
        return hp
    hp = {
        "hidden_channels": trial.suggest_categorical("hidden_channels", [64, 128, 256]),
        "num_layers": trial.suggest_categorical("num_layers", [2, 3, 4]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": args.batch_size,
    }
    if name == "pna_root":
        hp["pna_towers"] = trial.suggest_categorical("pna_towers", [1, 2, 4])
    if name in {"gcn_gmt", "gine_gmt"}:
        hp["gmt_heads"] = trial.suggest_categorical("gmt_heads", [2, 4, 8])
        hp["gmt_k"] = trial.suggest_categorical("gmt_k", [1, 4, 8])
    if name == "graphgps_vn":
        hp["gps_heads"] = trial.suggest_categorical("gps_heads", [2, 4, 8])
    if name == "hgp":
        hp["hgp_pool_ratio"] = trial.suggest_categorical("hgp_pool_ratio", [0.6, 0.7, 0.8, 0.9])
        hp["hgp_topk"] = trial.suggest_categorical("hgp_topk", [2, 4, 6])
        hp["hgp_lambda"] = trial.suggest_categorical("hgp_lambda", [0.5, 1.0, 2.0])
        hp["hgp_score_bias_ops"] = trial.suggest_categorical("hgp_score_bias_ops", [0.25, 0.5, 1.0])
        hp["hgp_score_bias_predicates"] = trial.suggest_categorical("hgp_score_bias_predicates", [0.1, 0.25, 0.5])
    if name == "gnnak_root":
        hp["gnnak_hops"] = trial.suggest_categorical("gnnak_hops", [1, 2, 3])
        hp["gnnak_inner_layers"] = trial.suggest_categorical("gnnak_inner_layers", [1, 2])
        hp["gnnak_hops_dim"] = trial.suggest_categorical("gnnak_hops_dim", [0, 16, 32])
        hp["gnnak_embs"] = parse_int_tuple_arg(
            trial.suggest_categorical("gnnak_embs", ["0", "0,1", "0,2", "0,1,2"])
        )
        hp["gnnak_embs_combine_mode"] = trial.suggest_categorical(
            "gnnak_embs_combine_mode", ["add", "concat"]
        )
    if supports_cardbench_root_augmentation(model_name):
        hp["use_aug"] = resolved_use_aug
        hp["aug_edge_drop_p1"] = args.aug_edge_drop_p1
        hp["aug_feat_mask_p1"] = args.aug_feat_mask_p1
        hp["aug_edge_drop_p2"] = args.aug_edge_drop_p2
        hp["aug_feat_mask_p2"] = args.aug_feat_mask_p2
        hp["aug_prob_cap"] = args.aug_prob_cap
        hp["contrast_weight"] = args.contrast_weight
        hp["contrast_temperature"] = args.contrast_temperature
        hp["use_flag"] = resolved_use_flag
        hp["flag_steps"] = args.flag_steps
        hp["flag_step_size"] = args.flag_step_size
        hp["flag_ops_scale"] = args.flag_ops_scale
        hp["flag_root_scale"] = args.flag_root_scale
    return hp


def tune_hparams(
    prepared: PreparedSplit,
    model_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    if optuna is None:
        raise SystemExit("Optuna is required when n_trials > 0. Install with `pip install optuna`.")
    resolved_use_aug, resolved_use_flag = resolve_requested_aug_flags(model_name, args.use_aug, args.use_flag)

    def objective(trial: "optuna.Trial") -> float:
        hparams = sample_hparams(trial, model_name, args)
        config = TrainConfig(
            max_epochs=args.hpo_max_epochs,
            early_stop=args.hpo_early_stop,
            batch_size=int(hparams["batch_size"]),
            lr=float(hparams["lr"]),
            weight_decay=float(hparams["weight_decay"]),
            verbose_every=max(1, min(args.verbose_every, args.hpo_max_epochs)),
            use_aug=resolved_use_aug,
            aug_edge_drop_p1=args.aug_edge_drop_p1,
            aug_feat_mask_p1=args.aug_feat_mask_p1,
            aug_edge_drop_p2=args.aug_edge_drop_p2,
            aug_feat_mask_p2=args.aug_feat_mask_p2,
            aug_prob_cap=args.aug_prob_cap,
            contrast_weight=args.contrast_weight,
            contrast_temperature=args.contrast_temperature,
            use_flag=resolved_use_flag,
            flag_steps=args.flag_steps,
            flag_step_size=args.flag_step_size,
            flag_ops_scale=args.flag_ops_scale,
            flag_root_scale=args.flag_root_scale,
        )
        print(f"[HPO] Trial {trial.number + 1}/{args.n_trials} | model={model_name} | seed={args.first_seed}", flush=True)
        _, best_val_metrics, _, _, _, _ = train_with_early_stopping(
            prepared=prepared,
            model_name=model_name,
            hparams=hparams,
            train_config=config,
            device=device,
            verbose=True,
            seed=args.first_seed,
            trial=trial,
            evaluate_test=False,
        )
        score = float(best_val_metrics["median_q_error"])
        trial.set_user_attr("best_val_metrics", best_val_metrics)
        return score

    is_mo = normalize_model_name(model_name) == "mo"
    is_compact_gca = is_gca_model_variant(model_name) and bool(getattr(args, "compact_hpo", False))
    is_small_budget_special = is_mo or is_compact_gca
    sampler = optuna.samplers.TPESampler(
        seed=args.first_seed,
        n_startup_trials=min(2, max(1, args.n_trials)) if is_small_budget_special else 10,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(2 if is_small_budget_special else 5, max(1, args.n_trials)),
        n_warmup_steps=10,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    if is_mo:
        study.enqueue_trial({
            "lr": 5e-4,
            "mo_gsl_epsilon": 0.85,
            "mo_gsl_lambda": 0.35,
        })
    elif is_compact_gca:
        study.enqueue_trial({
            "hidden_channels": 256,
            "num_layers": 4,
            "dropout": 0.05,
            "lr": 2e-4,
            "weight_decay": 1e-4,
        })
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    best = sample_hparams(optuna.trial.FixedTrial(study.best_trial.params), model_name, args)
    return best


def metrics_to_row(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def summarize_metric_values(values: Sequence[float]) -> Dict[str, float]:
    values = [float(v) for v in values]
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": float(mean(values)), "std": float(pstdev(values))}


def aggregate_runs_by_dataset(results: Sequence[RunResult]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[RunResult]] = defaultdict(list)
    for r in results:
        grouped[(r.dataset_name, r.npz_path)].append(r)
    rows: List[Dict[str, Any]] = []
    test_metric_names = list(results[0].test_metrics.keys()) if results else []
    efficiency_metric_names = list(results[0].efficiency_metrics.keys()) if results else []
    for (dataset_name, npz_path), runs in sorted(grouped.items()):
        workload_type = dataset_identity_from_path(npz_path)[2]
        base_name = dataset_identity_from_path(npz_path)[1]
        row: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "base_name": base_name,
            "workload_type": workload_type,
            "npz_path": npz_path,
            "n_runs": len(runs),
        }
        for metric_name in test_metric_names:
            stats = summarize_metric_values([r.test_metrics[metric_name] for r in runs])
            row[f"test_{metric_name}_mean"] = stats["mean"]
            row[f"test_{metric_name}_std"] = stats["std"]
        for metric_name in efficiency_metric_names:
            stats = summarize_metric_values([r.efficiency_metrics[metric_name] for r in runs])
            row[f"eff_{metric_name}_mean"] = stats["mean"]
            row[f"eff_{metric_name}_std"] = stats["std"]
        rows.append(row)
    return rows


def aggregate_family_macro(dataset_rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not dataset_rows:
        return {}
    keys = [
        k for k in dataset_rows[0].keys()
        if k.endswith("_mean") and (k.startswith("test_") or k.startswith("eff_"))
    ]
    out: Dict[str, float] = {}
    for key in keys:
        metric_name = key[:-5]
        out[f"macro_{metric_name}"] = float(np.mean([float(r[key]) for r in dataset_rows]))
    return out


def aggregate_family_micro(results: Sequence[RunResult]) -> Dict[str, float]:
    if not results:
        return {}
    y_true = np.concatenate([r.y_true_raw for r in results], axis=0)
    y_pred = np.concatenate([r.y_pred_raw for r in results], axis=0)
    return {f"micro_{k}": v for k, v in compute_metrics(y_true, y_pred, norm_mse=float("nan"), elapsed_seconds=float("nan")).items() if k != "avg_inference_ms_per_graph"}


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(title)
    for key in [
        "norm_mse",
        "mae",
        "rmse",
        "r2",
        "mean_q_error",
        "median_q_error",
        "p90_q_error",
        "p95_q_error",
        "max_q_error",
        "avg_inference_ms_per_graph",
    ]:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.6f}")


def print_efficiency_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(title)
    for key in [
        "num_epochs",
        "t_train_total",
        "t_train_epoch_avg",
        "train_max_memory_allocated_mb",
        "train_max_memory_reserved_mb",
        "t_infer",
        "infer_max_memory_allocated_mb",
        "infer_max_memory_reserved_mb",
    ]:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.6f}")


def summarize_cardbench_dataset(raw_dataset: RawGraphDataset) -> None:
    graphs = raw_dataset.graphs
    schema = raw_dataset.schema
    print("Raw dataset summary")
    print(f"  dataset_name: {raw_dataset.name}")
    print(f"  base_name: {raw_dataset.base_name}")
    print(f"  workload_type: {raw_dataset.workload_type}")
    print(f"  npz_path: {raw_dataset.npz_path}")
    print(f"  num_graphs: {len(graphs)}")
    print(f"  schema_edges: {sorted(schema.keys())}")
    first = graphs[0]
    print(f"  node_types: {list(first.nodes.keys())}")
    for node_type in first.nodes.keys():
        print(f"  node_features[{node_type}]: {list(first.nodes[node_type].keys())}")
    print(f"  edge_types: {list(first.edges.keys())}")
    cards = np.asarray([float(g.nodes[GRAPH_NODE_TYPE]['cardinality'][0]) for g in graphs], dtype=np.float64)
    print(f"  min_cardinality: {np.min(cards):.3f}")
    print(f"  median_cardinality: {np.median(cards):.3f}")
    print(f"  max_cardinality: {np.max(cards):.3f}")


def summarize_pyg_graphs(graphs: Sequence[Data], prefix: str) -> None:
    num_nodes = [int(g.num_nodes) for g in graphs]
    num_edges = [int(g.edge_index.size(1)) for g in graphs]
    y_raw = np.asarray([float(g.y_raw.item()) for g in graphs], dtype=np.float64)
    print(f"{prefix} summary")
    print(f"  num_graphs: {len(graphs)}")
    print(f"  avg_nodes_per_graph: {np.mean(num_nodes):.3f}")
    print(f"  avg_edges_per_graph: {np.mean(num_edges):.3f}")
    print(f"  min_cardinality: {np.min(y_raw):.3f}")
    print(f"  median_cardinality: {np.median(y_raw):.3f}")
    print(f"  max_cardinality: {np.max(y_raw):.3f}")
    first = graphs[0]
    print(f"  feature_dim: {first.x.size(1)}")
    print(f"  first_graph_x.shape: {tuple(first.x.shape)}")
    print(f"  first_graph_edge_index.shape: {tuple(first.edge_index.shape)}")
    print(f"  first_graph_root_index: {int(first.root_index.item())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CardBench instance benchmark for GINE/PNA/GMT/GraphGPS/AK/MO/GCA.")
    parser.add_argument("--npz_path", type=str, default="training_datasets/binary_join/consumer_binary_join.npz", help="Path, directory, or glob for CardBench .npz files.")
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
    parser.add_argument("--holdout_test_size", type=int, default=500)
    parser.add_argument("--cap_trainval", type=int, default=4500, help="Use -1 to disable capping and keep all remaining train/val graphs.")

    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--first_seed", type=int, default=2024)
    parser.add_argument("--n_trials", type=int, default=200, help="Optuna trials; set to 0 to use explicit CLI hyperparameters.")
    parser.add_argument("--use_best_hyper_params", action="store_true", help="Use the recorded configuration for each input dataset.")

    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=50)
    parser.add_argument("--hpo_max_epochs", type=int, default=60)
    parser.add_argument("--hpo_early_stop", type=int, default=20)
    parser.add_argument("--verbose_every", type=int, default=10)

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
    parser.add_argument(
        "--use_aug",
        action="store_true",
        help="Enable CardBench-safe GCA-style graph augmentation for gin_root/gine_root.",
    )
    parser.add_argument("--aug_edge_drop_p1", type=float, default=0.10)
    parser.add_argument("--aug_feat_mask_p1", type=float, default=0.05)
    parser.add_argument("--aug_edge_drop_p2", type=float, default=0.05)
    parser.add_argument("--aug_feat_mask_p2", type=float, default=0.10)
    parser.add_argument("--aug_prob_cap", type=float, default=0.35, help="Maximum per-edge/per-node augmentation probability.")
    parser.add_argument("--contrast_weight", type=float, default=0.10)
    parser.add_argument("--contrast_temperature", type=float, default=0.50)
    parser.add_argument(
        "--use_flag",
        action="store_true",
        help="Enable CardBench-safe FLAG-style adversarial feature perturbation for gin_root/gine_root.",
    )
    parser.add_argument("--flag_steps", type=int, default=3)
    parser.add_argument("--flag_step_size", type=float, default=0.002)
    parser.add_argument("--flag_ops_scale", type=float, default=0.25)
    parser.add_argument("--flag_root_scale", type=float, default=0.0)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--out_dir", type=str, default="cardbench_demo_results")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA GPU index to use. CPU is not supported.")
    return parser.parse_args()


def release_best_hparams(dataset_name: str, model_name: str) -> Dict[str, Any]:
    from configs.best import load_best_entry

    model_map = {
        "gine_root": "GINE", "gine_root_gca": "GCA", "gin_root_gca": "GCA",
        "pna_root": "PNA", "graphgps_vn": "GraphGPS", "gine_gmt": "GMT",
        "gcn_gmt": "GMT", "gnnak_root": "AK", "mo": "MO",
    }
    canonical = model_map.get(str(model_name).strip().lower())
    if canonical is None:
        raise KeyError(f"No release best-parameter manifest is defined for CardBench model {model_name!r}.")
    return dict(load_best_entry("CardBench", canonical, setting=dataset_name)["params"])


def main() -> None:
    args = parse_args()
    if args.cap_trainval is not None and args.cap_trainval <= 0:
        args.cap_trainval = None
    try:
        args.use_aug, args.use_flag = validate_cardbench_aug_args(
            args.model_name,
            args.use_aug,
            args.use_flag,
            args.flag_steps,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    if args.gpu < 0:
        raise SystemExit("--gpu must be a non-negative integer.")
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for this script, but no GPU is available.")
    if args.gpu >= torch.cuda.device_count():
        raise SystemExit(f"Requested --gpu={args.gpu}, but only {torch.cuda.device_count()} CUDA device(s) are available.")
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    npz_files = list_cardbench_npz_files(args.npz_path)
    print(f"Resolved {len(npz_files)} dataset file(s).")

    all_run_rows: List[Dict[str, Any]] = []
    all_results: List[RunResult] = []

    for npz_idx, npz_path in enumerate(npz_files, start=1):
        print("\n" + "=" * 88)
        print(f"[{npz_idx}/{len(npz_files)}] Processing dataset: {npz_path}")
        raw_dataset = load_raw_graph_dataset(npz_path, max_graphs=args.max_graphs)
        summarize_cardbench_dataset(raw_dataset)

        usable_graphs = [g for g in raw_dataset.graphs if is_usable_query_graph(g)]
        print(f"\nUsable graphs after official filtering (cardinality > 0 and at least one predicate): {len(usable_graphs)} / {len(raw_dataset.graphs)}")

        if len(usable_graphs) <= args.holdout_test_size + 2:
            print("Skipping dataset: not enough usable graphs after filtering.")
            continue

        seeds = [args.first_seed + i for i in range(args.n_runs)]
        canonical_seed = seeds[0]
        prepared_2024 = prepare_split(raw_dataset, canonical_seed, args.holdout_test_size, args.cap_trainval, debug=args.debug)
        print("\nOfficial CardBench-style split sizes")
        print(f"  train: {len(prepared_2024.train_graphs)}")
        print(f"  val:   {len(prepared_2024.val_graphs)}")
        print(f"  test:  {len(prepared_2024.test_graphs)}")
        # Validation/test run under model.eval(), so singleton final batches are acceptable there.
        # Training uses BatchNorm1d in multiple model components, so reject any candidate batch size
        # that would leave a singleton last training batch.
        assert_no_singleton_training_last_batch(
            dataset_name=raw_dataset.name,
            split_name="train",
            num_graphs=len(prepared_2024.train_graphs),
            batch_sizes=candidate_batch_sizes_for_model(args.model_name, args),
        )
        summarize_pyg_graphs(prepared_2024.train_graphs, "Train PyG")
        summarize_pyg_graphs(prepared_2024.val_graphs, "Validation PyG")
        summarize_pyg_graphs(prepared_2024.test_graphs, "Test PyG")

        if args.use_best_hyper_params:
            best_hparams = release_best_hparams(raw_dataset.name, args.model_name)
            for key, value in best_hparams.items():
                if hasattr(args, key):
                    setattr(args, key, value)
        elif args.n_trials > 0:
            print(f"\nTuning hyperparameters on seed={canonical_seed} with Optuna (n_trials={args.n_trials}) ...")
            best_hparams = tune_hparams(prepared_2024, args.model_name, args, device)
        else:
            best_hparams = default_hparams_from_args(args, args.model_name)

        print("Best / selected hyperparameters")
        for key, value in best_hparams.items():
            print(f"  {key}: {value}")

        dataset_result_dir = os.path.join(args.out_dir, raw_dataset.workload_type, raw_dataset.name, args.model_name)
        os.makedirs(dataset_result_dir, exist_ok=True)
        write_json(os.path.join(dataset_result_dir, "best_hparams.json"), best_hparams)

        for run_idx, seed in enumerate(seeds, start=1):
            print("\n" + "-" * 72)
            print(f"Run {run_idx}/{len(seeds)} | seed={seed}")
            prepared = prepared_2024 if seed == canonical_seed else prepare_split(raw_dataset, seed, args.holdout_test_size, args.cap_trainval, debug=args.debug)
            train_config = TrainConfig(
                max_epochs=args.max_epochs,
                early_stop=args.early_stop,
                batch_size=int(best_hparams.get("batch_size", args.batch_size)),
                lr=float(best_hparams.get("lr", args.lr)),
                weight_decay=float(best_hparams.get("weight_decay", args.weight_decay)),
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
            print(f"Training on device: {device}")
            best_epoch, best_val_metrics, test_metrics, efficiency_metrics, y_true_raw, y_pred_raw = train_with_early_stopping(
                prepared=prepared,
                model_name=args.model_name,
                hparams=best_hparams,
                train_config=train_config,
                device=device,
                verbose=True,
                seed=seed,
            )
            print(f"\nBest epoch (selected by validation median_q_error): {best_epoch}")
            print_metrics("\nBest validation metrics", best_val_metrics)
            print_metrics("\nCorresponding test metrics", test_metrics)

            run_result = RunResult(
                dataset_name=raw_dataset.name,
                base_name=raw_dataset.base_name,
                workload_type=raw_dataset.workload_type,
                npz_path=npz_path,
                model_name=args.model_name,
                seed=seed,
                best_epoch=best_epoch,
                hparams=dict(best_hparams),
                best_val_metrics=dict(best_val_metrics),
                test_metrics=dict(test_metrics),
                efficiency_metrics=dict(efficiency_metrics),
                y_true_raw=y_true_raw,
                y_pred_raw=y_pred_raw,
            )
            all_results.append(run_result)
            row = {
                "dataset_name": raw_dataset.name,
                "base_name": raw_dataset.base_name,
                "workload_type": raw_dataset.workload_type,
                "npz_path": npz_path,
                "model_name": args.model_name,
                "seed": seed,
                "best_epoch": best_epoch,
                **{f"hparam_{k}": v for k, v in best_hparams.items()},
                **metrics_to_row("val", best_val_metrics),
                **metrics_to_row("test", test_metrics),
                **metrics_to_row("eff", efficiency_metrics),
            }
            all_run_rows.append(row)

        dataset_rows = aggregate_runs_by_dataset([r for r in all_results if r.dataset_name == raw_dataset.name and r.model_name == args.model_name])
        write_csv(os.path.join(dataset_result_dir, "run_results.csv"), [r for r in all_run_rows if r["dataset_name"] == raw_dataset.name and r["model_name"] == args.model_name])
        write_csv(os.path.join(dataset_result_dir, "dataset_summary.csv"), dataset_rows)
        write_json(os.path.join(dataset_result_dir, "dataset_summary.json"), dataset_rows)

    if not all_results:
        raise SystemExit("No experiments were run.")

    all_run_csv = os.path.join(args.out_dir, f"{args.model_name}_all_run_results.csv")
    write_csv(all_run_csv, all_run_rows)

    dataset_summary_rows = aggregate_runs_by_dataset(all_results)
    dataset_summary_csv = os.path.join(args.out_dir, f"{args.model_name}_dataset_summary.csv")
    write_csv(dataset_summary_csv, dataset_summary_rows)
    macro_summary = aggregate_family_macro(dataset_summary_rows)
    micro_summary = aggregate_family_micro(all_results)
    family_summary = {
        "model_name": args.model_name,
        "num_datasets": len(dataset_summary_rows),
        "num_runs": len(all_results),
        **macro_summary,
        **micro_summary,
    }
    write_json(os.path.join(args.out_dir, f"{args.model_name}_family_summary.json"), family_summary)

    print("\n" + "=" * 88)
    print("Family-level summary")
    print(json.dumps(family_summary, indent=2, ensure_ascii=False))
    print("\nSaved artifacts:")
    print(f"  {all_run_csv}")
    print(f"  {dataset_summary_csv}")
    print(f"  {os.path.join(args.out_dir, f'{args.model_name}_family_summary.json')}")


if __name__ == "__main__":
    main()
