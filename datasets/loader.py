import os
import os.path as osp
import math
import tarfile
import json
import hashlib
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import AQSOL, LRGBDataset, TUDataset, ZINC
from torch_geometric.utils import degree, get_laplacian, to_networkx, to_scipy_sparse_matrix, to_undirected
from ogb.graphproppred import PygGraphPropPredDataset
import scipy.io as sio

from .good import is_good_dataset_name, load_official_good_splits

COUNTING_TASK_MAP = {
    '3-Cycle': 0, '4-Cycle': 1, '5-Cycle': 2, '6-Cycle': 3,
    '7-Cycle': 4, '8-Cycle': 5, '4-Path': 6, '5-Path': 7,
    '6-Path': 8, '4-Clique': 9, 'Tailed-Tri': 10,
    'Chordal-Cyc': 11, 'Tri-Rect': 12
}
COUNTING_DATASETS = list(COUNTING_TASK_MAP.keys())
OGB_DATASETS = {
    'ogbg-molbace', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-moltox21'
}
LRGB_DATASETS = {'peptides-func', 'peptides-struct'}
PYG_MOL_REGRESSION_DATASETS = {'zinc', 'aqsol'}
GCARE_DATASETS = {'gcare-aids', 'gcare-yago'}
GNCE_DATA_PREFIX = 'gnce-'
GNCE_KNOWLEDGE_GRAPHS = {'swdf', 'lubm', 'yago', 'wikidata'}
GNCE_QUERY_TYPES = {'path', 'star'}
GNCE_DATASETS = {
    f'{GNCE_DATA_PREFIX}{kg}-{query_type}'
    for kg in GNCE_KNOWLEDGE_GRAPHS
    for query_type in GNCE_QUERY_TYPES
}
GNCE_OFFICIAL_SPLIT_SEED = 4
GNCE_OFFICIAL_TRAIN_RATIO = 0.8
GNCE_VAL_RATIO_WITHIN_OFFICIAL_TRAIN = 0.1
GNCE_SPLIT_PROTOCOL = 'official_seed4_80-20_train90-10_v1'
TU_DATASETS = {
    'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB',
    'MUTAG', 'NCI1', 'DD', 'ENZYMES', 'PROTEINS',
}
CV_DEFAULTS = {
    'tu': {
        'fold_k': 10,
        'fold_seed': 2024,
        'group_by': 'graph label',
        'protocol': (
            'fold i is test, fold (i-1) mod k is validation, and all other '
            'folds are training'
        ),
        'split_uses_seed': True,
    },
    'gcare': {
        'fold_k': 5,
        'fold_seed': 2024,
        'group_by': 'query topology and ordered log-cardinality',
        'protocol': (
            'one deterministic balanced bucket is test, about 10% of all '
            'queries is selected from the remainder for validation, and all '
            'other queries are training'
        ),
        'split_uses_seed': False,
    },
}
PREPROCESS_CACHE_VERSION = 'v1'


def base_dataset_name(dataset_name):
    return str(dataset_name)


def dataset_loader_kwargs(_cfg):
    return {}


def dataset_cache_tag(cfg):
    return 'tu-fullattrs-v2_clean' if cfg.dataset in TU_DATASETS else 'clean'


def _cv_family(dataset_name):
    base_name = base_dataset_name(dataset_name)
    if base_name in TU_DATASETS:
        return 'tu'
    if base_name.lower() in GCARE_DATASETS:
        return 'gcare'
    return None


def resolve_cv_parameters(dataset_name, fold_k=None, fold_seed=2024, input_fn=input):
    """Resolve and guard the benchmark settings for TU and G-CARE.

    ``fold_k`` may be None or 0 to request the dataset-specific default.
    A non-default k or seed is allowed only after an explicit y/Y response.
    Datasets with an official/fixed split are returned unchanged and do not
    trigger a confirmation prompt.
    """
    family = _cv_family(dataset_name)
    if family is None:
        return fold_k, fold_seed

    spec = CV_DEFAULTS[family]
    expected_k = int(spec['fold_k'])
    expected_seed = int(spec['fold_seed'])
    resolved_k = expected_k if fold_k in (None, 0) else int(fold_k)
    resolved_seed = expected_seed if fold_seed is None else int(fold_seed)

    mismatches = []
    if resolved_k != expected_k:
        mismatches.append(f"k={resolved_k} (benchmark default: {expected_k})")
    if resolved_seed != expected_seed:
        mismatches.append(f"seed={resolved_seed} (benchmark default: {expected_seed})")

    if mismatches:
        print(
            f"[CV WARNING] {dataset_name} is about to use non-default CV settings: "
            f"{', '.join(mismatches)}."
        )
        print(
            f"[CV WARNING] Grouping/balancing={spec['group_by']}; "
            f"protocol={spec['protocol']}."
        )
        if not spec['split_uses_seed']:
            print(
                "[CV WARNING] This G-CARE split is deterministic and does not "
                f"use the seed; seed={resolved_seed} changes run-level "
                "randomness only."
            )
        try:
            answer = input_fn("Continue with these non-default CV settings? [y/N]: ")
        except EOFError as exc:
            raise RuntimeError(
                "Non-default CV settings require an interactive y/Y confirmation, "
                "but no input was available."
            ) from exc
        if str(answer).strip().lower() != 'y':
            raise RuntimeError(
                f"CV construction cancelled for {dataset_name}; only y/Y confirms "
                "non-default k or seed values."
            )

    seed_role = (
        'split construction only'
        if spec['split_uses_seed']
        else 'compatibility argument only; split is deterministic'
    )
    print(
        f"[CV] dataset={dataset_name}, k={resolved_k}, seed={resolved_seed}, "
        f"seed_role={seed_role}, grouping/balancing={spec['group_by']}"
    )
    print(f"[CV] protocol: {spec['protocol']}.")
    return resolved_k, resolved_seed


def _validate_rotating_cv_indices(folds, num_samples, expected_k, dataset_name):
    if len(folds) != expected_k:
        raise AssertionError(
            f"{dataset_name}: expected {expected_k} folds, got {len(folds)}."
        )

    universe = set(range(num_samples))
    test_counts = Counter()
    val_counts = Counter()
    split_sizes = []
    for fold_id, (train_idx, val_idx, test_idx) in enumerate(folds, start=1):
        train_set = set(int(i) for i in train_idx)
        val_set = set(int(i) for i in val_idx)
        test_set = set(int(i) for i in test_idx)

        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise AssertionError(f"{dataset_name}: overlap detected in fold {fold_id}.")
        if train_set | val_set | test_set != universe:
            raise AssertionError(
                f"{dataset_name}: fold {fold_id} does not cover all {num_samples} samples."
            )

        test_counts.update(test_set)
        val_counts.update(val_set)
        split_sizes.append((len(train_set), len(val_set), len(test_set)))

    if set(test_counts) != universe or any(count != 1 for count in test_counts.values()):
        raise AssertionError(
            f"{dataset_name}: every sample must appear exactly once as test."
        )
    if set(val_counts) != universe or any(count != 1 for count in val_counts.values()):
        raise AssertionError(
            f"{dataset_name}: every sample must appear exactly once as validation."
        )

    rendered_sizes = ', '.join(
        f"fold{i}=({n_train}/{n_val}/{n_test})"
        for i, (n_train, n_val, n_test) in enumerate(split_sizes, start=1)
    )
    print(f"[CV] verified train/val/test sizes: {rendered_sizes}")


def _validate_gcare_balanced_indices(
        folds, num_samples, expected_k, dataset_name='G-CARE', verbose=True):
    """Validate the deterministic G-CARE outer-test-fold protocol.

    Test buckets partition the data exactly once. Validation sets are selected
    independently from each fold's remainder, so unlike rotating k-fold CV a
    sample is not required to appear exactly once as validation.
    """
    if len(folds) != expected_k:
        raise AssertionError(
            f"{dataset_name}: expected {expected_k} folds, got {len(folds)}."
        )

    universe = set(range(num_samples))
    test_counts = Counter()
    split_sizes = []
    for fold_id, (train_idx, val_idx, test_idx) in enumerate(folds, start=1):
        train_set = set(int(i) for i in train_idx)
        val_set = set(int(i) for i in val_idx)
        test_set = set(int(i) for i in test_idx)

        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise AssertionError(f"{dataset_name}: overlap detected in fold {fold_id}.")
        if train_set | val_set | test_set != universe:
            raise AssertionError(
                f"{dataset_name}: fold {fold_id} does not cover all "
                f"{num_samples} samples."
            )
        if not train_set or not val_set or not test_set:
            raise AssertionError(
                f"{dataset_name}: fold {fold_id} contains an empty train, "
                "validation, or test split."
            )

        test_counts.update(test_set)
        split_sizes.append((len(train_set), len(val_set), len(test_set)))

    if set(test_counts) != universe or any(count != 1 for count in test_counts.values()):
        raise AssertionError(
            f"{dataset_name}: every sample must appear exactly once as test."
        )

    if verbose:
        rendered_sizes = ', '.join(
            f"fold{i}=({n_train}/{n_val}/{n_test})"
            for i, (n_train, n_val, n_test) in enumerate(split_sizes, start=1)
        )
        rendered_ratios = ', '.join(
            f"fold{i}=({n_train / num_samples:.1%}/"
            f"{n_val / num_samples:.1%}/{n_test / num_samples:.1%})"
            for i, (n_train, n_val, n_test) in enumerate(split_sizes, start=1)
        )
        print(
            "[G-CARE CV] deterministic topology/log-cardinality-balanced "
            f"{expected_k}-test-fold protocol; split seed is not used."
        )
        print(
            "[G-CARE CV] verified train/val/test sizes "
            f"(train/val/test): {rendered_sizes}"
        )
        print(f"[G-CARE CV] verified train/val/test ratios: {rendered_ratios}")


def _torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _set_dataset_storage(dataset, data, slices):
    if hasattr(dataset, '_data'):
        dataset._data = data
    else:
        dataset.data = data
    dataset.slices = slices
    if hasattr(dataset, '_data_list'):
        dataset._data_list = None


def _cache_bool(value):
    return '1' if value else '0'


def eigvec_normalizer(eigvecs, eigvals, normalization='L2', eps=1e-12):
    eigvals = eigvals.unsqueeze(0)
    if normalization == 'L1':
        denom = eigvecs.norm(p=1, dim=0, keepdim=True)
    elif normalization == 'L2':
        denom = eigvecs.norm(p=2, dim=0, keepdim=True)
    elif normalization == 'abs-max':
        denom = torch.max(eigvecs.abs(), dim=0, keepdim=True).values
    elif normalization == 'wavelength':
        denom = torch.max(eigvecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(eigvals)
        eigval_denom[eigvals < eps] = 1
        denom = denom * eigval_denom * 2 / np.pi
    else:
        raise ValueError(f"Unsupported LapPE eigvec normalization: {normalization}")

    return eigvecs / denom.clamp_min(eps).expand_as(eigvecs)


def add_lap_pe_stats(data, max_freqs=8, eigvec_norm='L2', laplacian_norm='none'):
    num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x.size(0)
    norm = None if str(laplacian_norm).lower() == 'none' else laplacian_norm
    edge_index = data.edge_index if data.is_undirected() else to_undirected(data.edge_index)
    laplacian = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=norm, num_nodes=num_nodes)
    )
    evals, evects = np.linalg.eigh(laplacian.toarray())

    idx = evals.argsort()[:max_freqs]
    evals = torch.from_numpy(np.real(evals[idx])).float().clamp_min(0)
    evects = torch.from_numpy(np.real(evects[:, idx])).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)

    if num_nodes < max_freqs:
        eigvecs = F.pad(evects, (0, max_freqs - num_nodes), value=float('nan'))
        eigvals = F.pad(evals, (0, max_freqs - num_nodes), value=float('nan')).unsqueeze(0)
    else:
        eigvecs = evects
        eigvals = evals.unsqueeze(0)

    data.EigVals = eigvals.repeat(num_nodes, 1).unsqueeze(2)
    data.EigVecs = eigvecs
    return data


def _ogb_preprocess_cache_path(root, name, use_fingerprints, fingerprint_dim,
                               use_rwpe, rwpe_dim, use_lappe, lappe_max_freqs,
                               augm_gca, aug_centrality):
    fp_dim = fingerprint_dim if use_fingerprints else 0
    rw_dim = rwpe_dim if use_rwpe else 0
    lap_dim = lappe_max_freqs if use_lappe else 0
    centrality = aug_centrality if augm_gca else 'none'
    filename = (
        f"{PREPROCESS_CACHE_VERSION}_{name}"
        f"_fp{_cache_bool(use_fingerprints)}d{fp_dim}"
        f"_rwpe{_cache_bool(use_rwpe)}d{rw_dim}"
        f"_lappe{_cache_bool(use_lappe)}k{lap_dim}"
        f"_gca{_cache_bool(augm_gca)}{centrality}.pt"
    )
    return osp.join(root, 'preprocessed_cache', filename)


def _save_preprocess_cache(cache_path, data, slices, max_degree):
    os.makedirs(osp.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    torch.save({'data': data, 'slices': slices, 'max_degree': max_degree}, tmp_path)
    os.replace(tmp_path, cache_path)


def _load_preprocess_cache(cache_path):
    cached = _torch_load(cache_path)
    if isinstance(cached, dict):
        return cached['data'], cached['slices'], cached.get('max_degree')
    data, slices = cached
    return data, slices, None


class GraphCountDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphCountDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        a = sio.loadmat(self.raw_paths[0])
        self.train_idx = torch.from_numpy(a['train_idx'][0])
        self.val_idx = torch.from_numpy(a['val_idx'][0])
        self.test_idx = torch.from_numpy(a['test_idx'][0])

    @property
    def raw_file_names(self):
        return ["data.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        a = sio.loadmat(self.raw_paths[0])
        A = a['A'][0]
        Y = a['Y']

        data_list = []
        for i in range(len(A)):
            E = np.where(A[i] > 0)
            edge_index = torch.Tensor(np.vstack((E[0], E[1]))).type(torch.int64)
            x = torch.ones(A[i].shape[0], 1).long()
            expy = Y[i]
            data_list.append(Data(edge_index=edge_index, x=x, y=expy))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def _canonical_gcare_name(name):
    name = name.lower()
    if name not in GCARE_DATASETS:
        raise ValueError(f"Unknown G-CARE dataset: {name}. Supported: {sorted(GCARE_DATASETS)}")
    return name[len('gcare-'):]


def _safe_extract_tar(tar_path, dst_dir):
    if not osp.exists(tar_path):
        return
    os.makedirs(dst_dir, exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tar:
        root = osp.abspath(dst_dir)
        for member in tar.getmembers():
            target = osp.abspath(osp.join(dst_dir, member.name))
            if not (target == root or target.startswith(root + os.sep)):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(dst_dir)


def _ensure_gcare_raw_layout(raw_dir):
    archive_map = {
        'queryset.tar.gz': 'queryset',
        'true_cardinality.tar.gz': 'true_cardinality',
        'dataset.tar.gz': 'dataset',
    }
    for archive, subdir in archive_map.items():
        subdir_path = osp.join(raw_dir, subdir)
        if not osp.isdir(subdir_path):
            _safe_extract_tar(osp.join(raw_dir, archive), subdir_path)


def _read_gcare_cardinality(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"Empty cardinality file: {path}")
    return float(text.split()[0])


def _parse_gcare_query(path):
    query_id = -1
    vertices = {}
    edges = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            parts = raw.strip().split()
            if not parts:
                continue
            if parts[0] == 't':
                try:
                    query_id = int(parts[-1])
                except ValueError:
                    query_id = -1
            elif parts[0] == 'v':
                if len(parts) < 4:
                    raise ValueError(f"Bad G-CARE vertex line in {path}: {raw}")
                vid = int(parts[1])
                vertices[vid] = (int(parts[2]), int(parts[3]))
            elif parts[0] == 'e':
                if len(parts) < 4:
                    raise ValueError(f"Bad G-CARE edge line in {path}: {raw}")
                edges.append((int(parts[1]), int(parts[2]), int(parts[3])))
    if not vertices:
        raise ValueError(f"G-CARE query has no vertices: {path}")
    return {'query_id': query_id, 'vertices': vertices, 'edges': edges}


def _gcare_topology_from_relpath(rel_path, dataset_name):
    parent = osp.basename(osp.dirname(rel_path))
    if parent in ('', '.', dataset_name):
        return dataset_name, 0
    if '_' in parent:
        topology, size = parent.rsplit('_', 1)
        try:
            return topology, int(size)
        except ValueError:
            return topology, 0
    return parent, 0


def _collect_gcare_queries(query_root, card_root, dataset_name):
    queries = []
    for dirpath, _, filenames in os.walk(query_root):
        for filename in filenames:
            if not filename.endswith('.txt'):
                continue
            query_path = osp.join(dirpath, filename)
            rel_path = osp.relpath(query_path, query_root)
            card_path = osp.join(card_root, rel_path)
            if not osp.exists(card_path):
                raise FileNotFoundError(f"Missing G-CARE true cardinality for query: {rel_path}")
            parsed = _parse_gcare_query(query_path)
            card = _read_gcare_cardinality(card_path)
            if card <= 0:
                continue
            topology, topology_size = _gcare_topology_from_relpath(rel_path, dataset_name)
            parsed.update({
                'rel_path': rel_path.replace('\\', '/'),
                'cardinality': card,
                'topology': topology,
                'topology_size': topology_size,
            })
            queries.append(parsed)
    queries.sort(key=lambda item: item['rel_path'])
    if not queries:
        raise RuntimeError(f"No positive-cardinality G-CARE queries found under: {query_root}")
    return queries


def _scan_gcare_data_graph_stats(path, bound_ids):
    label_counts = Counter()
    edge_label_counts = Counter()
    bound_degree = defaultdict(int)
    num_vertices = 0
    num_edges = 0
    if not osp.exists(path):
        return {
            'label_counts': label_counts,
            'edge_label_counts': edge_label_counts,
            'bound_degree': bound_degree,
            'num_vertices': 0,
            'num_edges': 0,
        }

    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            parts = raw.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                num_vertices += 1
                labels = [int(x) for x in parts[2:]] or [0]
                label_counts.update(labels)
            elif parts[0] == 'e':
                num_edges += 1
                src = int(parts[1])
                dst = int(parts[2])
                edge_label_counts[int(parts[3])] += 1
                if src in bound_ids:
                    bound_degree[src] += 1
                if dst in bound_ids:
                    bound_degree[dst] += 1

    return {
        'label_counts': label_counts,
        'edge_label_counts': edge_label_counts,
        'bound_degree': bound_degree,
        'num_vertices': num_vertices,
        'num_edges': num_edges,
    }


def _one_hot(index, size):
    out = torch.zeros(size, dtype=torch.float)
    out[index] = 1.0
    return out


def _make_gcare_data(parsed, node_label_to_idx, edge_label_to_idx, topology_to_idx, stats):
    vertices = parsed['vertices']
    edges = parsed['edges']
    num_nodes = max(vertices.keys()) + 1
    node_label_dim = len(node_label_to_idx)
    edge_label_dim = len(edge_label_to_idx)

    in_deg = torch.zeros(num_nodes, dtype=torch.float)
    out_deg = torch.zeros(num_nodes, dtype=torch.float)
    for src, dst, _ in edges:
        out_deg[src] += 1.0
        in_deg[dst] += 1.0

    x_rows = []
    x_blind_rows = []
    for vid in range(num_nodes):
        label, dvid = vertices.get(vid, (-1, -1))
        label_idx = node_label_to_idx.get(label, node_label_to_idx[-1])
        label_freq = stats['label_counts'].get(label, 0) if label != -1 else stats['num_vertices']
        bound_deg = stats['bound_degree'].get(dvid, 0) if dvid != -1 else 0
        blind_cont = torch.tensor([
            1.0 if label != -1 else 0.0,
            1.0 if dvid != -1 else 0.0,
            math.log1p(float(label_freq)),
            math.log1p(float(bound_deg)),
        ], dtype=torch.float)
        cont = torch.tensor([
            blind_cont[0].item(),
            blind_cont[1].item(),
            blind_cont[2].item(),
            blind_cont[3].item(),
            float(in_deg[vid].item()),
            float(out_deg[vid].item()),
            float((in_deg[vid] + out_deg[vid]).item()),
            math.log1p(float(num_nodes)),
            math.log1p(float(max(len(edges), 1))),
            float(parsed['topology_size']),
        ], dtype=torch.float)
        label_one_hot = _one_hot(label_idx, node_label_dim)
        x_rows.append(torch.cat([label_one_hot, cont], dim=0))
        x_blind_rows.append(torch.cat([label_one_hot, blind_cont], dim=0))
    x = torch.stack(x_rows, dim=0)
    x_blind = torch.stack(x_blind_rows, dim=0)

    edge_pairs = []
    edge_features = []
    edge_blind_features = []
    for src, dst, elabel in edges:
        edge_idx = edge_label_to_idx[elabel]
        edge_freq = stats['edge_label_counts'].get(elabel, 0)
        for is_forward, u, v in ((1.0, src, dst), (0.0, dst, src)):
            edge_pairs.append((u, v))
            blind_cont = torch.tensor([
                is_forward,
                math.log1p(float(edge_freq)),
            ], dtype=torch.float)
            cont = torch.tensor([
                blind_cont[0].item(),
                blind_cont[1].item(),
                math.log1p(float(num_nodes)),
                math.log1p(float(max(len(edges), 1))),
            ], dtype=torch.float)
            edge_one_hot = _one_hot(edge_idx, edge_label_dim)
            edge_features.append(torch.cat([edge_one_hot, cont], dim=0))
            edge_blind_features.append(torch.cat([edge_one_hot, blind_cont], dim=0))

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_features, dim=0)
        edge_attr_blind = torch.stack(edge_blind_features, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_label_dim + 4), dtype=torch.float)
        edge_attr_blind = torch.empty((0, edge_label_dim + 2), dtype=torch.float)

    card = float(parsed['cardinality'])
    y_log = math.log(card)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([y_log], dtype=torch.float),
    )
    data.x_blind = x_blind
    data.edge_attr_blind = edge_attr_blind
    data.y_log = torch.tensor([y_log], dtype=torch.float)
    data.y_card = torch.tensor([card], dtype=torch.float)
    data.query_n_nodes = torch.tensor([num_nodes], dtype=torch.long)
    data.query_n_edges = torch.tensor([len(edges)], dtype=torch.long)
    data.topology_id = torch.tensor([topology_to_idx[parsed['topology']]], dtype=torch.long)
    data.topology_size = torch.tensor([parsed['topology_size']], dtype=torch.long)
    data.query_id = torch.tensor([parsed['query_id']], dtype=torch.long)
    return data


def _gcare_balanced_buckets(items, num_buckets):
    buckets = [[] for _ in range(num_buckets)]
    grouped = defaultdict(list)
    for item in items:
        grouped[item['topology']].append(item)

    for topology in sorted(grouped):
        group = sorted(grouped[topology], key=lambda item: (item['log_cardinality'], item['idx']))
        for pos, item in enumerate(group):
            offset = pos % num_buckets
            cycle = pos // num_buckets
            bucket_id = offset if cycle % 2 == 0 else num_buckets - 1 - offset
            buckets[bucket_id].append(item)

    for bucket in buckets:
        bucket.sort(key=lambda item: item['idx'])
    return buckets


def _gcare_evenly_spaced_items(items, quota):
    if quota <= 0:
        return []
    group = sorted(items, key=lambda item: (item['log_cardinality'], item['idx']))
    n = len(group)
    if quota >= n:
        return group

    selected_positions = []
    used = set()
    for j in range(quota):
        pos = int((j + 0.5) * n / quota)
        pos = max(0, min(n - 1, pos))
        if pos in used:
            for delta in range(1, n):
                left = pos - delta
                right = pos + delta
                if left >= 0 and left not in used:
                    pos = left
                    break
                if right < n and right not in used:
                    pos = right
                    break
        used.add(pos)
        selected_positions.append(pos)

    return [group[pos] for pos in sorted(selected_positions)]


def _make_gcare_validation_items(trainval_items, total_n, val_ratio=0.10, min_topology_size=5):
    n_trainval = len(trainval_items)
    n_val = int(round(val_ratio * total_n))
    n_val = max(1, min(n_val, n_trainval - 1))

    grouped = defaultdict(list)
    for item in trainval_items:
        grouped[item['topology']].append(item)

    quotas = {}
    remainders = {}
    min_quotas = {}
    for topology, group in grouped.items():
        q_float = n_val * len(group) / n_trainval
        quotas[topology] = int(math.floor(q_float))
        remainders[topology] = q_float - quotas[topology]
        min_quotas[topology] = 1 if len(group) >= min_topology_size else 0
        if quotas[topology] == 0 and min_quotas[topology] == 1:
            quotas[topology] = 1

    while sum(quotas.values()) < n_val:
        candidates = [
            topology for topology, group in grouped.items()
            if quotas[topology] < len(group)
        ]
        if not candidates:
            break
        topology = max(candidates, key=lambda t: (remainders[t], len(grouped[t]), t))
        quotas[topology] += 1

    while sum(quotas.values()) > n_val:
        candidates = [
            topology for topology in grouped
            if quotas[topology] > min_quotas[topology]
        ]
        if not candidates:
            candidates = [topology for topology in grouped if quotas[topology] > 0]
        if not candidates:
            break
        topology = max(candidates, key=lambda t: (quotas[t], len(grouped[t]), -remainders[t], t))
        quotas[topology] -= 1

    val_items = []
    for topology in sorted(grouped):
        val_items.extend(_gcare_evenly_spaced_items(grouped[topology], quotas[topology]))
    val_items.sort(key=lambda item: item['idx'])
    return val_items


def _make_gcare_folds(split_items, outer_k=5):
    n = len(split_items)
    if n < outer_k:
        raise ValueError(f"G-CARE requires at least {outer_k} queries for {outer_k}-fold CV, got {n}.")

    outer_buckets = _gcare_balanced_buckets(split_items, outer_k)
    all_idx = set(item['idx'] for item in split_items)
    folds = []
    for outer_id, test_items in enumerate(outer_buckets):
        test_set = set(item['idx'] for item in test_items)
        remaining_items = [
            item for item in split_items if item['idx'] not in test_set
        ]

        val_items = _make_gcare_validation_items(remaining_items, total_n=n)
        val_set = set(item['idx'] for item in val_items)
        train_set = all_idx - test_set - val_set

        folds.append((
            torch.tensor(sorted(train_set), dtype=torch.long),
            torch.tensor(sorted(val_set), dtype=torch.long),
            torch.tensor(sorted(test_set), dtype=torch.long),
        ))
    _validate_gcare_balanced_indices(
        folds, n, outer_k, dataset_name='G-CARE', verbose=False
    )
    return folds


class GCAREQueryDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, fold_k=5, fold_seed=2024):
        self.gcare_name = _canonical_gcare_name(name)
        self.fold_k = fold_k
        self.fold_seed = fold_seed
        super().__init__(root, transform, pre_transform)
        cached = _torch_load(self.processed_paths[0])
        self.data = cached['data']
        self.slices = cached['slices']
        self.folds_idx = cached['folds_idx']
        self.dim_node_deg = cached['dim_node_deg']
        self.dim_node_label = 0
        self.dim_node_attr = 0
        self.num_edge_classes = cached['num_edge_classes']
        self.node_label_vocab = cached['node_label_vocab']
        self.edge_label_vocab = cached['edge_label_vocab']
        self.topology_vocab = cached['topology_vocab']
        _validate_gcare_balanced_indices(
            self.folds_idx,
            len(self),
            self.fold_k,
            dataset_name=f'G-CARE/{self.gcare_name}',
            verbose=True,
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # Keep the original default cache name so prior five-fold results and
        # cached indices remain directly reusable. Non-default k needs its own
        # cache; the seed is omitted because this split algorithm does not use it.
        if self.fold_k == 5:
            return f'{self.gcare_name}_5fold_v5.pt'
        return f'{self.gcare_name}_{self.fold_k}fold_balanced_v5.pt'

    def download(self):
        _ensure_gcare_raw_layout(self.raw_dir)

    def process(self):
        _ensure_gcare_raw_layout(self.raw_dir)
        query_root = osp.join(self.raw_dir, 'queryset', self.gcare_name)
        card_root = osp.join(self.raw_dir, 'true_cardinality', self.gcare_name)
        data_graph_path = osp.join(self.raw_dir, 'dataset', self.gcare_name, f'{self.gcare_name}.txt')

        # Compatibility with manually extracted true_cardinality.tar.gz at raw root.
        if not osp.isdir(card_root):
            card_root = osp.join(self.raw_dir, self.gcare_name)

        if not osp.isdir(query_root):
            raise FileNotFoundError(
                f"G-CARE queryset not found: {query_root}. "
                "Place queryset.tar.gz under data/GCARE/raw or extract it to raw/queryset/."
            )
        if not osp.isdir(card_root):
            raise FileNotFoundError(
                f"G-CARE true cardinalities not found: {card_root}. "
                "Place true_cardinality.tar.gz under data/GCARE/raw or extract it to raw/true_cardinality/."
            )

        parsed_queries = _collect_gcare_queries(query_root, card_root, self.gcare_name)
        node_labels = sorted({label for q in parsed_queries for label, _ in q['vertices'].values()} | {-1})
        edge_labels = sorted({label for q in parsed_queries for _, _, label in q['edges']})
        topologies = sorted({q['topology'] for q in parsed_queries})
        node_label_to_idx = {label: idx for idx, label in enumerate(node_labels)}
        edge_label_to_idx = {label: idx for idx, label in enumerate(edge_labels)}
        topology_to_idx = {topology: idx for idx, topology in enumerate(topologies)}
        bound_ids = {dvid for q in parsed_queries for _, dvid in q['vertices'].values() if dvid != -1}
        stats = _scan_gcare_data_graph_stats(data_graph_path, bound_ids)

        data_list = []
        split_items = []
        for sample_idx, q in enumerate(parsed_queries):
            data = _make_gcare_data(q, node_label_to_idx, edge_label_to_idx, topology_to_idx, stats)
            data_list.append(data)
            split_items.append({
                'idx': sample_idx,
                'topology': q['topology'],
                'log_cardinality': math.log(float(q['cardinality'])),
            })

        if self.pre_filter is not None:
            keep = [self.pre_filter(data) for data in data_list]
            data_list = [data for data, flag in zip(data_list, keep) if flag]
            filtered_items = []
            for item, flag in zip(split_items, keep):
                if flag:
                    filtered_items.append({
                        'idx': len(filtered_items),
                        'topology': item['topology'],
                        'log_cardinality': item['log_cardinality'],
                    })
            split_items = filtered_items
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        folds_idx = _make_gcare_folds(split_items, outer_k=self.fold_k)
        data, slices = self.collate(data_list)
        torch.save({
            'data': data,
            'slices': slices,
            'folds_idx': folds_idx,
            'dim_node_deg': int(data_list[0].x.size(-1)),
            'num_edge_classes': int(data_list[0].edge_attr.size(-1)) if data_list[0].edge_attr is not None else 0,
            'node_label_vocab': node_labels,
            'edge_label_vocab': edge_labels,
            'topology_vocab': topologies,
        }, self.processed_paths[0])


def _parse_gnce_name(name):
    name = name.lower()
    if not name.startswith(GNCE_DATA_PREFIX):
        raise ValueError(f"GNCE dataset names must start with '{GNCE_DATA_PREFIX}', got {name}.")
    parts = name.split('-')
    if len(parts) < 3:
        raise ValueError("GNCE dataset names must look like gnce-<kg>-<query_type>, e.g. gnce-swdf-star.")
    kg_name = parts[1]
    query_type = '-'.join(parts[2:])
    if kg_name not in GNCE_KNOWLEDGE_GRAPHS or query_type not in GNCE_QUERY_TYPES:
        raise ValueError(
            f"Unknown GNCE workload: {name}. Supported workloads: "
            f"{sorted(GNCE_DATASETS)}"
        )
    return kg_name, query_type


def _resolve_gnce_file(raw_dir, kg, query_type, kind):
    roots = [
        raw_dir,
        osp.abspath(osp.join(raw_dir, '..')),
    ]
    candidates = []
    if kind == 'queries':
        names = ['Joined_Queries.json', 'queries.json']
        for root in roots:
            candidates.extend(osp.join(root, kg, query_type, name) for name in names)
    elif kind == 'graph':
        names = [f'{kg}.nt', f'{kg}.ttl', 'graph.nt', 'graph.ttl']
        for root in roots:
            candidates.extend(osp.join(root, kg, 'graph', name) for name in names)
    elif kind == 'embeddings':
        names = [f'{kg}_embeddings.json', 'embeddings.json']
        for root in roots:
            candidates.extend(osp.join(root, kg, name) for name in names)
    else:
        raise ValueError(f"Unknown GNCE file kind: {kind}")

    for path in candidates:
        if osp.exists(path):
            return path
    return None


def _gnce_source_signature(raw_dir, kg, query_type):
    """Build a compact cache signature for query, graph, and embedding inputs."""
    records = []
    for kind in ('queries', 'graph', 'embeddings'):
        path = _resolve_gnce_file(raw_dir, kg, query_type, kind)
        if path is None:
            records.append((kind, None, None))
            continue
        stat = os.stat(path)
        records.append((kind, int(stat.st_size), int(stat.st_mtime_ns)))
    payload = json.dumps(records, sort_keys=True).encode('utf-8')
    return hashlib.md5(payload).hexdigest()[:12]


def _stable_hash01(value):
    digest = hashlib.md5(str(value).encode('utf-8')).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _strip_gnce_atom(value):
    if value is None:
        return ''
    value = str(value).strip()
    if value.startswith('<') and value.endswith('>'):
        value = value[1:-1]
    return value


def _canonical_gnce_query_key(query):
    """Canonicalize variable names for released-query duplicate removal."""
    variable_map = {}

    def canonical_atom(value):
        atom = _strip_gnce_atom(value)
        if atom.startswith('?'):
            if atom not in variable_map:
                variable_map[atom] = f"VAR{len(variable_map)}"
            return variable_map[atom]
        return atom

    triples = []
    for triple in query.get('triples', []):
        if len(triple) < 3:
            continue
        triples.append(tuple(canonical_atom(value) for value in triple[:3]))
    return tuple(sorted(triples))


def _scan_gnce_occurrences(graph_path):
    counts = Counter()
    if graph_path is None or not osp.exists(graph_path):
        return counts
    with open(graph_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            s, p, o = (_strip_gnce_atom(parts[0]), _strip_gnce_atom(parts[1]), _strip_gnce_atom(parts[2]))
            counts[s] += 1
            counts[p] += 1
            counts[o] += 1
    return counts


def _load_gnce_embeddings(path):
    if path is None or not osp.exists(path):
        return None, 0
    with open(path, 'r', encoding='utf-8') as f:
        embeddings = json.load(f)
    emb_dim = 0
    for item in embeddings.values():
        vec = item.get('embedding') if isinstance(item, dict) else None
        if vec is not None:
            emb_dim = len(vec)
            break
    return embeddings, emb_dim


def _gnce_embedding_and_occ(atom, embeddings, emb_dim, occurrences):
    atom = _strip_gnce_atom(atom)
    occ = occurrences.get(atom, 0)
    if embeddings is not None and atom in embeddings:
        stat = embeddings[atom]
        vec = stat.get('embedding', [0.0] * emb_dim)
        occ = stat.get('occurence', occ)
        if len(vec) != emb_dim:
            vec = (vec + [0.0] * emb_dim)[:emb_dim]
        return [float(v) for v in vec], float(occ)
    return [0.0] * emb_dim, float(occ)


def _gnce_node_features(atom, embeddings, emb_dim, occurrences, in_deg, out_deg):
    atom = _strip_gnce_atom(atom)
    is_var = 1.0 if atom.startswith('?') else 0.0
    if emb_dim > 0:
        vec, occ = _gnce_embedding_and_occ(atom, embeddings, emb_dim, occurrences)
        if is_var:
            vec = [1.0] * emb_dim
        return torch.tensor(vec + [is_var, 1.0 - is_var, math.log(float(occ) + 1.0)], dtype=torch.float)
    return torch.tensor([
        is_var,
        1.0 - is_var,
        math.log(float(occurrences.get(atom, 0)) + 1.0),
        _stable_hash01(atom) if not is_var else 0.0,
        float(in_deg),
        float(out_deg),
        float(in_deg + out_deg),
    ], dtype=torch.float)


def _gnce_node_blind_features(atom, embeddings, emb_dim, occurrences):
    atom = _strip_gnce_atom(atom)
    is_var = 1.0 if atom.startswith('?') else 0.0
    if emb_dim > 0:
        vec, occ = _gnce_embedding_and_occ(atom, embeddings, emb_dim, occurrences)
        if is_var:
            vec = [1.0] * emb_dim
        return torch.tensor(vec + [is_var, 1.0 - is_var, math.log(float(occ) + 1.0)], dtype=torch.float)
    return torch.tensor([
        is_var,
        1.0 - is_var,
        math.log(float(occurrences.get(atom, 0)) + 1.0),
        _stable_hash01(atom) if not is_var else 0.0,
    ], dtype=torch.float)


def _gnce_edge_features(predicate, direction, embeddings, emb_dim, occurrences):
    pred = _strip_gnce_atom(predicate)
    is_var = 1.0 if pred.startswith('?') else 0.0
    if emb_dim > 0:
        vec, occ = _gnce_embedding_and_occ(pred, embeddings, emb_dim, occurrences)
        if is_var:
            vec = [1.0] * emb_dim
        return torch.tensor(vec + [is_var, math.log(float(occ) + 1.0), float(direction)], dtype=torch.float)
    return torch.tensor([
        is_var,
        math.log(float(occurrences.get(pred, 0)) + 1.0),
        _stable_hash01(pred) if not is_var else 0.0,
        float(direction),
    ], dtype=torch.float)


def _make_gnce_data(query, query_type, embeddings, emb_dim, occurrences):
    triples = query.get('triples', [])
    if not triples:
        raise ValueError("GNCE query has no triples.")

    node_to_idx = {}
    directed_edges = []
    predicates = []
    in_deg = Counter()
    out_deg = Counter()

    for triple in triples:
        if len(triple) < 3:
            continue
        s = _strip_gnce_atom(triple[0])
        p = _strip_gnce_atom(triple[1])
        o = _strip_gnce_atom(triple[2])
        for atom in (s, o):
            if atom not in node_to_idx:
                node_to_idx[atom] = len(node_to_idx)
        src, dst = node_to_idx[s], node_to_idx[o]
        directed_edges.append((src, dst))
        predicates.append(p)
        out_deg[s] += 1
        in_deg[o] += 1

    if not node_to_idx:
        raise ValueError("GNCE query has no entity/variable nodes.")

    x_rows = []
    x_blind_rows = []
    for atom, _idx in sorted(node_to_idx.items(), key=lambda item: item[1]):
        x_rows.append(_gnce_node_features(atom, embeddings, emb_dim, occurrences, in_deg[atom], out_deg[atom]))
        x_blind_rows.append(_gnce_node_blind_features(atom, embeddings, emb_dim, occurrences))
    x = torch.stack(x_rows, dim=0)
    x_blind = torch.stack(x_blind_rows, dim=0)

    edge_pairs = []
    edge_attrs = []
    for (src, dst), pred in zip(directed_edges, predicates):
        edge_pairs.append((src, dst))
        edge_attrs.append(_gnce_edge_features(pred, 1.0, embeddings, emb_dim, occurrences))
        edge_pairs.append((dst, src))
        edge_attrs.append(_gnce_edge_features(pred, -1.0, embeddings, emb_dim, occurrences))

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attrs, dim=0)
    card = float(query['y'])
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([math.log(card)], dtype=torch.float),
    )
    data.y_log = torch.tensor([math.log(card)], dtype=torch.float)
    data.y_card = torch.tensor([card], dtype=torch.float)
    data.query_n_nodes = torch.tensor([x.size(0)], dtype=torch.long)
    data.query_n_edges = torch.tensor([len(triples)], dtype=torch.long)
    data.x_blind = x_blind
    data.edge_attr_blind = edge_attr
    return data


def _validate_gnce_split(train_idx, val_idx, test_idx, n):
    universe = set(range(int(n)))
    train_set = set(int(i) for i in train_idx)
    val_set = set(int(i) for i in val_idx)
    test_set = set(int(i) for i in test_idx)
    if not train_set or not val_set or not test_set:
        raise AssertionError("GNCE split contains an empty train, validation, or test set.")
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise AssertionError("GNCE train/validation/test indices overlap.")
    if train_set | val_set | test_set != universe:
        raise AssertionError(f"GNCE split does not cover all {n} queries.")


def _make_gnce_split(
        n, seed=GNCE_OFFICIAL_SPLIT_SEED,
        train_ratio=GNCE_OFFICIAL_TRAIN_RATIO,
        val_ratio_within_train=GNCE_VAL_RATIO_WITHIN_OFFICIAL_TRAIN):
    """Reproduce GNCE's released seed-4 80/20 split, then split train for val.

    The paper and released code define no validation set. We preserve the
    released 20% test membership exactly and take the final 10% of the
    shuffled official training portion as validation.
    """
    if int(seed) != GNCE_OFFICIAL_SPLIT_SEED:
        raise ValueError(
            f"GNCE uses the fixed official seed={GNCE_OFFICIAL_SPLIT_SEED}; "
            f"received seed={seed}."
        )
    if not math.isclose(float(train_ratio), GNCE_OFFICIAL_TRAIN_RATIO):
        raise ValueError(
            f"GNCE uses official train_ratio={GNCE_OFFICIAL_TRAIN_RATIO}; "
            f"received {train_ratio}."
        )
    if not math.isclose(
            float(val_ratio_within_train),
            GNCE_VAL_RATIO_WITHIN_OFFICIAL_TRAIN):
        raise ValueError(
            "GNCE validation must be the final "
            f"{GNCE_VAL_RATIO_WITHIN_OFFICIAL_TRAIN:.0%} of the official "
            f"training portion; received {val_ratio_within_train}."
        )
    if int(n) < 3:
        raise ValueError(f"GNCE requires at least 3 positive queries, got {n}.")

    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    trainval_n = int(n * train_ratio)
    trainval = indices[:trainval_n]
    test = indices[trainval_n:]
    val_n = int(round(len(trainval) * val_ratio_within_train))
    val_n = max(1, min(val_n, len(trainval) - 1))
    val = trainval[-val_n:]
    train = trainval[:-val_n]
    split = (
        torch.tensor(sorted(train), dtype=torch.long),
        torch.tensor(sorted(val), dtype=torch.long),
        torch.tensor(sorted(test), dtype=torch.long),
    )
    _validate_gnce_split(*split, n)
    return split


class GNCEQueryDataset(InMemoryDataset):
    def __init__(
            self, root, name, transform=None, pre_transform=None,
            split_seed=GNCE_OFFICIAL_SPLIT_SEED):
        self.gnce_name = name.lower()
        self.kg_name, self.query_type = _parse_gnce_name(name)
        self.split_seed = split_seed
        super().__init__(root, transform, pre_transform)
        cached = _torch_load(self.processed_paths[0])
        self.data = cached['data']
        self.slices = cached['slices']
        self.train_idx = cached['train_idx']
        self.val_idx = cached['val_idx']
        self.test_idx = cached['test_idx']
        self.dim_node_deg = 0
        self.dim_node_label = 0
        self.dim_node_attr = cached['dim_node_attr']
        self.num_edge_classes = cached['num_edge_classes']
        self.has_embeddings = cached.get('has_embeddings', False)
        self.positive_queries = cached.get('positive_queries', len(self.train_idx) + len(self.val_idx) + len(self.test_idx))
        self.filtered_queries = cached.get('filtered_queries', 0)
        self.duplicate_queries_removed = cached.get(
            'duplicate_queries_removed', 0
        )
        self.queries_path = cached.get('queries_path')
        self.graph_path = cached.get('graph_path')
        self.embeddings_path = cached.get('embeddings_path')
        self.feature_mode = cached.get('feature_mode', 'unknown')
        self.split_protocol = cached.get('split_protocol')
        if self.split_protocol != GNCE_SPLIT_PROTOCOL:
            raise RuntimeError(
                f"Stale GNCE cache protocol {self.split_protocol!r}; expected "
                f"{GNCE_SPLIT_PROTOCOL!r}. Remove only this processed cache "
                "and rerun so the fixed official split is regenerated."
            )
        _validate_gnce_split(
            self.train_idx, self.val_idx, self.test_idx, len(self)
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        source_signature = _gnce_source_signature(
            self.raw_dir, self.kg_name, self.query_type
        )
        return (
            f'{self.gnce_name}_{GNCE_SPLIT_PROTOCOL}_'
            f'src{source_signature}_v3.pt'
        )

    def download(self):
        pass

    def process(self):
        queries_path = _resolve_gnce_file(self.raw_dir, self.kg_name, self.query_type, 'queries')
        graph_path = _resolve_gnce_file(self.raw_dir, self.kg_name, self.query_type, 'graph')
        embeddings_path = _resolve_gnce_file(self.raw_dir, self.kg_name, self.query_type, 'embeddings')
        if queries_path is None:
            raise FileNotFoundError(
                f"GNCE queries not found for {self.gnce_name}. Expected "
                f"data/GNCE/raw/{self.kg_name}/{self.query_type}/Joined_Queries.json."
            )
        if graph_path is None:
            raise FileNotFoundError(
                f"GNCE knowledge graph not found for {self.gnce_name}. "
                f"Expected {self.kg_name}/graph/{self.kg_name}.nt or .ttl; "
                "occurrence features must not silently fall back to all zeros."
            )

        with open(queries_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        occurrences = _scan_gnce_occurrences(graph_path)
        embeddings, emb_dim = _load_gnce_embeddings(embeddings_path)
        if embeddings is None:
            print(
                f"[GNCE features] {self.gnce_name}: RDF2Vec embedding JSON "
                "not found; using deterministic occurrence/structure/hash "
                "features."
            )
        else:
            print(
                f"[GNCE features] {self.gnce_name}: loaded RDF2Vec-style "
                f"features from {embeddings_path} (dim={emb_dim})."
            )

        data_list = []
        filtered = 0
        duplicate_queries = 0
        seen_query_keys = set()
        for query in queries:
            card = float(query.get('y', 0))
            if card <= 0:
                filtered += 1
                continue
            query_key = _canonical_gnce_query_key(query)
            if query_key in seen_query_keys:
                duplicate_queries += 1
                continue
            seen_query_keys.add(query_key)
            data = _make_gnce_data(query, self.query_type, embeddings, emb_dim, occurrences)
            data_list.append(data)

        if not data_list:
            raise ValueError(f"No positive GNCE queries found for {self.gnce_name}.")

        if self.pre_filter is not None:
            before = len(data_list)
            data_list = [data for data in data_list if self.pre_filter(data)]
            filtered += before - len(data_list)
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        train_idx, val_idx, test_idx = _make_gnce_split(len(data_list), seed=self.split_seed)
        data, slices = self.collate(data_list)
        torch.save({
            'data': data,
            'slices': slices,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'dim_node_attr': int(data_list[0].x.size(-1)),
            'num_edge_classes': int(data_list[0].edge_attr.size(-1)) if data_list[0].edge_attr is not None else 0,
            'has_embeddings': embeddings is not None,
            'positive_queries': len(data_list),
            'filtered_queries': filtered,
            'duplicate_queries_removed': duplicate_queries,
            'split_protocol': GNCE_SPLIT_PROTOCOL,
            'queries_path': osp.abspath(queries_path),
            'graph_path': osp.abspath(graph_path),
            'embeddings_path': (
                osp.abspath(embeddings_path)
                if embeddings_path is not None else None
            ),
            'feature_mode': (
                'rdf2vec-occurrence'
                if embeddings is not None
                else 'occurrence-structure-hash'
            ),
        }, self.processed_paths[0])


def precompute_centrality(dataset, centrality_type='degree'):
    """
    Compute and inject centrality features

    data.node_c_log: Float Tensor [num_nodes]
    data.edge_c_log: Float Tensor [num_edges]
    """
    print(f"Computing {centrality_type} centrality...")
    data_list = [data for data in dataset]
    
    for data in tqdm(data_list):
        num_nodes = data.num_nodes

        if centrality_type == 'degree':
            # In-degree + Out-degree (Degree for undirected)
            c = degree(data.edge_index[0], num_nodes).float()
            c = c + 1e-6 # Avoid log(0)
            
        elif centrality_type == 'eigenvector':
            try:
                G = to_networkx(data, to_undirected=True)
                c_dict = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
                c = torch.tensor([c_dict.get(i, 0) for i in range(num_nodes)], dtype=torch.float) + 1e-6
            except:
                # Use degree centrality if the iteration does not converge.
                c = degree(data.edge_index[0], num_nodes).float() + 1e-6
                
        elif centrality_type == 'pagerank':
            try:
                G = to_networkx(data)
                c_dict = nx.pagerank(G)
                c = torch.tensor([c_dict.get(i, 0) for i in range(num_nodes)], dtype=torch.float) + 1e-6
            except:
                c = degree(data.edge_index[0], num_nodes).float() + 1e-6
        else:
            raise ValueError(f"Not supported centrality type: {centrality_type}")

        # Edge centrality = avg of endpoints
        row, col = data.edge_index
        c_edge = (c[row] + c[col]) / 2.0

        data.node_c_log = torch.log(c)
        data.edge_c_log = torch.log(c_edge)

    # Repack data
    new_data, new_slices = InMemoryDataset.collate(data_list)
    _set_dataset_storage(dataset, new_data, new_slices)
    
    return dataset


def _load_ogb_dataset(name, root, use_fingerprints, fingerprint_dim, use_rwpe,
                      rwpe_dim, use_lappe, lappe_max_freqs, lappe_eigvec_norm,
                      lappe_laplacian_norm, augm_gca, aug_centrality):
    dataset = PygGraphPropPredDataset(name=name, root=root)
    should_cache = use_fingerprints or use_rwpe or use_lappe or augm_gca
    cache_path = _ogb_preprocess_cache_path(
        root, name, use_fingerprints, fingerprint_dim, use_rwpe, rwpe_dim,
        use_lappe, lappe_max_freqs, augm_gca, aug_centrality
    )
    max_degree = None

    if should_cache and osp.exists(cache_path):
        print(f"Loading cached OGB preprocessing: {cache_path}")
        cached_data, cached_slices, max_degree = _load_preprocess_cache(cache_path)
        _set_dataset_storage(dataset, cached_data, cached_slices)
        if max_degree is not None:
            print(f"Global max node degree: {max_degree} (loaded from cache)")

    data_list = None
    cache_exists = should_cache and osp.exists(cache_path)
    need_preprocess = should_cache and not cache_exists
    if max_degree is None or need_preprocess:
        data_list = [data for data in dataset]

    # global max_degree
    if max_degree is None:
        print("Computing global max degree...")
        max_degree = 0
        for data in data_list:
            if data.edge_index.numel() > 0:
                d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
                max_degree = max(max_degree, int(d.max()))

        dim_node_degree = max_degree + 1
        print(f"Global max node degree: {max_degree} (Dim: {dim_node_degree})")

    if need_preprocess:
        print(
            "Preprocessing OGB dataset "
            f"(Fingerprints={use_fingerprints}, RWPE={use_rwpe}, LapPE={use_lappe})..."
        )

        rwpe_transform = None
        if use_rwpe:
            # Float Tensor [N, rwpe_dim]
            rwpe_transform = T.AddRandomWalkPE(walk_length=rwpe_dim, attr_name='random_walk_pe') 

        smiles_list = []
        if use_fingerprints:
            mapping_path = osp.join(dataset.root, 'mapping', 'mol.csv.gz')
            if not osp.exists(mapping_path):
                raise FileNotFoundError(f"SMILES file not found: {mapping_path}")
            df = pd.read_csv(mapping_path)
            smiles_list = df['smiles'].tolist()
            assert len(smiles_list) == len(data_list), "Len mismatch: SMILES vs Dataset"

        for i, data in enumerate(tqdm(data_list)):
            # Add Fingerprint
            if use_fingerprints:
                mol = Chem.MolFromSmiles(smiles_list[i])
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fingerprint_dim)
                    fp_vec = np.array(fp, dtype=np.float32)
                else:
                    fp_vec = np.zeros((fingerprint_dim,), dtype=np.float32)
                
                # Float Tensor, graph-level
                data.fingerprint = torch.tensor(fp_vec, dtype=torch.float).unsqueeze(0)

            # Add RWPE
            if use_rwpe:
                data = rwpe_transform(data)

            if use_lappe:
                data = add_lap_pe_stats(
                    data,
                    max_freqs=lappe_max_freqs,
                    eigvec_norm=lappe_eigvec_norm,
                    laplacian_norm=lappe_laplacian_norm,
                )

            data_list[i] = data

        # Repack data
        new_data, new_slices = InMemoryDataset.collate(data_list)
        _set_dataset_storage(dataset, new_data, new_slices)

        if augm_gca:
            dataset = precompute_centrality(dataset, centrality_type=aug_centrality)

        _save_preprocess_cache(cache_path, dataset._data if hasattr(dataset, '_data') else dataset.data, dataset.slices, max_degree)
        print(f"Saved OGB preprocessing cache: {cache_path}")

    split_idx = dataset.get_idx_split()
    train_dataset = dataset[split_idx['train']]
    val_dataset = dataset[split_idx['valid']]
    test_dataset = dataset[split_idx['test']]

    if name in ['ogbg-molhiv', 'ogbg-molbace']:
        dataset.task = 'b'
        dataset.key_metric = 'auc'
    elif name in ['ogbg-molpcba']:
        dataset.task = 'ml'
        dataset.key_metric = 'ap_macro'
    elif name in ['ogbg-moltox21']:
        dataset.task = 'ml'
        dataset.key_metric = 'auc_macro'

    dataset.output_dim = dataset.num_tasks
    
    return dataset, train_dataset, val_dataset, test_dataset, max_degree














def _load_tu_dataset(
        name, root, use_rwpe, rwpe_dim, use_lappe=False,
        lappe_max_freqs=8, lappe_eigvec_norm='L2',
        lappe_laplacian_norm='none', augm_gca=False,
        aug_centrality='degree', k=10, seed=2024):
    assert name in TU_DATASETS

    # PyG defaults both flags to False and silently slices continuous TU
    # attributes out of ``x``/``edge_attr`` after loading.  Request the full
    # official feature schema explicitly, then normalize it below.  Discrete
    # node/edge labels are retained by PyG in either mode.
    dataset = TUDataset(
        root=root,
        name=name,
        use_node_attr=True,
        use_edge_attr=True,
    )
    data_list = [data for data in dataset]

    # global max_degree
    print("Computing global max degree...")
    max_degree = 0
    for data in data_list:
        if data.edge_index.numel() > 0:
            d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))
    
    dim_node_degree = max_degree + 1
    print(f"Global max node degree: {max_degree} (Dim: {dim_node_degree})")

    if use_rwpe:
        rwpe_transform = T.AddRandomWalkPE(walk_length=rwpe_dim, attr_name='random_walk_pe')

    dim_node_label = int(getattr(dataset, 'num_node_labels', 0))
    dim_node_attr = int(getattr(dataset, 'num_node_attributes', 0))

    # Establish one node schema for the whole dataset before transformation;
    # every graph must retain the same optional-feature width for batching.
    raw_node_dims = set()
    for graph in data_list:
        raw_x = getattr(graph, 'x', None)
        if raw_x is not None:
            raw_node_dims.add(1 if raw_x.dim() == 1 else int(raw_x.size(-1)))
    if len(raw_node_dims) > 1:
        raise ValueError(
            f"TU dataset {name} has inconsistent raw node feature dimensions: "
            f"{sorted(raw_node_dims)}."
        )
    raw_node_dim = next(iter(raw_node_dims)) if raw_node_dims else 0
    metadata_node_dim = dim_node_label + dim_node_attr
    if raw_node_dim != metadata_node_dim:
        raise ValueError(
            f"TU dataset {name} node metadata says label({dim_node_label}) + "
            f"attr({dim_node_attr})={metadata_node_dim}, but PyG exposes "
            f"raw feature dimension {raw_node_dim}."
        )

    # TU collections are inconsistent about optional attributes.  Normalize
    # edge features once at the loader boundary so every model sees the same
    # schema: preserve a common native dimension when present; otherwise use
    # one constant channel. A constant 1 represents edge existence and avoids
    # erasing messages in multiplicative edge encoders.
    observed_edge_dims = set()
    for data in data_list:
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is not None:
            observed_edge_dims.add(
                1 if edge_attr.dim() == 1 else int(edge_attr.size(-1))
            )
    if len(observed_edge_dims) > 1:
        raise ValueError(
            f"TU dataset {name} has inconsistent edge feature dimensions: "
            f"{sorted(observed_edge_dims)}."
        )
    normalized_edge_dim = (
        next(iter(observed_edge_dims)) if observed_edge_dims else 1
    )
    has_native_edge_features = bool(observed_edge_dims)
    edge_default_graphs = 0
    
    for i, data in enumerate(data_list):
        num_nodes = data.num_nodes
        
        # Degree: One-Hot (Float)
        if data.edge_index.numel() > 0:
            d = degree(data.edge_index[0], num_nodes=num_nodes, dtype=torch.long)
        else:
            d = torch.zeros(num_nodes, dtype=torch.long)
        
        # [num_nodes, dim_degree]
        x_degree = F.one_hot(d, num_classes=dim_node_degree).float()

        # PyG stores TU node attributes first and one-hot node labels last.
        # We expose a common [degree | label | attribute] schema.  Completely
        # Featureless datasets use deterministic one-hot degree features.
        if raw_node_dim == 0:
            data.x = x_degree
        else:
            if data.x is None:
                print(
                    f"[TU features] WARNING: {name} graph {i} has no raw "
                    "node features; filling its optional label/attribute "
                    "channels with zeros while retaining one-hot degree."
                )
                x_origin = torch.zeros((num_nodes, raw_node_dim), dtype=torch.float)
            else:
                x_origin = data.x.float()
                if x_origin.dim() == 1:
                    x_origin = x_origin.view(-1, 1)
            if tuple(x_origin.shape) != (num_nodes, raw_node_dim):
                raise ValueError(
                    f"{name} graph {i} raw node feature shape "
                    f"{tuple(x_origin.shape)} does not match "
                    f"({num_nodes}, {raw_node_dim})."
                )
            x_attr = (
                x_origin[:, :dim_node_attr]
                if dim_node_attr else x_origin.new_zeros((num_nodes, 0))
            )
            x_label = (
                x_origin[:, dim_node_attr:]
                if dim_node_label else x_origin.new_zeros((num_nodes, 0))
            )
            data.x = torch.cat([x_degree, x_label, x_attr], dim=1)

        if data.x is None or data.x.dim() != 2 or data.x.size(0) != num_nodes:
            raise ValueError(
                f"{name} graph {i} produced invalid node features: "
                f"{None if data.x is None else tuple(data.x.shape)}."
            )
        if not torch.isfinite(data.x.float()).all():
            raise ValueError(f"{name} graph {i} has non-finite node features.")

        if data.edge_attr is None:
            # For a completely featureless TU collection, one constant
            # channel represents edge existence.  If the collection normally
            # has native edge features but an individual graph is missing
            # them, zeros mean "unknown/missing" without inventing all native
            # edge types at once.
            default_value = 0.0 if has_native_edge_features else 1.0
            data.edge_attr = torch.full(
                (data.edge_index.size(1), normalized_edge_dim),
                default_value,
                dtype=torch.float,
            )
            edge_default_graphs += 1
        else:
            if data.edge_attr.dim() == 1:
                data.edge_attr = data.edge_attr.view(-1, 1)
            data.edge_attr = data.edge_attr.float()
            if (
                data.edge_attr.size(0) != data.edge_index.size(1)
                or data.edge_attr.size(1) != normalized_edge_dim
            ):
                raise ValueError(
                    f"{name} graph {i} edge feature shape "
                    f"{tuple(data.edge_attr.shape)} does not match "
                    f"E={data.edge_index.size(1)}, dim={normalized_edge_dim}."
                )
            if not torch.isfinite(data.edge_attr).all():
                raise ValueError(f"{name} graph {i} has non-finite edge features.")

        # Add RWPE
        if use_rwpe:
            data = rwpe_transform(data)

        if use_lappe:
            data = add_lap_pe_stats(
                data,
                max_freqs=lappe_max_freqs,
                eigvec_norm=lappe_eigvec_norm,
                laplacian_norm=lappe_laplacian_norm,
            )
        
        data_list[i] = data

    # Repack data
    new_data, new_slices = InMemoryDataset.collate(data_list)
    dataset.data = new_data
    dataset.slices = new_slices

    dataset.dim_node_deg = dim_node_degree
    dataset.dim_node_label = dim_node_label
    dataset.dim_node_attr = dim_node_attr
    dataset.num_edge_classes = normalized_edge_dim
    print(
        f"[TU features] node schema=degree({dim_node_degree}) + "
        f"label({dim_node_label}) + attr({dim_node_attr}); edge_dim="
        f"{normalized_edge_dim}, constant-edge default graphs="
        f"{edge_default_graphs}/{len(data_list)}."
    )

    if augm_gca:
        dataset = precompute_centrality(dataset, centrality_type=aug_centrality)

    # K-Fold CV Split
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    labels = dataset.data.y.view(-1).cpu().numpy()
    
    train_indices_list = []
    test_indices_list = []
    for train_idx, test_idx in skf.split(np.zeros(len(dataset)), labels):
        train_indices_list.append(torch.from_numpy(train_idx).long())
        test_indices_list.append(torch.from_numpy(test_idx).long())

    folds = []
    for i in range(k):
        test_idx = test_indices_list[i]
        val_idx = test_indices_list[(i - 1) % k]
        train_all_idx = train_indices_list[i]
        
        train_idx_set = set(train_all_idx.tolist()) - set(val_idx.tolist())
        train_idx = torch.tensor(sorted(train_idx_set), dtype=torch.long)

        train_d = dataset[train_idx]
        val_d = dataset[val_idx]
        test_d = dataset[test_idx]
        
        folds.append((train_d, val_d, test_d))

    index_folds = []
    for i in range(k):
        test_idx = test_indices_list[i]
        val_idx = test_indices_list[(i - 1) % k]
        train_set = set(range(len(dataset))) - set(test_idx.tolist()) - set(val_idx.tolist())
        index_folds.append((
            torch.tensor(sorted(train_set), dtype=torch.long),
            val_idx,
            test_idx,
        ))
    _validate_rotating_cv_indices(index_folds, len(dataset), k, name)

    if name in ['IMDB-BINARY', 'REDDIT-BINARY', 'MUTAG', 'NCI1', 'DD', 'PROTEINS']:
        dataset.task = 'b'
        dataset.key_metric = 'acc'
        dataset.output_dim = 1
    elif name in ['IMDB-MULTI', 'ENZYMES', 'COLLAB']:
        dataset.task = 'mc'
        dataset.key_metric = 'acc'
        dataset.output_dim = dataset.num_classes

    return dataset, folds, max_degree


def _load_counting_dataset(
        name, root, use_rwpe=False, rwpe_dim=0, use_lappe=False,
        lappe_max_freqs=8, lappe_eigvec_norm='L2',
        lappe_laplacian_norm='none', augm_gca=False,
        aug_centrality='degree', gnn_type=None):
    task_idx = COUNTING_TASK_MAP[name]

    counting_root = osp.join(root, 'SubstructureCounting')
    dataset = GraphCountDataset(counting_root)
    data_list = [data for data in dataset]

    # normalize using train+val
    all_y = []
    for data in data_list:
        if isinstance(data.y, np.ndarray):
            y_val = float(data.y[task_idx])
        else:
            y_val = float(data.y[task_idx].item()) if data.y.dim() > 0 else float(data.y.item())
        all_y.append(y_val)

    all_y = np.array(all_y, dtype=np.float32)

    train_idx = dataset.train_idx.numpy()
    val_idx = dataset.val_idx.numpy()
    train_val_idx = np.concatenate([train_idx, val_idx])
    y_train_val = all_y[train_val_idx]
    y_mean = float(y_train_val.mean())
    y_std = float(y_train_val.std())
    if y_std < 1e-8:
        y_std = 1.0

    all_y_normalized = (all_y - y_mean) / y_std

    # Compute global max_degree and create one-hot degree node features
    print("Computing global max degree for counting dataset...")
    max_degree = 0
    max_num_nodes = 0
    for data in data_list:
        max_num_nodes = max(max_num_nodes, int(data.num_nodes))
        if data.edge_index.numel() > 0:
            d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

    dim_node_degree = max_degree + 1
    print(f"Global max node degree: {max_degree} (Dim: {dim_node_degree})")
    print(
        f"Global max graph size: {max_num_nodes} nodes "
        "(HyMN individualization samples must cover this value)."
    )

    rwpe_transform = None
    if use_rwpe:
        rwpe_transform = T.AddRandomWalkPE(walk_length=rwpe_dim, attr_name='random_walk_pe')

    # Check if the GNN type needs edge-feat
    from models import gnn_type_needs_edge_features
    needs_edge_features = gnn_type is not None and gnn_type_needs_edge_features(gnn_type)

    for i, data in enumerate(data_list):
        num_nodes = data.num_nodes

        if data.edge_index.numel() > 0:
            d = degree(data.edge_index[0], num_nodes=num_nodes, dtype=torch.long)
        else:
            d = torch.zeros(num_nodes, dtype=torch.long)

        x_degree = F.one_hot(d, num_classes=dim_node_degree).float()
        data.x = x_degree

        data.y = torch.tensor([all_y_normalized[i]], dtype=torch.float)

        # Initialize edge feat as zeros if the GNNs need it.
        if needs_edge_features:
            num_edges = data.edge_index.size(1)
            data.edge_attr = torch.zeros(num_edges, 1, dtype=torch.float)
        else:
            data.edge_attr = None

        if use_rwpe:
            data = rwpe_transform(data)

        if use_lappe:
            data = add_lap_pe_stats(
                data,
                max_freqs=lappe_max_freqs,
                eigvec_norm=lappe_eigvec_norm,
                laplacian_norm=lappe_laplacian_norm,
            )

        data_list[i] = data

    # Repack data
    new_data, new_slices = InMemoryDataset.collate(data_list)
    dataset.data = new_data
    dataset.slices = new_slices

    # Store normalization stats
    dataset.y_mean = y_mean
    dataset.y_std = y_std

    # Set dataset metadata
    dataset.dim_node_deg = dim_node_degree
    dataset.dim_node_label = 0
    dataset.dim_node_attr = 0
    dataset.num_edge_classes = 1 if needs_edge_features else 0
    dataset.task = 'reg'
    dataset.key_metric = 'r2'
    dataset.output_dim = 1

    if augm_gca:
        dataset = precompute_centrality(dataset, centrality_type=aug_centrality)

    train_d = dataset[dataset.train_idx]
    val_d = dataset[dataset.val_idx]
    test_d = dataset[dataset.test_idx]
    folds = [(train_d, val_d, test_d)]

    return dataset, folds, max_degree


def _canonical_lrgb_name(name):
    name = name.lower()
    if name not in LRGB_DATASETS:
        raise ValueError(f"Unknown LRGB dataset: {name}. Supported: {sorted(LRGB_DATASETS)}")
    return name


def _preprocess_lrgb_split(dataset, use_rwpe=False, rwpe_dim=0, augm_gca=False, aug_centrality='degree'):
    if not use_rwpe and not augm_gca:
        return dataset

    data_list = [data for data in dataset]
    rwpe_transform = T.AddRandomWalkPE(walk_length=rwpe_dim, attr_name='random_walk_pe') if use_rwpe else None

    for i, data in enumerate(data_list):
        if use_rwpe:
            data = rwpe_transform(data)
        data_list[i] = data

    new_data, new_slices = InMemoryDataset.collate(data_list)
    _set_dataset_storage(dataset, new_data, new_slices)

    if augm_gca:
        dataset = precompute_centrality(dataset, centrality_type=aug_centrality)

    return dataset


def _load_lrgb_dataset(name, root, use_rwpe=False, rwpe_dim=0, augm_gca=False, aug_centrality='degree'):
    name = _canonical_lrgb_name(name)
    train_dataset = LRGBDataset(root=root, name=name, split='train')
    val_dataset = LRGBDataset(root=root, name=name, split='val')
    test_dataset = LRGBDataset(root=root, name=name, split='test')

    print("Computing global max degree for LRGB dataset...")
    max_degree = 0
    for split_dataset in (train_dataset, val_dataset, test_dataset):
        for data in split_dataset:
            if data.edge_index.numel() > 0:
                d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
                max_degree = max(max_degree, int(d.max()))

    print(f"Global max node degree: {max_degree} (Dim: {max_degree + 1})")

    train_dataset = _preprocess_lrgb_split(train_dataset, use_rwpe, rwpe_dim, augm_gca, aug_centrality)
    val_dataset = _preprocess_lrgb_split(val_dataset, use_rwpe, rwpe_dim, augm_gca, aug_centrality)
    test_dataset = _preprocess_lrgb_split(test_dataset, use_rwpe, rwpe_dim, augm_gca, aug_centrality)

    dataset = train_dataset
    output_dim = int(train_dataset[0].y.view(-1).numel())
    dataset.output_dim = output_dim
    dataset.num_tasks = output_dim

    if name == 'peptides-func':
        dataset.task = 'ml'
        dataset.key_metric = 'ap_macro'
    elif name == 'peptides-struct':
        dataset.task = 'reg'
        dataset.key_metric = 'mae'

    return dataset, [(train_dataset, val_dataset, test_dataset)], max_degree


def _canonical_pyg_mol_name(name):
    name = name.lower()
    if name not in PYG_MOL_REGRESSION_DATASETS:
        raise ValueError(f"Unknown PyG molecular regression dataset: {name}. Supported: {sorted(PYG_MOL_REGRESSION_DATASETS)}")
    return name


def _preprocess_pyg_mol_split(dataset, use_rwpe=False, rwpe_dim=0,
                              use_lappe=False, lappe_max_freqs=8,
                              lappe_eigvec_norm='L2', lappe_laplacian_norm='none',
                              augm_gca=False, aug_centrality='degree'):
    data_list = [data for data in dataset]
    rwpe_transform = T.AddRandomWalkPE(walk_length=rwpe_dim, attr_name='random_walk_pe') if use_rwpe else None

    for i, data in enumerate(data_list):
        if data.x.dim() == 1:
            data.x = data.x.unsqueeze(-1)
        if data.edge_attr is None:
            data.edge_attr = data.edge_index.new_zeros((data.edge_index.size(1), 1))
        elif data.edge_attr.dim() == 1:
            data.edge_attr = data.edge_attr.unsqueeze(-1)

        if use_rwpe:
            data = rwpe_transform(data)

        if use_lappe:
            data = add_lap_pe_stats(
                data,
                max_freqs=lappe_max_freqs,
                eigvec_norm=lappe_eigvec_norm,
                laplacian_norm=lappe_laplacian_norm,
            )
        data_list[i] = data

    new_data, new_slices = InMemoryDataset.collate(data_list)
    _set_dataset_storage(dataset, new_data, new_slices)

    if augm_gca:
        dataset = precompute_centrality(dataset, centrality_type=aug_centrality)

    return dataset


def _load_pyg_mol_regression_dataset(name, root, use_rwpe=False, rwpe_dim=0,
                                     use_lappe=False, lappe_max_freqs=8,
                                     lappe_eigvec_norm='L2', lappe_laplacian_norm='none',
                                     augm_gca=False, aug_centrality='degree'):
    name = _canonical_pyg_mol_name(name)
    dataset_root = osp.join(root, 'ZINC' if name == 'zinc' else 'AQSOL')
    dataset_cls = ZINC if name == 'zinc' else AQSOL

    train_dataset = dataset_cls(root=dataset_root, split='train')
    val_dataset = dataset_cls(root=dataset_root, split='val')
    test_dataset = dataset_cls(root=dataset_root, split='test')

    print(f"Computing global max degree for {name.upper()} dataset...")
    max_degree = 0
    for split_dataset in (train_dataset, val_dataset, test_dataset):
        for data in split_dataset:
            if data.edge_index.numel() > 0:
                d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
                max_degree = max(max_degree, int(d.max()))

    print(f"Global max node degree: {max_degree} (Dim: {max_degree + 1})")

    train_dataset = _preprocess_pyg_mol_split(
        train_dataset, use_rwpe, rwpe_dim,
        use_lappe, lappe_max_freqs, lappe_eigvec_norm, lappe_laplacian_norm,
        augm_gca, aug_centrality
    )
    val_dataset = _preprocess_pyg_mol_split(
        val_dataset, use_rwpe, rwpe_dim,
        use_lappe, lappe_max_freqs, lappe_eigvec_norm, lappe_laplacian_norm,
        augm_gca, aug_centrality
    )
    test_dataset = _preprocess_pyg_mol_split(
        test_dataset, use_rwpe, rwpe_dim,
        use_lappe, lappe_max_freqs, lappe_eigvec_norm, lappe_laplacian_norm,
        augm_gca, aug_centrality
    )

    dataset = train_dataset
    output_dim = int(train_dataset[0].y.view(-1).numel())
    dataset.output_dim = output_dim
    dataset.num_tasks = output_dim
    dataset.task = 'reg'
    dataset.key_metric = 'mae'

    return dataset, [(train_dataset, val_dataset, test_dataset)], max_degree


def _normalize_gcare_split(data_list, indices, y_mean, y_std):
    out = []
    for idx in indices.tolist():
        data = data_list[idx].clone()
        y_log = data.y_log.float()
        data.y = (y_log - y_mean) / y_std
        data.y_mean = torch.tensor([y_mean], dtype=torch.float)
        data.y_std = torch.tensor([y_std], dtype=torch.float)
        out.append(data)
    return out


def _lap_pe_topology_key(data):
    """Return an order-independent key for topology-only LapPE reuse.

    G-CARE/GNCE workloads contain many query graphs with the same small
    topology.  Laplacian eigenvectors depend on the topology, not on node or
    edge features, so recomputing them for every one of the 100k+ GNCE
    queries is unnecessary.  Keep multiplicities in the key while treating
    the two stored directions of an undirected edge identically.
    """
    num_nodes = int(data.num_nodes)
    if data.edge_index.numel() == 0:
        return num_nodes, ()
    edge_index = data.edge_index.detach().cpu()
    lo = torch.minimum(edge_index[0], edge_index[1])
    hi = torch.maximum(edge_index[0], edge_index[1])
    edge_codes = (lo * max(num_nodes, 1) + hi).sort().values
    return num_nodes, tuple(int(value) for value in edge_codes.tolist())


def _preprocess_gcare_list(
        data_list, use_rwpe=False, rwpe_dim=0, augm_gca=False,
        aug_centrality='degree', use_lappe=False, lappe_max_freqs=8,
        lappe_eigvec_norm='L2', lappe_laplacian_norm='none'):
    if not use_rwpe and not use_lappe and not augm_gca:
        return data_list

    rwpe_transform = T.AddRandomWalkPE(walk_length=rwpe_dim, attr_name='random_walk_pe') if use_rwpe else None
    lap_pe_cache = {}
    processed = []
    for data in data_list:
        data = data.clone()
        if use_rwpe:
            data = rwpe_transform(data)
        if use_lappe:
            topology_key = _lap_pe_topology_key(data)
            cached_lap_pe = lap_pe_cache.get(topology_key)
            if cached_lap_pe is None:
                data = add_lap_pe_stats(
                    data,
                    max_freqs=lappe_max_freqs,
                    eigvec_norm=lappe_eigvec_norm,
                    laplacian_norm=lappe_laplacian_norm,
                )
                lap_pe_cache[topology_key] = (
                    data.EigVals.clone(), data.EigVecs.clone()
                )
            else:
                data.EigVals = cached_lap_pe[0].clone()
                data.EigVecs = cached_lap_pe[1].clone()
        processed.append(data)

    if use_lappe:
        print(
            f"[LapPE preprocess] graphs={len(processed)}; "
            f"unique_topologies={len(lap_pe_cache)}; "
            f"max_freqs={int(lappe_max_freqs)}; "
            f"eigvec_norm={lappe_eigvec_norm}; "
            f"laplacian_norm={lappe_laplacian_norm}."
        )

    if augm_gca:
        for data in processed:
            num_nodes = data.num_nodes
            if aug_centrality == 'degree':
                c = degree(data.edge_index[0], num_nodes).float() + 1e-6
            elif aug_centrality == 'eigenvector':
                try:
                    G = to_networkx(data, to_undirected=True)
                    c_dict = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
                    c = torch.tensor([c_dict.get(i, 0) for i in range(num_nodes)], dtype=torch.float) + 1e-6
                except Exception:
                    c = degree(data.edge_index[0], num_nodes).float() + 1e-6
            elif aug_centrality == 'pagerank':
                try:
                    G = to_networkx(data)
                    c_dict = nx.pagerank(G)
                    c = torch.tensor([c_dict.get(i, 0) for i in range(num_nodes)], dtype=torch.float) + 1e-6
                except Exception:
                    c = degree(data.edge_index[0], num_nodes).float() + 1e-6
            else:
                raise ValueError(f"Not supported centrality type: {aug_centrality}")
            row, col = data.edge_index
            c_edge = (c[row] + c[col]) / 2.0
            data.node_c_log = torch.log(c)
            data.edge_c_log = torch.log(c_edge)
    return processed


def _load_gcare_dataset(name, root, use_rwpe=False, rwpe_dim=0,
                        use_lappe=False, lappe_max_freqs=8,
                        lappe_eigvec_norm='L2',
                        lappe_laplacian_norm='none',
                        augm_gca=False, aug_centrality='degree',
                        fold_k=5, fold_seed=2024):
    gcare_root = osp.join(root, 'GCARE')
    dataset = GCAREQueryDataset(gcare_root, name, fold_k=fold_k, fold_seed=fold_seed)
    data_list = [data for data in dataset]

    print(f"Computing global max degree for {name} dataset...")
    max_degree = 0
    for data in data_list:
        if data.edge_index.numel() > 0:
            d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))
    print(f"Global max node degree: {max_degree} (Dim: {max_degree + 1})")

    data_list = _preprocess_gcare_list(
        data_list,
        use_rwpe=use_rwpe,
        rwpe_dim=rwpe_dim,
        augm_gca=augm_gca,
        aug_centrality=aug_centrality,
        use_lappe=use_lappe,
        lappe_max_freqs=lappe_max_freqs,
        lappe_eigvec_norm=lappe_eigvec_norm,
        lappe_laplacian_norm=lappe_laplacian_norm,
    )

    folds = []
    for train_idx, val_idx, test_idx in dataset.folds_idx:
        y_logs = torch.stack([data_list[int(i)].y_log.float().view(-1)[0] for i in train_idx])
        y_mean = float(y_logs.mean().item())
        y_std = float(y_logs.std(unbiased=False).item())
        if y_std < 1e-8:
            y_std = 1.0

        train_d = _normalize_gcare_split(data_list, train_idx, y_mean, y_std)
        val_d = _normalize_gcare_split(data_list, val_idx, y_mean, y_std)
        test_d = _normalize_gcare_split(data_list, test_idx, y_mean, y_std)
        folds.append((train_d, val_d, test_d))

    dataset.task = 'card'
    dataset.key_metric = 'medq'
    dataset.output_dim = 1
    dataset.num_tasks = 1
    return dataset, folds, max_degree


def _load_gnce_dataset(name, root, use_rwpe=False, rwpe_dim=0,
                       use_lappe=False, lappe_max_freqs=8,
                       lappe_eigvec_norm='L2',
                       lappe_laplacian_norm='none',
                       augm_gca=False, aug_centrality='degree',
                       fold_seed=GNCE_OFFICIAL_SPLIT_SEED):
    gnce_root = osp.join(root, 'GNCE')
    dataset = GNCEQueryDataset(gnce_root, name, split_seed=fold_seed)
    data_list = [data for data in dataset]

    print(f"Computing global max degree for {name} dataset...")
    max_degree = 0
    for data in data_list:
        if data.edge_index.numel() > 0:
            d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))
    print(f"Global max node degree: {max_degree} (Dim: {max_degree + 1})")

    data_list = _preprocess_gcare_list(
        data_list,
        use_rwpe=use_rwpe,
        rwpe_dim=rwpe_dim,
        augm_gca=augm_gca,
        aug_centrality=aug_centrality,
        use_lappe=use_lappe,
        lappe_max_freqs=lappe_max_freqs,
        lappe_eigvec_norm=lappe_eigvec_norm,
        lappe_laplacian_norm=lappe_laplacian_norm,
    )

    train_idx, val_idx, test_idx = dataset.train_idx, dataset.val_idx, dataset.test_idx
    print(
        "[GNCE split] "
        f"protocol={dataset.split_protocol}; official shuffle seed="
        f"{GNCE_OFFICIAL_SPLIT_SEED}; train/val/test="
        f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)} "
        f"({len(train_idx) / len(dataset):.1%}/"
        f"{len(val_idx) / len(dataset):.1%}/"
        f"{len(test_idx) / len(dataset):.1%})."
    )
    print(
        f"[GNCE queries] positive={dataset.positive_queries}; "
        f"nonpositive_filtered={dataset.filtered_queries}; "
        f"isomorphic_duplicates_removed={dataset.duplicate_queries_removed}."
    )
    print(
        f"[GNCE features] mode={dataset.feature_mode}; "
        f"queries={dataset.queries_path}; graph={dataset.graph_path}; "
        f"embeddings={dataset.embeddings_path or 'NA'}."
    )
    y_logs = torch.stack([data_list[int(i)].y_log.float().view(-1)[0] for i in train_idx])
    y_mean = float(y_logs.mean().item())
    y_std = float(y_logs.std(unbiased=False).item())
    if y_std < 1e-8:
        y_std = 1.0

    train_d = _normalize_gcare_split(data_list, train_idx, y_mean, y_std)
    val_d = _normalize_gcare_split(data_list, val_idx, y_mean, y_std)
    test_d = _normalize_gcare_split(data_list, test_idx, y_mean, y_std)

    dataset.task = 'card'
    dataset.key_metric = 'medq'
    dataset.output_dim = 1
    dataset.num_tasks = 1
    return dataset, [(train_d, val_d, test_d)], max_degree


def _preprocess_good_split(
        dataset, model_dataset_type, task, use_rwpe=False, rwpe_dim=0,
        use_lappe=False, lappe_max_freqs=8, lappe_eigvec_norm='L2',
        lappe_laplacian_norm='none', augm_gca=False,
        aug_centrality='degree'):
    """Adapt one official GOOD split without changing membership or labels."""
    data_list = [data.clone() for data in dataset]
    rwpe_transform = (
        T.AddRandomWalkPE(walk_length=rwpe_dim, attr_name='random_walk_pe')
        if use_rwpe else None
    )

    for index, data in enumerate(data_list):
        if data.x is None:
            data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)
        elif data.x.dim() == 1:
            data.x = data.x.unsqueeze(-1)

        # GOOD molecular releases use GOOD.utils.data.x_map/e_map categorical
        # columns.  Preserve those integer IDs for the exact GOOD encoders.
        if model_dataset_type == 'good_mol':
            data.x = data.x.long()
            if data.edge_attr is None:
                data.edge_attr = torch.zeros(
                    (data.edge_index.size(1), 3), dtype=torch.long
                )
            elif data.edge_attr.dim() == 1:
                data.edge_attr = data.edge_attr.unsqueeze(-1)
            data.edge_attr = data.edge_attr.long()
        else:
            data.x = data.x.float()
            if data.edge_attr is None:
                data.edge_attr = torch.zeros(
                    (data.edge_index.size(1), 1), dtype=torch.float
                )
            elif data.edge_attr.dim() == 1:
                data.edge_attr = data.edge_attr.unsqueeze(-1)
            data.edge_attr = data.edge_attr.float()

        if data.y is not None:
            # PyG concatenates graph attributes along dimension 0.  A
            # multi-task target therefore needs its leading graph dimension
            # so a batch is [num_graphs, num_tasks], not a flat
            # [num_graphs * num_tasks] vector.
            data.y = (
                data.y.reshape(1, -1)
                if task == 'ml' else data.y.view(-1)
            )

        if rwpe_transform is not None:
            data = rwpe_transform(data)
        if use_lappe:
            data = add_lap_pe_stats(
                data,
                max_freqs=lappe_max_freqs,
                eigvec_norm=lappe_eigvec_norm,
                laplacian_norm=lappe_laplacian_norm,
            )
        data_list[index] = data

    new_data, new_slices = InMemoryDataset.collate(data_list)
    _set_dataset_storage(dataset, new_data, new_slices)
    if augm_gca:
        dataset = precompute_centrality(
            dataset, centrality_type=aug_centrality
        )
    return dataset


def _good_split_ids(dataset):
    ids = []
    for data in dataset:
        value = getattr(data, 'idx', None)
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.view(-1)
            if value.numel() != 1:
                return None
            value = int(value.item())
        ids.append(value)
    return set(ids)


def _validate_good_split_membership(splits, canonical_name):
    """Use GOOD's stable source ``idx`` when present to guard overlap."""
    names = ['train', 'val', 'test']
    if splits.get('id_val') is not None:
        names.extend(['id_val', 'id_test'])
    id_sets = {key: _good_split_ids(splits[key]) for key in names}
    if any(value is None for value in id_sets.values()):
        print(
            f"[GOOD split] {canonical_name}: source idx is unavailable in at "
            "least one artifact; using the official split files as released."
        )
        return
    for left_idx, left in enumerate(names):
        for right in names[left_idx + 1:]:
            overlap = id_sets[left] & id_sets[right]
            if overlap:
                raise AssertionError(
                    f"{canonical_name}: official split artifacts {left} and "
                    f"{right} overlap on {len(overlap)} source graph ids."
                )


def _validate_good_split_roles(splits, shift, canonical_name):
    """Guard the official GOOD OOD/ID split semantics at runtime."""
    required = ('train', 'val', 'test')
    for role in required:
        split = splits.get(role)
        if split is None:
            raise AssertionError(
                f"{canonical_name}: missing official GOOD {role} split."
            )
        if getattr(split, 'shift', None) != shift:
            raise AssertionError(
                f"{canonical_name}: {role} has shift="
                f"{getattr(split, 'shift', None)!r}, expected {shift!r}."
            )
        if getattr(split, 'subset', None) != role:
            raise AssertionError(
                f"{canonical_name}: OpenGLT role {role!r} is backed by "
                f"GOOD subset={getattr(split, 'subset', None)!r}."
            )
        expected_file = f'{shift}_{role}.pt'
        if getattr(split, '_split_file_name', None) != expected_file:
            raise AssertionError(
                f"{canonical_name}: OpenGLT role {role!r} must load "
                f"{expected_file}, got "
                f"{getattr(split, '_split_file_name', None)!r}."
            )

    if shift == 'no_shift':
        if splits.get('id_val') is not None or splits.get('id_test') is not None:
            raise AssertionError(
                f"{canonical_name}: no-shift must not expose separate ID "
                "validation/test subsets."
            )
        print(
            f"[GOOD split roles] {canonical_name}: no-shift "
            "train/validation/test (OOD/ID distinction is not applicable)."
        )
        return

    for role in ('id_val', 'id_test'):
        split = splits.get(role)
        if split is None or getattr(split, 'subset', None) != role:
            raise AssertionError(
                f"{canonical_name}: missing or misrouted official GOOD "
                f"{role} split."
            )
        expected_file = f'{shift}_{role}.pt'
        if getattr(split, '_split_file_name', None) != expected_file:
            raise AssertionError(
                f"{canonical_name}: {role} must load {expected_file}."
            )

    if (
            splits['val'] is splits['id_val']
            or splits['test'] is splits['id_test']):
        raise AssertionError(
            f"{canonical_name}: OOD validation/test aliases an ID split."
        )
    print(
        f"[GOOD split roles] {canonical_name}: OpenGLT validation="
        f"OOD({shift}_val.pt), test=OOD({shift}_test.pt); "
        f"ID-only artifacts {shift}_id_val.pt/{shift}_id_test.pt are "
        "kept separate."
    )


def _load_good_dataset(
        name, root, use_rwpe=False, rwpe_dim=0, use_lappe=False,
        lappe_max_freqs=8, lappe_eigvec_norm='L2',
        lappe_laplacian_norm='none', augm_gca=False,
        aug_centrality='degree'):
    good_root = osp.join(root, 'GOOD')
    splits, spec, domain, shift, canonical_name = load_official_good_splits(
        name, good_root
    )
    _validate_good_split_roles(splits, shift, canonical_name)
    _validate_good_split_membership(splits, canonical_name)

    for split_name in ('train', 'id_val', 'id_test', 'val', 'test'):
        split = splits.get(split_name)
        if split is None:
            continue
        splits[split_name] = _preprocess_good_split(
            split,
            spec.model_dataset_type,
            spec.task,
            use_rwpe=use_rwpe,
            rwpe_dim=rwpe_dim,
            use_lappe=use_lappe,
            lappe_max_freqs=lappe_max_freqs,
            lappe_eigvec_norm=lappe_eigvec_norm,
            lappe_laplacian_norm=lappe_laplacian_norm,
            augm_gca=augm_gca,
            aug_centrality=aug_centrality,
        )

    max_degree = 0
    # PNA degree statistics and their tensor bound must be training-only.
    for data in splits['train']:
        if data.edge_index.numel() > 0:
            d = degree(
                data.edge_index[0],
                num_nodes=data.num_nodes,
                dtype=torch.long,
            )
            max_degree = max(max_degree, int(d.max()))

    dataset = splits['train']
    sample = dataset[0]
    observed_output_dim = int(sample.y.view(-1).numel())
    if spec.task != 'mc' and observed_output_dim != spec.output_dim:
        raise ValueError(
            f"{canonical_name}: expected output_dim={spec.output_dim}, "
            f"observed {observed_output_dim} in the official train artifact."
        )
    if spec.task == 'mc':
        labels = torch.cat([
            data.y.view(-1).long() for data in dataset
            if data.y is not None
        ])
        if labels.numel() and int(labels.max()) >= spec.output_dim:
            raise ValueError(
                f"{canonical_name}: observed class id {int(labels.max())}, "
                f"but output_dim={spec.output_dim}."
            )

    node_dim = int(sample.x.size(-1))
    edge_dim = (
        int(sample.edge_attr.size(-1))
        if sample.edge_attr is not None and sample.edge_attr.dim() > 1 else 1
    )
    dataset.task = spec.task
    dataset.key_metric = spec.key_metric
    dataset.output_dim = spec.output_dim
    dataset.num_tasks = spec.output_dim
    dataset.model_dataset_type = spec.model_dataset_type
    dataset.dim_node_deg = node_dim
    dataset.dim_node_label = 0
    dataset.dim_node_attr = 0
    dataset.num_edge_classes = edge_dim
    dataset.good_name = canonical_name
    dataset.good_official_name = spec.official_name
    dataset.good_domain = domain
    dataset.good_shift = shift
    dataset.good_validation_role = (
        'no_shift' if shift == 'no_shift' else 'ood'
    )
    dataset.good_test_role = (
        'no_shift' if shift == 'no_shift' else 'ood'
    )
    dataset.good_splits = splits

    split_sizes = {
        key: (len(value) if value is not None else 0)
        for key, value in splits.items()
        if key in ('train', 'id_val', 'id_test', 'val', 'test')
    }
    print(
        f"[GOOD] dataset={canonical_name}; official={spec.official_name}; "
        f"domain={domain}; shift={shift}; task={spec.task}; "
        f"key_metric={spec.key_metric}; model_features="
        f"{spec.model_dataset_type}."
    )
    print(
        f"[GOOD split] official released membership; sizes={split_sizes}; "
        f"validation_role={dataset.good_validation_role}; "
        f"test_role={dataset.good_test_role}."
    )
    if shift == 'no_shift':
        print(
            "[GOOD split] no-shift has train/val/test only; there are no "
            "separate id_val/id_test artifacts by official definition."
        )

    return dataset, [
        (splits['train'], splits['val'], splits['test'])
    ], max_degree


from engine import seed_everything

def load_dataset(dataset_name,
                 root='./data',
                 use_fingerprints=False,
                 fingerprint_dim=0,
                 use_rwpe=False,
                 rwpe_dim=0,
                 use_lappe=False,
                 lappe_max_freqs=8,
                 lappe_eigvec_norm='L2',
                 lappe_laplacian_norm='none',
                 augm_gca=False,
                 aug_centrality='degree',
                 fold_k=None,
                 fold_seed=2024,
                 gnn_type=None,
                 ):
    requested_dataset_name = dataset_name
    base_dataset_name = str(dataset_name)
    cv_k, cv_seed = resolve_cv_parameters(
        base_dataset_name, fold_k, fold_seed
    )
    seed_everything(cv_seed if cv_seed is not None else fold_seed)
    if base_dataset_name in OGB_DATASETS:
        dataset, train_d, val_d, test_d, max_degree = _load_ogb_dataset(
            base_dataset_name, root,
            use_fingerprints, fingerprint_dim, 
            use_rwpe, rwpe_dim,
            use_lappe, lappe_max_freqs, lappe_eigvec_norm, lappe_laplacian_norm,
            augm_gca, aug_centrality
        )
        fold = [(train_d, val_d, test_d)]
        
        return dataset, fold, max_degree

    elif base_dataset_name.startswith('ogbg-'):
        raise ValueError(
            f"Unknown OGB dataset: {base_dataset_name}. "
            f"Supported: {sorted(OGB_DATASETS)}"
        )

    elif is_good_dataset_name(base_dataset_name):
        dataset, folds, max_degree = _load_good_dataset(
            base_dataset_name,
            root,
            use_rwpe=use_rwpe,
            rwpe_dim=rwpe_dim,
            use_lappe=use_lappe,
            lappe_max_freqs=lappe_max_freqs,
            lappe_eigvec_norm=lappe_eigvec_norm,
            lappe_laplacian_norm=lappe_laplacian_norm,
            augm_gca=augm_gca,
            aug_centrality=aug_centrality,
        )
        return dataset, folds, max_degree

    elif base_dataset_name in TU_DATASETS:
        dataset, folds, max_degree = _load_tu_dataset(
            base_dataset_name, root,
            use_rwpe=use_rwpe,
            rwpe_dim=rwpe_dim,
            use_lappe=use_lappe,
            lappe_max_freqs=lappe_max_freqs,
            lappe_eigvec_norm=lappe_eigvec_norm,
            lappe_laplacian_norm=lappe_laplacian_norm,
            augm_gca=augm_gca,
            aug_centrality=aug_centrality,
            k=cv_k, seed=cv_seed,
        )

        return dataset, folds, max_degree

    elif base_dataset_name.startswith('Count-'):
        counting_name = base_dataset_name[len('Count-'):]  # e.g. 'Count-3-Cycle' -> '3-Cycle'
        if counting_name not in COUNTING_TASK_MAP:
            raise ValueError(f"Unknown counting task: {counting_name}. Supported: Count-{COUNTING_DATASETS}")

        dataset, folds, max_degree = _load_counting_dataset(
            counting_name, root,
            use_rwpe=use_rwpe, rwpe_dim=rwpe_dim,
            use_lappe=use_lappe,
            lappe_max_freqs=lappe_max_freqs,
            lappe_eigvec_norm=lappe_eigvec_norm,
            lappe_laplacian_norm=lappe_laplacian_norm,
            augm_gca=augm_gca, aug_centrality=aug_centrality,
            gnn_type=gnn_type,
        )

        return dataset, folds, max_degree

    elif base_dataset_name.lower() in LRGB_DATASETS:
        dataset, folds, max_degree = _load_lrgb_dataset(
            base_dataset_name,
            root,
            use_rwpe=use_rwpe,
            rwpe_dim=rwpe_dim,
            augm_gca=augm_gca,
            aug_centrality=aug_centrality
        )

        return dataset, folds, max_degree

    elif base_dataset_name.lower() in PYG_MOL_REGRESSION_DATASETS:
        dataset, folds, max_degree = _load_pyg_mol_regression_dataset(
            base_dataset_name,
            root,
            use_rwpe=use_rwpe,
            rwpe_dim=rwpe_dim,
            use_lappe=use_lappe,
            lappe_max_freqs=lappe_max_freqs,
            lappe_eigvec_norm=lappe_eigvec_norm,
            lappe_laplacian_norm=lappe_laplacian_norm,
            augm_gca=augm_gca,
            aug_centrality=aug_centrality
        )

        return dataset, folds, max_degree

    elif base_dataset_name.lower() in GCARE_DATASETS:
        dataset, folds, max_degree = _load_gcare_dataset(
            base_dataset_name,
            root,
            use_rwpe=use_rwpe,
            rwpe_dim=rwpe_dim,
            use_lappe=use_lappe,
            lappe_max_freqs=lappe_max_freqs,
            lappe_eigvec_norm=lappe_eigvec_norm,
            lappe_laplacian_norm=lappe_laplacian_norm,
            augm_gca=augm_gca,
            aug_centrality=aug_centrality,
            fold_k=cv_k,
            fold_seed=cv_seed,
        )

        return dataset, folds, max_degree

    elif base_dataset_name.lower().startswith(GNCE_DATA_PREFIX):
        dataset, folds, max_degree = _load_gnce_dataset(
            base_dataset_name,
            root,
            use_rwpe=use_rwpe,
            rwpe_dim=rwpe_dim,
            use_lappe=use_lappe,
            lappe_max_freqs=lappe_max_freqs,
            lappe_eigvec_norm=lappe_eigvec_norm,
            lappe_laplacian_norm=lappe_laplacian_norm,
            augm_gca=augm_gca,
            aug_centrality=aug_centrality,
            fold_seed=GNCE_OFFICIAL_SPLIT_SEED,
        )

        return dataset, folds, max_degree

    else:
        raise ValueError(f"Unknown dataset: {requested_dataset_name}")
