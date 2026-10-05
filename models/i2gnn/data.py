import os
import os.path as osp

import torch
from tqdm import tqdm

from .utils import create_subgraphs2


CACHE_VERSION = 'v1'


COUNTING_H_DEFAULTS = { # From I2GNN official code
    'Count-3-Cycle': 1,
    'Count-4-Cycle': 2,
    'Count-5-Cycle': 2,
    'Count-6-Cycle': 3,
    'Count-4-Clique': 1,
    'Count-Tri-Rect': 2,
}


def default_i2gnn_h(dataset):
    dataset_lower = dataset.lower()
    if dataset in {'ogbg-molhiv', 'ogbg-molbace'}:
        return 4
    if dataset.startswith('ogbg-'):
        return 3
    if dataset_lower in {'zinc', 'aqsol'}:
        return 3
    if dataset_lower in {'peptides-func', 'peptides-struct'}:
        return 2
    if dataset_lower in {'gcare-aids', 'gcare-yago'} or dataset_lower.startswith('gnce-'):
        return 2
    if dataset in {
        'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB',
        'MUTAG', 'NCI1', 'DD', 'ENZYMES', 'PROTEINS'
    }:
        return 2
    if dataset in COUNTING_H_DEFAULTS:
        return COUNTING_H_DEFAULTS[dataset]
    if dataset.startswith('Count-'):
        return 2
    return 3


def i2gnn_dataset_type(dataset):
    dataset_lower = dataset.lower()
    if dataset.startswith('ogbg-'):
        return 'ogb'
    if dataset.startswith('Count-'):
        return 'counting'
    if dataset_lower in {'zinc', 'aqsol'}:
        return 'zinc'
    if dataset_lower in {'peptides-func', 'peptides-struct'}:
        return 'ogb'
    if dataset_lower in {'gcare-aids', 'gcare-yago'} or dataset_lower.startswith('gnce-'):
        return 'generic'
    if dataset in {
        'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB',
        'MUTAG', 'NCI1', 'DD', 'ENZYMES', 'PROTEINS'
    }:
        return 'generic'
    raise ValueError(f"I2GNN runner does not support dataset={dataset}.")


def _torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _cache_path(cache_root, dataset_name, split_name, h, node_label, use_rd,
                max_nodes_per_hop, self_loop, center_idx):
    safe_dataset = dataset_name.replace('/', '_')
    mnph = max_nodes_per_hop if max_nodes_per_hop is not None else 'none'
    filename = (
        f"{CACHE_VERSION}_{safe_dataset}_{split_name}"
        f"_h{h}_{node_label}_rd{int(use_rd)}"
        f"_mnph{mnph}_self{int(self_loop)}_center{int(center_idx)}.pt"
    )
    return osp.join(cache_root, filename)


def prepare_base_data(data, dataset_type):
    data = data.clone()

    if dataset_type == 'zinc':
        if data.x is not None:
            data.x = data.x.view(-1).long()
        if data.edge_attr is None:
            data.edge_attr = data.edge_index.new_zeros(data.edge_index.size(1))
        else:
            data.edge_attr = data.edge_attr.view(-1).long()
    elif dataset_type == 'ogb':
        if data.x is not None and data.x.dtype == torch.float:
            data.x = data.x.long()
        if data.edge_attr is not None and data.edge_attr.dtype == torch.float:
            data.edge_attr = data.edge_attr.long()
    elif dataset_type == 'counting':
        data.edge_attr = None
    elif dataset_type == 'generic':
        if data.x is None:
            data.x = torch.ones(data.num_nodes, 1, dtype=torch.float)
        elif data.x.dim() == 1:
            data.x = data.x.unsqueeze(-1)
        data.x = data.x.float()
        if data.edge_attr is not None and data.edge_attr.dim() == 1:
            data.edge_attr = data.edge_attr.unsqueeze(-1)

    return data


def transform_i2gnn_data(data, dataset_type, h, node_label='spd',
                        use_rd=False, max_nodes_per_hop=None,
                        center_idx=True, self_loop=False):
    data = prepare_base_data(data, dataset_type)
    return create_subgraphs2(
        data,
        h,
        max_nodes_per_hop=max_nodes_per_hop,
        node_label=node_label,
        use_rd=use_rd,
        center_idx=center_idx,
        self_loop=self_loop,
    )


def build_i2gnn_split(dataset, dataset_name, split_name, cache_root,
                     h, node_label='spd', use_rd=False,
                     max_nodes_per_hop=None, center_idx=True,
                     self_loop=False, force_reload=False):
    os.makedirs(cache_root, exist_ok=True)
    dataset_type = i2gnn_dataset_type(dataset_name)
    cache_path = _cache_path(
        cache_root, dataset_name, split_name, h, node_label, use_rd,
        max_nodes_per_hop, self_loop, center_idx,
    )
    if osp.exists(cache_path) and not force_reload:
        return _torch_load(cache_path)

    processed = []
    for data in tqdm(dataset, desc=f"I2GNN preprocess {dataset_name}/{split_name}", ncols=80):
        processed.append(
            transform_i2gnn_data(
                data,
                dataset_type,
                h=h,
                node_label=node_label,
                use_rd=use_rd,
                max_nodes_per_hop=max_nodes_per_hop,
                center_idx=center_idx,
                self_loop=self_loop,
            )
        )

    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    torch.save(processed, tmp_path)
    os.replace(tmp_path, cache_path)
    return processed
