import os
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, subgraph, to_dense_adj
from tqdm import tqdm


def _torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def bfs(adj, start, max_sub_size=10):
    sub = []
    visited = [False for _ in range(adj.size(0))]
    queue = [int(start)]
    visited[int(start)] = True
    while queue:
        node = queue.pop(0)
        sub.append(node)
        if len(sub) >= max_sub_size:
            break
        neigh = torch.where(adj[node] > 0)[0].tolist()
        for nxt in neigh:
            if not visited[nxt]:
                visited[nxt] = True
                queue.append(nxt)
    sub.sort()
    return sub


def parse_bfs(graph, max_n_sub=5, max_sub_size=10, **kwargs):
    if graph.num_nodes == 0:
        return []
    if graph.edge_index.numel() == 0:
        return [list(range(min(graph.num_nodes, max_sub_size)))]
    adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes)[0]
    deg = degree(graph.edge_index[0], graph.num_nodes)
    seeds = deg.sort(descending=True)[1][:max_n_sub]
    seen = set()
    subs = []
    for seed in seeds:
        nodes = bfs(adj, int(seed), max_sub_size)
        key = tuple(nodes)
        if key not in seen:
            seen.add(key)
            subs.append(nodes)
    if not subs:
        subs.append(list(range(min(graph.num_nodes, max_sub_size))))
    return subs


def original(graph, **kwargs):
    return [list(range(int(graph.num_nodes)))]


PARSERS = {
    'parse_bfs': parse_bfs,
    'original': original,
}


def _graph_idx(graph, default):
    if hasattr(graph, 'idx'):
        idx = graph.idx
        return int(idx.item()) if torch.is_tensor(idx) else int(idx)
    return int(default)


def _full_edge_index(num_nodes):
    idx = torch.arange(num_nodes, dtype=torch.long)
    row, col = torch.meshgrid(idx, idx, indexing='ij')
    return torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)


def prepare_parsed_data(dataset, parsing='parse_bfs', parsing_params=None,
                        require_full_edge=False, desc='MOSGSL preprocess'):
    if parsing not in PARSERS:
        raise ValueError(f"Unknown MOSGSL parser: {parsing}. Supported: {sorted(PARSERS)}")
    parser = PARSERS[parsing]
    parsing_params = parsing_params or {}
    graphs2subgraphs = {}
    iterator = tqdm(
        enumerate(dataset), total=len(dataset), desc=desc,
        disable=desc is None,
    )
    for default_idx, graph in iterator:
        graph_idx = _graph_idx(graph, default_idx)
        subs_of_graph = []
        for nodes_sub in parser(graph, **parsing_params):
            nodes_sub = torch.tensor(nodes_sub, dtype=torch.long)
            if nodes_sub.numel() == 0:
                continue
            x_sub = graph.x[nodes_sub]
            edge_sub = subgraph(
                nodes_sub,
                graph.edge_index,
                relabel_nodes=True,
                num_nodes=graph.num_nodes,
            )[0]
            sub_data = Data(
                x=x_sub,
                edge_index=edge_sub,
                y=graph.y,
                belong=torch.tensor([graph_idx], dtype=torch.long),
                mapping=nodes_sub,
            )
            if require_full_edge:
                sub_data.full_edge_index = _full_edge_index(nodes_sub.numel())
            subs_of_graph.append(sub_data)
        if not subs_of_graph:
            node = torch.tensor([0], dtype=torch.long)
            sub_data = Data(
                x=graph.x[node],
                edge_index=torch.empty((2, 0), dtype=torch.long),
                y=graph.y,
                belong=torch.tensor([graph_idx], dtype=torch.long),
                mapping=node,
            )
            if require_full_edge:
                sub_data.full_edge_index = _full_edge_index(1)
            subs_of_graph.append(sub_data)
        graphs2subgraphs[graph_idx] = subs_of_graph
    return graphs2subgraphs


def _sanitize_feature(data):
    data = data.clone()
    num_nodes = data.num_nodes
    if data.x is None:
        data.x = torch.ones(num_nodes, 1, dtype=torch.float)
    elif data.x.dim() == 1:
        data.x = data.x.view(-1, 1).float()
    else:
        data.x = data.x.float()
    if data.y is not None and not torch.is_tensor(data.y):
        data.y = torch.tensor(data.y)
    return data


def _append_edge_context(data):
    """Lift directed edge semantics into node features for edge-blind MOSGSL.

    GNCE stores predicate/occurrence/direction information in ``edge_attr``.
    MOSGSL's learned adjacency is scalar-valued, so it cannot carry those raw
    vectors after graph reconstruction.  Incoming/outgoing mean aggregation
    preserves the information on the incident nodes without changing the
    query topology or using labels.
    """
    if data.edge_attr is None or data.edge_index.numel() == 0:
        return data
    edge_attr = data.edge_attr.float()
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.view(-1, 1)
    if edge_attr.size(0) != data.edge_index.size(1):
        raise ValueError(
            "MOSGSL edge-context adapter requires one edge feature row per edge."
        )

    src, dst = data.edge_index
    n_nodes = int(data.num_nodes)
    feat_dim = int(edge_attr.size(1))
    outgoing = edge_attr.new_zeros((n_nodes, feat_dim))
    incoming = edge_attr.new_zeros((n_nodes, feat_dim))
    outgoing.index_add_(0, src, edge_attr)
    incoming.index_add_(0, dst, edge_attr)

    one = edge_attr.new_ones(edge_attr.size(0))
    out_count = edge_attr.new_zeros(n_nodes)
    in_count = edge_attr.new_zeros(n_nodes)
    out_count.index_add_(0, src, one)
    in_count.index_add_(0, dst, one)
    outgoing = outgoing / out_count.clamp_min(1.0).unsqueeze(1)
    incoming = incoming / in_count.clamp_min(1.0).unsqueeze(1)

    data.x = torch.cat([data.x.float(), incoming, outgoing], dim=-1)
    data.mosgsl_edge_context_dim = torch.tensor([2 * feat_dim], dtype=torch.long)
    return data


def clone_with_global_idx(split, start_idx, use_edge_context=False):
    out = []
    next_idx = start_idx
    for data in split:
        data = _sanitize_feature(data)
        if use_edge_context:
            data = _append_edge_context(data)
        data.idx = torch.tensor([next_idx], dtype=torch.long)
        out.append(data)
        next_idx += 1
    return out, next_idx


def _cache_name(dataset_name, fold_name, parsing, parsing_params, require_full_edge):
    parts = [
        dataset_name.replace('/', '_'),
        fold_name,
        parsing,
        f"full{int(require_full_edge)}",
    ]
    for key in sorted((parsing_params or {}).keys()):
        parts.append(f"{key}{parsing_params[key]}")
    return "_".join(parts) + ".pt"


def build_parse_cache(data_list, dataset_name, fold_name, cache_dir,
                      parsing='parse_bfs', parsing_params=None,
                      require_full_edge=False, force_reload=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _cache_name(
        dataset_name, fold_name, parsing, parsing_params, require_full_edge
    )
    if cache_path.exists() and not force_reload:
        return _torch_load(cache_path)
    graphs2subgraphs = prepare_parsed_data(
        data_list,
        parsing=parsing,
        parsing_params=parsing_params,
        require_full_edge=require_full_edge,
        desc=f"MOSGSL preprocess {dataset_name}/{fold_name}",
    )
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    torch.save(graphs2subgraphs, tmp_path)
    os.replace(tmp_path, cache_path)
    return graphs2subgraphs


def build_mosgsl_fold(train_split, val_split, test_split, dataset_name, fold_name,
                      cache_dir, parsing='parse_bfs', parsing_params=None,
                      require_full_edge=False, force_reload=False,
                      include_test=True, use_edge_context=False):
    next_idx = 0
    train_data, next_idx = clone_with_global_idx(
        train_split, next_idx, use_edge_context=use_edge_context
    )
    val_data, next_idx = clone_with_global_idx(
        val_split, next_idx, use_edge_context=use_edge_context
    )
    if include_test:
        test_data, next_idx = clone_with_global_idx(
            test_split, next_idx, use_edge_context=use_edge_context
        )
    else:
        test_data = []
    all_data = train_data + val_data + test_data
    graphs2subgraphs = build_parse_cache(
        all_data,
        # The feature tag prevents reuse of an old cache whose subgraph x has
        # no GNCE edge-context channels.
        dataset_name=(
            f"{dataset_name}__edgecontext_v1"
            if use_edge_context else dataset_name
        ),
        fold_name=fold_name,
        cache_dir=cache_dir,
        parsing=parsing,
        parsing_params=parsing_params,
        require_full_edge=require_full_edge,
        force_reload=force_reload,
    )
    return train_data, val_data, test_data, graphs2subgraphs
