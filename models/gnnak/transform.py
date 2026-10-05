import re

import numpy as np
import torch
from torch_geometric.data import Data

from .transform_utils.sampling import (
    check_values_in_set,
    min_set_cover_sampling,
    random_sampling,
    shortest_path_sampling,
)
from .transform_utils.subgraph_extractors import k_hop_subgraph, random_walk_subgraph


class SubgraphsData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        num_nodes = self.num_nodes
        num_edges = self.edge_index.size(-1)
        if bool(re.search('(combined_subgraphs)', key)):
            return getattr(self, key[:-len('combined_subgraphs')] + 'subgraphs_nodes_mapper').size(0)
        if bool(re.search('(subgraphs_batch)', key)):
            return 1 + getattr(self, key)[-1]
        if bool(re.search('(nodes_mapper)|(selected_supernodes)', key)):
            return num_nodes
        if bool(re.search('(edges_mapper)', key)):
            return num_edges
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if bool(re.search('(combined_subgraphs)', key)):
            return -1
        return super().__cat_dim__(key, value, *args, **kwargs)


class SubgraphsTransform:
    def __init__(self, hops=2, walk_length=0, p=1, q=1, repeat=1,
                 sampling_mode=None, minimum_redundancy=2,
                 shortest_path_mode_stride=2, random_mode_sampling_rate=0.3,
                 random_init=False):
        self.num_hops = hops
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.repeat = repeat
        self.subsampling = sampling_mode is not None
        self.random_init = random_init
        self.sampling_mode = sampling_mode
        self.minimum_redundancy = minimum_redundancy
        self.shortest_path_mode_stride = shortest_path_mode_stride
        self.random_mode_sampling_rate = random_mode_sampling_rate

    def __call__(self, data):
        data = SubgraphsData(**{k: v for k, v in data})
        if data.x is None:
            data.x = torch.ones(data.num_nodes, 1, dtype=torch.long)

        subgraphs_nodes_mask, subgraphs_edges_mask, hop_indicator_dense = extract_subgraphs(
            data.edge_index,
            data.num_nodes,
            self.num_hops,
            self.walk_length,
            self.p,
            self.q,
            self.repeat,
        )
        subgraphs_nodes, subgraphs_edges, hop_indicator = to_sparse(
            subgraphs_nodes_mask, subgraphs_edges_mask, hop_indicator_dense
        )

        if self.subsampling:
            selected_subgraphs, node_selected_times = subsampling_subgraphs(
                data.edge_index,
                subgraphs_nodes if self.sampling_mode != 'min_set_cover' else subgraphs_nodes_mask,
                data.num_nodes,
                sampling_mode=self.sampling_mode,
                random_init=self.random_init,
                minimum_redundancy=self.minimum_redundancy,
                shortest_path_mode_stride=self.shortest_path_mode_stride,
                random_mode_sampling_rate=self.random_mode_sampling_rate,
            )
            data.subsampling_scale = subgraphs_nodes_mask.sum(0) / node_selected_times.clamp_min(1)
            data.selected_supernodes = torch.tensor(
                np.sort(selected_subgraphs), dtype=torch.long, device=data.edge_index.device
            )
            data.hops_to_selected, data.edges_between_two_hops = hops_to_selected_nodes(
                data.edge_index, selected_subgraphs, data.num_nodes
            )
            if min(subgraphs_edges.shape) > 0:
                subgraphs_nodes, subgraphs_edges, hop_indicator = select_subgraphs(
                    subgraphs_nodes, subgraphs_edges, hop_indicator, selected_subgraphs
                )

        combined_subgraphs = combine_subgraphs(
            data.edge_index,
            subgraphs_nodes,
            subgraphs_edges,
            num_selected=data.num_nodes,
            num_nodes=data.num_nodes,
        )
        data.subgraphs_batch = subgraphs_nodes[0]
        data.subgraphs_nodes_mapper = subgraphs_nodes[1]
        data.subgraphs_edges_mapper = subgraphs_edges[1]
        data.combined_subgraphs = combined_subgraphs
        data.hop_indicator = hop_indicator
        data.__num_nodes__ = data.num_nodes
        return data


def select_subgraphs(subgraphs_nodes, subgraphs_edges, hop_indicator, selected_subgraphs):
    selected_subgraphs = np.sort(selected_subgraphs)
    selected_t = torch.as_tensor(selected_subgraphs, dtype=torch.long,
                                 device=subgraphs_nodes.device)
    selected_nodes_mask = check_values_in_set(subgraphs_nodes[0], selected_t)
    selected_edges_mask = check_values_in_set(subgraphs_edges[0], selected_t)
    nodes_batch = subgraphs_nodes[0][selected_nodes_mask]
    edges_batch = subgraphs_edges[0][selected_edges_mask]
    batch_mapper = torch.zeros(1 + nodes_batch.max(), dtype=torch.long,
                               device=subgraphs_nodes.device)
    batch_mapper[selected_t] = torch.arange(len(selected_t), device=subgraphs_nodes.device)
    selected_subgraphs_nodes = batch_mapper[nodes_batch], subgraphs_nodes[1][selected_nodes_mask]
    selected_subgraphs_edges = batch_mapper[edges_batch], subgraphs_edges[1][selected_edges_mask]
    selected_hop_indicator = hop_indicator[selected_nodes_mask]
    return selected_subgraphs_nodes, selected_subgraphs_edges, selected_hop_indicator


def to_sparse(node_mask, edge_mask, hop_indicator):
    subgraphs_nodes = node_mask.nonzero().T
    subgraphs_edges = edge_mask.nonzero().T
    if hop_indicator is not None:
        hop_indicator = hop_indicator[subgraphs_nodes[0], subgraphs_nodes[1]]
    return subgraphs_nodes, subgraphs_edges, hop_indicator


def extract_subgraphs(edge_index, num_nodes, num_hops, walk_length=0, p=1,
                      q=1, repeat=1, sparse=False):
    if walk_length > 0:
        node_mask, hop_indicator = random_walk_subgraph(
            edge_index, num_nodes, walk_length, p=p, q=q, repeat=repeat,
            cal_hops=True,
        )
    else:
        node_mask, hop_indicator = k_hop_subgraph(edge_index, num_nodes, num_hops)
    edge_mask = node_mask[:, edge_index[0]] & node_mask[:, edge_index[1]]
    if sparse:
        return to_sparse(node_mask, edge_mask, hop_indicator)
    return node_mask, edge_mask, hop_indicator


def subsampling_subgraphs(edge_index, subgraphs_nodes, num_nodes=None,
                          sampling_mode='shortest_path', random_init=False,
                          minimum_redundancy=0, shortest_path_mode_stride=2,
                          random_mode_sampling_rate=0.5):
    assert sampling_mode in ['shortest_path', 'random', 'min_set_cover']
    if sampling_mode == 'random':
        return random_sampling(
            subgraphs_nodes, num_nodes=num_nodes, rate=random_mode_sampling_rate,
            minimum_redundancy=minimum_redundancy,
        )
    if sampling_mode == 'shortest_path':
        return shortest_path_sampling(
            edge_index, subgraphs_nodes, num_nodes=num_nodes,
            minimum_redundancy=minimum_redundancy,
            stride=max(1, shortest_path_mode_stride),
            random_init=random_init,
        )
    assert subgraphs_nodes.size(0) == num_nodes
    return min_set_cover_sampling(
        edge_index, subgraphs_nodes, minimum_redundancy=minimum_redundancy,
        random_init=random_init,
    )


def combine_subgraphs(edge_index, subgraphs_nodes, subgraphs_edges,
                      num_selected=None, num_nodes=None):
    if num_selected is None:
        num_selected = subgraphs_nodes[0][-1] + 1
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1
    combined_subgraphs = edge_index[:, subgraphs_edges[1]]
    node_label_mapper = edge_index.new_full((num_selected, num_nodes), -1)
    node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]] = torch.arange(
        len(subgraphs_nodes[1]), device=edge_index.device
    )
    node_label_mapper = node_label_mapper.reshape(-1)
    inc = torch.arange(num_selected, device=edge_index.device) * num_nodes
    combined_subgraphs = combined_subgraphs + inc[subgraphs_edges[0]]
    return node_label_mapper[combined_subgraphs]


def hops_to_selected_nodes(edge_index, selected_nodes, num_nodes=None):
    row, col = edge_index
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    selected_nodes = torch.as_tensor(selected_nodes, device=edge_index.device)
    hop_indicator = row.new_full((num_nodes,), -1)
    bipartite_indicator = row.new_full(row.shape, -1)
    hop_indicator[selected_nodes] = 0
    selected_mask = hop_indicator == 0
    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    i = 1
    while hop_indicator.min() < 0:
        source_near_edges = selected_mask[row]
        node_mask.fill_(False)
        node_mask[col[source_near_edges]] = True
        selected_mask = (hop_indicator == -1) & node_mask
        between = source_near_edges & selected_mask[col]
        bipartite_indicator[between] = i
        hop_indicator[selected_mask] = i
        i += 1
    return hop_indicator, bipartite_indicator
