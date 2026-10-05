import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import (
    add_remaining_self_loops,
    coalesce,
    dense_to_sparse,
    softmax,
    subgraph,
)
from torch_scatter import scatter_add
from torch_sparse import spspmm

from .sparsemax import Sparsemax


def topk_score(score, ratio, batch):
    perm = []
    for graph_id in torch.unique(batch, sorted=True):
        idx = (batch == graph_id).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        k = max(1, int(math.ceil(float(ratio) * idx.numel())))
        keep = idx[score[idx].topk(k).indices]
        perm.append(keep)
    if not perm:
        return batch.new_empty(0)
    return torch.cat(perm, dim=0)


def filter_adj(edge_index, edge_attr, perm, num_nodes):
    edge_index, edge_attr = subgraph(
        perm,
        edge_index,
        edge_attr=edge_attr,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )
    return edge_index, edge_attr


class TwoHopNeighborhood:
    @staticmethod
    def _spspmm(edge_index, value, num_nodes):
        """Compute A@A, using an exact CPU path when cuSPARSE cannot handle it."""
        try:
            return spspmm(
                edge_index, value, edge_index, value,
                num_nodes, num_nodes, num_nodes, True,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            recoverable = (
                edge_index.is_cuda
                and ('cusparsespgemm' in message or 'insufficient resources' in message)
            )
            if not recoverable:
                raise
            # The expanded topology is non-parametric. Moving only this sparse
            # integer/constant operation to CPU preserves the exact graph while
            # avoiding a cuSPARSE workspace-limit failure on high-degree graphs.
            cpu_index, cpu_value = spspmm(
                edge_index.cpu(), value.cpu(), edge_index.cpu(), value.cpu(),
                num_nodes, num_nodes, num_nodes, True,
            )
            return cpu_index.to(edge_index.device), cpu_value.to(value.device)

    @classmethod
    def _augment_one(cls, edge_index, edge_attr, num_nodes):
        if edge_index.size(1) == 0:
            return edge_index, edge_attr
        fill = 1e16
        value = edge_index.new_full((edge_index.size(1),), fill, dtype=torch.float)
        index, value = cls._spspmm(edge_index, value, num_nodes)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            return coalesce(edge_index, num_nodes=num_nodes), None
        value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
        value = value.expand(-1, *list(edge_attr.size())[1:])
        edge_attr = torch.cat([edge_attr, value], dim=0)
        edge_index, edge_attr = coalesce(
            edge_index, edge_attr, num_nodes=num_nodes, reduce='min'
        )
        edge_attr[edge_attr >= fill] = 0
        return edge_index, edge_attr

    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        num_nodes = data.num_nodes
        batch = getattr(data, 'batch', None)

        # Sparse multiplication over a PyG batch is mathematically block
        # diagonal. Compute each block separately to bound cuSPARSE workspace
        # without changing any graph or edge produced by A union A^2.
        if batch is None or batch.numel() == 0 or int(batch.max().item()) == 0:
            data.edge_index, data.edge_attr = self._augment_one(
                edge_index, edge_attr, num_nodes,
            )
            return data

        out_indices = []
        out_attrs = [] if edge_attr is not None else None
        for graph_id in torch.unique(batch, sorted=True):
            nodes = (batch == graph_id).nonzero(as_tuple=False).view(-1)
            local_index, local_attr = subgraph(
                nodes,
                edge_index,
                edge_attr=edge_attr,
                relabel_nodes=True,
                num_nodes=num_nodes,
            )
            local_index, local_attr = self._augment_one(
                local_index, local_attr, int(nodes.numel()),
            )
            out_indices.append(nodes[local_index])
            if out_attrs is not None:
                out_attrs.append(local_attr)

        data.edge_index = torch.cat(out_indices, dim=1)
        if out_attrs is not None:
            data.edge_attr = torch.cat(out_attrs, dim=0)
        return data


class GCN(MessagePassing):
    def __init__(self, in_channels, out_channels, cached=False, bias=True, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight.data)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias.data)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    f'Cached {self.cached_num_edges} edges, but found {edge_index.size(1)}'
                )
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class NodeInformationScore(MessagePassing):
    def __init__(self, improved=False, cached=False, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0, num_nodes)
        row, col = edge_index
        expand_deg = torch.zeros(edge_weight.size(0), dtype=dtype, device=edge_index.device)
        expand_deg[-num_nodes:] = 1
        return edge_index, expand_deg - deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    f'Cached {self.cached_num_edges} edges, but found {edge_index.size(1)}'
                )
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class HGPSLPool(nn.Module):
    def __init__(self, in_channels, ratio=0.8, sample=False, sparse=False,
                 sl=True, lamb=1.0, negative_slop=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl
        self.negative_slop = negative_slop
        self.lamb = lamb

        self.att = Parameter(torch.Tensor(1, self.in_channels * 2))
        nn.init.xavier_uniform_(self.att.data)
        self.sparse_attention = Sparsemax()
        self.neighbor_augment = TwoHopNeighborhood()
        self.calc_information_score = NodeInformationScore()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att.data)
        self.calc_information_score.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x_information_score = self.calc_information_score(x, edge_index, edge_attr)
        score = torch.sum(torch.abs(x_information_score), dim=1)

        original_x = x
        perm = topk_score(score, self.ratio, batch)
        x = x[perm]
        batch = batch[perm]
        induced_edge_index, induced_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        if not self.sl:
            return x, induced_edge_index, induced_edge_attr, batch

        if self.sample:
            k_hop = 3
            if edge_attr is None:
                edge_attr = torch.ones(edge_index.size(1), dtype=torch.float, device=edge_index.device)
            hop_data = Data(
                x=original_x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
                num_nodes=original_x.size(0),
            )
            for _ in range(k_hop - 1):
                hop_data = self.neighbor_augment(hop_data)
            new_edge_index, new_edge_attr = filter_adj(
                hop_data.edge_index, hop_data.edge_attr, perm, num_nodes=score.size(0)
            )
            new_edge_index, new_edge_attr = add_remaining_self_loops(new_edge_index, new_edge_attr, 0, x.size(0))
            row, col = new_edge_index
            weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop) + new_edge_attr.view(-1) * self.lamb
            # ``new_edge_index`` is already coalesced and self-loop complete.
            # The previous dense NxN round-trip only removed exact-zero edge
            # weights, while allocating across unrelated graphs in the batch.
            # Apply the same zero filtering directly in sparse form.
            nonzero = weights != 0
            new_edge_index = new_edge_index[:, nonzero]
            weights = weights[nonzero]
            row, col = new_edge_index
            if self.sparse:
                new_edge_attr = self.sparse_attention(weights, row)
            else:
                new_edge_attr = softmax(weights, row, num_nodes=x.size(0))
            nonzero = new_edge_attr != 0
            new_edge_index = new_edge_index[:, nonzero]
            new_edge_attr = new_edge_attr[nonzero]
        else:
            if induced_edge_attr is None:
                induced_edge_attr = torch.ones(
                    induced_edge_index.size(1), dtype=x.dtype, device=induced_edge_index.device
                )
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            shift_cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
            cum_num_nodes = num_nodes.cumsum(dim=0)
            adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
            for idx_i, idx_j in zip(shift_cum_num_nodes, cum_num_nodes):
                adj[int(idx_i.item()):int(idx_j.item()), int(idx_i.item()):int(idx_j.item())] = 1.0
            new_edge_index, _ = dense_to_sparse(adj)
            row, col = new_edge_index
            weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop)
            adj[row, col] = weights
            induced_row, induced_col = induced_edge_index
            adj[induced_row, induced_col] += induced_edge_attr.view(-1) * self.lamb
            weights = adj[row, col]
            if self.sparse:
                new_edge_attr = self.sparse_attention(weights, row)
            else:
                new_edge_attr = softmax(weights, row, num_nodes=x.size(0))
            adj[row, col] = new_edge_attr
            new_edge_index, new_edge_attr = dense_to_sparse(adj)

        return x, new_edge_index, new_edge_attr, batch
