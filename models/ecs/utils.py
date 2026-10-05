from collections import defaultdict

import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
from scipy import linalg
from torch_geometric.data import Data
from torch_geometric.utils import (
    add_self_loops,
    degree,
    remove_self_loops,
    to_scipy_sparse_matrix,
)


STRUCT_EMB_SIZE_WITH_RD = 1800
STRUCT_EMB_SIZE_NO_RD = 1700


def maybe_num_nodes(index, num_nodes=None):
    if num_nodes is not None:
        return int(num_nodes)
    if index.numel() == 0:
        return 0
    return int(index.max().item()) + 1


def k_hop_subgraph(node_idx, num_hops, edge_index, num_nodes,
                   node_label='hop', max_nodes_per_hop=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    subsets = [torch.tensor([node_idx], device=row.device).flatten()]
    visited = set(subsets[-1].tolist())
    label = defaultdict(list)
    for node in subsets[-1].tolist():
        label[node].append(1)
    hops = [torch.tensor([0], dtype=torch.long, device=row.device).flatten()]

    for hop in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        new_nodes = col[edge_mask]
        tmp = []
        for node in new_nodes.tolist():
            if node in visited:
                continue
            tmp.append(node)
            label[node].append(hop + 2)
        if len(tmp) == 0:
            break
        if max_nodes_per_hop is not None and max_nodes_per_hop < len(tmp):
            perm = torch.randperm(len(tmp))[:max_nodes_per_hop].tolist()
            tmp = [tmp[i] for i in perm]
        new_nodes_set = set(tmp)
        visited = visited.union(new_nodes_set)
        new_nodes = torch.tensor(list(new_nodes_set), device=row.device)
        subsets.append(new_nodes)
        hops.append(torch.tensor([hop + 1] * len(new_nodes), dtype=torch.long, device=row.device))

    subset = torch.cat(subsets)
    subset = subset[subset != node_idx]
    subset = torch.cat([torch.tensor([node_idx], device=row.device), subset])

    if node_label == 'hop':
        hop_values = torch.cat(hops)
        hop_values = hop_values[hop_values != 0]
        hop_values = torch.cat([torch.tensor([0], dtype=torch.long, device=row.device), hop_values])
        z = {ss: zz for ss, zz in zip(subset.tolist(), hop_values.tolist())}
    elif node_label.startswith('spd'):
        num_spd = int(node_label[3:]) if len(node_label) > 3 else 2
        z_tensor = torch.zeros([subset.size(0), num_spd], dtype=torch.long, device=row.device)
        for i, node in enumerate(subset.tolist()):
            dists = label[node][:num_spd]
            z_tensor[i][:min(num_spd, len(dists))] = torch.tensor(dists, device=row.device)
        z = {ss: zz for ss, zz in zip(subset.tolist(), z_tensor.tolist())}
    else:
        raise ValueError(f"Unsupported ECS node_label: {node_label}")

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]
    sub_edge_index = edge_index[:, edge_mask]
    return subset.tolist(), sub_edge_index, edge_mask, z


def _resistance_distance(edge_index, num_nodes):
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
    laplacian = ssp.csgraph.laplacian(adj).toarray()
    try:
        lap_inv = linalg.pinv(laplacian)
    except Exception:
        lap_inv = linalg.pinv(laplacian + 0.01 * np.eye(*laplacian.shape))
    lxx = lap_inv[0, 0]
    lyy = lap_inv[list(range(len(lap_inv))), list(range(len(lap_inv)))]
    lxy = lap_inv[0, :]
    lyx = lap_inv[:, 0]
    return torch.FloatTensor(lxx + lyy - lxy - lyx).unsqueeze(1)


def _structural_vector(subgraph, use_rd=False):
    num_nodes = int(subgraph.num_nodes)
    sub_degree = subgraph.sub_degree.long().clamp(min=0, max=199)
    z = subgraph.z.long().clamp(min=0, max=99)
    pos_enc = torch.cat(
        (
            F.one_hot(sub_degree, num_classes=200).view(num_nodes, -1),
            F.one_hot(z, num_classes=100).view(num_nodes, -1),
        ),
        dim=-1,
    )
    if use_rd:
        rd = subgraph.rd.long().clamp(min=0, max=99)
        pos_enc = torch.cat((pos_enc, F.one_hot(rd, num_classes=100).view(num_nodes, -1)), dim=-1)
    pos_enc = pos_enc.sum(dim=0)

    edge_index_no_loop = remove_self_loops(subgraph.edge_index)[0]
    if edge_index_no_loop.numel() == 0:
        edge_code = torch.zeros(1300, dtype=pos_enc.dtype)
    else:
        endpoint_z = z[edge_index_no_loop].transpose(0, 1).reshape(-1, 4)
        weights = torch.tensor([216, 36, 6, 1], dtype=torch.long, device=endpoint_z.device)
        code = (endpoint_z @ weights).clamp(min=0, max=1299)
        edge_code = F.one_hot(code, num_classes=1300).sum(dim=0).to(pos_enc.dtype)
    return torch.cat((pos_enc, edge_code), dim=0)


def create_ecs_data(data, h=3, node_label='hop', use_rd=True,
                    max_nodes_per_hop=None, self_loop=True):
    if not isinstance(data, Data):
        raise TypeError(f"ECS preprocessing expects a PyG Data object, got {type(data)}")

    x, edge_index = data.x, data.edge_index
    num_nodes = int(data.num_nodes)
    edge_attr = getattr(data, 'edge_attr', None)
    if self_loop:
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr=edge_attr)
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, num_nodes=num_nodes)

    pos_encs, pos_indices, pos_batches = [], [], []
    for edge_id, edge in enumerate(edge_index.t()):
        # The official efficient ESC implementation always labels each
        # edge-rooted neighborhood by hop distance here, even when the CLI
        # argument is set to "spd" for the non-efficient subgraph path.
        root_label = 'hop'
        nodes_0, _, edge_mask_0, z_0 = k_hop_subgraph(
            int(edge[0].item()), h, edge_index, num_nodes,
            node_label=root_label, max_nodes_per_hop=max_nodes_per_hop,
        )
        nodes_1, _, edge_mask_1, z_1 = k_hop_subgraph(
            int(edge[1].item()), h, edge_index, num_nodes,
            node_label=root_label, max_nodes_per_hop=max_nodes_per_hop,
        )
        nodes = [nodes_0[0], nodes_1[0]]
        nodes += [item for item in nodes_0 if item not in nodes]
        nodes += [item for item in nodes_1 if item not in nodes]

        z_rows = []
        for node in nodes:
            d0 = z_0[node] if node in z_0 else h + 1
            d1 = z_1[node] if node in z_1 else h + 1
            if isinstance(d0, list):
                d0 = d0[0]
            if isinstance(d1, list):
                d1 = d1[0]
            z_rows.append([d0, d1])
        z = torch.tensor(z_rows, dtype=torch.long)
        nodes_tensor = torch.tensor(nodes, dtype=torch.long)
        edge_mask = torch.logical_or(edge_mask_0, edge_mask_1)
        sub_edge_index = edge_index[:, edge_mask]
        node_idx = edge_index.new_full((num_nodes,), -1)
        node_idx[nodes_tensor] = torch.arange(nodes_tensor.size(0), device=edge_index.device)
        sub_edge_index = node_idx[sub_edge_index].cpu()

        subgraph = Data(
            x=x[nodes_tensor].cpu() if x is not None else None,
            edge_index=sub_edge_index,
            z=z.cpu(),
        )
        subgraph.num_nodes = int(nodes_tensor.numel())
        subgraph.sub_degree = degree(sub_edge_index[0], num_nodes=subgraph.num_nodes).cpu()
        if use_rd:
            subgraph.rd = _resistance_distance(sub_edge_index, subgraph.num_nodes)

        dense_struct = _structural_vector(subgraph, use_rd=use_rd)
        pos_index = torch.nonzero(dense_struct, as_tuple=False).view(-1)
        if pos_index.numel() == 0:
            pos_index = torch.zeros(1, dtype=torch.long)
            pos_value = torch.zeros(1, dtype=dense_struct.dtype)
        else:
            pos_value = dense_struct[pos_index].view(-1)
        pos_encs.append(pos_value)
        pos_indices.append(pos_index)
        pos_batches.append(torch.full((pos_index.numel(),), edge_id, dtype=torch.long))

    if pos_encs:
        pos_enc = torch.cat(pos_encs, dim=0)
        pos_index = torch.cat(pos_indices, dim=0)
        pos_batch = torch.cat(pos_batches, dim=0)
    else:
        pos_enc = torch.empty(0, dtype=torch.float)
        pos_index = torch.empty(0, dtype=torch.long)
        pos_batch = torch.empty(0, dtype=torch.long)

    new_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=data.y,
        pos_enc=pos_enc,
        pos_index=pos_index,
        pos_batch=pos_batch,
    )
    new_data.num_nodes = num_nodes
    for key, value in data:
        if key not in {
            'x', 'edge_index', 'edge_attr', 'y', 'pos', 'pos_enc',
            'pos_index', 'pos_batch', 'num_nodes',
        }:
            new_data[key] = value
    return new_data
