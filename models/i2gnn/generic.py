import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool


def center_pool(x, node_to_subgraph):
    unique, counts = torch.unique_consecutive(node_to_subgraph, return_counts=True)
    if unique.numel() == counts.numel() and unique.numel() > 0:
        starts = torch.cat([
            counts.new_zeros(1),
            torch.cumsum(counts, dim=0)[:-1],
        ])
        return x[starts]

    centers = []
    seen = set()
    for idx, group in enumerate(node_to_subgraph.detach().cpu().tolist()):
        if group not in seen:
            centers.append(idx)
            seen.add(group)
    return x[torch.tensor(centers, device=x.device, dtype=torch.long)]


def graph_pool(x, batch, mode):
    if mode in {'mean', 'avg'}:
        return global_mean_pool(x, batch)
    if mode in {'sum', 'add'}:
        return global_add_pool(x, batch)
    if mode == 'max':
        return global_max_pool(x, batch)
    raise ValueError(f"Unsupported pooling mode: {mode}")


class GenericI2GNN(nn.Module):
    """I2GNN variant for datasets whose node/edge features are already tensors.

    This keeps the official I2GNN hierarchical subgraph2 -> subgraph -> graph
    pooling pattern, but replaces molecule-specific AtomEncoder/BondEncoder
    with linear feature projections.
    """

    def __init__(self, node_input_dim, edge_input_dim, output_dim,
                 hidden_size=128, num_layers=4, dropout=0.0,
                 subgraph_pooling='mean', subgraph2_pooling='mean-center-side',
                 graph_pooling='mean', use_rd=False, use_pooling_nn=True,
                 gate=True, residual=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.subgraph_pooling = subgraph_pooling
        self.subgraph2_pooling = subgraph2_pooling
        self.graph_pooling = graph_pooling
        self.use_rd = use_rd
        self.use_pooling_nn = use_pooling_nn
        self.gate = gate
        self.residual = residual

        self.node_encoder = nn.Linear(max(1, node_input_dim), hidden_size)
        self.edge_encoder = nn.Linear(edge_input_dim, hidden_size) if edge_input_dim > 0 else None
        self.z_embeddings = nn.ModuleList([
            nn.Embedding(1000, hidden_size) for _ in range(num_layers + 1)
        ])
        self.rd_projections = nn.ModuleList([
            nn.Linear(2, hidden_size) for _ in range(num_layers + 1)
        ]) if use_rd else None

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_size, 2 * hidden_size),
                nn.BatchNorm1d(2 * hidden_size),
                nn.ReLU(),
                nn.Linear(2 * hidden_size, hidden_size),
            )
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_size))
            self.norms.append(nn.BatchNorm1d(hidden_size))

        subgraph2_dim = 1
        if subgraph2_pooling == 'mean-center':
            subgraph2_dim = 2
        elif subgraph2_pooling == 'mean-center-side':
            subgraph2_dim = 3
        pooled_dim_factor = subgraph2_dim
        if subgraph_pooling == 'mean-context':
            pooled_dim_factor += 1

        subgraph2_dim = subgraph2_dim * hidden_size
        pooled_dim = pooled_dim_factor * hidden_size
        if use_pooling_nn:
            self.edge_pooling_nn = nn.Sequential(
                nn.Linear(subgraph2_dim, subgraph2_dim), nn.ReLU(), nn.Linear(subgraph2_dim, subgraph2_dim)
            )
            self.node_pooling_nn = nn.Sequential(
                nn.Linear(pooled_dim, pooled_dim), nn.ReLU(), nn.Linear(pooled_dim, pooled_dim)
            )
        if gate:
            self.subgraph_gate = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())

        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim),
        )

    def encode_z(self, data, layer):
        z = data.z.long().clamp_min(0)
        z_emb = self.z_embeddings[layer](z)
        if z_emb.ndim == 3:
            z_emb = z_emb.sum(dim=1)
        if self.use_rd and hasattr(data, 'rd'):
            z_emb = z_emb + self.rd_projections[layer](data.rd.float())
        return z_emb

    def encode_edge(self, data, x):
        if data.edge_index.numel() == 0:
            return x.new_zeros((0, self.hidden_size))
        if self.edge_encoder is None or data.edge_attr is None:
            return x.new_zeros((data.edge_index.size(1), self.hidden_size))
        edge_attr = data.edge_attr.float()
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        return self.edge_encoder(edge_attr)

    def pool_i2(self, x, data, z_emb, layer):
        if self.subgraph_pooling == 'mean-context':
            x_node = global_mean_pool(x, data.node_to_original_node)

        if self.gate:
            x_gated = self.subgraph_gate(z_emb) * x
        else:
            x_gated = x

        if self.subgraph2_pooling == 'mean':
            x = global_mean_pool(x_gated, data.node_to_subgraph2)
        elif self.subgraph2_pooling in {'sum', 'add'}:
            x = global_add_pool(x_gated, data.node_to_subgraph2)
        elif self.subgraph2_pooling == 'center':
            x = x[data.center_idx[:, 0]] if hasattr(data, 'center_idx') else center_pool(x, data.node_to_subgraph2)
        elif self.subgraph2_pooling == 'mean-center':
            center = x[data.center_idx[:, 0]] if hasattr(data, 'center_idx') else center_pool(x, data.node_to_subgraph2)
            x = torch.cat([global_mean_pool(x_gated, data.node_to_subgraph2), center], dim=-1)
        elif self.subgraph2_pooling == 'mean-center-side':
            if not hasattr(data, 'center_idx'):
                raise ValueError("subgraph2_pooling='mean-center-side' requires center_idx in I2GNN data.")
            x = torch.cat([
                global_mean_pool(x_gated, data.node_to_subgraph2),
                x[data.center_idx[:, 0]],
                x[data.center_idx[:, 1]],
            ], dim=-1)
        else:
            raise ValueError(f"Unsupported subgraph2_pooling: {self.subgraph2_pooling}")

        if self.use_pooling_nn:
            x = self.edge_pooling_nn(x)

        if self.subgraph_pooling == 'mean':
            x = global_mean_pool(x, data.subgraph2_to_subgraph)
        elif self.subgraph_pooling in {'sum', 'add'}:
            x = global_add_pool(x, data.subgraph2_to_subgraph)
        elif self.subgraph_pooling == 'mean-context':
            x = torch.cat([global_mean_pool(x, data.subgraph2_to_subgraph), x_node], dim=-1)
        else:
            raise ValueError(f"Unsupported subgraph_pooling: {self.subgraph_pooling}")

        if self.use_pooling_nn:
            x = self.node_pooling_nn(x)

        return graph_pool(x, data.subgraph_to_graph, self.graph_pooling)

    def forward(self, data):
        x = data.x.float()
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x = self.node_encoder(x)
        edge_attr = self.encode_edge(data, x)

        for layer, conv in enumerate(self.convs):
            h = x + self.encode_z(data, layer)
            h = conv(h, data.edge_index, edge_attr)
            h = self.norms[layer](h)
            if layer < len(self.convs) - 1:
                h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual and h.shape == x.shape:
                h = h + x
            x = h

        z_emb = self.encode_z(data, len(self.convs))
        graph_emb = self.pool_i2(x, data, z_emb, len(self.convs))
        return self.head(graph_emb)
