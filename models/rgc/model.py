import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import degree

from models.baselines import MLP, TUNodeEncoder
from .backbone import GAT, GCN, GraphSAGE


EPS = 1e-8


class RGC(nn.Module):
    """Graph-level MotifRGC/RGC adapter.

    The official MotifRGC is node-centric: it combines a feature GNN, a
    structural/Riemannian branch, and a contrastive alignment loss. For
    graph-level benchmarks we keep this core dual-branch design and use
    graph-level pooling plus OpenGLT's supervised objective. The latest
    contrastive loss is exposed as ``latest_aux_loss`` for the runner.
    """

    def __init__(self, nhid, nout, dataset_type='tu',
                 dim_node_deg=0, dim_node_label=0, dim_node_attr=0,
                 backbone='gcn', num_layers=2, hidden_features=64,
                 embed_features=32, n_heads=4, drop_edge=0.0,
                 drop_node=0.0, struct_dim=16, temperature=0.2,
                 pooling='mean'):
        super().__init__()
        self.dataset_type = dataset_type
        self.backbone = backbone.lower()
        self.pooling = pooling
        self.temperature = float(temperature)
        self.struct_dim = int(struct_dim)
        self.latest_aux_loss = None

        if dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            self.input_encoder = AtomEncoder(emb_dim=nhid)
        else:
            self.input_encoder = TUNodeEncoder(nhid, dim_node_deg, dim_node_label, dim_node_attr)

        if self.backbone == 'gcn':
            self.encoder = GCN(num_layers, nhid, hidden_features, embed_features, drop_edge, drop_node)
            self.encoder_struct = GCN(num_layers, self.struct_dim, hidden_features, embed_features, drop_edge, drop_node)
        elif self.backbone == 'gat':
            self.encoder = GAT(num_layers, nhid, hidden_features, embed_features, n_heads, drop_edge, drop_node)
            self.encoder_struct = GAT(num_layers, self.struct_dim, hidden_features, embed_features, n_heads, drop_edge, drop_node)
        elif self.backbone == 'sage':
            self.encoder = GraphSAGE(num_layers, nhid, hidden_features, embed_features, drop_edge, drop_node)
            self.encoder_struct = GraphSAGE(num_layers, self.struct_dim, hidden_features, embed_features, drop_edge, drop_node)
        else:
            raise ValueError(f'Unknown RGC backbone: {backbone}')

        self.struct_proj = nn.Linear(4, self.struct_dim)
        self.norm = nn.LayerNorm(2 * embed_features)
        self.graph_head = MLP(2 * embed_features, embed_features, nlayer=2)
        self.output_decoder = MLP(2 * embed_features, nout, nlayer=2)

    def reset_parameters(self):
        if hasattr(self.input_encoder, 'reset_parameters'):
            self.input_encoder.reset_parameters()
        self.encoder.reset_parameters()
        self.encoder_struct.reset_parameters()
        self.struct_proj.reset_parameters()
        self.norm.reset_parameters()
        self.graph_head.reset_parameters()
        self.output_decoder.reset_parameters()
        self.latest_aux_loss = None

    def _encode_input(self, x):
        if self.dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            if x.dtype == torch.float:
                x = x.long()
        elif x.dtype == torch.long:
            x = x.float()
        return self.input_encoder(x)

    @staticmethod
    def _struct_features(edge_index, batch, num_nodes):
        row, col = edge_index
        deg = degree(row, num_nodes=num_nodes, dtype=torch.float)
        in_deg = degree(col, num_nodes=num_nodes, dtype=torch.float)
        graph_size = torch.bincount(batch, minlength=int(batch.max().item()) + 1).float().to(batch.device)
        rel_size = graph_size[batch].clamp(min=1.0)
        deg_norm = deg / rel_size
        in_deg_norm = in_deg / rel_size
        loop_like = (row == col).float()
        loop_count = torch.zeros(num_nodes, device=edge_index.device)
        if loop_like.numel() > 0:
            loop_count.index_add_(0, row, loop_like)
        return torch.stack([
            deg,
            in_deg,
            deg_norm,
            in_deg_norm + loop_count,
        ], dim=-1)

    def _pool(self, x, batch):
        if self.pooling in ('sum', 'add'):
            return global_add_pool(x, batch)
        if self.pooling == 'max':
            return global_max_pool(x, batch)
        return global_mean_pool(x, batch)

    def _contrastive_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        sim = torch.exp(torch.mm(z1, z2.t()) / self.temperature)
        pos = sim.diag()
        denom_row = sim.sum(dim=1) + EPS
        denom_col = sim.sum(dim=0) + EPS
        loss_1 = -torch.log(pos / denom_row.clamp(min=EPS)).mean()
        loss_2 = -torch.log(pos / denom_col.clamp(min=EPS)).mean()
        return 0.5 * (loss_1 + loss_2)

    def forward(self, x, edge_index, batch, edge_attr=None,
                fingerprint=None, rwpe=None, input_embedding=None, **kwargs):
        if input_embedding is not None:
            x = input_embedding
        else:
            x = self._encode_input(x)

        struct = self._struct_features(edge_index, batch, x.size(0)).to(x.dtype)
        struct = self.struct_proj(struct)
        z_feat = self.encoder(x, edge_index)
        z_struct = self.encoder_struct(struct, edge_index)
        self.latest_aux_loss = self._contrastive_loss(z_feat, z_struct)

        node_repr = self.norm(torch.cat([z_feat, z_struct], dim=-1))
        graph_emb = self._pool(node_repr, batch)
        graph_repr = self.graph_head(graph_emb)
        out = self.output_decoder(graph_emb)
        return out, graph_repr
