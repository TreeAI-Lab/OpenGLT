import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

from models.baselines import MLP, TUNodeEncoder


class DenseGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.act = nn.PReLU()
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if hasattr(self.act, 'reset_parameters'):
            self.act.reset_parameters()

    def forward(self, feat, adj):
        out = torch.bmm(adj, self.fc(feat))
        if self.bias is not None:
            out = out + self.bias
        return self.act(out)


class DenseGCN(nn.Module):
    def __init__(self, in_features, hidden_size, num_layers):
        super().__init__()
        layers = [DenseGCNLayer(in_features, hidden_size)]
        for _ in range(num_layers - 1):
            layers.append(DenseGCNLayer(hidden_size, hidden_size))
        self.layers = nn.ModuleList(layers)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, feat, adj, mask):
        h = feat
        graph_reprs = []
        for layer in self.layers:
            h = layer(h, adj)
            graph_reprs.append((h * mask.unsqueeze(-1).float()).sum(dim=1))
        return h, torch.cat(graph_reprs, dim=-1)


class MVGRLMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.PReLU(),
            nn.Linear(out_features, out_features),
            nn.PReLU(),
            nn.Linear(out_features, out_features),
            nn.PReLU(),
        )
        self.linear_shortcut = nn.Linear(in_features, out_features)

    def reset_parameters(self):
        for module in self.ffn:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.linear_shortcut.reset_parameters()

    def forward(self, x):
        return self.ffn(x) + self.linear_shortcut(x)


class MVGRL(nn.Module):
    """Graph-level MVGRL adapter.

    The official graph MVGRL learns graph representations from two views:
    normalized adjacency and PPR diffusion. This supervised adapter keeps the
    two dense GCN encoders and view-specific projection MLPs, then predicts from
    the sum of both graph-level views under OpenGLT's unified protocol.
    """

    def __init__(self, nhid, nout, num_layers=4, dataset_type='tu',
                 dim_node_deg=0, dim_node_label=0, dim_node_attr=0,
                 dropout=0.0, alpha=0.2):
        super().__init__()
        self.dataset_type = dataset_type
        self.nhid = int(nhid)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.alpha = float(alpha)

        if dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            self.input_encoder = AtomEncoder(emb_dim=nhid)
        else:
            self.input_encoder = TUNodeEncoder(nhid, dim_node_deg, dim_node_label, dim_node_attr)

        self.gnn_adj = DenseGCN(nhid, nhid, self.num_layers)
        self.gnn_diff = DenseGCN(nhid, nhid, self.num_layers)
        self.local_mlp = MVGRLMLP(nhid, nhid)
        self.global_mlp = MVGRLMLP(self.num_layers * nhid, nhid)
        self.graph_head = MLP(nhid, nhid, nlayer=2)
        self.output_decoder = MLP(nhid, nout, nlayer=2)

    def reset_parameters(self):
        if hasattr(self.input_encoder, 'reset_parameters'):
            self.input_encoder.reset_parameters()
        self.gnn_adj.reset_parameters()
        self.gnn_diff.reset_parameters()
        self.local_mlp.reset_parameters()
        self.global_mlp.reset_parameters()
        self.graph_head.reset_parameters()
        self.output_decoder.reset_parameters()

    def _encode_input(self, x):
        if self.dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            if x.dtype == torch.float:
                x = x.long()
        elif x.dtype == torch.long:
            x = x.float()
        return self.input_encoder(x)

    @staticmethod
    def _normalize_dense_adj(adj, mask):
        adj = adj * mask.unsqueeze(1).float() * mask.unsqueeze(2).float()
        eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype).unsqueeze(0)
        adj = adj + eye * mask.unsqueeze(1).float()
        deg = adj.sum(dim=-1).clamp(min=1e-12)
        deg_inv_sqrt = deg.pow(-0.5)
        return deg_inv_sqrt.unsqueeze(2) * adj * deg_inv_sqrt.unsqueeze(1)

    def _ppr(self, adj_norm, mask):
        batch_size, num_nodes, _ = adj_norm.shape
        eye = torch.eye(num_nodes, device=adj_norm.device, dtype=adj_norm.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        mat = eye - (1.0 - self.alpha) * adj_norm
        inv = torch.linalg.pinv(mat)
        diff = self.alpha * inv
        return diff * mask.unsqueeze(1).float() * mask.unsqueeze(2).float()

    def forward(self, x, edge_index, batch, edge_attr=None,
                fingerprint=None, rwpe=None, input_embedding=None, **kwargs):
        if input_embedding is not None:
            x = input_embedding
        else:
            x = self._encode_input(x)

        dense_x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch, max_num_nodes=dense_x.size(1)).to(dense_x.dtype)
        adj = self._normalize_dense_adj(adj, mask)
        diff = self._ppr(adj, mask)

        lv1, gv1 = self.gnn_adj(dense_x, adj, mask)
        lv2, gv2 = self.gnn_diff(dense_x, diff, mask)
        _ = self.local_mlp(lv1), self.local_mlp(lv2)
        gv1 = self.global_mlp(gv1)
        gv2 = self.global_mlp(gv2)
        graph_emb = F.dropout(gv1 + gv2, self.dropout, training=self.training)
        graph_repr = self.graph_head(graph_emb)
        out = self.output_decoder(graph_emb)
        return out, graph_repr
