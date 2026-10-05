import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

from models.baselines import TUNodeEncoder
from .layers import GCN, HGPSLPool


class HGPSL(nn.Module):
    """Hierarchical Graph Pooling with Structure Learning adapter.

    The implementation follows the official HGP-SL graph classification model:
    GCN -> HGPSLPool -> GCN -> HGPSLPool -> GCN, with mean+max graph readouts
    summed across hierarchy levels. The prediction head is kept as raw logits so
    OpenGLT can apply its unified classification/regression losses and metrics.
    """

    def __init__(self, nhid, nout, dataset_type='tu',
                 dim_node_deg=0, dim_node_label=0, dim_node_attr=0,
                 pooling_ratio=0.5, dropout=0.0, sample_neighbor=True,
                 sparse_attention=True, structure_learning=True, lamb=1.0):
        super().__init__()
        self.dataset_type = dataset_type
        self.nhid = int(nhid)
        self.nout = int(nout)
        self.dropout = float(dropout)

        if dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            self.input_encoder = AtomEncoder(emb_dim=nhid)
        else:
            self.input_encoder = TUNodeEncoder(nhid, dim_node_deg, dim_node_label, dim_node_attr)

        self.conv1 = GCNConv(nhid, nhid)
        self.conv2 = GCN(nhid, nhid)
        self.conv3 = GCN(nhid, nhid)

        self.pool1 = HGPSLPool(
            nhid, pooling_ratio, sample_neighbor, sparse_attention,
            structure_learning, lamb,
        )
        self.pool2 = HGPSLPool(
            nhid, pooling_ratio, sample_neighbor, sparse_attention,
            structure_learning, lamb,
        )

        self.lin1 = nn.Linear(nhid * 2, nhid)
        self.lin2 = nn.Linear(nhid, max(nhid // 2, 1))
        self.lin3 = nn.Linear(max(nhid // 2, 1), nout)

    def reset_parameters(self):
        if hasattr(self.input_encoder, 'reset_parameters'):
            self.input_encoder.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.pool1.reset_parameters()
        self.pool2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def _encode_input(self, x):
        if self.dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            if x.dtype == torch.float:
                x = x.long()
        elif x.dtype == torch.long:
            x = x.float()
        return self.input_encoder(x)

    @staticmethod
    def _readout(x, batch):
        return torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

    def forward(self, x, edge_index, batch, edge_attr=None,
                fingerprint=None, rwpe=None, input_embedding=None, **kwargs):
        if input_embedding is not None:
            x = input_embedding
        else:
            x = self._encode_input(x)

        edge_weight = None
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x, edge_index, edge_weight, batch = self.pool1(x, edge_index, edge_weight, batch)
        x1 = self._readout(x, batch)

        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x, edge_index, edge_weight, batch = self.pool2(x, edge_index, edge_weight, batch)
        x2 = self._readout(x, batch)

        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x3 = self._readout(x, batch)

        graph_emb = F.relu(x1) + F.relu(x2) + F.relu(x3)
        h = F.relu(self.lin1(graph_emb))
        h = F.dropout(h, p=self.dropout, training=self.training)
        graph_repr = h
        h = F.relu(self.lin2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.lin3(h)
        return out, graph_repr
