import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import dense_to_sparse, subgraph, to_dense_adj

from .backbone import VIBGAT, VIBGCN, VIBGIN
from .layers import GraphLearner


class VIBGSL(nn.Module):
    """Graph Structure Learning with Variational Information Bottleneck.

    This adapter keeps the official VIB-GSL recipe: learn a graph per input
    graph, run a shallow GNN backbone, split graph embeddings into mu/std,
    sample a bottleneck representation, and train with supervised loss plus
    beta times the variational KL term.
    """

    def __init__(self, nhid, nout, dataset_type='tu',
                 dim_node_deg=0, dim_node_label=0, dim_node_attr=0,
                 backbone='GCN', graph_type='prob', graph_metric_type='mlp',
                 repar=True, hidden_dim=16, ib_size=16, num_per=16,
                 feature_denoise=False, top_k=10, epsilon=0.3,
                 graph_skip_conn=0.0, graph_include_self=True,
                 beta=1e-5, dropout=0.5):
        super().__init__()
        self.dataset_type = dataset_type
        self.backbone = backbone.upper()
        self.repar = bool(repar)
        self.ib_size = int(ib_size)
        self.beta = float(beta)
        self.graph_skip_conn = graph_skip_conn
        self.graph_include_self = bool(graph_include_self)
        self.dropout = float(dropout)
        self.latest_kl = None

        if dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            self.input_encoder = AtomEncoder(emb_dim=nhid)
            graph_input_dim = nhid
        else:
            self.input_encoder = None
            graph_input_dim = int(dim_node_deg) + int(dim_node_label) + int(dim_node_attr)

        if graph_input_dim <= 0:
            raise ValueError('VIBGSL requires non-empty node features.')

        out_dim = self.ib_size * 2
        if self.backbone == 'GCN':
            self.backbone_gnn = VIBGCN(graph_input_dim, out_dim, hidden_dim)
        elif self.backbone == 'GIN':
            self.backbone_gnn = VIBGIN(graph_input_dim, out_dim, hidden_dim)
        elif self.backbone == 'GAT':
            self.backbone_gnn = VIBGAT(graph_input_dim, out_dim, hidden_dim)
        else:
            raise ValueError(f'Unknown VIBGSL backbone: {backbone}')

        self.graph_learner = GraphLearner(
            input_size=graph_input_dim,
            hidden_size=hidden_dim,
            graph_type=graph_type,
            top_k=top_k,
            epsilon=epsilon,
            num_pers=num_per,
            metric_type=graph_metric_type,
            feature_denoise=feature_denoise,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.ib_size, self.ib_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.ib_size, nout),
        )

    def reset_parameters(self):
        if hasattr(self.input_encoder, 'reset_parameters'):
            self.input_encoder.reset_parameters()
        self.backbone_gnn.reset_parameters()
        self.graph_learner.reset_parameters()
        for module in self.classifier:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.latest_kl = None

    def _encode_input(self, x):
        if self.input_encoder is None:
            return x.float() if x.dtype == torch.long else x
        if x.dtype == torch.float:
            x = x.long()
        return self.input_encoder(x)

    def learn_graph(self, node_features, init_adj):
        new_feature, new_adj = self.graph_learner(node_features)
        if self.graph_skip_conn in (0.0, None):
            if self.graph_include_self:
                new_adj = new_adj + torch.eye(new_adj.size(0), device=new_adj.device, dtype=new_adj.dtype)
        else:
            new_adj = float(self.graph_skip_conn) * init_adj + (1 - float(self.graph_skip_conn)) * new_adj
        return new_feature, new_adj

    @staticmethod
    def kl_loss(mu, std):
        return -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))

    def reparametrize(self, mu, std):
        if not self.repar:
            return mu
        eps = std.new_empty(std.size()).normal_()
        return mu + eps * std

    def _learn_batch_graphs(self, x, edge_index, batch):
        new_graphs = []
        for graph_id in torch.unique(batch, sorted=True):
            node_idx = (batch == graph_id).nonzero(as_tuple=False).view(-1)
            graph_x = x[node_idx]
            graph_edge_index, _ = subgraph(
                node_idx,
                edge_index,
                relabel_nodes=True,
                num_nodes=x.size(0),
            )
            raw_adj = to_dense_adj(
                graph_edge_index,
                max_num_nodes=graph_x.size(0),
            )[0].to(graph_x.device)
            new_feature, new_adj = self.learn_graph(graph_x, raw_adj)
            new_edge_index, new_edge_attr = dense_to_sparse(new_adj)
            new_graphs.append(Data(x=new_feature, edge_index=new_edge_index, edge_attr=new_edge_attr))
        return Batch.from_data_list(new_graphs).to(x.device)

    def forward(self, x, edge_index, batch, edge_attr=None,
                fingerprint=None, rwpe=None, input_embedding=None, **kwargs):
        if input_embedding is not None:
            x = input_embedding
        else:
            x = self._encode_input(x)

        learned_batch = self._learn_batch_graphs(x, edge_index, batch)
        node_embs = self.backbone_gnn(learned_batch.x, learned_batch.edge_index)
        graph_embs = global_mean_pool(node_embs, learned_batch.batch)

        mu = graph_embs[:, :self.ib_size]
        std = F.softplus(graph_embs[:, self.ib_size:] - self.ib_size, beta=1)
        std = std.clamp(min=1e-8)
        self.latest_kl = self.kl_loss(mu, std)
        graph_repr = self.reparametrize(mu, std)
        logits = self.classifier(graph_repr)
        return logits, mu
