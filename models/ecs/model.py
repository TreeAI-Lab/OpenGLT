import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool

from models.baselines import MLP, TUNodeEncoder
from .utils import STRUCT_EMB_SIZE_WITH_RD


class ECSGINConv(MessagePassing):
    def __init__(self, hidden_size, dataset_type='ogb', edge_dim=0):
        super().__init__(aggr='add')
        self.dataset_type = dataset_type
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.BatchNorm1d(2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, hidden_size),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        if dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            self.edge_encoder = BondEncoder(emb_dim=hidden_size)
        else:
            self.edge_encoder = nn.Linear(max(1, int(edge_dim)), hidden_size, bias=False)
        self.edge_struct_encoder = nn.Linear(hidden_size, hidden_size)

    def reset_parameters(self):
        for module in self.mlp:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        nn.init.zeros_(self.eps)
        if hasattr(self.edge_encoder, 'reset_parameters'):
            self.edge_encoder.reset_parameters()
        self.edge_struct_encoder.reset_parameters()

    def _encode_edge_attr(self, edge_attr, edge_index):
        num_edges = edge_index.size(1)
        if edge_attr is not None and not torch.is_tensor(edge_attr):
            edge_attr = None
        if self.dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            if edge_attr is None:
                edge_attr = edge_index.new_zeros((num_edges, 3))
            if edge_attr.dtype == torch.float:
                edge_attr = edge_attr.long()
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            return self.edge_encoder(edge_attr)

        if edge_attr is None:
            edge_attr = edge_index.new_zeros((num_edges, 1), dtype=torch.float)
        if edge_attr.dtype == torch.long:
            edge_attr = edge_attr.float()
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        return self.edge_encoder(edge_attr)

    def forward(self, x, edge_index, edge_attr, edge_struct):
        edge_embedding = self._encode_edge_attr(edge_attr, edge_index)
        edge_embedding = edge_embedding + self.edge_struct_encoder(edge_struct)
        out = self.mlp((1 + self.eps) * x + self.propagate(
            edge_index, x=x, edge_attr=edge_embedding
        ))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)


class ECSNodeEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dataset_type='ogb',
                 dim_node_deg=0, dim_node_label=0, dim_node_attr=0,
                 num_edge_classes=0, dropout=0.5, residual=False,
                 virtual_node=True, use_rd=True):
        super().__init__()
        self.dataset_type = dataset_type
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.dropout = float(dropout)
        self.residual = bool(residual)
        self.virtual_node = bool(virtual_node)
        self.use_rd = bool(use_rd)

        if dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            self.input_encoder = AtomEncoder(emb_dim=hidden_size)
        else:
            self.input_encoder = TUNodeEncoder(hidden_size, dim_node_deg, dim_node_label, dim_node_attr)

        self.struct_embedding = nn.Embedding(STRUCT_EMB_SIZE_WITH_RD, hidden_size)
        self.struct_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

        self.convs = nn.ModuleList([
            ECSGINConv(hidden_size, dataset_type=dataset_type, edge_dim=num_edge_classes)
            for _ in range(self.num_layers)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for _ in range(self.num_layers)
        ])

        if self.virtual_node:
            self.virtualnode_embedding = nn.Embedding(1, hidden_size)
            self.virtualnode_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, 2 * hidden_size),
                    nn.BatchNorm1d(2 * hidden_size),
                    nn.ReLU(),
                    nn.Linear(2 * hidden_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                )
                for _ in range(max(self.num_layers - 1, 0))
            ])

    def reset_parameters(self):
        if hasattr(self.input_encoder, 'reset_parameters'):
            self.input_encoder.reset_parameters()
        self.struct_embedding.reset_parameters()
        for module in self.struct_mlp:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        if self.virtual_node:
            nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            for mlp in self.virtualnode_mlps:
                for module in mlp:
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()

    def _encode_input(self, x):
        if self.dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            if x.dtype == torch.float:
                x = x.long()
        elif x.dtype == torch.long:
            x = x.float()
        return self.input_encoder(x)

    def _edge_structural_embedding(self, data):
        if not all(hasattr(data, name) for name in ('pos_index', 'pos_enc', 'pos_batch')):
            return data.edge_index.new_zeros((data.edge_index.size(1), self.hidden_size), dtype=torch.float)
        if data.pos_index.numel() == 0:
            return data.edge_index.new_zeros((data.edge_index.size(1), self.hidden_size), dtype=torch.float)
        pos_index = data.pos_index.long().clamp(max=self.struct_embedding.num_embeddings - 1)
        weighted = self.struct_embedding(pos_index) * data.pos_enc.to(self.struct_embedding.weight.dtype).view(-1, 1)
        edge_struct = global_add_pool(weighted, data.pos_batch.long(), size=data.edge_index.size(1))
        return self.struct_mlp(edge_struct)

    def forward(self, data, return_all=False):
        x = self._encode_input(data.x)
        edge_struct = self._edge_structural_embedding(data)
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = data.batch

        if self.virtual_node:
            virtual = self.virtualnode_embedding(
                torch.zeros(int(batch.max().item()) + 1, dtype=torch.long, device=batch.device)
            )

        h_list = [x]
        for layer in range(self.num_layers):
            h_in = h_list[layer]
            if self.virtual_node:
                h_in = h_in + virtual[batch]

            h = self.convs[layer](h_in, edge_index, edge_attr, edge_struct)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)
            if self.residual:
                h = h + h_list[layer]
            h_list.append(h)

            if self.virtual_node and layer < self.num_layers - 1:
                virtual_tmp = global_add_pool(h_list[layer], batch) + virtual
                update = self.virtualnode_mlps[layer](virtual_tmp)
                if self.residual:
                    virtual = virtual + F.dropout(update, self.dropout, training=self.training)
                else:
                    virtual = F.dropout(update, self.dropout, training=self.training)

        return h_list if return_all else h_list[-1]


class ECSGNN(nn.Module):
    """Efficient Substructure Counting GNN adapter for OpenGLT.

    This implements the official ESC-GNN efficient path: edge-rooted enclosing
    subgraphs are pre-encoded as sparse structural vectors, projected to edge
    embeddings, and injected into a GIN-style message passing network that runs
    once on the original graph.
    """

    def __init__(self, nhid, nout, num_layers=5, dataset_type='ogb',
                 dim_node_deg=0, dim_node_label=0, dim_node_attr=0,
                 num_edge_classes=0, dropout=0.5, pooling='mean',
                 residual=False, virtual_node=True, use_rd=True,
                 concat_readout=False):
        super().__init__()
        self.pooling = pooling
        self.dropout = float(dropout)
        self.concat_readout = bool(concat_readout)
        self.num_layers = int(num_layers)
        self.node_encoder = ECSNodeEncoder(
            hidden_size=nhid,
            num_layers=num_layers,
            dataset_type=dataset_type,
            dim_node_deg=dim_node_deg,
            dim_node_label=dim_node_label,
            dim_node_attr=dim_node_attr,
            num_edge_classes=num_edge_classes,
            dropout=dropout,
            residual=residual,
            virtual_node=virtual_node,
            use_rd=use_rd,
        )
        readout_dim = nhid * (self.num_layers + 1) if self.concat_readout else nhid
        self.graph_head = MLP(readout_dim, nhid, nlayer=2)
        decoder_in = nhid if self.concat_readout else readout_dim
        self.output_decoder = MLP(decoder_in, nout, nlayer=2)

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.graph_head.reset_parameters()
        self.output_decoder.reset_parameters()

    def _pool(self, x, batch):
        if self.pooling in ('sum', 'add'):
            return global_add_pool(x, batch)
        if self.pooling == 'max':
            return global_max_pool(x, batch)
        return global_mean_pool(x, batch)

    def forward(self, x, edge_index, batch, edge_attr=None,
                fingerprint=None, rwpe=None, input_embedding=None, **kwargs):
        data = kwargs.get('data', None)
        if data is None:
            from torch_geometric.data import Data
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        if input_embedding is not None:
            data.x = input_embedding
        node_repr = self.node_encoder(data, return_all=self.concat_readout)
        if self.concat_readout:
            node_repr = torch.cat(node_repr, dim=-1)
        graph_emb = self._pool(node_repr, data.batch)
        graph_emb = F.dropout(graph_emb, self.dropout, training=self.training)
        graph_repr = self.graph_head(graph_emb)
        out = self.output_decoder(graph_repr if self.concat_readout else graph_emb)
        return out, graph_repr
