import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_scatter import scatter

from ..good_encoders import GOODAtomEncoder, GOODBondEncoder
from .model_utils import pyg_gnn_wrapper as gnn_wrapper
from .model_utils.elements import DiscreteEncoder, Identity, MLP, VNUpdate


class GNN(nn.Module):
    def __init__(self, nin, nout, nlayer, gnn_type, dropout=0, bn=True,
                 bias=True, res=True):
        super().__init__()
        self.convs = nn.ModuleList([
            getattr(gnn_wrapper, gnn_type)(nin, nin, bias=not bn)
            for _ in range(nlayer)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(nin) if bn else Identity()
            for _ in range(nlayer)
        ])
        self.output_encoder = (
            MLP(nin, nout, nlayer=1, with_final_activation=False, bias=bias)
            if nin != nout else Identity()
        )
        self.dropout = dropout
        self.res = res

    def reset_parameters(self):
        self.output_encoder.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch=None):
        previous_x = x
        for layer, norm in zip(self.convs, self.norms):
            x = layer(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x = x + previous_x
                previous_x = x
        return self.output_encoder(x)


class SubgraphGNNKernel(nn.Module):
    def __init__(self, nin, nout, nlayer, gnn_types, dropout=0, hop_dim=16,
                 bias=True, res=True, pooling='mean', embs=(0, 1, 2),
                 embs_combine_mode='add', mlp_layers=1, subsampling=False,
                 online=True):
        super().__init__()
        assert max(embs) <= 2 and min(embs) >= 0
        assert embs_combine_mode in ['add', 'concat']
        use_hops = hop_dim > 0
        nhid = nout // len(gnn_types)
        self.hop_embedder = nn.Embedding(64, hop_dim)
        self.gnns = nn.ModuleList([
            GNN(nin + hop_dim if use_hops else nin, nhid, nlayer, gnn_type,
                dropout=dropout, res=res)
            for gnn_type in gnn_types
        ])
        self.subgraph_transform = MLP(nout, nout, nlayer=mlp_layers,
                                      with_final_activation=True)
        self.context_transform = MLP(nout, nout, nlayer=mlp_layers,
                                     with_final_activation=True)
        self.out_encoder = MLP(
            nout if embs_combine_mode == 'add' else nout * len(embs),
            nout,
            nlayer=mlp_layers,
            with_final_activation=False,
            bias=bias,
            with_norm=True,
        )
        self.use_hops = use_hops
        self.gate_mapper_subgraph = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid())
        self.gate_mapper_context = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid())
        self.gate_mapper_centroid = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid())
        self.subsampling = subsampling
        self.dropout = dropout
        self.online = online
        self.pooling = pooling
        self.embs = tuple(embs)
        self.embs_combine_mode = embs_combine_mode

    def reset_parameters(self):
        self.hop_embedder.reset_parameters()
        for gnn in self.gnns:
            gnn.reset_parameters()
        self.subgraph_transform.reset_parameters()
        self.context_transform.reset_parameters()
        self.out_encoder.reset_parameters()
        reset(self.gate_mapper_context)
        reset(self.gate_mapper_subgraph)
        reset(self.gate_mapper_centroid)

    def forward(self, data):
        combined_x = data.x[data.subgraphs_nodes_mapper]
        combined_edge_index = data.combined_subgraphs
        combined_edge_attr = data.edge_attr[data.subgraphs_edges_mapper]
        combined_batch = data.subgraphs_batch
        if self.use_hops:
            hop_emb = self.hop_embedder((data.hop_indicator + 1).clamp_min(0).clamp_max(63))
            combined_x = torch.cat([combined_x, hop_emb], dim=-1)
        else:
            hop_emb = None

        combined_x = torch.cat([
            gnn(combined_x, combined_edge_index, combined_edge_attr, combined_batch)
            for gnn in self.gnns
        ], dim=-1)

        if self.subsampling and self.training:
            selected_mask = data.subgraphs_nodes_mapper == data.selected_supernodes[combined_batch]
            centroid_x_selected = combined_x[selected_mask]
            subgraph_x_selected = (
                self.subgraph_transform(F.dropout(combined_x, self.dropout, training=self.training))
                if len(self.embs) > 1 else combined_x
            )
            context_x_selected = (
                self.context_transform(F.dropout(combined_x, self.dropout, training=self.training))
                if len(self.embs) > 1 else combined_x
            )
            if self.use_hops:
                centroid_x_selected = centroid_x_selected * self.gate_mapper_centroid(hop_emb[selected_mask])
                subgraph_x_selected = subgraph_x_selected * self.gate_mapper_subgraph(hop_emb)
                context_x_selected = context_x_selected * self.gate_mapper_context(hop_emb)
            subgraph_x_selected = scatter(subgraph_x_selected, combined_batch,
                                          dim=0, reduce=self.pooling)
            # Some nodes may be absent from every sampled subgraph (in
            # particular, an absent node can be the last node in a PyG
            # batch).  Letting scatter infer its output size would then drop
            # those trailing nodes, while centroid_x, subgraph_x and
            # subsampling_scale still use the original node count.  Keep all
            # node-level representations aligned with data.x.
            context_x = scatter(
                context_x_selected,
                data.subgraphs_nodes_mapper,
                dim=0,
                dim_size=data.x.size(0),
                reduce=self.pooling,
            )
            centroid_x = data.x.new_zeros((len(data.x), centroid_x_selected.size(-1)))
            centroid_x[data.selected_supernodes] = centroid_x_selected
            subgraph_x = data.x.new_zeros((len(data.x), subgraph_x_selected.size(-1)))
            subgraph_x[data.selected_supernodes] = subgraph_x_selected
            max_hop = int(data.edges_between_two_hops.max().item()) if data.edges_between_two_hops.numel() else 0
            for i in range(1, max_hop + 1):
                bipartite = data.edge_index[:, data.edges_between_two_hops == i]
                scatter(centroid_x[bipartite[0]], bipartite[1], dim=0,
                        reduce='mean', out=centroid_x)
                scatter(subgraph_x[bipartite[0]], bipartite[1], dim=0,
                        reduce='mean', out=subgraph_x)
            if self.pooling == 'add':
                context_x = context_x * data.subsampling_scale.unsqueeze(-1)
        else:
            centroid_mask = data.subgraphs_nodes_mapper == combined_batch
            centroid_x = combined_x[centroid_mask]
            subgraph_x = (
                self.subgraph_transform(F.dropout(combined_x, self.dropout, training=self.training))
                if len(self.embs) > 1 else combined_x
            )
            context_x = (
                self.context_transform(F.dropout(combined_x, self.dropout, training=self.training))
                if len(self.embs) > 1 else combined_x
            )
            if self.use_hops:
                centroid_x = centroid_x * self.gate_mapper_centroid(hop_emb[centroid_mask])
                subgraph_x = subgraph_x * self.gate_mapper_subgraph(hop_emb)
                context_x = context_x * self.gate_mapper_context(hop_emb)
            subgraph_x = scatter(subgraph_x, combined_batch, dim=0,
                                 reduce=self.pooling)
            context_x = scatter(
                context_x,
                data.subgraphs_nodes_mapper,
                dim=0,
                dim_size=data.x.size(0),
                reduce=self.pooling,
            )

        parts = [centroid_x, subgraph_x, context_x]
        selected = [parts[i] for i in self.embs]
        if self.embs_combine_mode == 'add':
            return sum(selected)
        x = torch.cat(selected, dim=-1)
        return self.out_encoder(F.dropout(x, self.dropout, training=self.training))


class GNNAsKernel(nn.Module):
    def __init__(self, nfeat_node, nfeat_edge, nhid, nout, nlayer_outer,
                 nlayer_inner, gnn_types, dropout=0, hop_dim=0,
                 node_embedding=False, use_normal_gnn=False, bn=True, vn=False,
                 res=True, pooling='mean', embs=(0, 1, 2),
                 embs_combine_mode='add', mlp_layers=1, subsampling=False,
                 online=True, dataset_type=None):
        super().__init__()
        edge_emb_dim = nhid if nlayer_inner == 0 else nhid + hop_dim
        if dataset_type == 'good_mol':
            # Keep GNNAK on exactly the same released GOOD categorical schema
            # used by the other molecular baselines.  The former generic
            # 512-way encoders preserved the integer IDs, but used a different
            # parameterization and initialization from the GOOD experiments.
            self.input_encoder = GOODAtomEncoder(nhid)
            self.edge_encoders = nn.ModuleList([
                GOODBondEncoder(edge_emb_dim)
                for _ in range(nlayer_outer)
            ])
        else:
            self.input_encoder = (
                DiscreteEncoder(nhid) if nfeat_node is None
                else MLP(nfeat_node, nhid, 1)
            )
            self.edge_encoders = nn.ModuleList([
                DiscreteEncoder(edge_emb_dim) if nfeat_edge is None
                else MLP(nfeat_edge, edge_emb_dim, 1)
                for _ in range(nlayer_outer)
            ])
        self.subgraph_layers = nn.ModuleList([
            SubgraphGNNKernel(
                nhid, nhid, nlayer_inner, gnn_types, dropout,
                hop_dim=hop_dim, bias=not bn, res=res, pooling=pooling,
                embs=embs, embs_combine_mode=embs_combine_mode,
                mlp_layers=mlp_layers, subsampling=subsampling, online=online,
            )
            for _ in range(nlayer_outer)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(nhid) if bn else Identity()
            for _ in range(nlayer_outer)
        ])
        self.output_decoder = MLP(nhid, nout, nlayer=2,
                                  with_final_activation=False)
        self.traditional_gnns = nn.ModuleList([
            getattr(gnn_wrapper, gnn_types[0])(nhid, nhid, bias=not bn)
            for _ in range(nlayer_outer)
        ])
        self.vn_aggregators = nn.ModuleList([
            VNUpdate(nhid) for _ in range(nlayer_outer)
        ])
        self.gnn_type = gnn_types[0]
        self.dropout = dropout
        self.num_inner_layers = nlayer_inner
        self.node_embedding = node_embedding
        self.use_normal_gnn = use_normal_gnn
        self.hop_dim = hop_dim
        self.vn = vn
        self.res = res
        self.pooling = pooling

    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        self.output_decoder.reset_parameters()
        for edge_encoder, layer, norm, old, vn in zip(
            self.edge_encoders, self.subgraph_layers, self.norms,
            self.traditional_gnns, self.vn_aggregators,
        ):
            edge_encoder.reset_parameters()
            layer.reset_parameters()
            norm.reset_parameters()
            old.reset_parameters()
            vn.reset_parameters()

    def forward(self, data):
        x = data.x if len(data.x.shape) <= 2 else data.x.squeeze(-1)
        x = self.input_encoder(x)
        ori_edge_attr = data.edge_attr
        if ori_edge_attr is None:
            ori_edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        if ori_edge_attr.dim() == 1:
            ori_edge_attr = ori_edge_attr.unsqueeze(-1)

        previous_x = x
        virtual_node = None
        for edge_encoder, subgraph_layer, normal_gnn, norm, vn_aggregator in zip(
            self.edge_encoders, self.subgraph_layers, self.traditional_gnns,
            self.norms, self.vn_aggregators,
        ):
            data.edge_attr = edge_encoder(ori_edge_attr)
            data.x = x
            if self.num_inner_layers == 0:
                x = normal_gnn(data.x, data.edge_index, data.edge_attr)
            elif self.use_normal_gnn:
                edge_attr_normal = data.edge_attr[:, :-self.hop_dim] if self.hop_dim > 0 else data.edge_attr
                x = subgraph_layer(data) + normal_gnn(data.x, data.edge_index, edge_attr_normal)
            else:
                x = subgraph_layer(data)

            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.vn:
                virtual_node, x = vn_aggregator(virtual_node, x, data.batch)
            if self.res:
                x = x + previous_x
                previous_x = x

        if not self.node_embedding:
            x = scatter(x, data.batch, dim=0, reduce=self.pooling)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.output_decoder(x)
