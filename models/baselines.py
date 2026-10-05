import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import add_remaining_self_loops, degree
import torch_geometric.nn as gnn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, GINConv, GatedGraphConv, SAGEConv, PNAConv, GINEConv, TransformerConv, NNConv, ResGatedGraphConv, EdgePooling, TopKPooling, GraphMultisetTransformer, global_add_pool, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
import numpy as np

from .good_encoders import GOODAtomEncoder, GOODBondEncoder


MLP_BASELINE_BACKBONES = {'MLPFeature', 'MLPFeatureStruct', 'MLP', 'MLPStruct'}
TABULAR_BASELINE_BACKBONES = MLP_BASELINE_BACKBONES | {'LightGBM'}

def gnn_type_needs_edge_features(gnn_type):
    """
    Check if a GNN type requires edge features.

    Args:
        gnn_type: str, the GNN type name

    Returns:
        bool: True if the GNN type needs edge features, False otherwise
    """
    edge_feature_gnn_types = [
        'GINE', 'GCNplus', 'SAGEplus', 'GAT', 'GATv2', 'Transformer', 'NN', 'SimplifiedPNA', 'PNA'
    ]
    return gnn_type in edge_feature_gnn_types


def gnn_type_supports_edgepool(gnn_type):
    return gnn_type in ('GCN', 'GIN', 'SAGE')


def get_edge_score_method(method):
    method = method.lower()
    if method == 'softmax':
        return EdgePooling.compute_edge_score_softmax
    if method == 'tanh':
        return EdgePooling.compute_edge_score_tanh
    if method == 'sigmoid':
        return EdgePooling.compute_edge_score_sigmoid
    raise ValueError(f"Unknown edge_score_method: {method}")


def make_norm(norm_type, hidden_size):
    norm_type = norm_type.lower()
    if norm_type == 'none':
        return nn.Identity()
    if norm_type == 'batch_norm':
        return nn.BatchNorm1d(hidden_size)
    if norm_type == 'layer_norm':
        return nn.LayerNorm(hidden_size)
    raise ValueError(f"Unknown norm_type: {norm_type}")


def make_activation(act_type, hidden_size):
    act_type = act_type.lower()
    if act_type == 'relu':
        return nn.ReLU()
    if act_type == 'prelu':
        return nn.PReLU(hidden_size)
    if act_type == 'swish':
        return nn.SiLU()
    raise ValueError(f"Unknown act_type: {act_type}")


def _safe_scatter(src, index, dim_size, reduce):
    if src.numel() == 0:
        return src.new_zeros((dim_size, src.size(-1)))
    return scatter(src, index, dim=0, dim_size=dim_size, reduce=reduce)


def _pool_feature_stats(values, index, dim_size, include_sum=True):
    if values is None:
        values = torch.zeros((0, 1))
    if values.dim() == 1:
        values = values.unsqueeze(-1)
    values = values.float()
    if values.numel() == 0:
        feat_dim = values.size(-1)
        zero = values.new_zeros((dim_size, feat_dim))
        stats = [zero]
        if include_sum:
            stats.append(zero)
        stats.extend([zero, zero, zero])
        return torch.cat(stats, dim=-1)

    count = torch.bincount(index, minlength=dim_size).clamp(min=1).to(values.device).view(-1, 1).float()
    summed = _safe_scatter(values, index, dim_size, 'sum')
    mean = summed / count
    centered = values - mean[index]
    var = _safe_scatter(centered * centered, index, dim_size, 'sum') / count
    std = torch.sqrt(var.clamp(min=0.0) + 1e-12)
    maxv = _safe_scatter(values, index, dim_size, 'max')
    minv = _safe_scatter(values, index, dim_size, 'min')
    has_value = (torch.bincount(index, minlength=dim_size).to(values.device) > 0).view(-1, 1)
    maxv = torch.where(has_value, maxv, torch.zeros_like(maxv))
    minv = torch.where(has_value, minv, torch.zeros_like(minv))
    stats = [mean]
    if include_sum:
        stats.append(summed)
    stats.extend([std, maxv, minv])
    return torch.cat(stats, dim=-1)


def graph_feature_dim(node_dim, edge_dim, include_structure=False,
                      include_sum=True, include_topology=False):
    edge_dim = max(int(edge_dim), 1)
    num_stats = 5 if include_sum else 4
    dim = num_stats * int(node_dim) + num_stats * edge_dim
    if include_structure:
        dim += 16
    if include_topology:
        dim += 1
    return dim


def build_graph_feature_batch(x, edge_index, batch, edge_attr=None,
                              include_structure=False, include_sum=True,
                              topology_size=None, include_topology=False):
    if x.dtype == torch.long:
        x = x.float()
    if edge_attr is not None and edge_attr.dtype == torch.long:
        edge_attr = edge_attr.float()
    if edge_attr is not None and edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(-1)

    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    node_stats = _pool_feature_stats(x, batch, num_graphs, include_sum=include_sum)

    if edge_index.numel() > 0:
        edge_batch = batch[edge_index[0]]
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.size(1), 1))
    else:
        edge_batch = batch.new_empty((0,))
        edge_attr = x.new_zeros((0, 1)) if edge_attr is None else edge_attr
    edge_stats = _pool_feature_stats(edge_attr, edge_batch, num_graphs, include_sum=include_sum)
    features = [node_stats, edge_stats]

    if include_structure:
        device = x.device
        node_count = torch.bincount(batch, minlength=num_graphs).float().to(device)
        edge_count = torch.bincount(edge_batch, minlength=num_graphs).float().to(device)
        row, col = edge_index
        out_deg = x.new_zeros(x.size(0))
        in_deg = x.new_zeros(x.size(0))
        if edge_index.numel() > 0:
            out_deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float, device=device))
            in_deg.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float, device=device))
        total_deg = in_deg + out_deg
        deg_mean = _safe_scatter(total_deg.view(-1, 1), batch, num_graphs, 'mean').squeeze(-1)
        deg_max = _safe_scatter(total_deg.view(-1, 1), batch, num_graphs, 'max').squeeze(-1)
        deg_min = _safe_scatter(total_deg.view(-1, 1), batch, num_graphs, 'min').squeeze(-1)
        deg_std = _pool_feature_stats(total_deg.view(-1, 1), batch, num_graphs)[:, 2]
        out_mean = _safe_scatter(out_deg.view(-1, 1), batch, num_graphs, 'mean').squeeze(-1)
        in_mean = _safe_scatter(in_deg.view(-1, 1), batch, num_graphs, 'mean').squeeze(-1)
        possible_directed = (node_count * (node_count - 1.0)).clamp(min=1.0)
        density = edge_count / possible_directed
        self_loop_count = x.new_zeros(num_graphs)
        if edge_index.numel() > 0:
            loop_mask = row == col
            if loop_mask.any():
                self_loop_count.scatter_add_(0, edge_batch[loop_mask], torch.ones(int(loop_mask.sum().item()), device=device))
        isolated = (total_deg == 0).float()
        isolated_ratio = _safe_scatter(isolated.view(-1, 1), batch, num_graphs, 'mean').squeeze(-1)
        struct = torch.stack([
            torch.log1p(node_count),
            torch.log1p(edge_count),
            density,
            deg_mean,
            deg_std,
            deg_max,
            deg_min,
            in_mean,
            out_mean,
            isolated_ratio,
            self_loop_count / edge_count.clamp(min=1.0),
            (edge_count / node_count.clamp(min=1.0)),
            torch.sqrt(node_count.clamp(min=0.0)),
            torch.sqrt(edge_count.clamp(min=0.0)),
            (deg_max / node_count.clamp(min=1.0)),
            (deg_std / (deg_mean.abs() + 1e-6)),
        ], dim=-1)
        features.append(struct)

    if include_topology:
        if topology_size is None:
            topo = x.new_zeros((num_graphs, 1))
        else:
            topo = topology_size.to(x.device).float().view(-1, 1)
            if topo.size(0) != num_graphs:
                topo = topo[:num_graphs]
        features.append(topo)

    return torch.cat(features, dim=-1)


class GraphFeatureMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size=128, num_layers=3,
                 dropout=0.1, norm_type='batch_norm', act_type='relu',
                 include_structure=False, include_sum=True,
                 include_topology=False):
        super().__init__()
        self.include_structure = include_structure
        self.include_sum = include_sum
        self.include_topology = include_topology
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        hidden_size = int(hidden_size)
        num_layers = max(1, int(num_layers))

        layers = []
        last_dim = self.in_dim
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(make_norm(norm_type, hidden_size))
            layers.append(make_activation(act_type, hidden_size))
            layers.append(nn.Dropout(dropout))
            last_dim = hidden_size
        layers.append(nn.Linear(last_dim, self.out_dim))
        self.net = nn.Sequential(*layers)

    def reset_parameters(self):
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, x, edge_index, batch, edge_attr=None, fingerprint=None, rwpe=None,
                input_embedding=None, topology_size=None, **kwargs):
        graph_x = build_graph_feature_batch(
            x, edge_index, batch, edge_attr=edge_attr,
            include_structure=self.include_structure,
            include_sum=self.include_sum,
            topology_size=topology_size,
            include_topology=self.include_topology,
        )
        out = self.net(graph_x)
        return out, graph_x


class FeedForwardBlock(nn.Module):
    def __init__(self, hidden_size, expansion=2, dropout=0.0, act_type='relu'):
        super().__init__()
        inner_size = int(hidden_size * expansion)
        self.lin1 = nn.Linear(hidden_size, inner_size)
        self.act = make_activation(act_type, inner_size)
        self.dropout1 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(inner_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if hasattr(self.act, 'reset_parameters'):
            self.act.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.lin2(x)
        x = self.dropout2(x)
        return x


class TUNodeEncoder(nn.Module):
    """
    dataset_type='tu':
    Split x (Degree|Label|Attr) -> Concat
    1|2|3 *nhid -> nhid
    """
    def __init__(self, nhid, dim_degree, dim_label, dim_attr):
        super().__init__()
        self.dim_degree = dim_degree
        self.dim_label = dim_label
        self.dim_attr = dim_attr
        self.nhid = nhid

        self.deg_encoder = nn.Linear(dim_degree, nhid, bias=False)

        if dim_label > 0:
            self.lbl_encoder = nn.Linear(dim_label, nhid, bias=False)

        if dim_attr > 0:
            self.attr_encoder = nn.Linear(dim_attr, nhid, bias=True)

        concat_dim = nhid
        if dim_label > 0: concat_dim += nhid
        if dim_attr > 0: concat_dim += nhid
        
        # concat_dim -> nhid
        self.fusion_mlp = MLP(concat_dim, nhid, nlayer=2, with_final_activation=True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.deg_encoder.weight)
        
        if hasattr(self, 'lbl_encoder'):
            nn.init.xavier_uniform_(self.lbl_encoder.weight)
            
        if hasattr(self, 'attr_encoder'):
            nn.init.xavier_uniform_(self.attr_encoder.weight)
            if self.attr_encoder.bias is not None:
                nn.init.zeros_(self.attr_encoder.bias)

        self.fusion_mlp.reset_parameters()
    
    def forward(self, x):
        slices = []
        pointer = 0

        # Degree
        x_deg = x[:, pointer : pointer + self.dim_degree]
        h_deg = self.deg_encoder(x_deg)
        slices.append(h_deg)
        pointer += self.dim_degree

        # Label
        if self.dim_label > 0:
            x_label = x[:, pointer : pointer + self.dim_label]
            h_label = self.lbl_encoder(x_label)
            slices.append(h_label)
            pointer += self.dim_label

        # Attr
        if self.dim_attr > 0:
            x_attr = x[:, pointer : pointer + self.dim_attr]
            h_attr = self.attr_encoder(x_attr)
            slices.append(h_attr)

        # Concat
        out = torch.cat(slices, dim=1) # [N, 2*nhid] or [N, 3*nhid]
        out = self.fusion_mlp(out)     # [N, nhid]
        
        return out


class GLGNN(nn.Module): # Basic GNNs for Graph-Level Tasks
    def __init__(self, nhid, nout, n_layer_basicgnn, gnn_types, 
                 dataset_type='ogb', dim_node_deg=0, dim_node_label=0, dim_node_attr=0, num_edge_classes=0, 
                 dropout=0, readout_type='mean', pooling_method='none',
                 pool_ratio=0.9, num_pool_layers=1, pool_readout='mean',
                 edgepool_dropout=0.2, edge_score_method='softmax',
                 gmt_k=8, gmt_heads=1, gmt_blocks=1, gmt_layer_norm=False,
                 res=True,
                 deg=None, use_virtual_node=False, use_fingerprints=False, fingerprint_dim=2048, 
                 use_rwpe=False, rwpe_dim=20, norm_type='batch_norm', act_type='relu',
                 use_ffn=False, ffn_expansion=2, **kwargs):
        super().__init__()

        assert dataset_type in [
            'ogb', 'tu', 'counting', 'lrgb', 'pyg_mol', 'good_mol'
        ]
        self.dataset_type = dataset_type

        if self.dataset_type == 'good_mol':
            self.input_encoder = GOODAtomEncoder(emb_dim=nhid)
            self.edge_encoder = GOODBondEncoder(emb_dim=nhid)
        elif self.dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            self.input_encoder = AtomEncoder(emb_dim=nhid)
            self.edge_encoder = BondEncoder(emb_dim=nhid)
        else:
            self.input_encoder = TUNodeEncoder(nhid, dim_node_deg, dim_node_label, dim_node_attr)
            if num_edge_classes > 0:
                self.edge_encoder = nn.Linear(num_edge_classes, nhid, bias=False)
            else:
                self.edge_encoder = nn.Linear(1, nhid, bias=False)

        self.use_virtual_node = use_virtual_node
        if self.use_virtual_node:
            self.virtual_node_embedding = nn.Embedding(1, nhid)
            self.virtual_node_mlps = nn.ModuleList()
            # For each GNN layer: MLP for updating Virtual Node
            for _ in range(n_layer_basicgnn - 1):
                self.virtual_node_mlps.append(
                    nn.Sequential(
                        MLP(nhid, nhid, nlayer=2, with_final_activation=True),
                        nn.LayerNorm(nhid), 
                        nn.Dropout(dropout)
                    )
                )

        self.gnn_layers = nn.ModuleList()
        if n_layer_basicgnn > len(gnn_types):
            gnn_types = gnn_types + [gnn_types[-1]] * (n_layer_basicgnn - len(gnn_types))
        self.gnn_types = gnn_types
        
        self.gnn_layers = nn.ModuleList([get_gnn_conv(gnn_types[i], nhid, nhid, deg, **kwargs) for i in range(n_layer_basicgnn)])

        self.norm_type = norm_type
        self.act_type = act_type
        self.use_ffn = use_ffn
        self.ffn_expansion = ffn_expansion
        self.norms = nn.ModuleList([make_norm(norm_type, nhid) for _ in range(n_layer_basicgnn)])
        self.activations = nn.ModuleList([make_activation(act_type, nhid) for _ in range(n_layer_basicgnn)])
        if self.use_ffn:
            self.ffn_norms = nn.ModuleList([make_norm(norm_type, nhid) for _ in range(n_layer_basicgnn)])
            self.ffn_mlps = nn.ModuleList([
                FeedForwardBlock(nhid, expansion=ffn_expansion, dropout=dropout, act_type=act_type)
                for _ in range(n_layer_basicgnn)
            ])

        self.dropout = dropout
        self.res = res
        self.readout_type = readout_type
        self.pooling_method = pooling_method.lower()
        self.pool_readout = pool_readout.lower()
        if self.pooling_method not in ('none', 'topk', 'edgepool', 'gmt'):
            raise ValueError(f"Unknown pooling_method: {pooling_method}")
        if self.readout_type not in ('mean', 'sum', 'add', 'max', 'attn'):
            raise ValueError(f"Unknown readout_type: {self.readout_type}")
        if self.pool_readout not in ('mean', 'mean_max'):
            raise ValueError(f"Unknown pool_readout: {pool_readout}")
        if self.readout_type == 'attn':
            self.attn_pool_layer = AttentionalAggregation(gate_nn=nn.Linear(nhid, 1))

        self.n_layer_basicgnn = n_layer_basicgnn
        self.use_fingerprints = use_fingerprints
        self.use_rwpe = use_rwpe

        # Pooling layers
        self.num_pool_layers = min(max(int(num_pool_layers), 0), n_layer_basicgnn)
        self.topk_pools = nn.ModuleList()
        self.edge_pools = nn.ModuleList()
        self.pool_readout_proj = None
        self.gmt_pool = None

        if self.pooling_method == 'topk':
            self.topk_pools = nn.ModuleList([
                TopKPooling(nhid, ratio=pool_ratio)
                for _ in range(self.num_pool_layers)
            ])
            if self.pool_readout == 'mean_max':
                self.pool_readout_proj = nn.Linear(2 * nhid, nhid)
        elif self.pooling_method == 'edgepool':
            invalid = [g for g in self.gnn_types if not gnn_type_supports_edgepool(g)]
            if invalid:
                raise ValueError(
                    "pooling_method='edgepool' only supports node-only backbones "
                    f"GCN / GIN / SAGE, but got: {sorted(set(invalid))}"
                )
            score_method = get_edge_score_method(edge_score_method)
            self.edge_pools = nn.ModuleList([
                EdgePooling(nhid, edge_score_method=score_method, dropout=edgepool_dropout)
                for _ in range(self.num_pool_layers)
            ])
            if self.pool_readout == 'mean_max':
                self.pool_readout_proj = nn.Linear(2 * nhid, nhid)
        elif self.pooling_method == 'gmt':
            self.gmt_pool = GraphMultisetTransformer(
                channels=nhid,
                k=gmt_k,
                num_encoder_blocks=gmt_blocks,
                heads=gmt_heads,
                layer_norm=gmt_layer_norm,
                dropout=dropout,
            )
        
        self.graph_head = MLP(nhid, nhid, nlayer=2)
        
        if self.use_fingerprints:
            final_in_dim = nhid + fingerprint_dim
        else:
            final_in_dim = nhid
        
        if self.use_rwpe:
            self.rwpe_proj = nn.Linear(nhid + rwpe_dim, nhid)
        
        self.output_decoder = MLP(final_in_dim, nout, nlayer=2)

        self.reset_parameters()

    def reset_parameters(self):
        if self.dataset_type in ('ogb', 'lrgb', 'pyg_mol', 'good_mol'):
            for name, param in self.input_encoder.named_parameters():
                if 'weight' in name: nn.init.xavier_uniform_(param)
        elif self.dataset_type in ('tu', 'counting'):
            self.input_encoder.reset_parameters()

        if isinstance(self.edge_encoder, nn.Embedding):
             nn.init.xavier_uniform_(self.edge_encoder.weight)
        else:
             for name, param in self.edge_encoder.named_parameters():
                if 'weight' in name: nn.init.xavier_uniform_(param)

        for gnn_layer in self.gnn_layers:
            if hasattr(gnn_layer, 'reset_parameters'):
                gnn_layer.reset_parameters()

        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
        for act in self.activations:
            if hasattr(act, 'reset_parameters'):
                act.reset_parameters()
        if self.use_ffn:
            for norm in self.ffn_norms:
                if hasattr(norm, 'reset_parameters'):
                    norm.reset_parameters()
            for ffn in self.ffn_mlps:
                ffn.reset_parameters()

        if hasattr(self.output_decoder, 'reset_parameters'):
            self.output_decoder.reset_parameters()
        
        if hasattr(self.graph_head, 'reset_parameters'):
            self.graph_head.reset_parameters()

        for pool in self.topk_pools:
            if hasattr(pool, 'reset_parameters'):
                pool.reset_parameters()
        for pool in self.edge_pools:
            if hasattr(pool, 'reset_parameters'):
                pool.reset_parameters()
        if self.pool_readout_proj is not None:
            self.pool_readout_proj.reset_parameters()
        if self.gmt_pool is not None and hasattr(self.gmt_pool, 'reset_parameters'):
            self.gmt_pool.reset_parameters()
        
        if self.use_virtual_node:
            nn.init.constant_(self.virtual_node_embedding.weight.data, 0)
            for mlp_container in self.virtual_node_mlps:
                for layer in mlp_container:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
        
        if self.use_rwpe and hasattr(self, 'rwpe_proj'):
            self.rwpe_proj.reset_parameters()

    def _global_readout(self, x, batch, readout_type=None):
        readout_type = readout_type or self.readout_type
        if readout_type == 'attn':
            return self.attn_pool_layer(x, batch)
        reduce = 'sum' if readout_type == 'add' else readout_type
        return scatter(x, batch, dim=0, reduce=reduce)

    def _pool_readout(self, x, batch):
        if self.pool_readout == 'mean':
            return global_mean_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_max = scatter(x, batch, dim=0, reduce='max')
        return self.pool_readout_proj(torch.cat([x_mean, x_max], dim=-1))

    def _gmt_readout(self, x, batch):
        try:
            return self.gmt_pool(x, index=batch)
        except TypeError as index_error:
            try:
                return self.gmt_pool(x, batch=batch)
            except TypeError:
                raise index_error

    def _conv_block(self, i, x, edge_index, edge_attr, batch, virtual_node_feat=None):
        if self.use_virtual_node:
            x = x + virtual_node_feat[batch]
        x_in = x

        gnn_layer = self.gnn_layers[i]
        gnn_type = self.gnn_types[i]

        if gnn_type_needs_edge_features(gnn_type):
            x = gnn_layer(x, edge_index, edge_attr)
        else:
            x = gnn_layer(x, edge_index)

        x = self.norms[i](x)
        x = self.activations[i](x)
        x = F.dropout(x, self.dropout, training=self.training)

        if self.res:
            x = x + x_in

        if self.use_ffn:
            x = x + self.ffn_mlps[i](self.ffn_norms[i](x))

        if self.use_virtual_node and i < self.n_layer_basicgnn - 1:
            vn_agg = global_add_pool(x, batch)
            vn_temp = self.virtual_node_mlps[i](vn_agg)
            vn_temp = F.dropout(vn_temp, self.dropout, training=self.training)
            virtual_node_feat = virtual_node_feat + vn_temp

        return x, virtual_node_feat
    
    def forward(self, x, edge_index, batch, edge_attr=None, fingerprint=None, rwpe=None, input_embedding=None, **kwargs):
        if input_embedding is not None:
            x = input_embedding
        else:
            assert len(x.shape) == 2
            if self.dataset_type in ('ogb', 'lrgb', 'pyg_mol', 'good_mol'):
                if x.dtype == torch.float: x = x.long()
                x = self.input_encoder(x)
            else: # tu | counting
                if x.dtype == torch.long: x = x.float()
                x = self.input_encoder(x)

        if self.use_rwpe:
            if rwpe is None:
                raise ValueError("RWPE expected but is None")
            x = self.rwpe_proj(torch.cat([x, rwpe.float()], dim=-1))

        if self.dataset_type in ('ogb', 'lrgb', 'pyg_mol', 'good_mol'):
            edge_attr = edge_attr if edge_attr is not None else edge_index.new_zeros((edge_index.size(1), 3))
            if edge_attr.dtype == torch.float:
                edge_attr = edge_attr.long()
            edge_attr = self.edge_encoder(edge_attr)
        else: # tu | counting
            edge_attr = edge_attr if edge_attr is not None else edge_index.new_zeros((edge_index.size(1), 1))
            if edge_attr.dtype == torch.long:
                edge_attr = edge_attr.float()
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1) # [E] -> [E, 1]
            edge_attr = self.edge_encoder(edge_attr) # [E, num_edge_classes] -> [E, nhid]

        virtual_node_feat = None
        if self.use_virtual_node:
            num_graphs = batch.max().item() + 1
            vn_emb = self.virtual_node_embedding.weight 
            virtual_node_feat = vn_emb.expand(num_graphs, -1).clone() # [batch_size, hid]

        pooled_outputs = []
        last_pool_layer = -1
        for i in range(self.n_layer_basicgnn):
            x, virtual_node_feat = self._conv_block(i, x, edge_index, edge_attr, batch, virtual_node_feat)

            if self.pooling_method == 'topk' and i < self.num_pool_layers:
                x, edge_index, edge_attr, batch, _, _ = self.topk_pools[i](
                    x, edge_index, edge_attr=edge_attr, batch=batch
                )
                pooled_outputs.append(self._pool_readout(x, batch))
                last_pool_layer = i
            elif self.pooling_method == 'edgepool' and i < self.num_pool_layers:
                x, edge_index, batch, _ = self.edge_pools[i](x, edge_index, batch=batch)
                edge_attr = None
                pooled_outputs.append(self._pool_readout(x, batch))
                last_pool_layer = i

        if self.pooling_method in ('topk', 'edgepool'):
            if not pooled_outputs or last_pool_layer != self.n_layer_basicgnn - 1:
                pooled_outputs.append(self._pool_readout(x, batch))
            x = torch.stack(pooled_outputs, dim=0).sum(dim=0)
        elif self.pooling_method == 'gmt':
            x = self._gmt_readout(x, batch)
        else:
            x = self._global_readout(x, batch)

        x = F.dropout(x, self.dropout, training=self.training)

        graph_emb = self.graph_head(x)

        if self.use_fingerprints:
            if fingerprint is None:
                raise ValueError("Model expects fingerprints but got None!")
            x = torch.cat([x, fingerprint], dim=1)

        x = self.output_decoder(x)

        return x, graph_emb


def get_gnn_conv(gnn_type, in_dim, out_dim, deg, **kwargs):
    """
    - gnn_type: str, choice = 'GCN', 'GIN', 'GINE', 'SAGE', 'PNA', 'GCNplus', 'SAGEplus'
    - in_dim:	int
    - out_dim:	int
    - kwargs:	options
    	- GAT:              n_heads_gat(4), concat_gat(True), dropout_gat(0.5)
        - GatedGraph:       gated_n_layers(1)
        - SAGE:             aggr_sage(mean)
        - SimplifiedPNA:    aggrs_pna(['mean', 'min', 'max', 'std']), scalers(None)
        - Transformer:      n_heads_trans(4)
    """
    if gnn_type in ('GAT', 'GATv2'):
        n_heads = kwargs.get('n_heads_gat', 4)
        concat = kwargs.get('concat_gat', True)
        dropout = kwargs.get('dropout_gat', 0.5)
        # assert out_dim % n_heads == 0
        return GATv2Conv(in_channels=in_dim, 
                         out_channels=out_dim // n_heads if concat else out_dim,
                         heads=n_heads,
                         concat=concat,
                         dropout=dropout,
                         edge_dim=in_dim)

    elif gnn_type == 'GCN':
        return GCNConv(in_channels=in_dim, out_channels=out_dim)

    elif gnn_type == 'GCNplus':
        return GCNplusConv(in_channels=in_dim, out_channels=out_dim, edge_dim=in_dim)
    
    elif gnn_type == 'GIN':
        nn_mlp = MLP(in_dim, out_dim, nlayer=2)
        return GINConv(nn=nn_mlp, train_eps=kwargs.get('train_eps_gin', False))

    elif gnn_type == 'GatedGraph':
        return GatedGraphConv(out_channels=out_dim, num_layers=kwargs.get('gated_n_layers', 1))

    elif gnn_type in ('SAGE'):
        aggr = kwargs.get('aggr_sage', 'mean') # 'mean', 'sum', 'max', 'lstm'
        return SAGEConv(in_channels=in_dim, out_channels=out_dim, aggr=aggr)

    elif gnn_type == 'SAGEplus':
        aggr = kwargs.get('aggr_sage', 'mean') # 'mean', 'sum', 'max'
        return SAGEplusConv(in_channels=in_dim, out_channels=out_dim, edge_dim=in_dim, aggr=aggr)

    elif gnn_type == 'SimplifiedPNA':
        aggregators = kwargs.get('aggrs_pna', ['mean', 'min', 'max', 'std'])
        scalers = kwargs.get('scalers', None) # 缩放器
        return SimplifiedPNAConv(in_dim, out_dim, aggregators=aggregators, scalers=scalers, deg=deg)
    
    elif gnn_type == 'PNA':
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        return PNAConv(in_channels=in_dim, 
                       out_channels=out_dim, 
                       aggregators=aggregators, 
                       scalers=scalers, 
                       deg=deg, 
                       edge_dim=in_dim,
                       towers=kwargs.get('towers_pna', 1),
                       pre_layers=kwargs.get('pre_layers_pna', 1),
                       post_layers=kwargs.get('post_layers_pna', 1),
                       divide_input=kwargs.get('divide_input_pna', False))

    elif gnn_type == 'GINE':
        return GINEConv(MLP(in_dim, out_dim, nlayer=2, use_norm=False),
                        train_eps=kwargs.get('train_eps_gin', False))

    elif gnn_type == 'Transformer':
        n_heads = kwargs.get('n_heads_trans', 4)
        return TransformerConv(in_channels=in_dim, out_channels=out_dim // n_heads, heads=n_heads, edge_dim=in_dim)

    elif gnn_type == 'NN':
        nn_mlp = MLP(in_dim, in_dim * out_dim, nlayer=2)
        return NNConv(in_channels=in_dim, out_channels=out_dim, nn=nn_mlp)

    elif gnn_type == 'ResGatedGraph':
        return ResGatedGraphConv(in_channels=in_dim, out_channels=out_dim)

    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")


class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=1, bias=True, with_final_activation=False, use_norm=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.use_norm = use_norm

        for i in range(nlayer):
            in_d = nin if i == 0 else (nin if i < nlayer else nin) # hid_dim = in_dim
            out_d = nout if i == nlayer - 1 else nin
            layer_components = [nn.Linear(in_d, out_d, bias=bias)]
            if use_norm:
                layer_components.append(nn.BatchNorm1d(out_d))
            self.layers.append(nn.Sequential(*layer_components))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.nlayer - 1 or self.with_final_activation:
                x = F.relu(x)
        return x
    
    def reset_parameters(self):
        for layer in self.layers:
            for sub_layer in layer:
                if isinstance(sub_layer, nn.Linear):
                    sub_layer.reset_parameters()
                elif isinstance(sub_layer, nn.BatchNorm1d):
                    sub_layer.reset_parameters()


class GCNplusConv(gnn.MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=None, bias=True):
        super().__init__(aggr='add', node_dim=0)
        edge_dim = edge_dim or in_channels
        self.lin_node = Linear(in_channels, out_channels, bias=False)
        self.lin_edge = Linear(edge_dim, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_node.reset_parameters()
        self.lin_edge.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        num_nodes = x.size(0)
        x = self.lin_node(x)

        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.size(1), self.lin_edge.in_features))
        edge_index, edge_attr = add_remaining_self_loops(
            edge_index,
            edge_attr,
            fill_value=0.,
            num_nodes=num_nodes
        )
        edge_index, edge_weight = gcn_norm(
            edge_index,
            edge_weight=None,
            num_nodes=num_nodes,
            add_self_loops=False,
            dtype=x.dtype
        )

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, edge_attr, edge_weight):
        return edge_weight.view(-1, 1) * x_j + self.lin_edge(edge_attr)


class SAGEplusConv(gnn.MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=None, aggr='mean', bias=True):
        super().__init__(aggr=aggr, node_dim=0)
        edge_dim = edge_dim or in_channels
        self.lin_neigh = Linear(in_channels, out_channels, bias=False)
        self.lin_edge = Linear(edge_dim, out_channels, bias=False)
        self.lin_root = Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_neigh.reset_parameters()
        self.lin_edge.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.size(1), self.lin_edge.in_features))
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out + self.lin_root(x)

    def message(self, x_j, edge_attr):
        return self.lin_neigh(x_j) + self.lin_edge(edge_attr)


class SimplifiedPNAConv(gnn.MessagePassing):
    def __init__(self, nin, nout, aggregators=['mean', 'min', 'max', 'std'], scalers=None, deg=None, bias=True, **kwargs):
        kwargs.setdefault('aggr', None)
        super().__init__(node_dim=0, **kwargs)

        self.aggregators = aggregators
        self.deg = deg
        self.pre_nn = MLP(3*nin, nin, nlayer=2)
        self.post_nn = MLP((len(aggregators)+2)*nin, nout, nlayer=2, bias=bias)
        self.deg_embedder = nn.Embedding(deg.size(0), nin)

    def reset_parameters(self):
        self.pre_nn.reset_parameters()
        self.post_nn.reset_parameters()
        self.deg_embedder.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = torch.cat([x, out], dim=-1)
        out = self.post_nn(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        if edge_attr is not None:
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)
        return self.pre_nn(h)

    def aggregate(self, inputs, index, dim_size=None):
        outs = []
        degree_values = degree(index, dim_size, dtype=torch.long)
        
        degree_values = torch.clamp(degree_values, max=self.deg.size(0)-1)

        outs.append(self.deg_embedder(degree_values))

        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None, dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(F.relu_(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            outs.append(out)

        return torch.cat(outs, dim=-1)
