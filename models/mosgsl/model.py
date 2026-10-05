import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, GINConv, SAGEConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn import inits
from torch_geometric.utils import coalesce, dense_to_sparse, scatter, softmax, subgraph

from ..good_encoders import GOODCategoricalEncoder


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _activation(name):
    name = str(name or 'relu').lower()
    if name == 'relu':
        return F.relu
    if name == 'gelu':
        return F.gelu
    if name in ('silu', 'swish'):
        return F.silu
    if name == 'elu':
        return F.elu
    if name == 'tanh':
        return torch.tanh
    raise ValueError(f"Unsupported MOSGSL activation: {name}")


def _norm_layer(norm, hidden):
    if norm is None:
        return None
    norm = str(norm)
    if norm in ('', 'none', 'None'):
        return None
    if norm in ('BatchNorm', 'batch_norm', 'batch'):
        return SafeBatchNorm1d(hidden)
    if norm in ('LayerNorm', 'layer_norm', 'layer'):
        return nn.LayerNorm(hidden)
    raise ValueError(f"Unsupported MOSGSL norm: {norm}")


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input):
        if self.training and input.dim() == 2 and input.size(0) <= 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(input)


def _pool(x, batch, mode):
    mode = str(mode or 'mean').lower()
    if mode in ('mean', 'avg'):
        return global_mean_pool(x, batch)
    if mode in ('sum', 'add'):
        return global_add_pool(x, batch)
    if mode == 'max':
        return global_max_pool(x, batch)
    raise ValueError(f"Unsupported MOSGSL pooling: {mode}")


def _batch_sum(values, batch, size=None):
    if size is None:
        size = int(batch.max().item()) + 1 if batch.numel() else 0
    out = values.new_zeros(size)
    if batch.numel():
        out.index_add_(0, batch, values)
    return out


def _ptr_from_batch(batch, size=None):
    if size is None:
        size = int(batch.max().item()) + 1 if batch.numel() else 0
    counts = torch.bincount(batch, minlength=size)
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    return ptr


def _topk_per_graph(score, ratio, batch, min_score=None):
    selected = []
    n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    for graph_id in range(n_graphs):
        idx = torch.where(batch == graph_id)[0]
        if idx.numel() == 0:
            continue
        if min_score is not None:
            keep = idx[score[idx] > min_score]
            if keep.numel() == 0:
                keep = idx[score[idx].argmax().view(1)]
        else:
            if ratio >= 1:
                k = min(int(ratio), idx.numel())
            else:
                k = max(1, int(math.ceil(float(ratio) * idx.numel())))
            keep = idx[score[idx].topk(k).indices]
        selected.append(keep)
    if not selected:
        return batch.new_empty(0)
    return torch.cat(selected, dim=0)


class MLPEncoder(nn.Module):
    def __init__(self, n_feat, n_hidden, n_class, n_layers, dropout=0.5,
                 use_bn=True, activation='relu', **kwargs):
        super().__init__()
        self.n_feat = n_feat
        self.dropout = dropout
        self.use_bn = use_bn
        self.act = _activation(activation)
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if n_layers <= 1:
            self.lins.append(nn.Linear(n_feat, n_class))
        else:
            self.lins.append(nn.Linear(n_feat, n_hidden))
            self.bns.append(SafeBatchNorm1d(n_hidden))
            for _ in range(n_layers - 2):
                self.lins.append(nn.Linear(n_hidden, n_hidden))
                self.bns.append(SafeBatchNorm1d(n_hidden))
            self.lins.append(nn.Linear(n_hidden, n_class))

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, **kwargs):
        if isinstance(x, Batch):
            x = x.x
        for i, lin in enumerate(self.lins[:-1]):
            x = self.act(lin(x))
            if self.use_bn:
                x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lins[-1](x)


class GINConvPlus(GINConv):
    def forward(self, x, edge_index, size=None, edge_weight=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)
        # Some PyG releases retain GINConv's generated ``propagate`` signature
        # in subclasses, so passing our additional edge_weight keyword fails
        # before ``message`` is reached.  Explicit sum aggregation is exactly
        # GIN's source-to-target propagation and is stable across PyG versions.
        source, target = edge_index[0], edge_index[1]
        messages = x[0].index_select(0, source)
        if edge_weight is not None:
            messages = messages * edge_weight.view(-1, 1).to(messages)
        num_target = (
            x[1].size(0) if x[1] is not None
            else (size[1] if size is not None else x[0].size(0))
        )
        out = scatter(
            messages, target, dim=0, dim_size=num_target, reduce='sum'
        )
        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r
        return self.nn(out)

    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class SAGEConvPlus(SAGEConv):
    def forward(self, x, edge_index, size=None, edge_weight=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)
        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])
        # Match SAGEConv's default mean aggregation while avoiding the same
        # generated-propagate incompatibility described for GINConvPlus.
        source, target = edge_index[0], edge_index[1]
        messages = x[0].index_select(0, source)
        if edge_weight is not None:
            messages = messages * edge_weight.view(-1, 1).to(messages)
        num_target = (
            x[1].size(0) if x[1] is not None
            else (size[1] if size is not None else x[0].size(0))
        )
        out = scatter(
            messages, target, dim=0, dim_size=num_target, reduce='mean'
        )
        out = self.lin_l(out)
        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)
        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        return out

    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class GNNEncoder(nn.Module):
    def __init__(self, n_feat, n_class, n_hidden, n_layers=2, dropout=0.5,
                 norm=None, act='relu', output_layer=False, n_layers_output=1,
                 pool='mean', backbone='gcn', residual=False,
                 categorical_feature_dims=None, **kwargs):
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.pool = pool
        self.backbone = str(backbone).lower()
        self.output_layer = output_layer
        self.residual = residual
        self.act = _activation(act)
        self.input_encoder = None
        effective_n_feat = n_feat
        if categorical_feature_dims:
            self.input_encoder = GOODCategoricalEncoder(
                categorical_feature_dims,
                n_hidden,
                feature_name='MOSGSL node',
            )
            effective_n_feat = n_hidden

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(n_layers):
            in_dim = effective_n_feat if layer == 0 else n_hidden
            out_dim = n_hidden if (layer < n_layers - 1 or output_layer) else n_class
            if self.backbone == 'gcn':
                conv = GCNConv(in_dim, out_dim)
            elif self.backbone == 'sage':
                # MOSGSL learns scalar edge weights.  The ordinary PyG
                # SAGEConv ignores them; use the weighted implementation that
                # was already provided in this module (and used upstream).
                conv = SAGEConvPlus(in_dim, out_dim)
            elif self.backbone == 'gin':
                mlp = MLPEncoder(in_dim, out_dim, out_dim, n_layers=2, dropout=0.0, use_bn=False)
                conv = GINConvPlus(mlp)
            else:
                raise ValueError(f"Unsupported MOSGSL GNN backbone: {backbone}")
            self.convs.append(conv)
            self.norms.append(_norm_layer(norm, out_dim))

        if output_layer:
            self.output_linear = MLPEncoder(
                n_feat=n_hidden,
                n_hidden=n_hidden,
                n_class=n_class,
                n_layers=n_layers_output,
                dropout=dropout,
            )
        else:
            self.output_linear = None

    def reset_parameters(self):
        if self.input_encoder is not None:
            self.input_encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if norm is not None and hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
        if self.output_linear is not None:
            self.output_linear.reset_parameters()

    def _edge_weight(self, edge_attr):
        if edge_attr is None:
            return None
        edge_weight = edge_attr.float()
        if edge_weight.dim() > 1:
            if edge_weight.size(-1) == 1:
                edge_weight = edge_weight.view(-1)
            else:
                edge_weight = edge_weight.mean(dim=-1)
        return edge_weight

    def forward(self, x=None, edge_index=None, edge_attr=None, batch=None,
                get_cls=True, return_before_pool=False):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        edge_weight = self._edge_weight(edge_attr)
        h = (
            self.input_encoder(x)
            if self.input_encoder is not None else x.float()
        )
        for i, conv in enumerate(self.convs):
            if self.backbone in {'gcn', 'sage', 'gin'}:
                h_next = conv(h, edge_index, edge_weight=edge_weight)
            else:
                h_next = conv(h, edge_index)
            if i < self.n_layers - 1 or self.output_layer:
                norm = self.norms[i]
                if norm is not None:
                    h_next = norm(h_next)
                h_next = self.act(h_next)
                h_next = F.dropout(h_next, p=self.dropout, training=self.training)
            if self.residual and h_next.shape == h.shape:
                h_next = h_next + h
            h = h_next

        pooled = _pool(h, batch, self.pool) if self.pool else h
        out = self.output_linear(pooled) if (self.output_layer and get_cls) else pooled
        if return_before_pool:
            return out, h
        return out


class WeightedCosine(nn.Module):
    def __init__(self, d_in, num_pers=16, weighted=True, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.w = nn.Parameter(torch.empty(num_pers, d_in)) if weighted else None
        self.reset_parameters()

    def reset_parameters(self):
        if self.w is not None:
            nn.init.xavier_uniform_(self.w)

    def forward(self, x, y=None, non_negative=False):
        y = x if y is None else y
        x_ctx = x.unsqueeze(0)
        y_ctx = y.unsqueeze(0)
        if self.w is not None:
            w = self.w.unsqueeze(1)
            x_ctx = x_ctx * w
            y_ctx = y_ctx * w
        if self.normalize:
            x_ctx = F.normalize(x_ctx, p=2, dim=-1)
            y_ctx = F.normalize(y_ctx, p=2, dim=-1)
        adj = torch.matmul(x_ctx, y_ctx.transpose(-1, -2)).mean(0)
        if non_negative:
            adj = adj.clamp_min(0)
        return adj


class EpsilonNN(nn.Module):
    def __init__(self, epsilon=0.5, set_value=None):
        super().__init__()
        self.epsilon = epsilon
        self.set_value = set_value

    def forward(self, adj):
        mask = adj > self.epsilon
        if self.set_value is None:
            return adj * mask.to(adj.dtype)
        return mask.to(adj.dtype) * self.set_value


class RemoveSelfLoop(nn.Module):
    def forward(self, adj):
        if adj.numel() == 0:
            return adj
        adj = adj.clone()
        adj.fill_diagonal_(0)
        return adj


class ProjectionEstimator(nn.Module):
    def __init__(self, n_feat, act='sigmoid', **kwargs):
        super().__init__()
        self.n_feat = n_feat
        self.act_name = str(act).lower()
        self.projection_vector = nn.Parameter(torch.empty(n_feat))
        self.reset_parameters()

    def reset_parameters(self):
        inits.uniform(self.n_feat, self.projection_vector)

    def forward(self, z, batch):
        score = z @ self.projection_vector
        if self.act_name == 'softmax':
            return softmax(score, batch)
        score = score / self.projection_vector.norm(p=2).clamp_min(1e-12)
        if self.act_name == 'sigmoid':
            return torch.sigmoid(score)
        return score


class SimpleParsingModule(nn.Module):
    def __init__(self, parsing='parse_bfs', parsing_params=None,
                 return_new_mapping=True, require_full_edge=False,
                 reparsing=False, **kwargs):
        super().__init__()
        self.parsing = parsing
        self.parsing_params = parsing_params or {}
        self.graphs2subgraphs = None
        self.return_new_mapping = return_new_mapping
        self.require_full_edge = require_full_edge
        self.reparsing = reparsing

    def _idx(self, graph):
        idx = graph.idx
        return int(idx.item()) if torch.is_tensor(idx) else int(idx)

    @torch.no_grad()
    def forward(self, batch, only_idx=False):
        if self.graphs2subgraphs is None:
            raise RuntimeError("MOSGSL parsing cache is not initialized.")
        device = batch.x.device
        graphs = batch.to_data_list()
        n_nodes_graph = scatter(batch.batch.new_ones(batch.batch.shape[0]), batch.batch)
        cum_nodes = torch.cat([n_nodes_graph.new_zeros(1), n_nodes_graph.cumsum(0)])

        if only_idx:
            nodes_list = []
            belong_list = []
            for graph_pos, graph in enumerate(graphs):
                subs = self.graphs2subgraphs[self._idx(graph)]
                belong_list.append(torch.full((len(subs),), graph_pos, dtype=torch.long, device=device))
                nodes_list.extend([sub.mapping.to(device) + cum_nodes[graph_pos] for sub in subs])
            belong = torch.cat(belong_list, dim=0) if belong_list else batch.batch.new_empty(0)
            return nodes_list, belong

        subgraphs = []
        belong_list = []
        for graph_pos, graph in enumerate(graphs):
            subs = self.graphs2subgraphs[self._idx(graph)]
            subgraphs.extend(subs)
            belong_list.append(torch.full((len(subs),), graph_pos, dtype=torch.long, device=device))
        belong = torch.cat(belong_list, dim=0) if belong_list else batch.batch.new_empty(0)
        batch_subs = Batch.from_data_list(subgraphs).to(device)
        if self.return_new_mapping:
            new_mapping = batch_subs.mapping + cum_nodes[belong[batch_subs.batch]]
        else:
            new_mapping = None
        return batch_subs, belong, new_mapping


class SubgraphSelectModule(nn.Module):
    def __init__(self, n_feat, topk=0.8, min_score=None, conf_encoder=None,
                 conf_estimator=None, select_flag=True, **kwargs):
        super().__init__()
        if topk is None and min_score is None:
            raise ValueError("MOSGSL selector needs either topk or min_score.")
        self.n_feat = n_feat
        self.topk = topk
        self.min_score = min_score
        self.select_flag = select_flag
        self.conf_encoder = conf_encoder or {}
        self.conf_estimator = conf_estimator or {}
        if select_flag:
            encoder_type = self.conf_encoder.get('type', 'gnn')
            if encoder_type == 'mlp':
                self.encoder = MLPEncoder(n_feat, **self.conf_encoder)
            elif encoder_type == 'gnn':
                self.encoder = GNNEncoder(n_feat, **self.conf_encoder)
            else:
                raise ValueError(f"Unsupported selector encoder: {encoder_type}")
            if self.conf_estimator.get('type', 'projection') == 'mlp':
                self.estimator = MLPEncoder(
                    n_feat=self.conf_encoder['n_class'],
                    n_class=1,
                    **self.conf_estimator,
                )
                self.estimator_needs_batch = False
            else:
                self.estimator = ProjectionEstimator(
                    n_feat=self.conf_encoder['n_class'],
                    **self.conf_estimator,
                )
                self.estimator_needs_batch = True
        else:
            self.encoder = None
            self.estimator = None
            self.estimator_needs_batch = False

    def reset_parameters(self):
        if self.encoder is not None:
            self.encoder.reset_parameters()
        if self.estimator is not None:
            self.estimator.reset_parameters()

    def forward(self, batch_sub, belong, new_mapping):
        if self.select_flag:
            z_sub, z_node = self.encoder(
                x=batch_sub.x,
                edge_index=batch_sub.edge_index,
                batch=batch_sub.batch,
                return_before_pool=True,
            )
            if self.estimator_needs_batch:
                score = self.estimator(z_sub, belong).view(-1)
            else:
                score = self.estimator(z_sub).view(-1)
            sub_index = _topk_per_graph(score, self.topk, belong, self.min_score).sort()[0]
            sub_mask = sub_index.new_full((batch_sub.num_graphs,), -1)
            sub_mask[sub_index] = torch.arange(sub_index.numel(), device=sub_index.device)
            batch_after_select = sub_mask[batch_sub.batch]
            node_index = torch.where(batch_after_select >= 0)[0]
            new_edge_index, _ = subgraph(
                node_index,
                batch_sub.edge_index,
                relabel_nodes=True,
                num_nodes=batch_sub.num_nodes,
            )
            new_x = batch_sub.x[node_index]
            new_batch = batch_after_select[node_index]
            new_belong = belong[sub_index]
            new_mapping = new_mapping[node_index] if new_mapping is not None else batch_sub.mapping[node_index]
            if hasattr(batch_sub, 'full_edge_index'):
                new_full_edge_index, _ = subgraph(
                    node_index,
                    batch_sub.full_edge_index,
                    relabel_nodes=True,
                    num_nodes=batch_sub.num_nodes,
                )
            else:
                new_full_edge_index = None
            return (
                new_x, z_node[node_index], z_sub[sub_index], new_edge_index,
                new_batch, new_belong, new_mapping, score[sub_index],
                new_full_edge_index,
            )

        score = torch.ones(belong.shape, device=belong.device, dtype=batch_sub.x.dtype)
        full_edge_index = batch_sub.full_edge_index if hasattr(batch_sub, 'full_edge_index') else None
        return (
            batch_sub.x, None, None, batch_sub.edge_index, batch_sub.batch,
            belong, new_mapping, score, full_edge_index,
        )


class BasicSubGraphLearner(nn.Module):
    def __init__(self, n_feat, conf_encoder, n_hidden=None,
                 metric_type='weighted_cosine', conf_metric=None, epsilon=0.5,
                 lamb1=0.5, gsl_one_by_one=False, **kwargs):
        super().__init__()
        self.n_feat = n_feat
        self.conf_encoder = conf_encoder or {}
        self.encoder_type = self.conf_encoder.get('type')
        if self.encoder_type == 'gnn':
            self.encoder = GNNEncoder(n_feat, **self.conf_encoder)
        elif self.encoder_type == 'mlp':
            self.encoder = MLPEncoder(n_feat, **self.conf_encoder)
        else:
            self.encoder = None
        self.n_hidden = self.conf_encoder.get('n_hidden', n_hidden or n_feat)
        if metric_type != 'weighted_cosine':
            raise ValueError(f"Unsupported MOSGSL metric: {metric_type}")
        self.metric = WeightedCosine(d_in=self.n_hidden, **(conf_metric or {}))
        self.postprocess = nn.ModuleList([EpsilonNN(epsilon), RemoveSelfLoop()])
        self.lamb1 = float(lamb1)
        self.gsl_one_by_one = bool(gsl_one_by_one)

    def reset_parameters(self):
        if self.encoder is not None:
            self.encoder.reset_parameters()
        self.metric.reset_parameters()

    def _encode(self, x, edge_index, batch):
        if self.encoder is None:
            return x
        z = self.encoder(x, edge_index=edge_index, batch=batch, return_before_pool=True)
        return z[1] if isinstance(z, tuple) else z

    def _postprocess(self, adj):
        for layer in self.postprocess:
            adj = layer(adj)
        return adj

    def forward(self, x, selected_edge_index, selected_batch, selected_mapping,
                selected_belong, selected_score, full_edge_index=None,
                return_sub=False, raw_edge_index=None, raw_num_nodes=None,
                **kwargs):
        raw_edge_attr = x.new_ones(raw_edge_index.size(1))
        z_node = self._encode(x, selected_edge_index, selected_batch)
        score_sum = _batch_sum(
            selected_score,
            selected_belong,
            size=int(selected_belong.max().item()) + 1 if selected_belong.numel() else 0,
        )[selected_belong].clamp_min(1e-12)
        selected_score = selected_score / score_sum

        if self.gsl_one_by_one:
            edge_index_out = selected_edge_index.new_empty((2, 0))
            edge_index_out_subs = selected_edge_index.new_empty((2, 0))
            edge_attr_out = x.new_empty(0)
            edge_attr_out_subs = x.new_empty(0)
            ptr = _ptr_from_batch(selected_batch)
            for sub_id in range(ptr.numel() - 1):
                start, end = int(ptr[sub_id]), int(ptr[sub_id + 1])
                adj_sub = self._postprocess(self.metric(z_node[start:end]))
                edge_index, edge_attr = dense_to_sparse(adj_sub)
                edge_index_out_subs = torch.cat([edge_index_out_subs, edge_index + start], dim=1)
                edge_attr_out_subs = torch.cat([edge_attr_out_subs, edge_attr], dim=0)
                weighted_attr = edge_attr * selected_score[sub_id] * self.lamb1
                edge_index = selected_mapping[edge_index + start]
                edge_index_out = torch.cat([edge_index_out, edge_index], dim=1)
                edge_attr_out = torch.cat([edge_attr_out, weighted_attr], dim=0)
        else:
            if full_edge_index is None:
                raise RuntimeError("full_edge_index is required when gsl_one_by_one=False.")
            adj_all = self.metric(z_node)
            mask = adj_all.new_zeros(adj_all.size())
            mask[full_edge_index[0], full_edge_index[1]] = 1
            adj_all = self._postprocess(adj_all * mask)
            edge_index_out_subs, edge_attr_out_subs = dense_to_sparse(adj_all)
            weights = selected_score[selected_batch[edge_index_out_subs[0]]]
            edge_attr_out = edge_attr_out_subs * weights * self.lamb1
            edge_index_out = selected_mapping[edge_index_out_subs]

        edge_index_out = torch.cat([edge_index_out, raw_edge_index], dim=1)
        edge_attr_out = torch.cat([edge_attr_out, raw_edge_attr * (1.0 - self.lamb1)], dim=0)
        edge_index_out, edge_attr_out = coalesce(edge_index_out, edge_attr_out, num_nodes=raw_num_nodes)
        if return_sub:
            return edge_index_out_subs, edge_attr_out_subs, edge_index_out, edge_attr_out
        return edge_index_out, edge_attr_out


class MotifVector(nn.Module):
    def __init__(self, n_hidden, n_motif_per_class, n_class, sim_type='cosine',
                 temperature=0.2, update_type='kmeans', k2=1, tau=0.99,
                 weighted_mean=False, device=None, **kwargs):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_motif_per_class = int(n_motif_per_class)
        self.n_class = int(n_class)
        self.n_motif = self.n_motif_per_class * self.n_class
        self.sim_type = sim_type
        self.temperature = temperature
        self.update_type = update_type
        self.k2 = k2
        self.tau = tau
        self.weighted_mean = weighted_mean
        self.register_buffer('mapping', torch.zeros(self.n_motif, self.n_class))
        for idx in range(self.n_motif):
            self.mapping[idx, idx // self.n_motif_per_class] = 1
        self.Motif_Vector = nn.Parameter(torch.randn(self.n_motif, n_hidden), requires_grad=False)
        if device is not None:
            self.to(device)

    def reset_parameters(self):
        nn.init.normal_(self.Motif_Vector)

    def _similarity(self, z):
        if self.sim_type == 'cosine':
            sim = F.normalize(z, p=2, dim=-1) @ F.normalize(self.Motif_Vector, p=2, dim=-1).T
            return sim, -sim
        distance = torch.sum((z.unsqueeze(1) - self.Motif_Vector) ** 2, dim=2)
        if self.sim_type == 'euclidean_1':
            sim = torch.log((distance + 1) / (distance + 1e-4))
        elif self.sim_type == 'euclidean_2':
            sim = 1.0 / (1.0 + distance)
        else:
            sim = -distance
        return sim, distance

    def loss_contrastive(self, z, y):
        if z.numel() == 0:
            return z.new_tensor(0.0)
        y = y.view(-1).long()
        valid = (y >= 0) & (y < self.n_class)
        if valid.sum() == 0:
            return z.new_tensor(0.0)
        z, y = z[valid], y[valid]
        similarity, _ = self._similarity(z)
        true_motifs = self.mapping.to(z.device)[:, y].bool().T
        similarities = torch.exp(similarity / self.temperature)
        sim_pos = torch.max(
            similarities[true_motifs].reshape(-1, self.n_motif_per_class), dim=1
        )[0]
        if self.n_class <= 1:
            return z.new_tensor(0.0)
        sim_neg = similarities[~true_motifs].reshape(-1, (self.n_class - 1) * self.n_motif_per_class)
        return -torch.log(sim_pos / (sim_neg.sum(1) + sim_pos).clamp_min(1e-12)).mean()

    @torch.no_grad()
    def init_motif(self, z, y):
        self._update_from_embeddings(z, y, init=True)

    @torch.no_grad()
    def update_motif(self, z, y):
        self._update_from_embeddings(z, y, init=False)

    @torch.no_grad()
    def _update_from_embeddings(self, z, y, init=False):
        if z.numel() == 0:
            return
        device = z.device
        y = y.view(-1).long()
        new_motifs = []
        z_cpu = z.detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy()
        old = self.Motif_Vector.detach().cpu().numpy()
        for cls in range(self.n_class):
            z_cls = z_cpu[y_cpu == cls]
            old_cls = old[
                self.n_motif_per_class * cls:self.n_motif_per_class * (cls + 1)
            ]
            if z_cls.shape[0] == 0:
                centroids = old_cls
            elif z_cls.shape[0] < self.n_motif_per_class:
                repeat = math.ceil(self.n_motif_per_class / z_cls.shape[0])
                centroids = z_cls.repeat(repeat, axis=0)[:self.n_motif_per_class]
            elif self.update_type == 'kmeans' or init:
                km = KMeans(
                    n_clusters=self.n_motif_per_class,
                    random_state=0,
                    init=old_cls if not init else 'k-means++',
                    n_init=1 if not init else 10,
                ).fit(z_cls)
                centroids = km.cluster_centers_
            else:
                centroids = z_cls[:self.n_motif_per_class]
            new_motifs.append(torch.tensor(centroids, dtype=z.dtype, device=device))
        new_motifs = torch.cat(new_motifs, dim=0)
        if not init:
            new_motifs = (1.0 - self.tau) * new_motifs + self.tau * self.Motif_Vector.data
        self.Motif_Vector.data = new_motifs.detach().clone()


class MOSGSL(nn.Module):
    def __init__(self, conf, n_feat, n_class, motif_n_class=None):
        super().__init__()
        self.conf = conf
        self.n_feat = n_feat
        self.n_class = n_class
        self.use_select = bool(conf.use_select) and conf.parsing_module['parsing'] != 'original'
        self.use_gsl = bool(conf.use_gsl)
        self.use_motif = bool(conf.use_motif) and motif_n_class is not None
        self.use_seperate_encoder = bool(conf.use_seperate_encoder) and self.use_motif

        self.parsing = SimpleParsingModule(
            **conf.parsing_module,
            return_new_mapping=self.use_gsl,
            require_full_edge=not conf.gsl_module['gsl_one_by_one'],
        )
        self.selecting = SubgraphSelectModule(
            n_feat=self.n_feat,
            select_flag=self.use_select,
            **conf.selecting_module,
        )
        self.gsl = BasicSubGraphLearner(
            n_feat=self.n_feat,
            n_hidden=self.selecting.conf_encoder.get('n_hidden', n_feat),
            **conf.gsl_module,
        )
        self.backbone = GNNEncoder(n_feat=self.n_feat, n_class=self.n_class, **conf.backbone)
        self.classifier = MLPEncoder(
            n_feat=self.selecting.conf_encoder['n_class'],
            n_class=n_class,
            **conf.classifier,
        )
        self.backbone2 = GNNEncoder(n_feat=self.n_feat, n_class=self.n_class, **conf.backbone)
        motif_hidden = self.backbone.output_linear.n_feat if self.backbone.output_linear is not None else conf.backbone['n_hidden']
        self.motif = MotifVector(
            n_hidden=motif_hidden,
            n_class=motif_n_class or n_class,
            **conf.motif,
        )

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def _labels_for_subgraphs(self, graph_y, selected_belong):
        y = graph_y[selected_belong]
        return y.view(-1).long()

    def forward_wo_gsl(self, batch, return_sub=False, only_backbone=False):
        if only_backbone:
            return self.backbone(batch.x, batch.edge_index, batch=batch.batch), 0
        batch_subs, belong, new_mapping = self.parsing(batch)
        (
            selected_x, _, _, selected_edge_index, selected_batch, selected_belong,
            _, selected_score, _,
        ) = self.selecting(batch_subs, belong, new_mapping)
        encoder = self.backbone2 if self.use_seperate_encoder else self.backbone
        z_sub = encoder(selected_x, edge_index=selected_edge_index, batch=selected_batch, get_cls=False)
        z_sub_agg = z_sub * selected_score.unsqueeze(1)
        z_global = global_add_pool(z_sub_agg, batch=selected_belong)
        output = encoder.output_linear(z_global)
        if return_sub:
            return output, 0, z_sub, selected_belong, selected_score
        return output, 0

    def forward(self, batch, return_sub=False):
        if not self.use_gsl:
            return self.forward_wo_gsl(batch, return_sub=return_sub)
        batch_subs, belong, new_mapping = self.parsing(batch)
        (
            selected_x, selected_z_node, _, selected_edge_index, selected_batch,
            selected_belong, selected_mapping, selected_score, full_edge_index,
        ) = self.selecting(batch_subs, belong, new_mapping)
        x_gsl = selected_x if self.gsl.encoder_type else selected_z_node
        if x_gsl is None:
            x_gsl = selected_x
        edge_index_sub, edge_attr_sub, edge_index_out, edge_attr_out = self.gsl(
            x_gsl,
            selected_edge_index=selected_edge_index,
            selected_batch=selected_batch,
            selected_mapping=selected_mapping,
            selected_belong=selected_belong,
            selected_score=selected_score,
            full_edge_index=full_edge_index,
            return_sub=True,
            raw_edge_index=batch.edge_index,
            raw_num_nodes=batch.num_nodes,
        )
        output = self.backbone(batch.x, edge_index=edge_index_out, edge_attr=edge_attr_out, batch=batch.batch)
        loss_con = batch.x.new_tensor(0.0)
        z_sub = None
        if self.use_motif:
            encoder = self.backbone2 if self.use_seperate_encoder else self.backbone
            z_sub = encoder(
                selected_x,
                edge_index=edge_index_sub,
                edge_attr=edge_attr_sub,
                batch=selected_batch,
                get_cls=False,
            )
            y_sub = self._labels_for_subgraphs(batch.y, selected_belong)
            loss_con = self.motif.loss_contrastive(z_sub, y_sub)
        if return_sub:
            if z_sub is None:
                z_sub = self.backbone(
                    selected_x,
                    edge_index=edge_index_sub,
                    edge_attr=edge_attr_sub,
                    batch=selected_batch,
                    get_cls=False,
                )
            return output, loss_con, z_sub, selected_belong, selected_score
        return output, loss_con
