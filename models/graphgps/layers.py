import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_scatter import scatter
from torch_geometric.nn import Linear as PygLinear
from torch_geometric.utils import to_dense_batch

try:
    from performer_pytorch import SelfAttention
except ImportError:  # Optional dependency.
    SelfAttention = None


def make_activation(act_type):
    act_type = act_type.lower()
    if act_type == 'relu':
        return nn.ReLU
    if act_type == 'gelu':
        return nn.GELU
    if act_type == 'prelu':
        return nn.PReLU
    if act_type in ('swish', 'silu'):
        return nn.SiLU
    raise ValueError(f"Unknown GraphGPS activation: {act_type}")


class GPSLayer(nn.Module):
    """Local MPNN + global attention layer adapted from GraphGPS.

    This keeps the GraphGPS residual structure while removing GraphGym cfg
    dependencies so it can be used inside the OpenGLT benchmark runners.
    """

    def __init__(self, dim_h, local_gnn_type='GINE', global_model_type='Transformer',
                 num_heads=4, act='relu', dropout=0.0, attn_dropout=0.0,
                 layer_norm=False, batch_norm=True, pna_degrees=None):
        super().__init__()

        if layer_norm and batch_norm:
            raise ValueError("GraphGPS cannot use layer_norm and batch_norm together.")

        self.dim_h = dim_h
        self.local_gnn_type = local_gnn_type
        self.global_model_type = global_model_type
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.norm1_local = None
        self.norm1_attn = None
        self.norm2 = None
        activation = make_activation(act)

        self.local_gnn_with_edge_attr = True
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GCN':
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GCNConv(dim_h, dim_h)
        elif local_gnn_type == 'GIN':
            self.local_gnn_with_edge_attr = False
            gin_nn = nn.Sequential(PygLinear(dim_h, dim_h), activation(), PygLinear(dim_h, dim_h))
            self.local_model = pygnn.GINConv(gin_nn)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(PygLinear(dim_h, dim_h), activation(), PygLinear(dim_h, dim_h))
            self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            if dim_h % num_heads != 0:
                raise ValueError(
                    f"dim_h={dim_h} must be divisible by num_heads={num_heads} for GAT."
                )
            self.local_model = pygnn.GATConv(
                in_channels=dim_h,
                out_channels=dim_h // num_heads,
                heads=num_heads,
                edge_dim=dim_h,
            )
        elif local_gnn_type == 'PNA':
            if pna_degrees is None:
                raise ValueError("pna_degrees is required when local_gnn_type='PNA'.")
            self.local_model = pygnn.PNAConv(
                dim_h,
                dim_h,
                aggregators=['mean', 'max', 'sum'],
                scalers=['identity'],
                deg=pna_degrees.cpu(),
                edge_dim=dim_h,
                towers=1,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(
                dim_h,
                dim_h,
                dropout=dropout,
                residual=True,
                act=act,
            )
        else:
            raise ValueError(f"Unsupported GraphGPS local_gnn_type: {local_gnn_type}")

        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'Transformer':
            self.self_attn = nn.MultiheadAttention(
                dim_h, num_heads, dropout=attn_dropout, batch_first=False
            )
        elif global_model_type == 'Performer':
            if SelfAttention is None:
                raise ImportError("global_model_type='Performer' requires performer-pytorch.")
            self.self_attn = SelfAttention(dim=dim_h, heads=num_heads, dropout=attn_dropout, causal=False)
        else:
            raise ValueError(f"Unsupported GraphGPS global_model_type: {global_model_type}")

        if self.local_model is None and self.self_attn is None:
            raise ValueError("At least one of local or global GraphGPS modules must be enabled.")

        if layer_norm:
            self.norm1_local = nn.LayerNorm(dim_h)
            self.norm1_attn = nn.LayerNorm(dim_h)
            self.norm2 = nn.LayerNorm(dim_h)
        elif batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
            self.norm2 = nn.BatchNorm1d(dim_h)

        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = activation()
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, edge_attr=None):
        h_in = x
        h_out_list = []
        next_edge_attr = edge_attr

        if self.local_model is not None:
            if self.local_gnn_type == 'CustomGatedGCN':
                h_local, next_edge_attr = self.local_model(x, edge_index, edge_attr)
            elif self.local_gnn_with_edge_attr:
                h_local = self.local_model(x, edge_index, edge_attr)
            else:
                h_local = self.local_model(x, edge_index)
            if self.local_gnn_type != 'CustomGatedGCN':
                h_local = h_in + self.dropout_local(h_local)
            h_local = self._norm(h_local, self.norm1_local)
            h_out_list.append(h_local)

        if self.self_attn is not None:
            h_dense, mask = to_dense_batch(x, batch)
            mask = mask.bool()
            if self.global_model_type == 'Transformer':
                key_padding_mask = torch.logical_not(mask)
                h_dense_t = h_dense.transpose(0, 1)
                h_attn_dense = self.self_attn(
                    h_dense_t, h_dense_t, h_dense_t,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )[0].transpose(0, 1)
                h_attn = h_attn_dense[mask]
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            else:
                raise RuntimeError(f"Unexpected global_model_type={self.global_model_type}")
            h_attn = h_in + self.dropout_attn(h_attn)
            h_attn = self._norm(h_attn, self.norm1_attn)
            h_out_list.append(h_attn)

        h = sum(h_out_list)
        h = h + self._ff_block(h)
        h = self._norm(h, self.norm2)
        return h, next_edge_attr

    def _norm(self, x, norm):
        if norm is not None:
            return norm(x)
        return x

    def _ff_block(self, x):
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))


class GatedGCNLayer(pygnn.conv.MessagePassing):
    """Residual GatedGCN layer used by the official GraphGPS molhiv recipe."""

    def __init__(self, in_dim, out_dim, dropout, residual=True, act='relu', **kwargs):
        super().__init__(**kwargs)
        activation = make_activation(act)
        self.A = pygnn.Linear(in_dim, out_dim, bias=True)
        self.B = pygnn.Linear(in_dim, out_dim, bias=True)
        self.C = pygnn.Linear(in_dim, out_dim, bias=True)
        self.D = pygnn.Linear(in_dim, out_dim, bias=True)
        self.E = pygnn.Linear(in_dim, out_dim, bias=True)

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.act_fn_x = activation()
        self.act_fn_e = activation()
        self.dropout = dropout
        self.residual = residual
        self._edge_out = None

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is None:
            raise ValueError("CustomGatedGCN requires encoded edge_attr.")

        x_in = x
        e_in = edge_attr

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(edge_attr)
        Dx = self.D(x)
        Ex = self.E(x)

        x, edge_attr = self.propagate(
            edge_index,
            Bx=Bx,
            Dx=Dx,
            Ex=Ex,
            Ce=Ce,
            Ax=Ax,
        )

        x = self.bn_node_x(x)
        edge_attr = self.bn_edge_e(edge_attr)

        x = self.act_fn_x(x)
        edge_attr = self.act_fn_e(edge_attr)

        x = F.dropout(x, self.dropout, training=self.training)
        edge_attr = F.dropout(edge_attr, self.dropout, training=self.training)

        if self.residual:
            x = x_in + x
            edge_attr = e_in + edge_attr

        return x, edge_attr

    def message(self, Dx_i, Ex_j, Ce):
        edge_out = Dx_i + Ex_j + Ce
        self._edge_out = edge_out
        return torch.sigmoid(edge_out)

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        dim_size = Bx.shape[0]
        numerator = scatter(sigma_ij * Bx_j, index, dim=0, dim_size=dim_size, reduce='sum')
        denominator = scatter(sigma_ij, index, dim=0, dim_size=dim_size, reduce='sum')
        return numerator / (denominator + 1e-6)

    def update(self, aggr_out, Ax):
        edge_out = self._edge_out
        self._edge_out = None
        return Ax + aggr_out, edge_out
