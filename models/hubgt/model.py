import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.utils import to_dense_adj, to_dense_batch

from models.baselines import MLP, TUNodeEncoder


INF8 = 255


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        return self.layer2(self.gelu(self.layer1(x)))


class StructuralEmbedding(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.linear_bias = nn.Embedding(INF8 + 1, num_heads, padding_idx=INF8)

    def forward(self, attn_bias):
        attn_bias = attn_bias.squeeze(3)
        mask_off = attn_bias == INF8
        bias = self.linear_bias(attn_bias.long())
        bias[mask_off] = -1e9
        return bias.permute(0, 3, 1, 2)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, dp_bias):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for HubGT.")
        self.num_heads = num_heads
        self.dp_bias = dp_bias
        self.att_size = hidden_size // num_heads
        self.scale = self.att_size ** -0.5
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.bias_enc = StructuralEmbedding(num_heads)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()
        batch_size = q.size(0)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, self.att_size)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, self.att_size)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, self.att_size)
        q = q.transpose(1, 2) * self.scale
        k = k.transpose(1, 2).transpose(2, 3)
        v = v.transpose(1, 2)

        x = torch.matmul(q, k)
        if attn_bias is not None:
            bias = self.bias_enc(attn_bias)
            if self.training and self.dp_bias > 0:
                drop = torch.rand_like(bias) < self.dp_bias
                eye = torch.eye(bias.size(2), dtype=torch.bool, device=bias.device)
                drop &= ~eye.unsqueeze(0).unsqueeze(0)
                bias = bias + torch.where(drop, bias.new_full((), -1e9), bias.new_zeros(()))
            x = x + bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.att_size)
        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, dp_bias):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads, dp_bias
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        x = x + self.self_attention_dropout(y)
        y = self.ffn_norm(x)
        y = self.ffn(y)
        return x + self.ffn_dropout(y)


def compute_spd_batch(edge_index, batch, max_nodes, device):
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    adj = to_dense_adj(edge_index, batch, max_num_nodes=max_nodes).cpu()
    spd = torch.full((num_graphs, max_nodes, max_nodes), INF8, dtype=torch.long)

    for graph_id in range(num_graphs):
        n = int((batch == graph_id).sum().item())
        if n == 0:
            continue
        graph_adj = adj[graph_id, :n, :n]
        dist = torch.full((n, n), INF8, dtype=torch.long)
        idx = torch.arange(n)
        dist[idx, idx] = 0
        dist[graph_adj > 0] = 1
        for k_node in range(n):
            dist = torch.minimum(dist, dist[:, k_node:k_node + 1] + dist[k_node:k_node + 1, :])
        spd[graph_id, :n, :n] = dist.clamp(max=INF8 - 1)

    return spd.unsqueeze(-1).to(device)


class HubGT(nn.Module):
    """Graph-level HubGT adapter.

    HubGT uses shortest-path distance as a structural attention bias in a dense
    graph Transformer. The original model is node-level; this adapter pools the
    valid node outputs into a graph representation.
    """

    def __init__(self, nhid, nout, n_layers=4, num_heads=4,
                 dataset_type='tu', dim_node_deg=0, dim_node_label=0,
                 dim_node_attr=0, dropout=0.3, ffn_dim=256,
                 dp_attn=0.1, dp_bias=0.0, pooling='mean'):
        super().__init__()
        self.dataset_type = dataset_type
        self.pooling = pooling
        self.dropout_val = dropout

        if dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            self.input_encoder = AtomEncoder(emb_dim=nhid)
        else:
            self.input_encoder = TUNodeEncoder(nhid, dim_node_deg, dim_node_label, dim_node_attr)

        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(nhid, ffn_dim, dropout, dp_attn, num_heads, dp_bias)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(nhid)
        self.graph_head = MLP(nhid, nhid, nlayer=2)
        self.output_decoder = MLP(nhid, nout, nlayer=2)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def reset_parameters(self):
        if hasattr(self.input_encoder, 'reset_parameters'):
            self.input_encoder.reset_parameters()
        for layer in self.layers:
            layer.apply(lambda module: init_params(module, n_layers=max(len(self.layers), 1)))
        self.final_ln.reset_parameters()
        self.graph_head.reset_parameters()
        self.output_decoder.reset_parameters()

    def _encode_input(self, x):
        if self.dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            if x.dtype == torch.float:
                x = x.long()
        elif x.dtype == torch.long:
            x = x.float()
        return self.input_encoder(x)

    def _pool(self, output, mask):
        mask_float = mask.unsqueeze(-1).float()
        if self.pooling in ('sum', 'add'):
            return (output * mask_float).sum(dim=1)
        if self.pooling == 'max':
            output = output.masked_fill(~mask.unsqueeze(-1), -1e9)
            return output.max(dim=1).values
        return (output * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)

    def forward(self, x, edge_index, batch, edge_attr=None,
                fingerprint=None, rwpe=None, input_embedding=None, **kwargs):
        h = input_embedding if input_embedding is not None else self._encode_input(x)
        h_dense, mask = to_dense_batch(h, batch)
        mask_float = mask.unsqueeze(-1).to(h_dense.dtype)
        h_dense = h_dense * mask_float
        spd = compute_spd_batch(edge_index, batch, h_dense.size(1), h_dense.device)

        output = self.input_dropout(h_dense)
        for enc_layer in self.layers:
            output = enc_layer(output, spd)
            output = output * mask_float
        output = self.final_ln(output)
        output = output * mask_float

        graph_emb = self._pool(output, mask)
        graph_emb = F.dropout(graph_emb, self.dropout_val, training=self.training)
        graph_repr = self.graph_head(graph_emb)
        out = self.output_decoder(graph_emb)
        return out, graph_repr
