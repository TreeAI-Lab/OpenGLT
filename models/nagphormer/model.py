import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.utils import to_dense_adj, to_dense_batch

from models.baselines import MLP, TUNodeEncoder


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


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads for NAGphormer.")
        self.num_heads = num_heads
        self.att_size = hidden_size // num_heads
        self.scale = self.att_size ** -0.5
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
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
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.att_size)
        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
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


def compute_khop_features_batch(x_dense, adj_dense, mask, hops):
    batch_size, max_nodes, hidden = x_dense.shape
    khop = x_dense.new_zeros(batch_size, max_nodes, hops + 1, hidden)
    mask_float = mask.unsqueeze(-1).to(x_dense.dtype)
    khop[:, :, 0, :] = x_dense * mask_float

    eye = torch.eye(max_nodes, dtype=adj_dense.dtype, device=adj_dense.device).unsqueeze(0)
    valid_pair = mask.unsqueeze(1) & mask.unsqueeze(2)
    adj_dense = (adj_dense + eye * mask.unsqueeze(1).to(adj_dense.dtype)).masked_fill(~valid_pair, 0)
    deg_inv_sqrt = adj_dense.sum(dim=-1).clamp(min=1).pow(-0.5)
    adj_norm = deg_inv_sqrt.unsqueeze(-1) * adj_dense * deg_inv_sqrt.unsqueeze(1)

    current = x_dense * mask_float
    for hop in range(hops):
        current = torch.bmm(adj_norm, current) * mask_float
        khop[:, :, hop + 1, :] = current
    return khop


class NAGphormer(nn.Module):
    """Graph-level NAGphormer adapter.

    NAGphormer represents each node by a sequence of 0..K-hop aggregated
    features, applies Transformer layers over this hop sequence, then pools node
    representations to graph predictions.
    """

    def __init__(self, nhid, nout, n_layers=2, num_heads=4, hops=3,
                 dataset_type='tu', dim_node_deg=0, dim_node_label=0,
                 dim_node_attr=0, dropout=0.3, attention_dropout=0.1,
                 pooling='mean', ffn_dim=None):
        super().__init__()
        self.dataset_type = dataset_type
        self.hops = int(hops)
        if self.hops < 1:
            raise ValueError("NAGphormer expects hops >= 1.")
        self.seq_len = self.hops + 1
        self.pooling = pooling
        self.dropout_val = dropout

        if dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            self.input_encoder = AtomEncoder(emb_dim=nhid)
        else:
            self.input_encoder = TUNodeEncoder(nhid, dim_node_deg, dim_node_label, dim_node_attr)

        ffn_dim = int(ffn_dim) if ffn_dim is not None else 2 * nhid
        self.layers = nn.ModuleList([
            EncoderLayer(nhid, ffn_dim, dropout, attention_dropout, num_heads)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(nhid)
        self.attn_layer = nn.Linear(2 * nhid, 1)
        self.out_proj = nn.Linear(nhid, nhid)
        self.graph_head = MLP(nhid, nhid, nlayer=2)
        self.output_decoder = MLP(nhid, nout, nlayer=2)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def reset_parameters(self):
        if hasattr(self.input_encoder, 'reset_parameters'):
            self.input_encoder.reset_parameters()
        for layer in self.layers:
            layer.apply(lambda module: init_params(module, n_layers=max(len(self.layers), 1)))
        self.final_ln.reset_parameters()
        nn.init.xavier_uniform_(self.attn_layer.weight)
        if self.attn_layer.bias is not None:
            nn.init.zeros_(self.attn_layer.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        self.graph_head.reset_parameters()
        self.output_decoder.reset_parameters()

    def _encode_input(self, x):
        if self.dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            if x.dtype == torch.float:
                x = x.long()
        elif x.dtype == torch.long:
            x = x.float()
        return self.input_encoder(x)

    def _pool(self, node_repr, mask):
        mask_float = mask.unsqueeze(-1).float()
        if self.pooling in ('sum', 'add'):
            return (node_repr * mask_float).sum(dim=1)
        if self.pooling == 'max':
            node_repr = node_repr.masked_fill(~mask.unsqueeze(-1), -1e9)
            return node_repr.max(dim=1).values
        return (node_repr * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)

    def forward(self, x, edge_index, batch, edge_attr=None,
                fingerprint=None, rwpe=None, input_embedding=None, **kwargs):
        h = input_embedding if input_embedding is not None else self._encode_input(x)
        h_dense, mask = to_dense_batch(h, batch)
        h_dense = h_dense * mask.unsqueeze(-1).to(h_dense.dtype)
        batch_size, max_nodes, hidden = h_dense.shape
        adj_dense = to_dense_adj(edge_index, batch, max_num_nodes=max_nodes)
        khop = compute_khop_features_batch(h_dense, adj_dense, mask, self.hops)

        tensor = khop.view(batch_size * max_nodes, self.seq_len, hidden)
        for enc_layer in self.layers:
            tensor = enc_layer(tensor)
        output = self.final_ln(tensor)

        target = output[:, 0, :].unsqueeze(1).repeat(1, self.seq_len - 1, 1)
        node_tensor, neighbor_tensor = torch.split(output, [1, self.seq_len - 1], dim=1)
        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        layer_atten = F.softmax(layer_atten, dim=1)
        neighbor_tensor = torch.sum(neighbor_tensor * layer_atten, dim=1, keepdim=True)
        node_repr = (node_tensor + neighbor_tensor).squeeze(1)
        node_repr = torch.relu(self.out_proj(node_repr)).view(batch_size, max_nodes, -1)
        node_repr = node_repr * mask.unsqueeze(-1).to(node_repr.dtype)

        graph_emb = self._pool(node_repr, mask)
        graph_emb = F.dropout(graph_emb, self.dropout_val, training=self.training)
        graph_repr = self.graph_head(graph_emb)
        out = self.output_decoder(graph_emb)
        return out, graph_repr
