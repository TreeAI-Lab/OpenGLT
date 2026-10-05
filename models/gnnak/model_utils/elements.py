import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool


class Identity(nn.Module):
    def forward(self, x):
        return x

    def reset_parameters(self):
        pass


class DiscreteEncoder(nn.Module):
    def __init__(self, hidden_channels, max_num_features=16, max_num_values=512):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(max_num_values, hidden_channels)
            for _ in range(max_num_features)
        ])

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if x.size(1) > len(self.embeddings):
            raise ValueError(
                f"DiscreteEncoder received {x.size(1)} feature columns, "
                f"but max_num_features={len(self.embeddings)}."
            )
        out = 0
        x = x.long().clamp_min(0)
        for i in range(x.size(1)):
            idx = x[:, i].clamp_max(self.embeddings[i].num_embeddings - 1)
            out = out + self.embeddings[i](idx)
        return out


class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True,
                 with_norm=True, bias=True):
        super().__init__()
        n_hid = nin
        self.layers = nn.ModuleList([
            nn.Linear(
                nin if i == 0 else n_hid,
                n_hid if i < nlayer - 1 else nout,
                bias=(i == nlayer - 1 and not with_final_activation and bias)
                or (not with_norm),
            )
            for i in range(nlayer)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(n_hid if i < nlayer - 1 else nout)
            if with_norm else Identity()
            for i in range(nlayer)
        ])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < self.nlayer - 1 or self.with_final_activation:
                x = norm(x)
                x = F.relu(x)
        return x


class VNUpdate(nn.Module):
    def __init__(self, dim, with_norm=True):
        super().__init__()
        self.mlp = MLP(dim, dim, with_norm=with_norm,
                       with_final_activation=True, bias=False)

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, vn, x, batch):
        graph_emb = global_add_pool(x, batch)
        if vn is not None:
            graph_emb = graph_emb + vn
        vn = self.mlp(graph_emb)
        x = x + vn[batch]
        return vn, x
