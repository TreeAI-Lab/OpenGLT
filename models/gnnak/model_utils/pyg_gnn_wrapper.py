import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import degree
from torch_scatter import scatter

from .elements import MLP


class GINEConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.nn = MLP(nin, nout, 2, False, bias=bias)
        self.layer = pyg_nn.GINEConv(self.nn, train_eps=True)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)


class GINConv(GINEConv):
    pass


class GCNConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.layer = pyg_nn.GCNConv(nin, nout, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        return self.layer(x, edge_index)


class GATConv(nn.Module):
    def __init__(self, nin, nout, bias=True, nhead=1):
        super().__init__()
        self.layer = pyg_nn.GATConv(nin, nout // nhead, nhead, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        return self.layer(x, edge_index)


class SAGEConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.layer = pyg_nn.SAGEConv(nin, nout, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        return self.layer(x, edge_index)


class SimplifiedPNAConv(pyg_nn.MessagePassing):
    def __init__(self, nin, nout, bias=True,
                 aggregators=('mean', 'min', 'max', 'std'), **kwargs):
        kwargs.setdefault('aggr', None)
        super().__init__(node_dim=0, **kwargs)
        self.aggregators = aggregators
        self.pre_nn = MLP(3 * nin, nin, 2, False)
        self.post_nn = MLP((len(aggregators) + 2) * nin, nout, 2, False, bias=bias)
        self.deg_embedder = nn.Embedding(512, nin)

    def reset_parameters(self):
        self.pre_nn.reset_parameters()
        self.post_nn.reset_parameters()
        self.deg_embedder.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.post_nn(torch.cat([x, out], dim=-1))

    def message(self, x_i, x_j, edge_attr):
        return self.pre_nn(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def aggregate(self, inputs, index, dim_size=None):
        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='max')
            elif aggregator in ('var', 'std'):
                mean = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='mean')
                mean_sq = scatter(inputs * inputs, index, dim=0,
                                  dim_size=dim_size, reduce='mean')
                out = mean_sq - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(F.relu(out) + 1e-5)
            else:
                raise ValueError(f"Unknown aggregator {aggregator}.")
            outs.append(out)
        deg = degree(index, dim_size, dtype=index.dtype).clamp_max(
            self.deg_embedder.num_embeddings - 1
        )
        outs.append(self.deg_embedder(deg))
        return torch.cat(outs, dim=-1)
