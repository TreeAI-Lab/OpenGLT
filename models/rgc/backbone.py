import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import dropout_edge


class GCN(nn.Module):
    def __init__(self, n_layers, in_features, hidden_features, out_features,
                 drop_edge=0.0, drop_feats=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_features, hidden_features))
        for _ in range(max(n_layers - 2, 0)):
            self.layers.append(GCNConv(hidden_features, hidden_features))
        self.layers.append(GCNConv(hidden_features, out_features))
        self.drop_edge = drop_edge
        self.drop = nn.Dropout(drop_feats)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index):
        edge = dropout_edge(edge_index, self.drop_edge, training=self.training)[0]
        for layer in self.layers[:-1]:
            x = self.drop(F.relu(layer(x, edge)))
        return self.layers[-1](x, edge)


class GAT(nn.Module):
    def __init__(self, n_layers, in_features, hidden_features, out_features,
                 heads=4, drop_edge=0.0, drop_feats=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_features, hidden_features, heads, dropout=drop_feats, concat=False))
        for _ in range(max(n_layers - 2, 0)):
            self.layers.append(GATConv(hidden_features, hidden_features, heads, dropout=drop_feats, concat=False))
        self.layers.append(GATConv(hidden_features, out_features, heads, dropout=drop_feats, concat=False))
        self.drop_edge = drop_edge

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index):
        edge = dropout_edge(edge_index, self.drop_edge, training=self.training)[0]
        for layer in self.layers:
            x = layer(x, edge)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, n_layers, in_features, hidden_features, out_features,
                 drop_edge=0.0, drop_feats=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_features, hidden_features))
        for _ in range(max(n_layers - 2, 0)):
            self.layers.append(SAGEConv(hidden_features, hidden_features))
        self.layers.append(SAGEConv(hidden_features, out_features))
        self.drop_edge = drop_edge
        self.drop = nn.Dropout(drop_feats)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index):
        edge = dropout_edge(edge_index, self.drop_edge, training=self.training)[0]
        for layer in self.layers[:-1]:
            x = self.drop(F.relu(layer(x, edge)))
        return self.layers[-1](x, edge)
