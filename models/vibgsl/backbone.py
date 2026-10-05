import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GATConv, GCNConv, GINConv


class VIBGCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class VIBGIN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(in_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                BN(hidden_dim),
            ),
            train_eps=True,
        )
        self.conv2 = GINConv(
            Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, out_dim),
                ReLU(),
                BN(out_dim),
            ),
            train_eps=True,
        )

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class VIBGAT(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=4, dropout=0.5)
        self.conv2 = GATConv(hidden_dim * 4, out_dim, heads=1, concat=False, dropout=0.5)
        self.relu = torch.nn.LeakyReLU(0.2)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
