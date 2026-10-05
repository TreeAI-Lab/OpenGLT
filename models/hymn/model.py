import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.data import Data
from torch_geometric.nn import GINConv as PyGGINConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from models.baselines import TUNodeEncoder


def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


class Reshaper(nn.Module):
    def __init__(self, num_feats, num_samples, out=False, stacked_samples=True):
        super().__init__()
        self.out = out
        self.num_feats = num_feats
        self.num_samples = num_samples
        self.stacked_samples = stacked_samples

    def reset_parameters(self):
        return

    def forward(self, x):
        if self.stacked_samples:
            if self.out:
                num_nodes = int(x.shape[0] / self.num_samples)
                return x.reshape(num_nodes, self.num_samples * self.num_feats)
            return x.reshape(x.shape[0] * self.num_samples, self.num_feats)

        if self.out:
            return x.reshape(-1, self.num_samples * self.num_feats)
        return x.reshape(-1, self.num_samples, self.num_feats)


class ColourCatter(nn.Module):
    def __init__(self, num_feats, num_colours, num_samples, out_reshape=True):
        super().__init__()
        self.feat_resh = Reshaper(num_feats, num_samples, out=False, stacked_samples=False)
        self.colour_resh = Reshaper(num_colours, num_samples, out=False, stacked_samples=False)
        self.out_resh = Reshaper(
            num_feats + num_colours, num_samples, out=True, stacked_samples=False
        ) if out_reshape else None

    def reset_parameters(self):
        self.feat_resh.reset_parameters()
        self.colour_resh.reset_parameters()
        if self.out_resh is not None:
            self.out_resh.reset_parameters()

    def forward(self, x, c):
        x = self.feat_resh(x)
        c = self.colour_resh(c)
        h = torch.cat([x, c], dim=2)
        return self.out_resh(h) if self.out_resh is not None else h


class SharedMLP(nn.Module):
    def __init__(self, in_dim, emb_dim, num_samples, multiplier=1, bn=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, multiplier * emb_dim),
            nn.BatchNorm1d(multiplier * emb_dim) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(multiplier * emb_dim, emb_dim),
        )
        self.in_resh = Reshaper(in_dim, num_samples, out=False, stacked_samples=True)
        self.out_resh = Reshaper(emb_dim, num_samples, out=True, stacked_samples=True)
        self.layers = nn.Sequential(self.in_resh, self.mlp, self.out_resh)

    def reset_parameters(self):
        reset(self.layers)

    def forward(self, x):
        return self.layers(x)


class SharedLinear(nn.Module):
    """Apply one shared linear predictor independently to every sample."""

    def __init__(self, in_dim, emb_dim, num_samples):
        super().__init__()
        self.layers = nn.Sequential(
            Reshaper(in_dim, num_samples, out=False, stacked_samples=True),
            nn.Linear(in_dim, emb_dim),
            Reshaper(emb_dim, num_samples, out=True, stacked_samples=True),
        )

    def reset_parameters(self):
        reset(self.layers)

    def forward(self, x):
        return self.layers(x)


class ColourCatSharedGINConv(nn.Module):
    def __init__(self, in_dim, emb_dim, colour_dim, num_samples, multiplier=1, bn=True):
        super().__init__()
        self.colour_catter = ColourCatter(in_dim, colour_dim, num_samples, out_reshape=True)
        mlp = SharedMLP(in_dim + colour_dim, emb_dim, num_samples, multiplier=multiplier, bn=bn)
        self.layer = PyGGINConv(nn=mlp, train_eps=True)

    def reset_parameters(self):
        self.colour_catter.reset_parameters()
        self.layer.reset_parameters()

    def forward(self, x, edge_index, c, edge_attr=None):
        return self.layer(self.colour_catter(x, c), edge_index)


class SharedGINConv(nn.Module):
    def __init__(self, in_dim, emb_dim, num_samples, multiplier=1, bn=True):
        super().__init__()
        mlp = SharedMLP(in_dim, emb_dim, num_samples, multiplier=multiplier, bn=bn)
        self.layer = PyGGINConv(nn=mlp, train_eps=True)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        return self.layer(x, edge_index)


class SumReadout(nn.Module):
    def reset_parameters(self):
        return

    def forward(self, h, index):
        h = h[-1] if isinstance(h, list) else h
        return global_add_pool(h, index)


class MeanReadout(nn.Module):
    def reset_parameters(self):
        return

    def forward(self, h, index):
        h = h[-1] if isinstance(h, list) else h
        return global_mean_pool(h, index)


class MaxReadout(nn.Module):
    def reset_parameters(self):
        return

    def forward(self, h, index):
        h = h[-1] if isinstance(h, list) else h
        return global_max_pool(h, index)


class MeanAveraging(nn.Module):
    def __init__(self, emb_dim, num_samples):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_samples = num_samples

    def reset_parameters(self):
        return

    def forward(self, h, sample_mask=None):
        h = h.reshape(-1, self.num_samples, self.emb_dim)
        if sample_mask is None:
            return h.mean(1)
        if sample_mask.shape != h.shape[:2]:
            raise ValueError(
                f"HyMN sample mask has shape {tuple(sample_mask.shape)}, "
                f"expected {tuple(h.shape[:2])}."
            )
        weights = sample_mask.to(dtype=h.dtype).unsqueeze(-1)
        denominator = weights.sum(1).clamp_min_(1.0)
        return (h * weights).sum(1) / denominator


class ColourCatSharedGNN(nn.Module):
    def __init__(self, num_layers, in_dim, emb_dim, colour_dim, num_samples,
                 readout_module, prediction_module, averaging_module,
                 residual=False, multiplier=1, inject_colours=True,
                 bn=True, bn_between_convs=False):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.readout_module = readout_module
        self.prediction_module = prediction_module
        self.averaging_module = averaging_module
        self.residual = residual
        self.inject_colours = inject_colours

        if bn_between_convs:
            self.in_resh = Reshaper(emb_dim, num_samples, out=False, stacked_samples=True)
            self.out_resh = Reshaper(emb_dim, num_samples, out=True, stacked_samples=True)
            self.batch_norms = nn.ModuleList()
        else:
            self.batch_norms = None

        for layer_idx in range(num_layers):
            use_colour = layer_idx == 0 or inject_colours
            if use_colour:
                conv = ColourCatSharedGINConv(
                    emb_dim if layer_idx != 0 else in_dim,
                    emb_dim,
                    colour_dim,
                    num_samples,
                    multiplier=multiplier,
                    bn=bn,
                )
            else:
                conv = SharedGINConv(
                    emb_dim,
                    emb_dim,
                    num_samples,
                    multiplier=multiplier,
                    bn=bn,
                )
            self.convs.append(conv)
            if bn_between_convs:
                self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def reset_parameters(self):
        self.readout_module.reset_parameters()
        self.prediction_module.reset_parameters()
        self.averaging_module.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norms is not None:
            for bn in self.batch_norms:
                bn.reset_parameters()

    def forward(self, batched_data):
        current = batched_data.x_repeated
        c_samples = batched_data.c_samples
        edge_index = batched_data.edge_index
        batch = batched_data.batch
        h_list = [current]
        for layer_idx, conv in enumerate(self.convs):
            if layer_idx > 0 and not self.inject_colours:
                h = conv(current, edge_index, None)
            else:
                h = conv(current, edge_index, c_samples, None)
            if self.batch_norms is not None:
                h = self.out_resh(self.batch_norms[layer_idx](self.in_resh(h)))
            if layer_idx != self.num_layers - 1:
                h = F.relu(h)
            current = h + current if layer_idx > 0 and self.residual else h
            h_list.append(current)
        hs = self.readout_module(h_list, batch)
        ys = self.prediction_module(hs)
        return self.averaging_module(
            ys, sample_mask=getattr(batched_data, 'sample_mask', None)
        )


class HyMN(nn.Module):
    """Graph-level HyMN adapter with shared colour-aware GIN samples.

    The official HyMN counting implementation paints each graph multiple times
    by individualizing nodes and shares a colour-aware GIN across samples.  The
    generic adapter also retains random categorical colourings for non-counting
    workloads.
    """

    def __init__(self, nhid, nout, num_layers=4, num_samples=3,
                 num_colours=2, dataset_type='tu', dim_node_deg=0,
                 dim_node_label=0, dim_node_attr=0, dropout=0.0,
                 pooling='sum', residual=True, inject_colours=True,
                 multiplier=1, predictor_multiplier=1, bn=True,
                 colouring='uniform', counting_input='degree',
                 predictor='mlp'):
        super().__init__()
        self.dataset_type = dataset_type
        self.num_samples = int(num_samples)
        self.num_colours = int(num_colours)
        self.dropout = float(dropout)
        self.colouring = str(colouring).lower()
        self.counting_input = str(counting_input).lower()
        self.predictor = str(predictor).lower()
        self._coverage_warning_printed = False

        if dataset_type == 'counting' and self.counting_input == 'constant':
            # This is the reference HyMN counting input: x_v = 1.  In
            # particular, it does not insert OpenGLT's degree encoder before
            # the first colour-aware GIN layer.
            self.input_encoder = nn.Identity()
            input_dim = 1
        elif dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            self.input_encoder = AtomEncoder(emb_dim=nhid)
            input_dim = nhid
        else:
            self.input_encoder = TUNodeEncoder(nhid, dim_node_deg, dim_node_label, dim_node_attr)
            input_dim = nhid

        if pooling in ('sum', 'add'):
            readout_module = SumReadout()
        elif pooling == 'max':
            readout_module = MaxReadout()
        else:
            readout_module = MeanReadout()

        if self.predictor == 'linear':
            prediction_module = SharedLinear(nhid, nout, self.num_samples)
        elif self.predictor == 'mlp':
            prediction_module = SharedMLP(
                nhid,
                nout,
                self.num_samples,
                multiplier=int(predictor_multiplier),
                bn=bn,
            )
        else:
            raise ValueError(
                f"Unsupported HyMN predictor '{predictor}'. Use 'linear' or 'mlp'."
            )
        averaging_module = MeanAveraging(nout, self.num_samples)
        self.gnn = ColourCatSharedGNN(
            num_layers=num_layers,
            in_dim=input_dim,
            emb_dim=nhid,
            colour_dim=self.num_colours,
            num_samples=self.num_samples,
            readout_module=readout_module,
            prediction_module=prediction_module,
            averaging_module=averaging_module,
            residual=residual,
            multiplier=multiplier,
            inject_colours=inject_colours,
            bn=bn,
        )

    def reset_parameters(self):
        if hasattr(self.input_encoder, 'reset_parameters'):
            self.input_encoder.reset_parameters()
        self.gnn.reset_parameters()

    def _encode_input(self, x):
        if self.dataset_type == 'counting' and self.counting_input == 'constant':
            return torch.ones((x.size(0), 1), dtype=torch.float, device=x.device)
        if self.dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            if x.dtype == torch.float:
                x = x.long()
        elif x.dtype == torch.long:
            x = x.float()
        return self.input_encoder(x)

    def _paint(self, data):
        num_nodes = data.x.size(0)
        if self.colouring == 'uniform':
            colours = torch.randint(
                0,
                self.num_colours,
                (num_nodes * self.num_samples,),
                device=data.x.device,
            )
            c_onehot = F.one_hot(colours, self.num_colours).to(data.x.dtype)
            data.c_samples = c_onehot.reshape(
                num_nodes, self.num_samples * self.num_colours
            )
            num_graphs = int(data.batch.max().item()) + 1 if num_nodes else 0
            data.sample_mask = torch.ones(
                (num_graphs, self.num_samples),
                dtype=torch.bool,
                device=data.x.device,
            )
        elif self.colouring == 'individualize':
            if self.num_colours != 1:
                raise ValueError(
                    "HyMN individualize colouring requires num_colours=1 "
                    "(a scalar indicator for the selected node)."
                )
            num_graphs = int(data.batch.max().item()) + 1 if num_nodes else 0
            graph_sizes = torch.bincount(data.batch, minlength=num_graphs)
            starts = torch.cat([
                graph_sizes.new_zeros(1), graph_sizes.cumsum(0)[:-1]
            ])
            local_node_index = torch.arange(
                num_nodes, device=data.x.device
            ) - starts[data.batch]
            sample_index = torch.arange(
                self.num_samples, device=data.x.device
            )
            # [N, S]: sample s marks node s in the local node ordering.  When
            # S >= |V| this enumerates every one-node individualization and is
            # permutation invariant after sample averaging.
            data.c_samples = (
                local_node_index.unsqueeze(1) == sample_index.unsqueeze(0)
            ).to(data.x.dtype)
            data.sample_mask = (
                sample_index.unsqueeze(0) < graph_sizes.unsqueeze(1)
            )
            if (
                not self._coverage_warning_printed
                and graph_sizes.numel()
                and int(graph_sizes.max().item()) > self.num_samples
            ):
                print(
                    "[HyMN counting warning] num_samples="
                    f"{self.num_samples} is smaller than a graph with "
                    f"{int(graph_sizes.max().item())} nodes; only the first "
                    "num_samples node individualizations are used. Use a "
                    "num_samples value at least as large as the dataset's "
                    "maximum graph size for the full reference protocol."
                )
                self._coverage_warning_printed = True
        else:
            raise ValueError(
                f"Unsupported HyMN colouring '{self.colouring}'. "
                "Use 'uniform' or 'individualize'."
            )
        data.x_repeated = data.x.repeat(1, self.num_samples)
        data.edge_attr = None
        data.edge_attr_repeated = None
        return data

    def forward(self, x, edge_index, batch, edge_attr=None,
                fingerprint=None, rwpe=None, input_embedding=None, **kwargs):
        h = input_embedding if input_embedding is not None else self._encode_input(x)
        h = F.dropout(h, p=self.dropout, training=self.training)
        data = Data(x=h, edge_index=edge_index, batch=batch, edge_attr=None)
        data = self._paint(data)
        out = self.gnn(data)
        return out, None
