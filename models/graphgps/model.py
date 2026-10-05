import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import (
    AttentionalAggregation,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from .layers import GPSLayer, make_activation
from models.good_encoders import GOODAtomEncoder, GOODBondEncoder


class TUFeatureEncoder(nn.Module):
    def __init__(self, hidden_size, dim_node_deg=0, dim_node_label=0, dim_node_attr=0,
                 num_edge_classes=0, edge_hidden_size=None):
        super().__init__()
        edge_hidden_size = hidden_size if edge_hidden_size is None else edge_hidden_size
        node_dim = dim_node_deg + dim_node_label + dim_node_attr
        if node_dim <= 0:
            node_dim = 1
        edge_dim = max(num_edge_classes, 1)
        self.node_encoder = nn.Linear(node_dim, hidden_size)
        self.edge_encoder = nn.Linear(edge_dim, edge_hidden_size)

    def forward(self, x, edge_attr, edge_index):
        if x.dtype == torch.long:
            x = x.float()
        x = self.node_encoder(x)
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.size(1), self.edge_encoder.in_features))
        elif edge_attr.dtype == torch.long:
            edge_attr = edge_attr.float()
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        edge_attr = self.edge_encoder(edge_attr)
        return x, edge_attr


class MolecularFeatureEncoder(nn.Module):
    def __init__(self, hidden_size, node_hidden_size=None):
        super().__init__()
        node_hidden_size = hidden_size if node_hidden_size is None else node_hidden_size
        self.node_encoder = AtomEncoder(emb_dim=node_hidden_size)
        self.edge_encoder = BondEncoder(emb_dim=hidden_size)

    def forward(self, x, edge_attr, edge_index):
        if x.dtype == torch.float:
            x = x.long()
        x = self.node_encoder(x)
        if edge_attr is None:
            edge_attr = edge_index.new_zeros((edge_index.size(1), 3))
        if edge_attr.dtype == torch.float:
            edge_attr = edge_attr.long()
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        edge_attr = self.edge_encoder(edge_attr)
        return x, edge_attr


class GOODMolecularFeatureEncoder(nn.Module):
    """The exact categorical schema used by GOOD.utils.data.from_smiles."""

    def __init__(self, hidden_size, node_hidden_size=None):
        super().__init__()
        node_hidden_size = (
            hidden_size if node_hidden_size is None else node_hidden_size
        )
        self.node_encoder = GOODAtomEncoder(emb_dim=node_hidden_size)
        self.edge_encoder = GOODBondEncoder(emb_dim=hidden_size)

    def forward(self, x, edge_attr, edge_index):
        x = self.node_encoder(x.long())
        if edge_attr is None:
            edge_attr = edge_index.new_zeros((edge_index.size(1), 3))
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        edge_attr = self.edge_encoder(edge_attr.long())
        return x, edge_attr


class LapPEEncoder(nn.Module):
    """Laplace PE encoder matching GraphGPS' DeepSet/Transformer PE block."""

    def __init__(self, dim_pe=8, max_freqs=8, model_type='DeepSet',
                 num_layers=2, num_heads=4, raw_norm_type='none'):
        super().__init__()
        self.model_type = model_type
        self.max_freqs = max_freqs
        self.linear_A = nn.Linear(2, dim_pe)
        self.raw_norm = nn.BatchNorm1d(max_freqs) if raw_norm_type.lower() == 'batchnorm' else None

        activation = nn.ReLU
        if model_type == 'Transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim_pe,
                nhead=num_heads,
                batch_first=True,
            )
            self.pe_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif model_type == 'DeepSet':
            layers = []
            if num_layers == 1:
                layers.append(activation())
            else:
                self.linear_A = nn.Linear(2, 2 * dim_pe)
                layers.append(activation())
                for _ in range(num_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported LapPE model_type: {model_type}")

    def forward(self, LapPE):
        if LapPE is None:
            raise ValueError("LapPE is enabled but EigVals/EigVecs were not provided.")
        EigVals = LapPE.get('EigVals') if isinstance(LapPE, dict) else getattr(LapPE, 'EigVals', None)
        EigVecs = LapPE.get('EigVecs') if isinstance(LapPE, dict) else getattr(LapPE, 'EigVecs', None)
        if EigVals is None or EigVecs is None:
            raise ValueError("LapPE must contain 'EigVals' and 'EigVecs'.")

        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip = torch.where(sign_flip >= 0.5, 1.0, -1.0)
            EigVecs = EigVecs * sign_flip.unsqueeze(0)

        pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2)
        empty_mask = torch.isnan(pos_enc)
        pos_enc = pos_enc.masked_fill(empty_mask, 0.0)

        if self.raw_norm is not None:
            pos_enc = self.raw_norm(pos_enc)

        pos_enc = self.linear_A(pos_enc)
        if self.model_type == 'Transformer':
            pos_enc = self.pe_encoder(pos_enc, src_key_padding_mask=empty_mask[:, :, 0])
        else:
            pos_enc = self.pe_encoder(pos_enc)

        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2), 0.0)
        return torch.sum(pos_enc, dim=1)


class RWSEEncoder(nn.Module):
    """GraphGPS-style random-walk structural encoder.

    ``rwpe_dim`` is the number of random-walk steps in the input, whereas
    ``embed_dim`` is the hidden width reserved for the encoded RWSE. This
    mirrors GraphGPS' ``KernelPENodeEncoder``.
    """

    def __init__(self, rwpe_dim=20, embed_dim=20, model_type='linear',
                 num_layers=1, raw_norm_type='BatchNorm'):
        super().__init__()
        self.raw_norm = (
            nn.BatchNorm1d(rwpe_dim)
            if str(raw_norm_type).lower() == 'batchnorm'
            else None
        )
        model_type = str(model_type).lower()
        if model_type == 'linear':
            self.encoder = nn.Linear(rwpe_dim, embed_dim)
        elif model_type == 'mlp':
            layers = []
            if num_layers <= 1:
                layers.extend([nn.Linear(rwpe_dim, embed_dim), nn.ReLU()])
            else:
                layers.extend([nn.Linear(rwpe_dim, 2 * embed_dim), nn.ReLU()])
                for _ in range(num_layers - 2):
                    layers.extend([
                        nn.Linear(2 * embed_dim, 2 * embed_dim), nn.ReLU()
                    ])
                layers.extend([nn.Linear(2 * embed_dim, embed_dim), nn.ReLU()])
            self.encoder = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported GraphGPS RWSE model_type: {model_type}")

    def forward(self, rwpe):
        rwpe = rwpe.float()
        if self.raw_norm is not None:
            rwpe = self.raw_norm(rwpe)
        return self.encoder(rwpe)


class MLPHead(nn.Module):
    def __init__(self, hidden_size, out_size, act_type='relu', num_layers=2):
        super().__init__()
        activation = make_activation(act_type)
        layers = []
        in_dim = hidden_size
        for _ in range(max(num_layers - 1, 0)):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(activation())
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, out_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SANGraphHead(nn.Module):
    """Official GraphGPS graph head: pool, shrink MLP, output."""

    def __init__(self, hidden_size, out_size, act_type='relu', num_layers=2):
        super().__init__()
        activation = make_activation(act_type)
        layers = []
        in_dim = hidden_size
        for layer_idx in range(num_layers):
            out_dim = hidden_size // (2 ** (layer_idx + 1))
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, out_size))
        self.net = nn.Sequential(*layers)

    def forward(self, graph_emb):
        return self.net(graph_emb)


class GraphGPSModel(nn.Module):
    def __init__(self, hidden_size, out_size, num_layers,
                 dataset_type='ogb', dim_node_deg=0, dim_node_label=0,
                 dim_node_attr=0, num_edge_classes=0,
                 local_gnn_type='GINE', global_model_type='Transformer',
                 num_heads=4, dropout=0.0, attn_dropout=0.0,
                 act_type='relu', layer_norm=False, batch_norm=True,
                 readout_type='mean', use_rwpe=False, rwpe_dim=20,
                 rwpe_encoder_type='graphgps', rwpe_embed_dim=20,
                 rwpe_model='linear', rwpe_layers=1,
                 rwpe_raw_norm_type='BatchNorm',
                 use_lappe=False, lappe_dim=8, lappe_max_freqs=8,
                 lappe_model='DeepSet', lappe_layers=2, lappe_heads=4,
                 lappe_raw_norm_type='none', pna_degrees=None, head_layers=2):
        super().__init__()
        self.dataset_type = dataset_type
        self.readout_type = readout_type
        self.use_rwpe = use_rwpe
        self.use_lappe = use_lappe
        self.rwpe_encoder_type = str(rwpe_encoder_type).lower()

        reserved_pe_dim = lappe_dim if use_lappe else 0
        if use_rwpe and self.rwpe_encoder_type == 'graphgps':
            reserved_pe_dim += rwpe_embed_dim
        node_hidden_size = hidden_size - reserved_pe_dim
        if node_hidden_size <= 0:
            raise ValueError(
                f"hidden_size={hidden_size} must exceed the total reserved PE "
                f"dimension ({reserved_pe_dim})."
            )

        if dataset_type == 'good_mol':
            self.encoder = GOODMolecularFeatureEncoder(
                hidden_size, node_hidden_size=node_hidden_size
            )
        elif dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
            self.encoder = MolecularFeatureEncoder(hidden_size, node_hidden_size=node_hidden_size)
        else:
            self.encoder = TUFeatureEncoder(
                node_hidden_size,
                dim_node_deg=dim_node_deg,
                dim_node_label=dim_node_label,
                dim_node_attr=dim_node_attr,
                num_edge_classes=num_edge_classes,
                edge_hidden_size=hidden_size,
            )

        if use_lappe:
            self.lappe_encoder = LapPEEncoder(
                dim_pe=lappe_dim,
                max_freqs=lappe_max_freqs,
                model_type=lappe_model,
                num_layers=lappe_layers,
                num_heads=lappe_heads,
                raw_norm_type=lappe_raw_norm_type,
            )

        if use_rwpe and self.rwpe_encoder_type == 'graphgps':
            self.rwpe_encoder = RWSEEncoder(
                rwpe_dim=rwpe_dim,
                embed_dim=rwpe_embed_dim,
                model_type=rwpe_model,
                num_layers=rwpe_layers,
                raw_norm_type=rwpe_raw_norm_type,
            )
        elif use_rwpe and self.rwpe_encoder_type == 'legacy_project':
            self.rwpe_proj = nn.Linear(hidden_size + rwpe_dim, hidden_size)
        elif use_rwpe:
            raise ValueError(
                "rwpe_encoder_type must be 'graphgps' or 'legacy_project', "
                f"got {rwpe_encoder_type!r}."
            )

        if hidden_size % num_heads != 0 and global_model_type != 'None':
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}."
            )

        self.layers = nn.ModuleList([
            GPSLayer(
                dim_h=hidden_size,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=num_heads,
                act=act_type,
                dropout=dropout,
                attn_dropout=attn_dropout,
                layer_norm=layer_norm,
                batch_norm=batch_norm,
                pna_degrees=pna_degrees,
            )
            for _ in range(num_layers)
        ])

        if readout_type == 'attn':
            self.attn_readout = AttentionalAggregation(gate_nn=nn.Linear(hidden_size, 1))
        elif readout_type not in ('mean', 'sum', 'add', 'max'):
            raise ValueError(f"Unknown GraphGPS readout_type: {readout_type}")

        self.graph_head = nn.Identity()
        self.output_decoder = SANGraphHead(hidden_size, out_size, act_type=act_type, num_layers=head_layers)

    def forward(self, x, edge_index, batch, edge_attr=None, fingerprint=None, rwpe=None,
                input_embedding=None, LapPE=None):
        if input_embedding is None:
            # The reference GraphGPS counting loader uses a constant one-valued
            # edge feature before LinearEdge. Keep this compatibility local to
            # GraphGPS so the shared OpenGLT loader and other models are not
            # silently changed.
            if self.dataset_type == 'counting':
                edge_attr = x.new_ones((edge_index.size(1), 1)).float()
            x, edge_attr = self.encoder(x, edge_attr, edge_index)
        else:
            x = input_embedding

        if self.use_lappe:
            lap_pe = self.lappe_encoder(LapPE)
            x = torch.cat([x, lap_pe], dim=-1)

        if self.use_rwpe:
            if rwpe is None:
                raise ValueError("GraphGPS expected random_walk_pe but got None.")
            if self.rwpe_encoder_type == 'graphgps':
                x = torch.cat([x, self.rwpe_encoder(rwpe)], dim=-1)
            else:
                x = self.rwpe_proj(torch.cat([x, rwpe.float()], dim=-1))

        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, batch, edge_attr)

        graph_emb = self._readout(x, batch)
        hidden_graph_emb = self.graph_head(graph_emb)
        out = self.output_decoder(graph_emb)
        return out, hidden_graph_emb

    def _readout(self, x, batch):
        if self.readout_type == 'mean':
            return global_mean_pool(x, batch)
        if self.readout_type in ('sum', 'add'):
            return global_add_pool(x, batch)
        if self.readout_type == 'max':
            return global_max_pool(x, batch)
        return self.attn_readout(x, batch)
