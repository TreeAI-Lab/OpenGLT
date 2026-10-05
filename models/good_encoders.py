"""Categorical molecule encoders matching GOOD's released feature schema."""

import torch
import torch.nn as nn


# Copied from GOOD.utils.data.x_map/e_map in the official GOOD repository.
GOOD_ATOM_FEATURE_DIMS = (119, 4, 11, 12, 9, 5, 8, 2, 2)
GOOD_BOND_FEATURE_DIMS = (5, 6, 2)


class GOODCategoricalEncoder(nn.Module):
    def __init__(self, feature_dims, emb_dim, feature_name):
        super().__init__()
        self.feature_dims = tuple(int(dim) for dim in feature_dims)
        self.feature_name = feature_name
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, emb_dim) for dim in self.feature_dims
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, values):
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        if values.size(1) != len(self.embeddings):
            raise ValueError(
                f"GOOD {self.feature_name} encoder expects "
                f"{len(self.embeddings)} categorical columns, got "
                f"{values.size(1)}."
            )
        values = values.long()
        encoded = 0
        for column, (embedding, cardinality) in enumerate(
                zip(self.embeddings, self.feature_dims)):
            column_values = values[:, column]
            if column_values.numel():
                min_value = int(column_values.min())
                max_value = int(column_values.max())
                if min_value < 0 or max_value >= cardinality:
                    raise ValueError(
                        f"GOOD {self.feature_name} column {column} has range "
                        f"[{min_value}, {max_value}], expected "
                        f"[0, {cardinality - 1}]."
                    )
            encoded = encoded + embedding(column_values)
        return encoded


class GOODAtomEncoder(GOODCategoricalEncoder):
    def __init__(self, emb_dim):
        super().__init__(
            GOOD_ATOM_FEATURE_DIMS, emb_dim, feature_name='atom'
        )


class GOODBondEncoder(GOODCategoricalEncoder):
    def __init__(self, emb_dim):
        super().__init__(
            GOOD_BOND_FEATURE_DIMS, emb_dim, feature_name='bond'
        )

