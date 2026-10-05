from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CE_LABEL_COLUMNS, GENERAL_LABEL_COLUMNS, TOPOLOGY_COLUMNS


@dataclass(frozen=True)
class BenchmarkData:
    general_performance: pd.DataFrame
    ce_performance: pd.DataFrame
    general_topology: pd.DataFrame
    ce_topology: pd.DataFrame
    general_features: pd.DataFrame
    ce_features: pd.DataFrame


def _frame(archive: np.lib.npyio.NpzFile, name: str) -> pd.DataFrame:
    columns = archive[f"{name}__columns"].tolist()
    return pd.DataFrame({column: archive[f"{name}__{column}"] for column in columns})


def _indexed(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return frame.set_index("dataset").loc[:, columns].astype(float)


def load_data(path: Path) -> BenchmarkData:
    with np.load(path, allow_pickle=False) as archive:
        general_performance = _frame(archive, "general_performance")
        ce_performance = _frame(archive, "ce_performance")
        general_topology = _frame(archive, "general_topology")
        ce_topology = _frame(archive, "ce_topology")
        general_features = _frame(archive, "general_features")
        ce_features = _frame(archive, "ce_features")

    return BenchmarkData(
        general_performance=general_performance,
        ce_performance=ce_performance,
        general_topology=_indexed(general_topology, TOPOLOGY_COLUMNS),
        ce_topology=_indexed(ce_topology, TOPOLOGY_COLUMNS),
        general_features=_indexed(general_features, GENERAL_LABEL_COLUMNS),
        ce_features=_indexed(ce_features, CE_LABEL_COLUMNS),
    )
