from __future__ import annotations

import json
from pathlib import Path


BEST_DIR = Path(__file__).resolve().parents[1] / "best_hparams"

MODEL_ALIASES = {
    "gcnplus": "GCNPlus", "sageplus": "SAGEPlus", "gps": "GraphGPS",
    "graphgps": "GraphGPS", "topk": "TopkPool", "topkpool": "TopkPool",
    "edgepool": "EdgePool", "hgt": "HGT", "nag": "NAG", "hymn": "HMN",
    "ak": "AK", "ak+": "AK", "i2": "I2GNN", "i2gnn": "I2GNN",
    "mo": "MO", "hgp": "HGP", "vib": "VIB", "rgc": "RGC",
    "mvg": "MVG", "mlp-f": "MLP-F", "mlp-s": "MLP-S",
    "lgbm": "LGBM",
}


def canonical_model(name):
    text = str(name).strip()
    return MODEL_ALIASES.get(text.lower(), text)


def dataset_family(dataset):
    name = str(dataset)
    lower = name.lower()
    if lower.startswith("gnce-"):
        return "gnce"
    if lower.startswith("good-"):
        return "good"
    if lower.startswith("gcare-"):
        return "gcare"
    if name.startswith("Count-"):
        return "counting"
    if lower.startswith("ogbg-"):
        return "ogb"
    if lower.startswith("peptides-"):
        return "lrgb"
    if lower in {"zinc", "aqsol"}:
        return "pyg_molecular"
    return "tu"


def load_best_entry(dataset, model, setting=None):
    family = "cardbench" if setting is not None else dataset_family(dataset)
    key = setting or dataset
    payload = json.loads((BEST_DIR / f"{family}.yaml").read_text(encoding="utf-8"))
    datasets = payload["datasets"]
    if family == "good" and setting is None:
        normalized = str(dataset).lower().replace("_", "-")
        benchmark = "GOOD-HIV" if normalized.startswith("good-hiv") else "GOOD-ZINC"
        if normalized.endswith("no-shift"):
            key = f"{benchmark}::no-shift"
        else:
            setting_name = next(
                (name for name in (
                    "scaffold-covariate", "scaffold-concept",
                    "size-covariate", "size-concept",
                ) if name in normalized),
                None,
            )
            if setting_name is not None:
                key = f"{benchmark}::{setting_name}"
    if key not in datasets:
        lower_map = {str(name).lower(): name for name in datasets}
        key = lower_map.get(str(key).lower(), key)
    if key not in datasets:
        raise KeyError(f"No best-parameter entry for {key!r} in {family}.yaml")
    canonical = canonical_model(model)
    if canonical not in datasets[key]:
        raise KeyError(
            f"No best hyperparameters available since OOM/TLE: "
            f"{key} x {canonical}."
        )
    return datasets[key][canonical]


def merge_cfg(cfg, values):
    """Recursively apply a plain mapping to a yacs configuration."""
    for key, value in values.items():
        target = getattr(cfg, key)
        if isinstance(value, dict):
            merge_cfg(target, value)
        else:
            setattr(cfg, key, value)
    return cfg
