from __future__ import annotations

import argparse
import importlib
import sys


BASELINES = {
    "GCN": ("GCN", "none"), "GCNPlus": ("GCNplus", "none"),
    "GIN": ("GIN", "none"), "GINE": ("GINE", "none"),
    "SAGE": ("SAGE", "none"), "SAGEPlus": ("SAGEplus", "none"),
    "PNA": ("PNA", "none"), "EdgePool": ("GIN", "edgepool"),
    "TopkPool": ("GIN", "topk"), "GMT": ("GIN", "gmt"),
    "MLP-F": ("MLPFeature", "none"),
    "MLP-S": ("MLPFeatureStruct", "none"), "LGBM": ("LightGBM", "none"),
}

MODULES = {
    "GraphGPS": "graphgps_tuning", "HGT": "hubgt_tuning",
    "NAG": "nagphormer_tuning", "HMN": "hymn_tuning",
    "AK": "gnnak_tuning", "ECS": "ecs_tuning", "I2GNN": "i2gnn_tuning",
    "MO": "mosgsl_tuning", "HGP": "hgpsl_tuning", "VIB": "vibgsl_tuning",
    "RGC": "rgc_tuning", "MVG": "mvgrl_tuning", "GCA": "gca_tuning",
}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", required=True, choices=tuple(BASELINES) + tuple(MODULES))
    known, remaining = parser.parse_known_args()
    if "--n_trials" not in remaining:
        remaining.extend(["--n_trials", "200"])
    if known.model in BASELINES:
        backbone, pooling = BASELINES[known.model]
        remaining.extend(["--backbone", backbone, "--pooling_method", pooling])
        module_name = "main_baselines_tuning"
    else:
        module_name = MODULES[known.model]
    sys.argv = [sys.argv[0], *remaining]
    importlib.import_module(f"tuning.{module_name}").main()


if __name__ == "__main__":
    main()
