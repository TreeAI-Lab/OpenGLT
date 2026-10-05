import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.hubgt import HubGT
from example.uni_runner import run_uni_baseline

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def build_model(cfg, dataset_info):
    return HubGT(
        nhid=int(cfg.hubgt.hidden_size),
        nout=int(dataset_info['output_dim']),
        n_layers=int(cfg.hubgt.num_layers),
        num_heads=int(cfg.hubgt.num_heads),
        dataset_type=dataset_info['type'],
        dim_node_deg=dataset_info.get('dim_node_deg', 0),
        dim_node_label=dataset_info.get('dim_node_label', 0),
        dim_node_attr=dataset_info.get('dim_node_attr', 0),
        dropout=float(cfg.hubgt.dropout),
        ffn_dim=int(cfg.hubgt.ffn_dim),
        dp_attn=float(cfg.hubgt.dp_attn),
        dp_bias=float(cfg.hubgt.dp_bias),
        pooling=str(cfg.hubgt.pooling),
    )


def run_hubgt(cfg=None, key_metric_override=None, config_string=None,
              prune_callback=None, evaluate_test=True):
    return run_uni_baseline(
        'HubGT',
        build_model,
        cfg=cfg,
        key_metric_override=key_metric_override,
        config_string=config_string,
        prune_callback=prune_callback,
        evaluate_test=evaluate_test,
    )


def main():
    return run_hubgt()


if __name__ == '__main__':
    main()
