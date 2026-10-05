import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from example.uni_runner import run_uni_baseline
from models.hgpsl import HGPSL

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def build_model(cfg, dataset_info):
    return HGPSL(
        nhid=int(cfg.hgpsl.hidden_size),
        nout=int(dataset_info['output_dim']),
        dataset_type=dataset_info['type'],
        dim_node_deg=dataset_info.get('dim_node_deg', 0),
        dim_node_label=dataset_info.get('dim_node_label', 0),
        dim_node_attr=dataset_info.get('dim_node_attr', 0),
        pooling_ratio=float(cfg.hgpsl.pooling_ratio),
        dropout=float(cfg.hgpsl.dropout),
        sample_neighbor=bool(cfg.hgpsl.sample_neighbor),
        sparse_attention=bool(cfg.hgpsl.sparse_attention),
        structure_learning=bool(cfg.hgpsl.structure_learning),
        lamb=float(cfg.hgpsl.lamb),
    )


def run_hgpsl(cfg=None, key_metric_override=None, config_string=None,
              prune_callback=None, evaluate_test=True):
    return run_uni_baseline(
        'HGPSL',
        build_model,
        cfg=cfg,
        key_metric_override=key_metric_override,
        config_string=config_string,
        prune_callback=prune_callback,
        evaluate_test=evaluate_test,
    )


def main():
    return run_hgpsl()


if __name__ == '__main__':
    main()
