import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from example.uni_runner import run_uni_baseline
from models.mvgrl import MVGRL

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def build_model(cfg, dataset_info):
    return MVGRL(
        nhid=int(cfg.mvgrl.hidden_size),
        nout=int(dataset_info['output_dim']),
        num_layers=int(cfg.mvgrl.num_layers),
        dataset_type=dataset_info['type'],
        dim_node_deg=dataset_info.get('dim_node_deg', 0),
        dim_node_label=dataset_info.get('dim_node_label', 0),
        dim_node_attr=dataset_info.get('dim_node_attr', 0),
        dropout=float(cfg.mvgrl.dropout),
        alpha=float(cfg.mvgrl.alpha),
    )


def run_mvgrl(cfg=None, key_metric_override=None, config_string=None,
              prune_callback=None, evaluate_test=True):
    return run_uni_baseline(
        'MVGRL',
        build_model,
        cfg=cfg,
        key_metric_override=key_metric_override,
        config_string=config_string,
        prune_callback=prune_callback,
        evaluate_test=evaluate_test,
    )


def main():
    return run_mvgrl()


if __name__ == '__main__':
    main()
