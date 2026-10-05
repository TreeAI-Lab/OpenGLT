import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.nagphormer import NAGphormer
from example.uni_runner import run_uni_baseline

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def build_model(cfg, dataset_info):
    ffn_dim = int(cfg.nagphormer.ffn_dim)
    return NAGphormer(
        nhid=int(cfg.nagphormer.hidden_size),
        nout=int(dataset_info['output_dim']),
        n_layers=int(cfg.nagphormer.num_layers),
        num_heads=int(cfg.nagphormer.num_heads),
        hops=int(cfg.nagphormer.hops),
        dataset_type=dataset_info['type'],
        dim_node_deg=dataset_info.get('dim_node_deg', 0),
        dim_node_label=dataset_info.get('dim_node_label', 0),
        dim_node_attr=dataset_info.get('dim_node_attr', 0),
        dropout=float(cfg.nagphormer.dropout),
        attention_dropout=float(cfg.nagphormer.attention_dropout),
        pooling=str(cfg.nagphormer.pooling),
        ffn_dim=None if ffn_dim <= 0 else ffn_dim,
    )


def run_nagphormer(cfg=None, key_metric_override=None, config_string=None,
                   prune_callback=None, evaluate_test=True):
    return run_uni_baseline(
        'NAGphormer',
        build_model,
        cfg=cfg,
        key_metric_override=key_metric_override,
        config_string=config_string,
        prune_callback=prune_callback,
        evaluate_test=evaluate_test,
    )


def main():
    return run_nagphormer()


if __name__ == '__main__':
    main()
