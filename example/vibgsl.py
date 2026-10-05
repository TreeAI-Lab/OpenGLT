import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from example.uni_runner import run_uni_baseline
from models.vibgsl import VIBGSL


def build_model(cfg, dataset_info):
    return VIBGSL(
        nhid=int(cfg.vibgsl.nhid),
        nout=int(dataset_info['output_dim']),
        dataset_type=dataset_info['type'],
        dim_node_deg=dataset_info.get('dim_node_deg', 0),
        dim_node_label=dataset_info.get('dim_node_label', 0),
        dim_node_attr=dataset_info.get('dim_node_attr', 0),
        backbone=str(cfg.vibgsl.backbone),
        graph_type=str(cfg.vibgsl.graph_type),
        graph_metric_type=str(cfg.vibgsl.graph_metric_type),
        repar=bool(cfg.vibgsl.repar),
        hidden_dim=int(cfg.vibgsl.hidden_dim),
        ib_size=int(cfg.vibgsl.ib_size),
        num_per=int(cfg.vibgsl.num_per),
        feature_denoise=bool(cfg.vibgsl.feature_denoise),
        top_k=int(cfg.vibgsl.top_k),
        epsilon=float(cfg.vibgsl.epsilon),
        graph_skip_conn=float(cfg.vibgsl.graph_skip_conn),
        graph_include_self=bool(cfg.vibgsl.graph_include_self),
        beta=float(cfg.vibgsl.beta),
        dropout=float(cfg.vibgsl.dropout),
    )


def model_forward(model, data, dataset_type):
    if dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
        x = data.x.long() if data.x.dtype == torch.float else data.x
    else:
        x = data.x.float() if data.x.dtype == torch.long else data.x
    return model(
        x,
        data.edge_index,
        data.batch,
        edge_attr=getattr(data, 'edge_attr', None),
    )


def extra_loss(model, data, out, emb, supervised_loss, cfg, dataset_type):
    kl_loss = model.latest_kl
    if kl_loss is None:
        raise FloatingPointError("VIBGSL did not produce a KL term.")
    return float(cfg.vibgsl.beta) * kl_loss


def run_vibgsl(cfg=None, key_metric_override=None, config_string=None,
               prune_callback=None, evaluate_test=True):
    return run_uni_baseline(
        'VIBGSL',
        build_model,
        cfg=cfg,
        key_metric_override=key_metric_override,
        config_string=config_string,
        prune_callback=prune_callback,
        model_forward_fn=model_forward,
        extra_loss_fn=extra_loss,
        evaluate_test=evaluate_test,
    )


def main():
    return run_vibgsl()


if __name__ == '__main__':
    main()
