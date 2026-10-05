import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from example.uni_runner import run_uni_baseline
from models.rgc import RGC


def build_model(cfg, dataset_info):
    return RGC(
        nhid=int(cfg.rgc.nhid),
        nout=int(dataset_info['output_dim']),
        dataset_type=dataset_info['type'],
        dim_node_deg=dataset_info.get('dim_node_deg', 0),
        dim_node_label=dataset_info.get('dim_node_label', 0),
        dim_node_attr=dataset_info.get('dim_node_attr', 0),
        backbone=str(cfg.rgc.backbone),
        num_layers=int(cfg.rgc.num_layers),
        hidden_features=int(cfg.rgc.hidden_features),
        embed_features=int(cfg.rgc.embed_features),
        n_heads=int(cfg.rgc.n_heads),
        drop_edge=float(cfg.rgc.drop_edge),
        drop_node=float(cfg.rgc.drop_node),
        struct_dim=int(cfg.rgc.struct_dim),
        temperature=float(cfg.rgc.temperature),
        pooling=str(cfg.rgc.pooling),
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
    aux_loss = model.latest_aux_loss
    if aux_loss is not None and float(cfg.rgc.aux_weight) > 0:
        return float(cfg.rgc.aux_weight) * aux_loss
    return None


def run_rgc(cfg=None, key_metric_override=None, config_string=None,
            prune_callback=None, evaluate_test=True):
    return run_uni_baseline(
        'RGC',
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
    return run_rgc()


if __name__ == '__main__':
    main()
