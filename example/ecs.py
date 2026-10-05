import time

import torch

from example.uni_runner import run_uni_baseline
from models.ecs import ECSGNN
from models.ecs.batch import ECSBatch, ECSDataLoader
from models.ecs.utils import create_ecs_data


def build_model(cfg, dataset_info):
    is_counting = dataset_info['type'] == 'counting'
    return ECSGNN(
        nhid=int(cfg.ecs.hidden_size),
        nout=int(dataset_info['output_dim']),
        num_layers=int(cfg.ecs.num_layers),
        dataset_type=dataset_info['type'],
        dim_node_deg=dataset_info.get('dim_node_deg', 0),
        dim_node_label=dataset_info.get('dim_node_label', 0),
        dim_node_attr=dataset_info.get('dim_node_attr', 0),
        num_edge_classes=dataset_info.get('num_edge_classes', 0),
        dropout=float(cfg.ecs.dropout),
        pooling=str(cfg.ecs.pooling),
        residual=bool(cfg.ecs.residual),
        virtual_node=bool(cfg.ecs.virtual_node) and not is_counting,
        use_rd=bool(cfg.ecs.use_rd),
        concat_readout=is_counting,
    )


def ecs_forward(model, data, dataset_type):
    if dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
        if data.x.dtype == torch.float:
            data.x = data.x.long()
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is not None and edge_attr.dtype == torch.float:
            data.edge_attr = edge_attr.long()
    return model(
        data.x,
        data.edge_index,
        data.batch,
        edge_attr=getattr(data, 'edge_attr', None),
        data=data,
    )


def preprocess_split(split, cfg, logger, split_name):
    t0 = time.time()
    processed = []
    for idx, data in enumerate(split):
        processed.append(
            create_ecs_data(
                data,
                h=int(cfg.ecs.h),
                node_label=str(cfg.ecs.node_label),
                use_rd=bool(cfg.ecs.use_rd),
                max_nodes_per_hop=None if int(cfg.ecs.max_nodes_per_hop) <= 0 else int(cfg.ecs.max_nodes_per_hop),
                self_loop=bool(cfg.ecs.self_loop),
            )
        )
        if (idx + 1) % 500 == 0:
            logger.info(f"  ECS preprocessing {split_name}: {idx + 1}/{len(split)} graphs")
    logger.info(f"ECS preprocessing {split_name}: {len(processed)} graphs in {time.time() - t0:.2f}s")
    return processed


def preprocess_folds(folds_data, cfg, dataset_type, logger, include_test=True):
    processed = []
    for fold_idx, (train_d, val_d, test_d) in enumerate(folds_data):
        prefix = f"fold{fold_idx + 1}" if dataset_type == 'tu' else "split"
        processed.append((
            preprocess_split(train_d, cfg, logger, f"{prefix}/train"),
            preprocess_split(val_d, cfg, logger, f"{prefix}/val"),
            preprocess_split(test_d, cfg, logger, f"{prefix}/test")
            if include_test else None,
        ))
    return processed


def efficiency_transform(cfg):
    return lambda data: create_ecs_data(
        data,
        h=int(cfg.ecs.h),
        node_label=str(cfg.ecs.node_label),
        use_rd=bool(cfg.ecs.use_rd),
        max_nodes_per_hop=(
            None if int(cfg.ecs.max_nodes_per_hop) <= 0
            else int(cfg.ecs.max_nodes_per_hop)
        ),
        self_loop=bool(cfg.ecs.self_loop),
    )


def run_ecs(cfg=None, key_metric_override=None, config_string=None,
            prune_callback=None, evaluate_test=True):
    return run_uni_baseline(
        'ECS',
        build_model,
        cfg=cfg,
        key_metric_override=key_metric_override,
        config_string=config_string,
        prune_callback=prune_callback,
        model_forward_fn=ecs_forward,
        preprocess_folds_fn=preprocess_folds,
        loader_cls=ECSDataLoader,
        efficiency_transform_fn=efficiency_transform(cfg),
        efficiency_collate_fn=lambda data: ECSBatch.from_data_list([data]),
        evaluate_test=evaluate_test,
    )


def main():
    return run_ecs()


if __name__ == '__main__':
    main()
