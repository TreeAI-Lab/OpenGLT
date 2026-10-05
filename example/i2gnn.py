import copy
import datetime
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import init_cfg, parse_args, update_cfg
from datasets import (
    load_dataset,
    base_dataset_name,
    dataset_cache_tag,
    dataset_loader_kwargs,
)
from engine import evaluate, get_criterion, seed_everything
from example.main_baselines import evaluate_cardinality_outputs, is_cuda_device, setup_logger, stable_sigmoid
from models.i2gnn import CountI2GNN, DataLoader, GenericI2GNN, OGBMolI2GNN, ZincI2GNN
from models.i2gnn.batch import Batch as I2Batch
from models.i2gnn.data import (
    build_i2gnn_split,
    default_i2gnn_h,
    i2gnn_dataset_type,
    transform_i2gnn_data,
)
from efficiency.runtime import (
    benchmark_session_if_enabled,
    efficiency_enabled,
    finish_preprocessing,
    save_efficiency_checkpoint_if_enabled,
)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
I2GNN_CACHE_DIR = DATA_DIR / 'i2gnn_cache'


def i2gnn_cache_root(cfg):
    """Resolve a cache shared by trials in one job or a persistent namespace."""
    if efficiency_enabled(cfg) and str(getattr(cfg.efficiency, 'cache_root', '')):
        return Path(str(cfg.efficiency.cache_root)) / 'i2gnn'

    policy = os.environ.get('OPENGLT_SUBGRAPH_CACHE_POLICY', 'persistent').lower()
    if policy != 'job':
        return I2GNN_CACHE_DIR
    job_id = os.environ.get('OPENGLT_SUBGRAPH_CACHE_JOB_ID', '')
    if not job_id or re.fullmatch(r'[A-Za-z0-9._-]+', job_id) is None:
        raise RuntimeError(
            'Job-scoped I2GNN cache requires a safe, non-empty '
            'OPENGLT_SUBGRAPH_CACHE_JOB_ID.'
        )
    return I2GNN_CACHE_DIR / 'jobs' / job_id


def supported_dataset_type(dataset):
    dataset = base_dataset_name(dataset)
    dataset_lower = dataset.lower()
    supported_ogb = {'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molpcba', 'ogbg-moltox21'}
    supported_tu = {
        'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB',
        'MUTAG', 'NCI1', 'DD', 'ENZYMES', 'PROTEINS',
    }
    supported_lrgb = {'peptides-func', 'peptides-struct'}
    supported_pyg_mol = {'zinc', 'aqsol'}
    supported_gcare = {'gcare-aids', 'gcare-yago'}

    if dataset in supported_ogb:
        return 'ogb'
    if dataset in supported_tu:
        return 'tu'
    if dataset_lower in supported_lrgb:
        return 'lrgb'
    if dataset_lower in supported_pyg_mol:
        return 'pyg_mol'
    if dataset_lower in supported_gcare or dataset_lower.startswith('gnce-'):
        return 'tu'
    if dataset.startswith('Count-'):
        return 'counting'
    raise ValueError(
        "I2GNN runner supports TU, OGB molecular datasets, Count-* datasets, "
        "zinc, aqsol, peptides-func, and peptides-struct."
    )


def set_loss_for_dataset(cfg, dataset_type):
    dataset = base_dataset_name(cfg.dataset)
    dataset_lower = dataset.lower()
    if dataset in {'ogbg-molhiv', 'ogbg-molbace', 'IMDB-BINARY', 'REDDIT-BINARY', 'MUTAG', 'NCI1', 'DD', 'PROTEINS'}:
        cfg.train.loss_type = 'BCE'
    elif dataset in {'IMDB-MULTI', 'ENZYMES', 'COLLAB'}:
        cfg.train.loss_type = 'CE'
    elif dataset in {'ogbg-molpcba', 'ogbg-moltox21'} or dataset_lower == 'peptides-func':
        cfg.train.loss_type = 'BCE'
    elif dataset_lower in {'peptides-struct', 'zinc', 'aqsol', 'gcare-aids', 'gcare-yago'} or dataset_lower.startswith('gnce-'):
        cfg.train.loss_type = 'L1'
    elif dataset_type == 'counting':
        cfg.train.loss_type = 'L1'
    else:
        raise ValueError(f"Unsupported I2GNN dataset type: {dataset_type}")


def validate_i2gnn_cfg(cfg, dataset_type):
    if dataset_type == 'counting' and cfg.i2gnn.rni:
        raise ValueError("I2GNN counting model does not support i2gnn.rni=True in this implementation.")


def metric_higher_better(key_metric):
    if key_metric in ('acc', 'auc', 'auc_macro', 'auc_micro', 'ap', 'ap_macro', 'ap_micro', 'r2'):
        return True
    if key_metric in ('mae', 'mre', 'mse', 'rmse', 'medq', 'p90q', 'p95q', 'p99q', 'meanq', 'maxq', 'log_mae'):
        return False
    raise ValueError(f"Check key_metric: {key_metric}")


def make_scheduler(cfg, optimizer, key_metric):
    scheduler_name = str(cfg.train.scheduler).lower() if cfg.train.scheduler is not None else 'none'
    if scheduler_name == 'steplr':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.train.lr_patience, gamma=cfg.train.lr_decay
        ), scheduler_name
    if scheduler_name == 'reducelr':
        mode = 'max' if metric_higher_better(key_metric) else 'min'
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=cfg.train.factor_R,
            patience=cfg.train.lr_patience_R, min_lr=1e-6
        ), scheduler_name
    if scheduler_name == 'cosrestart':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.train.t_0, T_mult=cfg.train.t_mult,
            eta_min=cfg.train.eta_min
        ), scheduler_name
    if scheduler_name in ('none', 'null', ''):
        return None, scheduler_name
    raise ValueError(f"Unknown scheduler: {cfg.train.scheduler}")


def i2gnn_h(cfg):
    return (
        int(cfg.i2gnn.h)
        if int(cfg.i2gnn.h) > 0
        else default_i2gnn_h(base_dataset_name(cfg.dataset))
    )


def resolve_i2gnn_use_rd(cfg, dataset_type):
    if cfg.i2gnn.use_rd_auto:
        return dataset_type in {'ogb', 'pyg_mol'}
    return bool(cfg.i2gnn.use_rd)


def i2gnn_max_nodes_per_hop(cfg):
    value = int(cfg.i2gnn.max_nodes_per_hop)
    return value if value > 0 else None


def prepare_i2gnn_folds(folds_data, cfg, dataset_type, logger, include_test=True):
    h = i2gnn_h(cfg)
    max_nodes_per_hop = i2gnn_max_nodes_per_hop(cfg)
    use_rd = resolve_i2gnn_use_rd(cfg, dataset_type)
    logger.info(
        f"I2GNN preprocessing: h={h}, node_label={cfg.i2gnn.node_label}, "
        f"use_rd={use_rd}, max_nodes_per_hop={max_nodes_per_hop}"
    )

    cache_dir = str(i2gnn_cache_root(cfg))
    prepared = []
    for fold_id, (train_d, val_d, test_d) in enumerate(folds_data):
        split_suffix = f"fold{fold_id}_{dataset_cache_tag(cfg)}"
        train_i2 = build_i2gnn_split(
            train_d, cfg.dataset, f"{split_suffix}_train", cache_dir,
            h=h, node_label=cfg.i2gnn.node_label, use_rd=use_rd,
            max_nodes_per_hop=max_nodes_per_hop, center_idx=True,
            self_loop=cfg.i2gnn.self_loop, force_reload=cfg.i2gnn.force_reload,
        )
        val_i2 = build_i2gnn_split(
            val_d, cfg.dataset, f"{split_suffix}_val", cache_dir,
            h=h, node_label=cfg.i2gnn.node_label, use_rd=use_rd,
            max_nodes_per_hop=max_nodes_per_hop, center_idx=True,
            self_loop=cfg.i2gnn.self_loop, force_reload=cfg.i2gnn.force_reload,
        )
        test_i2 = (
            build_i2gnn_split(
                test_d, cfg.dataset, f"{split_suffix}_test", cache_dir,
                h=h, node_label=cfg.i2gnn.node_label, use_rd=use_rd,
                max_nodes_per_hop=max_nodes_per_hop, center_idx=True,
                self_loop=cfg.i2gnn.self_loop, force_reload=cfg.i2gnn.force_reload,
            )
            if include_test else None
        )
        prepared.append((train_i2, val_i2, test_i2))
    return prepared


def infer_feature_dims(dataset):
    data = dataset[0]
    if data.x is None:
        node_input_dim = 1
    elif data.x.dim() == 1:
        node_input_dim = 1
    else:
        node_input_dim = data.x.size(-1)

    if data.edge_attr is None:
        edge_input_dim = 0
    elif data.edge_attr.dim() == 1:
        edge_input_dim = 1
    else:
        edge_input_dim = data.edge_attr.size(-1)
    return node_input_dim, edge_input_dim


def build_model(cfg, dataset_info, train_dataset):
    dataset_type = dataset_info['type']
    use_rd = resolve_i2gnn_use_rd(cfg, dataset_type)
    if dataset_type in {'ogb', 'lrgb'}:
        encoder_dataset = cfg.dataset if dataset_type == 'ogb' else 'ogbg-molhiv'
        return OGBMolI2GNN(
            encoder_dataset,
            'n2gnn',
            dataset_info['output_dim'],
            num_layer=cfg.model.num_layers,
            emb_dim=cfg.model.hidden_size,
            gnn_type=cfg.i2gnn.gnn_type,
            virtual_node=cfg.i2gnn.virtual_node,
            residual=cfg.i2gnn.residual,
            drop_ratio=cfg.i2gnn.drop_ratio,
            graph_pooling=cfg.i2gnn.graph_pooling,
            subgraph_pooling=cfg.i2gnn.subgraph2_pooling,
            use_pooling_nn=cfg.i2gnn.use_pooling_nn,
            use_rd=use_rd,
            gate=cfg.i2gnn.gate,
            RNI=cfg.i2gnn.rni,
        ).to(cfg.device)

    if dataset_type == 'counting':
        return CountI2GNN(
            train_dataset,
            num_layers=cfg.model.num_layers,
            subgraph_pooling=cfg.i2gnn.subgraph_pooling,
            use_rd=use_rd,
            RNI=cfg.i2gnn.rni,
            y_ndim=1,
        ).to(cfg.device)

    if dataset_type == 'pyg_mol':
        return ZincI2GNN(
            train_dataset,
            num_layers=cfg.model.num_layers,
            subgraph_pooling=cfg.i2gnn.subgraph_pooling,
            subgraph2_pooling=cfg.i2gnn.subgraph2_pooling,
            use_pooling_nn=cfg.i2gnn.use_pooling_nn,
            use_virtual_node=cfg.i2gnn.virtual_node,
            use_rd=use_rd,
            RNI=cfg.i2gnn.rni,
            drop_ratio=cfg.i2gnn.drop_ratio,
            double_pooling=cfg.i2gnn.double_pooling,
            gate=cfg.i2gnn.gate,
        ).to(cfg.device)

    if dataset_type == 'tu':
        node_input_dim, edge_input_dim = infer_feature_dims(train_dataset)
        return GenericI2GNN(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            output_dim=dataset_info['output_dim'],
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.i2gnn.drop_ratio,
            subgraph_pooling=cfg.i2gnn.subgraph_pooling,
            subgraph2_pooling=cfg.i2gnn.subgraph2_pooling,
            graph_pooling=cfg.i2gnn.graph_pooling,
            use_rd=use_rd,
            use_pooling_nn=cfg.i2gnn.use_pooling_nn,
            gate=cfg.i2gnn.gate,
            residual=cfg.i2gnn.residual,
        ).to(cfg.device)

    raise ValueError(f"Unsupported I2GNN dataset type: {dataset_type}")


def train_epoch(train_loader, model, criterion, optimizer, device, cfg, dataset_type):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    track_cuda = is_cuda_device(device)
    if track_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.time()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        y = data.y
        out = model(data)
        if not torch.isfinite(out).all():
            raise FloatingPointError("Non-finite I2GNN output during training.")

        if cfg.train.loss_type in ('L1', 'MSE'):
            y_target = y.float()
            out_for_loss = out
            if y_target.shape != out_for_loss.shape:
                y_target = y_target.view_as(out_for_loss)
            finite_mask = torch.isfinite(y_target)
            if finite_mask.sum() == 0:
                continue
            loss = criterion(out_for_loss[finite_mask], y_target[finite_mask])
            num_graphs = y.size(0)
        elif cfg.train.loss_type == 'CE':
            y_target = y.squeeze().long()
            loss = criterion(out, y_target)
            num_graphs = y.size(0)
        else:
            y_target = y.float()
            mask = ~torch.isnan(y_target)
            if mask.sum() == 0:
                continue
            if out.size(-1) == 1:
                mask_1d = mask.squeeze(-1) if mask.dim() > 1 else mask
                loss = criterion(out.squeeze(-1)[mask_1d], y_target.squeeze(-1)[mask_1d])
            else:
                loss = criterion(out[mask], y_target[mask])
            num_graphs = y.size(0)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite I2GNN loss: {loss.item()}")

        loss.backward()
        if dataset_type != 'counting':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item()) * num_graphs
        total_graphs += num_graphs

    t_end = time.time()
    mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if track_cuda else 0
    mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2) if track_cuda else 0
    avg_loss = total_loss / max(total_graphs, 1)
    return avg_loss, t_end - t_start, mem_allocated, mem_reserved


@torch.no_grad()
def eval_epoch(loader, model, device, dataset_task='b', dataset_type='ogb'):
    model.eval()
    y_preds = []
    y_trues = []
    y_cards = []
    y_means = []
    y_stds = []
    t_forward_sum = 0.0
    track_cuda = is_cuda_device(device)
    if track_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    for data in loader:
        data = data.to(device)
        if track_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        out = model(data)
        if not torch.isfinite(out).all():
            raise FloatingPointError("Non-finite I2GNN output during evaluation.")
        if track_cuda:
            torch.cuda.synchronize()
        t_forward_sum += time.time() - t0
        y_preds.append(out.detach().cpu())
        y_trues.append(data.y.detach().cpu())
        if dataset_task == 'card':
            y_cards.append(data.y_card.detach().cpu())
            y_means.append(data.y_mean.detach().cpu())
            y_stds.append(data.y_std.detach().cpu())

    mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if track_cuda else 0
    mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2) if track_cuda else 0

    y_preds = torch.cat(y_preds, dim=0).numpy()
    y_trues = torch.cat(y_trues, dim=0).numpy()
    if not np.isfinite(y_preds).all():
        raise FloatingPointError("Non-finite I2GNN raw predictions during evaluation.")

    if dataset_task == 'reg':
        if y_preds.ndim > 1 and y_preds.shape[1] == 1:
            y_preds = y_preds.squeeze(1)
        if y_trues.ndim > 1 and y_trues.shape[1] == 1:
            y_trues = y_trues.squeeze(1)
        results = evaluate(y_trues, y_preds, task='reg')
    elif dataset_task == 'card':
        y_cards = torch.cat(y_cards, dim=0).numpy()
        y_means = torch.cat(y_means, dim=0).numpy()
        y_stds = torch.cat(y_stds, dim=0).numpy()
        results = evaluate_cardinality_outputs(y_preds, y_cards, y_means, y_stds)
    elif dataset_task == 'mc':
        y_preds_prob = torch.softmax(torch.tensor(y_preds), dim=1).numpy()
        results = evaluate(y_trues, y_preds_prob, task=dataset_task)
    else:
        if y_preds.ndim > 1 and y_preds.shape[1] == 1:
            y_preds = y_preds.squeeze(1)
        y_preds_prob = stable_sigmoid(y_preds)
        results = evaluate(y_trues, y_preds_prob, task=dataset_task)

    return results, t_forward_sum, mem_allocated, mem_reserved


def run_single_session(session_id, train_loader, val_loader, test_loader,
                       dataset_info, cfg, logger, prune_callback=None,
                       prune_step_offset=0, evaluate_test=True,
                       efficiency_raw_test=None, efficiency_transform_fn=None):
    model = build_model(cfg, dataset_info, train_loader.dataset)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    num_trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    logger.info(
        f"Model parameters | total: {num_parameters:,}, "
        f"trainable: {num_trainable_parameters:,}"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
    criterion = get_criterion(cfg.train.loss_type).to(cfg.device)
    key_metric = dataset_info['key_metric']
    scheduler, scheduler_name = make_scheduler(cfg, optimizer, key_metric)
    higher_better = metric_higher_better(key_metric)
    best_val_metric = -float('inf') if higher_better else float('inf')
    best_val_perf = {}
    best_model_state = None
    patience_count = 0
    epoch_count = 0
    best_epoch = 0
    total_train_time = 0.0
    cycle_times = []
    max_mem_train = 0.0
    max_mem_train_res = 0.0

    logger.info(f"Start Session {session_id} | Key Metric: {key_metric}")

    for epoch in range(1, cfg.train.epochs + 1):
        if patience_count >= cfg.train.patience:
            logger.info(f"Session {session_id}: Early stopping at epoch {epoch}.")
            break

        epoch_count += 1
        train_loss, t_train, mem_tr_alloc, mem_tr_res = train_epoch(
            train_loader, model, criterion, optimizer, cfg.device, cfg, dataset_info['type']
        )
        total_train_time += t_train
        max_mem_train = max(max_mem_train, mem_tr_alloc)
        max_mem_train_res = max(max_mem_train_res, mem_tr_res)

        val_perf, t_val, mem_val_alloc, mem_val_res = eval_epoch(
            val_loader, model, cfg.device,
            dataset_task=dataset_info['task'],
            dataset_type=dataset_info['type'],
        )
        cycle_times.append(t_train + t_val)
        if key_metric not in val_perf:
            raise KeyError(f"Key metric '{key_metric}' not found. Available: {sorted(val_perf)}")

        current_val = val_perf[key_metric]
        if scheduler_name == 'reducelr':
            scheduler.step(current_val)
        elif scheduler is not None:
            scheduler.step()

        improved = current_val > best_val_metric if higher_better else current_val < best_val_metric
        if improved:
            best_val_metric = current_val
            best_val_perf = copy.deepcopy(val_perf)
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1

        if prune_callback is not None:
            prune_callback(current_val, prune_step_offset + epoch)

        if epoch == 1 or epoch % 10 == 0 or patience_count >= cfg.train.patience:
            logger.info(
                f"  Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                f"Val {key_metric}: {current_val:.4f} (Best: {best_val_metric:.4f}) | "
                f"Train | time: {t_train:.2f} s, mem_allocated: {mem_tr_alloc:.2f} MB, "
                f"mem_reserved: {mem_tr_res:.2f} MB | "
                f"Val-infer | time: {t_val:.2f} s, mem_allocated: {mem_val_alloc:.2f} MB, "
                f"mem_reserved: {mem_val_res:.2f} MB"
            )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(
            f"Session {session_id}: Loading best model for "
            f"{'final testing' if evaluate_test else 'validation-only trial completion'}..."
        )
    else:
        logger.warning(f"Session {session_id}: No best model saved! Using last epoch model.")

    if evaluate_test:
        test_perf, t_test, mem_test_alloc, mem_test_res = eval_epoch(
            test_loader, model, cfg.device,
            dataset_task=dataset_info['task'],
            dataset_type=dataset_info['type'],
        )
    else:
        test_perf = {}
        t_test = mem_test_alloc = mem_test_res = 0.0

    steady_cycles = cycle_times[int(cfg.efficiency.timed_train_warmup_epochs):] if efficiency_enabled(cfg) else []
    training_efficiency = {
        'cycle_times_s': cycle_times,
        'steady_cycle_median_s': float(np.median(steady_cycles)) if steady_cycles else None,
        'epochs_executed': epoch_count,
        'best_epoch': best_epoch,
        'train_peak_allocated_mb': max(max_mem_train, mem_val_alloc),
        'train_peak_reserved_mb': max(max_mem_train_res, mem_val_res),
    }
    strict_result = None
    if evaluate_test and efficiency_enabled(cfg, 'strict'):
        strict_result = benchmark_session_if_enabled(
            cfg,
            model,
            test_loader.dataset,
            efficiency_raw_test,
            lambda current_model, data: current_model(data),
            session_id=session_id,
            transform_fn=efficiency_transform_fn,
            collate_fn=lambda data: I2Batch.from_data_list([data]),
            throughput_loader=test_loader,
            extra={'training': training_efficiency},
        )

    checkpoint_path = save_efficiency_checkpoint_if_enabled(
        cfg, model, session_id=session_id, best_epoch=best_epoch,
    )

    del model, optimizer, scheduler, criterion, best_model_state
    if is_cuda_device(cfg.device):
        torch.cuda.empty_cache()

    def dict_to_str(d):
        return ", ".join([f"{k}: {v:.4f}" for k, v in d.items()])

    test_line = (
        f"  Test Metrics: {dict_to_str(test_perf)}\n"
        f"  Test-Infer | Time: {t_test:.2f} s, Mem Alloc: {mem_test_alloc:.2f} MB, "
        f"Res: {mem_test_res:.2f} MB\n"
        if evaluate_test else
        "  Test Metrics: skipped (validation-only tuning trial)\n"
    )
    logger.info(
        test_line
        + f"  Best Val Metrics: {dict_to_str(best_val_perf)}\n"
        + f"  Train | Total Time: {total_train_time:.2f}s, Epochs: {epoch_count}, "
        f"Mem Alloc: {max_mem_train:.2f} MB, Res: {max_mem_train_res:.2f} MB\n"
    )

    return {
        'test_perf': test_perf,
        'best_val_perf': best_val_perf,
        'train_time': total_train_time,
        'epoch_count': epoch_count,
        'best_epoch': best_epoch,
        'test_time': t_test,
        'train_mem_alloc': max_mem_train,
        'train_mem_res': max_mem_train_res,
        'test_mem_alloc': mem_test_alloc,
        'test_mem_res': mem_test_res,
        'num_parameters': num_parameters,
        'num_trainable_parameters': num_trainable_parameters,
        'strict_efficiency': strict_result,
        'training_efficiency': training_efficiency,
        'checkpoint_path': checkpoint_path,
    }


def build_dataset_info(dataset, dataset_type):
    return {
        'type': dataset_type,
        'output_dim': dataset.output_dim,
        'task': getattr(dataset, 'task', 'b'),
        'key_metric': getattr(dataset, 'key_metric', 'acc'),
    }


def run_i2gnn(cfg=None, key_metric_override=None, config_string=None,
              prune_callback=None, evaluate_test=True):
    args = None
    if cfg is None:
        args = parse_args()
        cfg = init_cfg()
        cfg = update_cfg(cfg, args=args)

    dataset_type = supported_dataset_type(cfg.dataset)
    set_loss_for_dataset(cfg, dataset_type)
    validate_i2gnn_cfg(cfg, dataset_type)

    if config_string is None:
        config_string = f"{cfg.dataset}_I2GNN_{int(time.time())}"
    logger = setup_logger(cfg, config_string)
    logger.info(f"Config:\n{cfg}\n")
    logger.info(f'Start Time: {datetime.datetime.now().strftime("%Y/%m/%d - %H:%M")}\n')

    prep_started = time.perf_counter()
    efficiency_data_root = str(getattr(cfg.efficiency, 'data_root', '')) if efficiency_enabled(cfg) else ''
    dataset, folds_data, _ = load_dataset(
            dataset_name=cfg.dataset,
            root=efficiency_data_root or str(DATA_DIR),
            use_fingerprints=False,
            use_rwpe=False,
            augm_gca=False,
            fold_k=cfg.cv.fold_k,
            fold_seed=cfg.cv.seed,
            gnn_type=None,
            **dataset_loader_kwargs(cfg),
    )
    raw_folds_data = folds_data
    dataset_info = build_dataset_info(dataset, dataset_type)
    if key_metric_override is not None:
        dataset_info['key_metric'] = key_metric_override
        logger.info(f"Using key_metric override: {key_metric_override}")

    # Subgraph extraction may call random.sample when max_nodes_per_hop is
    # active. Seed preprocessing explicitly so repeated runs are reproducible
    # and the three model seeds differ only in training.
    seed_everything(
        cfg.seed,
        deterministic=cfg.deterministic,
        warn_only=cfg.deterministic_warn_only,
    )
    folds_data = prepare_i2gnn_folds(
        folds_data, cfg, dataset_type, logger,
        include_test=(evaluate_test or efficiency_enabled(cfg)),
    )

    if efficiency_enabled(cfg, 'prep'):
        result = finish_preprocessing(
            cfg, prep_started, folds_data, raw_folds_data,
            extra={
                'transform': 'I2GNN nested-subgraph extraction',
                'cache_root': str(getattr(cfg.efficiency, 'cache_root', '')),
            },
        )
        return result

    is_gnce = cfg.dataset.lower().startswith('gnce-')
    use_fold_sessions = dataset_type == 'tu' and not is_gnce

    if use_fold_sessions:
        num_sessions = len(folds_data)
        cv_dataset = 'G-CARE' if cfg.dataset.lower().startswith('gcare-') else 'TU'
        logger.info(f"{cv_dataset} Dataset (K-Fold = {num_sessions})")
    elif is_gnce:
        num_sessions = cfg.train.runs
        logger.info(f"GNCE Dataset (Repeated Runs = {num_sessions})")
    else:
        num_sessions = cfg.train.runs
        logger.info(f"{dataset_type.upper()} Dataset (Repeated Runs = {num_sessions})")

    session_indices = list(range(num_sessions))
    if efficiency_enabled(cfg):
        selected = int(cfg.efficiency.fold_index) if use_fold_sessions else 0
        if selected < 0 or selected >= num_sessions:
            raise IndexError(f'efficiency.fold_index={selected} outside 0..{num_sessions - 1}')
        session_indices = [selected]

    stats_collector = {
        'test_perf': {},
        'best_val_perf': {},
        'train_time': [],
        'epoch_count': [],
        'best_epoch': [],
        'test_time': [],
        'train_mem_alloc': [],
        'train_mem_res': [],
        'test_mem_alloc': [],
        'test_mem_res': [],
        'num_parameters': [],
        'num_trainable_parameters': [],
        'training_efficiency': [],
    }

    h = i2gnn_h(cfg)
    max_nodes_per_hop = i2gnn_max_nodes_per_hop(cfg)
    use_rd = resolve_i2gnn_use_rd(cfg, dataset_type)
    transform_for_efficiency = lambda data: transform_i2gnn_data(
        data,
        i2gnn_dataset_type(cfg.dataset),
        h=h,
        node_label=cfg.i2gnn.node_label,
        use_rd=use_rd,
        max_nodes_per_hop=max_nodes_per_hop,
        center_idx=True,
        self_loop=cfg.i2gnn.self_loop,
    )

    for session_id in session_indices:
        if use_fold_sessions:
            train_d, val_d, test_d = folds_data[session_id]
            current_seed = cfg.seed
        else:
            train_d, val_d, test_d = folds_data[0]
            current_seed = cfg.seed + session_id
        seed_everything(
            current_seed,
            deterministic=cfg.deterministic,
            warn_only=cfg.deterministic_warn_only,
        )
        train_loader = DataLoader(
            train_d, cfg.train.batch_size, shuffle=True,
            num_workers=cfg.num_workers, drop_last=False
        )
        val_loader = DataLoader(
            val_d, cfg.train.batch_size, shuffle=False,
            num_workers=cfg.num_workers
        )
        test_loader = (
            DataLoader(
                test_d, cfg.train.batch_size, shuffle=False,
                num_workers=cfg.num_workers
            )
            if evaluate_test else None
        )

        results_run = run_single_session(
            session_id + 1, train_loader, val_loader, test_loader,
            dataset_info, cfg, logger,
            prune_callback=prune_callback,
            prune_step_offset=session_id * cfg.train.epochs,
            evaluate_test=evaluate_test,
            efficiency_raw_test=raw_folds_data[session_id if use_fold_sessions else 0][2],
            efficiency_transform_fn=transform_for_efficiency,
        )
        stats_collector['training_efficiency'].append(results_run['training_efficiency'])

        for key in [
            'train_time', 'epoch_count', 'best_epoch', 'test_time',
            'train_mem_alloc', 'train_mem_res',
            'test_mem_alloc', 'test_mem_res',
            'num_parameters', 'num_trainable_parameters',
        ]:
            stats_collector[key].append(results_run[key])
        for m, v in results_run['test_perf'].items():
            stats_collector['test_perf'].setdefault(m, []).append(v)
        for m, v in results_run['best_val_perf'].items():
            stats_collector['best_val_perf'].setdefault(m, []).append(v)

    logger.info("\n" + "=" * 60)
    measured_sessions = len(session_indices)
    logger.info(f"Final Aggregated Results ({measured_sessions} sessions)")
    logger.info("=" * 60)

    def agg_values(values):
        v = torch.tensor(values).float()
        return {'mean': float(v.mean().item()), 'std': float(v.std(unbiased=False).item())}

    aggregate = {
        'test_perf': {m: agg_values(v) for m, v in stats_collector['test_perf'].items()},
        'best_val_perf': {m: agg_values(v) for m, v in stats_collector['best_val_perf'].items()},
        'efficiency': {
            'train_time': agg_values(stats_collector['train_time']),
            'epoch_count': agg_values(stats_collector['epoch_count']),
            'best_epoch': agg_values(stats_collector['best_epoch']),
            'test_time': agg_values(stats_collector['test_time']),
            'train_mem_alloc': agg_values(stats_collector['train_mem_alloc']),
            'train_mem_res': agg_values(stats_collector['train_mem_res']),
            'test_mem_alloc': agg_values(stats_collector['test_mem_alloc']),
            'test_mem_res': agg_values(stats_collector['test_mem_res']),
            'num_parameters': agg_values(stats_collector['num_parameters']),
            'num_trainable_parameters': agg_values(stats_collector['num_trainable_parameters']),
        },
    }

    for section, metrics in [('Test Effectiveness', aggregate['test_perf']), ('Best Val', aggregate['best_val_perf'])]:
        logger.info(f"[{section}]")
        for m, stat in metrics.items():
            logger.info(f"{m.upper()}: {stat['mean']:.4f} +/- {stat['std']:.4f}")
    logger.info("[Efficiency]")
    for m, stat in aggregate['efficiency'].items():
        logger.info(f"{m}: {stat['mean']:.4f} +/- {stat['std']:.4f}")
    logger.info("=" * 60)

    for handler in list(logger.handlers):
        handler.flush()
        logger.removeHandler(handler)
        if isinstance(handler, logging.FileHandler):
            handler.close()

    return {
        'dataset_info': dataset_info,
        'num_sessions': measured_sessions,
        'raw': stats_collector,
        'aggregate': aggregate,
        'best_val_perf': aggregate['best_val_perf'],
        'test_perf': aggregate['test_perf'],
        'efficiency': aggregate['efficiency'],
    }


def main():
    return run_i2gnn()


if __name__ == '__main__':
    main()
