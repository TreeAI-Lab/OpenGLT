import copy
import datetime
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from yacs.config import CfgNode as CN

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import init_cfg, parse_args
from datasets import (
    good_loss_type,
    good_model_dataset_type,
    is_good_dataset_name,
    load_dataset,
    base_dataset_name,
    dataset_loader_kwargs,
)
from engine import evaluate, get_criterion, seed_everything
from efficiency.runtime import (
    benchmark_session_if_enabled,
    efficiency_enabled,
    finish_preprocessing,
    load_prepared_input,
    restore_dataset_metadata,
    save_prepared_input,
    dataset_metadata,
    save_efficiency_checkpoint_if_enabled,
)
from datasets.loader import add_lap_pe_stats
from example.main_baselines import evaluate_cardinality_outputs, is_cuda_device, setup_logger, stable_sigmoid
from models.graphgps import GraphGPSModel

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / 'configs'
DATA_DIR = BASE_DIR / 'data'


REQUIRED_GRAPHGPS_FIELDS = (
    'hidden_size',
    'num_layers',
    'local_gnn_type',
    'global_model_type',
    'num_heads',
    'dropout',
    'attn_dropout',
    'act_type',
    'layer_norm',
    'batch_norm',
    'readout_type',
    'head_layers',
    'use_rwpe',
    'rwpe_dim',
    'use_lappe',
    'lappe_dim',
    'lappe_max_freqs',
    'lappe_eigvec_norm',
    'lappe_laplacian_norm',
    'lappe_model',
    'lappe_layers',
    'lappe_heads',
    'lappe_raw_norm_type',
)


def ensure_graphgps_cfg(cfg):
    if not hasattr(cfg, 'graphgps'):
        cfg.graphgps = CN(new_allowed=True)
    cfg.graphgps.set_new_allowed(True)
    return cfg


def resolve_graphgps_config_path(cfg, args):
    if args is not None and args.config:
        return Path(args.config)
    return CONFIG_DIR / f'{base_dataset_name(cfg.dataset)}_GPS.yaml'


def infer_dataset_from_opts(opts, default):
    for idx, opt in enumerate(opts):
        if opt == 'dataset' and idx + 1 < len(opts):
            return opts[idx + 1]
        if opt.startswith('dataset='):
            return opt.split('=', 1)[1]
    return default


def load_graphgps_yaml(cfg, args):
    config_path = resolve_graphgps_config_path(cfg, args)
    if not config_path.exists():
        if is_good_dataset_name(cfg.dataset) and not (args and args.config):
            print(
                f"GraphGPS: {config_path} does not exist; using built-in "
                "GraphGPS defaults for the GOOD example."
            )
            return cfg, None
        raise FileNotFoundError(
            f"GraphGPS config not found: {config_path}. "
            f"Create configs/{cfg.dataset}_GPS.yaml or pass --config explicitly."
        )
    cfg.merge_from_file(str(config_path))
    return cfg, config_path


def validate_graphgps_cfg(cfg):
    missing = [key for key in REQUIRED_GRAPHGPS_FIELDS if key not in cfg.graphgps]
    if missing:
        raise KeyError(
            "Missing GraphGPS config fields in YAML: " + ", ".join(missing)
        )
    if cfg.graphgps.hidden_size % cfg.graphgps.num_heads != 0 and cfg.graphgps.global_model_type != 'None':
        raise ValueError(
            f"graphgps.hidden_size={cfg.graphgps.hidden_size} must be divisible by "
            f"graphgps.num_heads={cfg.graphgps.num_heads}."
        )
    if cfg.graphgps.use_lappe and cfg.graphgps.hidden_size <= cfg.graphgps.lappe_dim:
        raise ValueError(
            f"graphgps.hidden_size={cfg.graphgps.hidden_size} must be larger than "
            f"graphgps.lappe_dim={cfg.graphgps.lappe_dim}."
        )
    if (
        cfg.graphgps.use_lappe
        and cfg.graphgps.lappe_model == 'Transformer'
        and cfg.graphgps.lappe_dim % cfg.graphgps.lappe_heads != 0
    ):
        raise ValueError(
            f"graphgps.lappe_dim={cfg.graphgps.lappe_dim} must be divisible by "
            f"graphgps.lappe_heads={cfg.graphgps.lappe_heads} for Transformer LapPE."
        )
    if (
        cfg.graphgps.use_lappe
        and cfg.graphgps.lappe_model == 'Transformer'
        and cfg.graphgps.lappe_heads % 2 != 0
    ):
        raise ValueError(
            f"graphgps.lappe_heads={cfg.graphgps.lappe_heads} must be even for Transformer LapPE."
        )
    rwpe_encoder_type = str(
        getattr(cfg.graphgps, 'rwpe_encoder_type', 'graphgps')
    ).lower()
    if rwpe_encoder_type not in {'graphgps', 'legacy_project'}:
        raise ValueError(
            "graphgps.rwpe_encoder_type must be 'graphgps' or "
            f"'legacy_project', got {rwpe_encoder_type!r}."
        )
    if cfg.graphgps.use_rwpe and rwpe_encoder_type == 'graphgps':
        reserved = int(
            getattr(cfg.graphgps, 'rwpe_embed_dim', cfg.graphgps.rwpe_dim)
        )
        if cfg.graphgps.use_lappe:
            reserved += int(cfg.graphgps.lappe_dim)
        if int(cfg.graphgps.hidden_size) <= reserved:
            raise ValueError(
                f"graphgps.hidden_size={cfg.graphgps.hidden_size} must exceed "
                f"the total reserved PE dimension ({reserved})."
            )


def supported_dataset_type(dataset):
    dataset = base_dataset_name(dataset)
    supported_ogb = {'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molpcba', 'ogbg-moltox21'}
    supported_tu = {
        'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB',
        'MUTAG', 'NCI1', 'DD', 'ENZYMES', 'PROTEINS'
    }
    supported_lrgb = {'peptides-func', 'peptides-struct'}
    supported_pyg_mol = {'zinc', 'aqsol'}
    supported_gcare = {'gcare-aids', 'gcare-yago'}
    dataset_lower = dataset.lower()

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
    if is_good_dataset_name(dataset):
        return good_model_dataset_type(dataset)
    if dataset.startswith('Count-'):
        return 'counting'
    raise ValueError(f"Dataset {dataset} not supported by GraphGPS runner.")


def set_loss_for_dataset(cfg, dataset_type):
    dataset = base_dataset_name(cfg.dataset)
    dataset_lower = dataset.lower()
    if is_good_dataset_name(cfg.dataset):
        cfg.train.loss_type = good_loss_type(cfg.dataset)
    elif dataset in ['ogbg-molhiv', 'ogbg-molbace', 'IMDB-BINARY', 'REDDIT-BINARY', 'MUTAG', 'NCI1', 'DD', 'PROTEINS']:
        cfg.train.loss_type = 'BCE'
    elif dataset in ['IMDB-MULTI', 'ENZYMES', 'COLLAB']:
        cfg.train.loss_type = 'CE'
    elif dataset in ['ogbg-molpcba', 'ogbg-moltox21'] or dataset_lower == 'peptides-func':
        cfg.train.loss_type = 'BCE'
    elif dataset_lower in {'peptides-struct', 'zinc', 'aqsol', 'gcare-aids', 'gcare-yago'} or dataset_lower.startswith('gnce-') or dataset_type == 'counting':
        cfg.train.loss_type = 'L1'
    else:
        raise ValueError(f"Cannot infer loss_type for dataset={cfg.dataset}.")


def compute_degree_histogram(train_dataset, max_degree):
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        if data.edge_index.numel() == 0:
            continue
        num_nodes_safe = max(data.num_nodes, int(data.edge_index.max().item()) + 1)
        d = degree(data.edge_index[1], num_nodes=num_nodes_safe, dtype=torch.long)
        max_d = int(d.max())
        if max_d >= deg.size(0):
            deg = torch.cat([deg, torch.zeros(max_d - deg.size(0) + 1, dtype=torch.long)])
        deg += torch.bincount(d, minlength=deg.size(0))
    return deg


def make_scheduler(cfg, optimizer, key_metric):
    scheduler_name = str(cfg.train.scheduler).lower() if cfg.train.scheduler is not None else 'none'
    if scheduler_name == 'steplr':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.train.lr_patience, gamma=cfg.train.lr_decay
        ), scheduler_name
    if scheduler_name == 'reducelr':
        if key_metric in ('acc', 'auc', 'auc_macro', 'auc_micro', 'ap', 'ap_macro', 'ap_micro', 'r2'):
            mode = 'max'
        elif key_metric in ('mae', 'mre', 'mse', 'rmse', 'medq', 'p90q', 'p95q', 'p99q', 'meanq', 'maxq', 'log_mae'):
            mode = 'min'
        else:
            raise ValueError(f"Check key_metric for ReduceLROnPlateau: {key_metric}")
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=cfg.train.factor_R,
            patience=cfg.train.lr_patience_R, min_lr=1e-6
        ), scheduler_name
    if scheduler_name == 'cosrestart':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.train.t_0, T_mult=cfg.train.t_mult,
            eta_min=cfg.train.eta_min
        ), scheduler_name
    if scheduler_name == 'cosine_with_warmup':
        num_warmup_epochs = int(cfg.train.num_warmup_epochs)
        max_epoch = int(cfg.train.epochs)

        def lr_lambda(current_step):
            if current_step < num_warmup_epochs:
                return max(1e-6, float(current_step) / float(max(1, num_warmup_epochs)))
            progress = float(current_step - num_warmup_epochs) / float(max(1, max_epoch - num_warmup_epochs))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), scheduler_name
    if scheduler_name in ('none', 'null', ''):
        return None, scheduler_name
    raise ValueError(f"Unknown scheduler: {cfg.train.scheduler}")


def is_higher_better_metric(key_metric):
    if key_metric in ('acc', 'auc', 'auc_macro', 'auc_micro', 'ap', 'ap_macro', 'ap_micro', 'r2'):
        return True
    if key_metric in ('mae', 'mre', 'mse', 'rmse', 'medq', 'p90q', 'p95q', 'p99q', 'meanq', 'maxq', 'log_mae'):
        return False
    raise ValueError(f"Check key_metric: {key_metric}")


def get_lap_pe(data):
    if hasattr(data, 'EigVals') or hasattr(data, 'EigVecs'):
        return {
            'EigVals': getattr(data, 'EigVals', None),
            'EigVecs': getattr(data, 'EigVecs', None),
        }
    return None


def train_epoch(train_loader, model, criterion, optimizer, device, cfg, dataset_type='ogb'):
    model.train()

    total_loss = 0
    track_cuda = is_cuda_device(device)
    if track_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.time()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        if dataset_type in ('ogb', 'lrgb', 'pyg_mol', 'good_mol'):
            x = data.x.long() if data.x.dtype == torch.float else data.x
            edge_attr = data.edge_attr
            if edge_attr is not None and edge_attr.dtype == torch.float:
                edge_attr = edge_attr.long()
        else:
            x = data.x
            edge_attr = data.edge_attr

        fingerprint = getattr(data, 'fingerprint', None)
        rwpe = getattr(data, 'random_walk_pe', None)
        LapPE = get_lap_pe(data)
        y = data.y

        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue

        out, _ = model(
            x,
            data.edge_index,
            data.batch,
            edge_attr,
            fingerprint,
            rwpe=rwpe,
            LapPE=LapPE,
        )
        if not torch.isfinite(out).all():
            raise FloatingPointError("Non-finite model output during training.")

        if cfg.train.loss_type in ('L1', 'MSE'):
            y_target = y.float()
            out_for_loss = out
            if y_target.shape != out_for_loss.shape:
                y_target = y_target.squeeze(-1)
                out_for_loss = out_for_loss.squeeze(-1)
            finite_mask = torch.isfinite(y_target)
            if finite_mask.sum() == 0:
                continue
            loss_label = criterion(out_for_loss[finite_mask], y_target[finite_mask])
        elif cfg.train.loss_type == 'CE':
            y_target = y.squeeze().long()
            loss_label = criterion(out, y_target)
        else:
            y_target = y.float()
            if out.size(-1) == 1:
                mask_1d = mask.squeeze(-1) if mask.dim() > 1 else mask
                loss_label = criterion(out.squeeze(-1)[mask_1d], y_target.squeeze(-1)[mask_1d])
            else:
                loss_label = criterion(out[mask], y_target[mask])

        loss = loss_label
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite supervised loss during training: {loss.item()}")

        loss.backward()
        batch_loss = loss.item()

        configured_clip = float(getattr(cfg.train, 'clip_grad_norm', 0.0))
        if configured_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=configured_clip
            )
        elif dataset_type != 'counting':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += batch_loss

    if track_cuda:
        torch.cuda.synchronize(device)
    t_end = time.time()
    mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if track_cuda else 0
    mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2) if track_cuda else 0

    return total_loss / len(train_loader), t_end - t_start, mem_allocated, mem_reserved


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

        if dataset_type in ('ogb', 'lrgb', 'pyg_mol', 'good_mol'):
            x = data.x.long() if data.x.dtype == torch.float else data.x
            edge_attr = data.edge_attr
            if edge_attr is not None and edge_attr.dtype == torch.float:
                edge_attr = edge_attr.long()
        else:
            x = data.x
            edge_attr = data.edge_attr

        fingerprint = getattr(data, 'fingerprint', None)
        rwpe = getattr(data, 'random_walk_pe', None)
        LapPE = get_lap_pe(data)

        if track_cuda:
            torch.cuda.synchronize()
        t0 = time.time()

        out, _ = model(
            x,
            data.edge_index,
            data.batch,
            edge_attr,
            fingerprint,
            rwpe=rwpe,
            LapPE=LapPE,
        )
        if not torch.isfinite(out).all():
            raise FloatingPointError("Non-finite model output during evaluation.")

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
        raise FloatingPointError("Non-finite raw predictions during evaluation.")

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
        if not np.isfinite(y_preds_prob).all():
            raise FloatingPointError("Non-finite prediction probabilities during evaluation.")
        results = evaluate(y_trues, y_preds_prob, task=dataset_task)
    else:
        if y_preds.ndim > 1 and y_preds.shape[1] == 1:
            y_preds = y_preds.squeeze(1)
        y_preds_prob = stable_sigmoid(y_preds)
        if not np.isfinite(y_preds_prob).all():
            raise FloatingPointError("Non-finite prediction probabilities during evaluation.")
        results = evaluate(y_trues, y_preds_prob, task=dataset_task)

    return results, t_forward_sum, mem_allocated, mem_reserved


def run_single_session(session_id, train_loader, val_loader, test_loader,
                       dataset_info, cfg, logger, prune_callback=None,
                       prune_step_offset=0, evaluate_test=True):
    pna_degrees = compute_degree_histogram(train_loader.dataset, dataset_info['max_degree']).to(cfg.device)
    model = GraphGPSModel(
        hidden_size=cfg.graphgps.hidden_size,
        out_size=dataset_info['output_dim'],
        num_layers=cfg.graphgps.num_layers,
        dataset_type=dataset_info['type'],
        dim_node_deg=dataset_info.get('dim_node_deg', 0),
        dim_node_label=dataset_info.get('dim_node_label', 0),
        dim_node_attr=dataset_info.get('dim_node_attr', 0),
        num_edge_classes=dataset_info.get('num_edge_classes', 0),
        local_gnn_type=cfg.graphgps.local_gnn_type,
        global_model_type=cfg.graphgps.global_model_type,
        num_heads=cfg.graphgps.num_heads,
        dropout=cfg.graphgps.dropout,
        attn_dropout=cfg.graphgps.attn_dropout,
        act_type=cfg.graphgps.act_type,
        layer_norm=cfg.graphgps.layer_norm,
        batch_norm=cfg.graphgps.batch_norm,
        readout_type=cfg.graphgps.readout_type,
        use_rwpe=cfg.graphgps.use_rwpe,
        rwpe_dim=cfg.graphgps.rwpe_dim,
        rwpe_encoder_type=getattr(
            cfg.graphgps, 'rwpe_encoder_type', 'graphgps'
        ),
        rwpe_embed_dim=getattr(
            cfg.graphgps, 'rwpe_embed_dim', cfg.graphgps.rwpe_dim
        ),
        rwpe_model=getattr(cfg.graphgps, 'rwpe_model', 'linear'),
        rwpe_layers=getattr(cfg.graphgps, 'rwpe_layers', 1),
        rwpe_raw_norm_type=getattr(
            cfg.graphgps, 'rwpe_raw_norm_type', 'BatchNorm'
        ),
        use_lappe=cfg.graphgps.use_lappe,
        lappe_dim=cfg.graphgps.lappe_dim,
        lappe_max_freqs=cfg.graphgps.lappe_max_freqs,
        lappe_model=cfg.graphgps.lappe_model,
        lappe_layers=cfg.graphgps.lappe_layers,
        lappe_heads=cfg.graphgps.lappe_heads,
        lappe_raw_norm_type=cfg.graphgps.lappe_raw_norm_type,
        pna_degrees=pna_degrees,
        head_layers=cfg.graphgps.head_layers,
    ).to(cfg.device)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    num_trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    logger.info(
        f"Model parameters | total: {num_parameters:,}, "
        f"trainable: {num_trainable_parameters:,}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
    key_metric = dataset_info['key_metric']
    scheduler, scheduler_name = make_scheduler(cfg, optimizer, key_metric)
    criterion = get_criterion(cfg.train.loss_type).to(cfg.device)
    higher_better = is_higher_better_metric(key_metric)
    best_val_metric = -float('inf') if higher_better else float('inf')
    best_val_perf = {}
    best_model_state = None
    best_epoch = 0
    patience_count = 0

    logger.info(f"Start Session {session_id} | Key Metric: {key_metric}")

    total_train_time = 0
    epoch_count = 0
    max_mem_train = 0
    max_mem_train_res = 0
    cycle_times = []

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
                f"Train | time: {t_train:.2f} s, mem_allocated: {mem_tr_alloc:.2f} MB, mem_reserved: {mem_tr_res:.2f} MB | "
                f"Val-infer | time: {t_val:.2f} s, mem_allocated: {mem_val_alloc:.2f} MB, mem_reserved: {mem_val_res:.2f} MB"
            )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(
            f"Session {session_id}: Loading best model for "
            f"{'final testing' if evaluate_test else 'validation-only trial completion'}..."
        )

    test_perf = {}
    t_test = 0.0
    mem_test_alloc = 0.0
    mem_test_res = 0.0
    if evaluate_test:
        test_perf, t_test, mem_test_alloc, mem_test_res = eval_epoch(
            test_loader, model, cfg.device,
            dataset_task=dataset_info['task'],
            dataset_type=dataset_info['type'],
        )

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
        def strict_forward(current_model, data):
            if dataset_info['type'] in ('ogb', 'lrgb', 'pyg_mol', 'good_mol'):
                x = data.x.long() if data.x.dtype == torch.float else data.x
                edge_attr = data.edge_attr
                if edge_attr is not None and edge_attr.dtype == torch.float:
                    edge_attr = edge_attr.long()
            else:
                x, edge_attr = data.x, data.edge_attr
            return current_model(
                x, data.edge_index, data.batch, edge_attr,
                getattr(data, 'fingerprint', None),
                rwpe=getattr(data, 'random_walk_pe', None),
                LapPE=get_lap_pe(data),
            )

        rw_transform = (
            T.AddRandomWalkPE(
                walk_length=int(cfg.graphgps.rwpe_dim),
                attr_name='random_walk_pe',
            ) if bool(cfg.graphgps.use_rwpe) else None
        )

        def strict_transform(data):
            data = data.clone()
            for attr in ('random_walk_pe', 'EigVals', 'EigVecs'):
                if hasattr(data, attr):
                    delattr(data, attr)
            if rw_transform is not None:
                data = rw_transform(data)
            if bool(cfg.graphgps.use_lappe):
                data = add_lap_pe_stats(
                    data,
                    max_freqs=int(cfg.graphgps.lappe_max_freqs),
                    eigvec_norm=str(cfg.graphgps.lappe_eigvec_norm),
                    laplacian_norm=str(cfg.graphgps.lappe_laplacian_norm),
                )
            return data

        strict_result = benchmark_session_if_enabled(
            cfg, model, test_loader.dataset, test_loader.dataset,
            strict_forward,
            session_id=session_id,
            transform_fn=strict_transform,
            throughput_loader=test_loader,
            extra={'training': training_efficiency},
        )

    checkpoint_path = save_efficiency_checkpoint_if_enabled(
        cfg, model, session_id=session_id, best_epoch=best_epoch,
    )

    del model, optimizer, scheduler, criterion, best_model_state
    if is_cuda_device(cfg.device):
        torch.cuda.empty_cache()

    test_log = (
        "  Test Metrics: "
        + ", ".join([f"{k}: {v:.4f}" for k, v in test_perf.items()])
        + "\n"
        + f"  Test-Infer | Time: {t_test:.2f} s, Mem Alloc: "
        f"{mem_test_alloc:.2f} MB, Res: {mem_test_res:.2f} MB\n"
        if evaluate_test else
        "  Test Metrics: skipped (validation-only tuning trial)\n"
    )
    logger.info(
        test_log
        + "  Best Val Metrics: "
        + ", ".join([f"{k}: {v:.4f}" for k, v in best_val_perf.items()])
        + "\n"
        + f"  Train | Total Time: {total_train_time:.2f}s, Epochs: {epoch_count}, "
        + f"Best Epoch: {best_epoch}, "
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


def build_dataset_info(cfg, dataset, max_degree, dataset_type):
    info = {
        'type': dataset_type,
        'max_degree': max_degree,
        'output_dim': dataset.output_dim,
        'task': getattr(dataset, 'task', 'b'),
        'key_metric': getattr(dataset, 'key_metric', 'acc'),
    }
    if dataset_type in ('tu', 'counting'):
        info.update({
            'dim_node_deg': dataset.dim_node_deg,
            'dim_node_label': dataset.dim_node_label,
            'dim_node_attr': dataset.dim_node_attr,
            'num_edge_classes': dataset.num_edge_classes,
        })
    return info


def run_graphgps(cfg=None, key_metric_override=None, config_string=None,
                 prune_callback=None, evaluate_test=True):
    args = None
    config_path = None
    if cfg is None:
        args = parse_args()
        cfg = init_cfg()
        cfg = ensure_graphgps_cfg(cfg)
        if args.config == "":
            cfg.dataset = infer_dataset_from_opts(args.opts, cfg.dataset)
        cfg, config_path = load_graphgps_yaml(cfg, args)
        if args is not None and args.opts:
            cfg.merge_from_list(args.opts)
    else:
        cfg = ensure_graphgps_cfg(cfg)
    validate_graphgps_cfg(cfg)

    dataset_type = supported_dataset_type(cfg.dataset)
    set_loss_for_dataset(cfg, dataset_type)
    if cfg.use_augm_gca:
        print("GraphGPS runner currently disables GCA augmentation; set cfg.use_augm_gca=False.")
        cfg.use_augm_gca = False
    cfg.model.use_fingerprints = False
    cfg.model.use_rwpe = cfg.graphgps.use_rwpe
    cfg.model.rwpe_dim = cfg.graphgps.rwpe_dim

    if config_string is None:
        config_string = f"{cfg.dataset}_GraphGPS_{int(time.time())}"
    logger = setup_logger(cfg, config_string)
    logger.info(f"GraphGPS config file: {config_path}")
    logger.info(f"Config:\n{cfg}\n")
    logger.info(f'Start Time: {datetime.datetime.now().strftime("%Y/%m/%d - %H:%M")}\n')

    prep_started = time.perf_counter()
    efficiency_data_root = str(getattr(cfg.efficiency, 'data_root', '')) if efficiency_enabled(cfg) else ''
    prepared = load_prepared_input(cfg)
    if prepared is not None:
        dataset = restore_dataset_metadata(prepared['dataset_metadata'])
        folds_data = prepared['folds_data']
        max_degree = prepared['max_degree']
    else:
        dataset, folds_data, max_degree = load_dataset(
        dataset_name=cfg.dataset,
        root=efficiency_data_root or str(DATA_DIR),
        use_fingerprints=False,
        fingerprint_dim=0,
        use_rwpe=cfg.graphgps.use_rwpe,
        rwpe_dim=cfg.graphgps.rwpe_dim,
        use_lappe=cfg.graphgps.use_lappe,
        lappe_max_freqs=cfg.graphgps.lappe_max_freqs,
        lappe_eigvec_norm=cfg.graphgps.lappe_eigvec_norm,
        lappe_laplacian_norm=cfg.graphgps.lappe_laplacian_norm,
        augm_gca=False,
        aug_centrality='degree',
        fold_k=cfg.cv.fold_k,
        fold_seed=cfg.cv.seed,
        gnn_type=cfg.graphgps.local_gnn_type,
        **dataset_loader_kwargs(cfg),
        )
    # Keep all G-CARE folds through preprocessing; select the requested
    # official fold only in the training-session block below.
    dataset_info = build_dataset_info(cfg, dataset, max_degree, dataset_type)
    if efficiency_enabled(cfg, 'prep'):
        result = finish_preprocessing(cfg, prep_started, folds_data)
        # Only Count GraphGPS entries need an untimed, serialized copy of the
        # in-memory RWSE transform.  Other GraphGPS entries are reconstructed
        # entirely from the finalized PyG processed/ tree in ``data_root``.
        if bool(getattr(cfg.efficiency, 'require_prepared_input', False)):
            save_prepared_input(cfg, {
                'dataset_metadata': dataset_metadata(dataset),
                'folds_data': folds_data,
                'max_degree': max_degree,
            })
        return result
    if key_metric_override is not None:
        dataset_info['key_metric'] = key_metric_override
    else:
        configured_metric = str(
            getattr(cfg.train, 'key_metric', 'auto')
        ).lower()
        if configured_metric != 'auto':
            dataset_info['key_metric'] = configured_metric

    is_gnce = cfg.dataset.lower().startswith('gnce-')
    is_good = is_good_dataset_name(cfg.dataset)
    use_fold_sessions = (
        dataset_type == 'tu' and not is_gnce and not is_good
    )

    if use_fold_sessions:
        num_sessions = len(folds_data)
        cv_dataset = 'G-CARE' if cfg.dataset.lower().startswith('gcare-') else 'TU'
        logger.info(f"{cv_dataset} Dataset (K-Fold = {num_sessions})")
    elif is_gnce:
        num_sessions = cfg.train.runs
        logger.info(f"GNCE Dataset (Repeated Runs = {num_sessions})")
    elif is_good:
        num_sessions = cfg.train.runs
        logger.info(
            f"GOOD official fixed split "
            f"({getattr(dataset, 'good_name', cfg.dataset)}; "
            f"Repeated Runs = {num_sessions}; val/test=OOD for shifts)"
        )
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

    for session_id in session_indices:
        if use_fold_sessions:
            train_d, val_d, test_d = folds_data[session_id]
            current_seed = cfg.seed
        else:
            train_d, val_d, test_d = folds_data[0]
            current_seed = cfg.seed + session_id

        seed_everything(current_seed, deterministic=cfg.deterministic, warn_only=cfg.deterministic_warn_only)
        train_loader = DataLoader(train_d, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False)
        val_loader = DataLoader(val_d, cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)
        test_loader = (
            DataLoader(
                test_d, cfg.train.batch_size, shuffle=False,
                num_workers=cfg.num_workers,
            )
            if evaluate_test else None
        )

        results_run = run_single_session(
            session_id + 1, train_loader, val_loader, test_loader,
            dataset_info, cfg, logger,
            prune_callback=prune_callback,
            prune_step_offset=session_id * cfg.train.epochs,
            evaluate_test=evaluate_test,
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
        v = torch.tensor(values, dtype=torch.float64)
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
            'num_trainable_parameters': agg_values(
                stats_collector['num_trainable_parameters']
            ),
        },
    }

    sections = [('Best Val', aggregate['best_val_perf'])]
    if evaluate_test:
        sections.insert(0, ('Test Effectiveness', aggregate['test_perf']))
    else:
        logger.info("[Test Effectiveness] skipped (validation-only tuning trial)")
    for section, metrics in sections:
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
    return run_graphgps()


if __name__ == '__main__':
    main()
