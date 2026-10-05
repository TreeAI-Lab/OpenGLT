import copy
import datetime
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import init_cfg, parse_args, update_cfg
from datasets import (
    good_loss_type,
    good_model_dataset_type,
    is_good_dataset_name,
    load_dataset,
    base_dataset_name,
    dataset_loader_kwargs,
)
from engine import compute_nt_xent_loss, evaluate, get_criterion, seed_everything
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
from example.main_baselines import evaluate_cardinality_outputs, is_cuda_device, setup_logger, stable_sigmoid, validate_pooling_cfg
from models import GLGNN

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / 'configs'
DATA_DIR = BASE_DIR / 'data'


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
    raise ValueError(f"Dataset {dataset} not supported by GCA runner.")


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


def normalize_drop_scheme(drop_scheme):
    drop_scheme = str(drop_scheme).lower()
    if drop_scheme in {'degree', 'de'}:
        return 'degree'
    if drop_scheme in {'pr', 'pagerank', 'page_rank'}:
        return 'pagerank'
    if drop_scheme in {'evc', 'eigenvector', 'eigenvector_centrality'}:
        return 'eigenvector'
    raise ValueError("gca.drop_scheme must be one of: degree / pr / evc.")


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
    if scheduler_name in ('none', 'null', ''):
        return None, scheduler_name
    raise ValueError(f"Unknown scheduler: {cfg.train.scheduler}")


def is_higher_better_metric(key_metric):
    if key_metric in ('acc', 'auc', 'auc_macro', 'auc_micro', 'ap', 'ap_macro', 'ap_micro', 'r2'):
        return True
    if key_metric in ('mae', 'mre', 'mse', 'rmse', 'medq', 'p90q', 'p95q', 'p99q', 'meanq', 'maxq', 'log_mae'):
        return False
    raise ValueError(f"Check key_metric: {key_metric}")


def get_batch_inputs(data, dataset_type):
    if dataset_type in ('ogb', 'lrgb', 'pyg_mol', 'good_mol'):
        x = data.x.long() if data.x.dtype == torch.float else data.x
        edge_attr = data.edge_attr
        if edge_attr is not None and edge_attr.dtype == torch.float:
            edge_attr = edge_attr.long()
    else:
        x = data.x
        edge_attr = data.edge_attr
    return x, edge_attr


def encode_node_features(model, x, dataset_type):
    if dataset_type in ('ogb', 'lrgb', 'pyg_mol', 'good_mol'):
        if x.dtype == torch.float:
            x = x.long()
        return model.input_encoder(x)
    if x.dtype == torch.long:
        x = x.float()
    return model.input_encoder(x)


def per_graph_drop_prob(scores, batch_index, p, p_tau, p_corr):
    prob = torch.zeros_like(scores)
    for graph_id in batch_index.unique(sorted=True):
        mask = batch_index == graph_id
        graph_scores = scores[mask]
        s_max = graph_scores.max()
        if p_corr == 'mean':
            s_base = graph_scores.mean()
        elif p_corr == 'min':
            s_base = graph_scores.min()
        else:
            raise ValueError("gca.p_corr must be 'mean' or 'min'.")
        denom = s_max - s_base
        if denom < 1e-6:
            continue
        prob[mask] = ((s_max - graph_scores) / denom * p).clamp(0, p_tau)
    return prob


def adaptive_edge_drop(data, p_e, p_tau, p_corr):
    if p_e <= 0 or data.edge_index.numel() == 0:
        return data.edge_index, data.edge_attr
    if not hasattr(data, 'edge_c_log'):
        raise ValueError("GCA requires edge_c_log; load dataset with GCA centrality preprocessing.")

    edge_batch = data.batch[data.edge_index[0]]
    edge_scores = data.edge_c_log.to(data.edge_index.device)
    prob_drop = per_graph_drop_prob(edge_scores, edge_batch, p_e, p_tau, p_corr)
    keep_mask = ~torch.bernoulli(prob_drop).bool()

    for graph_id in edge_batch.unique(sorted=True):
        graph_edge_mask = edge_batch == graph_id
        if graph_edge_mask.any() and not keep_mask[graph_edge_mask].any():
            first_edge = graph_edge_mask.nonzero(as_tuple=False)[0]
            keep_mask[first_edge] = True

    edge_index = data.edge_index[:, keep_mask]
    edge_attr = data.edge_attr[keep_mask] if data.edge_attr is not None else None
    return edge_index, edge_attr


def adaptive_feature_drop(data, x_emb, p_f, p_tau, p_corr):
    if p_f <= 0:
        return x_emb
    if not hasattr(data, 'node_c_log'):
        raise ValueError("GCA requires node_c_log; load dataset with GCA centrality preprocessing.")

    node_batch = data.batch
    node_importance = data.node_c_log.to(x_emb.device).exp().clamp_min(1e-6)
    x_new = x_emb.clone()
    num_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 0

    for graph_id in range(num_graphs):
        node_mask = node_batch == graph_id
        if not node_mask.any():
            continue
        graph_x = x_emb.detach()[node_mask].abs()
        graph_importance = node_importance[node_mask].view(-1, 1)
        feature_scores = (graph_x * graph_importance).sum(dim=0)
        score_max = feature_scores.max()
        if p_corr == 'mean':
            score_base = feature_scores.mean()
        elif p_corr == 'min':
            score_base = feature_scores.min()
        else:
            raise ValueError("gca.p_corr must be 'mean' or 'min'.")
        denom = score_max - score_base
        if denom < 1e-6:
            continue
        prob_mask = ((score_max - feature_scores) / denom * p_f).clamp(0, p_tau)
        feature_mask = torch.bernoulli(prob_mask).bool()
        if feature_mask.any():
            keep = (~feature_mask).to(x_emb.dtype).view(1, -1)
            x_new[node_mask] = x_new[node_mask] * keep
    return x_new


def adaptive_gca_view(data, x_emb, p_e, p_f, p_tau, p_corr):
    edge_index, edge_attr = adaptive_edge_drop(data, p_e, p_tau, p_corr)
    x_aug = adaptive_feature_drop(data, x_emb, p_f, p_tau, p_corr)
    return edge_index, edge_attr, x_aug


def pretrain_gca_epoch(train_loader, model, optimizer, device, cfg, dataset_type='ogb'):
    model.train()
    total_loss = 0
    track_cuda = is_cuda_device(device)
    if track_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.time()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        x, _ = get_batch_inputs(data, dataset_type)
        fingerprint = getattr(data, 'fingerprint', None)
        x_emb = encode_node_features(model, x, dataset_type)

        ei1, ea1, x1 = adaptive_gca_view(
            data, x_emb, cfg.gca.pe1, cfg.gca.pf1, cfg.gca.p_tau, cfg.gca.p_corr
        )
        ei2, ea2, x2 = adaptive_gca_view(
            data, x_emb, cfg.gca.pe2, cfg.gca.pf2, cfg.gca.p_tau, cfg.gca.p_corr
        )

        _, z1 = model(None, ei1, data.batch, ea1, fingerprint=fingerprint, input_embedding=x1)
        _, z2 = model(None, ei2, data.batch, ea2, fingerprint=fingerprint, input_embedding=x2)
        loss = compute_nt_xent_loss(z1, z2, temperature=cfg.gca.temperature)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite GCA pretrain loss: {loss.item()}")

        loss.backward()
        if dataset_type != 'counting':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    t_end = time.time()
    mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if track_cuda else 0
    mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2) if track_cuda else 0
    return total_loss / len(train_loader), t_end - t_start, mem_allocated, mem_reserved


def supervised_train_epoch(train_loader, model, criterion, optimizer, device, cfg, dataset_type='ogb'):
    model.train()
    total_loss = 0
    track_cuda = is_cuda_device(device)
    if track_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.time()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        x, edge_attr = get_batch_inputs(data, dataset_type)
        fingerprint = getattr(data, 'fingerprint', None)
        y = data.y

        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue

        out, _ = model(x, data.edge_index, data.batch, edge_attr, fingerprint)
        if not torch.isfinite(out).all():
            raise FloatingPointError("Non-finite model output during supervised training.")

        if cfg.train.loss_type in ('L1', 'MSE'):
            y_target = y.float()
            out_for_loss = out
            if y_target.shape != out_for_loss.shape:
                y_target = y_target.squeeze(-1)
                out_for_loss = out_for_loss.squeeze(-1)
            finite_mask = torch.isfinite(y_target)
            if finite_mask.sum() == 0:
                continue
            loss = criterion(out_for_loss[finite_mask], y_target[finite_mask])
        elif cfg.train.loss_type == 'CE':
            y_target = y.squeeze().long()
            loss = criterion(out, y_target)
        else:
            y_target = y.float()
            if out.size(-1) == 1:
                mask_1d = mask.squeeze(-1) if mask.dim() > 1 else mask
                loss = criterion(out.squeeze(-1)[mask_1d], y_target.squeeze(-1)[mask_1d])
            else:
                loss = criterion(out[mask], y_target[mask])

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite supervised loss: {loss.item()}")

        loss.backward()
        if dataset_type != 'counting':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

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
        x, edge_attr = get_batch_inputs(data, dataset_type)
        fingerprint = getattr(data, 'fingerprint', None)

        if track_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        out, _ = model(x, data.edge_index, data.batch, edge_attr, fingerprint)
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


def reset_prediction_head(model):
    if hasattr(model, 'output_decoder') and hasattr(model.output_decoder, 'reset_parameters'):
        model.output_decoder.reset_parameters()


def build_model(cfg, dataset_info, deg):
    extra_model_kwargs = {
        'n_heads_gat': 4,
        'train_eps_gin': cfg.model.train_eps_gin,
        'aggr_sage': cfg.model.aggr_sage,
        'towers_pna': cfg.model.towers_pna,
        'pre_layers_pna': cfg.model.pre_layers_pna,
        'post_layers_pna': cfg.model.post_layers_pna,
        'divide_input_pna': cfg.model.divide_input_pna,
    }
    return GLGNN(
        nhid=cfg.model.hidden_size,
        nout=dataset_info['output_dim'],
        n_layer_basicgnn=cfg.model.num_layers,
        gnn_types=[cfg.model.backbone],
        dataset_type=dataset_info['type'],
        dim_node_deg=dataset_info.get('dim_node_deg', 0),
        dim_node_label=dataset_info.get('dim_node_label', 0),
        dim_node_attr=dataset_info.get('dim_node_attr', 0),
        num_edge_classes=dataset_info.get('num_edge_classes', 0),
        dropout=cfg.model.dropout,
        readout_type=cfg.model.readout_type,
        pooling_method=cfg.model.pooling_method,
        pool_ratio=cfg.model.pool_ratio,
        num_pool_layers=cfg.model.num_pool_layers,
        pool_readout=cfg.model.pool_readout,
        edgepool_dropout=cfg.model.edgepool_dropout,
        edge_score_method=cfg.model.edge_score_method,
        gmt_k=cfg.model.gmt_k,
        gmt_heads=cfg.model.gmt_heads,
        gmt_blocks=cfg.model.gmt_blocks,
        gmt_layer_norm=cfg.model.gmt_layer_norm,
        res=cfg.model.res,
        deg=deg,
        use_virtual_node=cfg.model.use_virtual_node,
        use_fingerprints=cfg.model.use_fingerprints,
        fingerprint_dim=cfg.model.fingerprint_dim,
        use_rwpe=False,
        rwpe_dim=cfg.model.rwpe_dim,
        norm_type=cfg.model.norm_type,
        act_type=cfg.model.act_type,
        use_ffn=cfg.model.use_ffn,
        ffn_expansion=cfg.model.ffn_expansion,
        **extra_model_kwargs
    ).to(cfg.device)


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


def run_single_session(session_id, train_loader, val_loader, test_loader,
                       dataset_info, cfg, logger, prune_callback=None,
                       prune_step_offset=0, evaluate_test=True):
    deg = compute_degree_histogram(train_loader.dataset, dataset_info['max_degree']).to(cfg.device)
    model = build_model(cfg, dataset_info, deg)
    model.reset_parameters()

    total_train_time = 0
    max_mem_train = 0
    max_mem_train_res = 0
    pretrain_epochs = int(cfg.gca.pretrain_epochs)
    pretrain_times = []

    if pretrain_epochs > 0:
        pretrain_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.gca.pretrain_lr,
            weight_decay=cfg.gca.pretrain_wd,
        )
        for epoch in range(1, pretrain_epochs + 1):
            loss, t_pre, mem_pre_alloc, mem_pre_res = pretrain_gca_epoch(
                train_loader, model, pretrain_optimizer, cfg.device, cfg, dataset_info['type']
            )
            total_train_time += t_pre
            pretrain_times.append(t_pre)
            max_mem_train = max(max_mem_train, mem_pre_alloc)
            max_mem_train_res = max(max_mem_train_res, mem_pre_res)
            if epoch == 1 or epoch % 10 == 0 or epoch == pretrain_epochs:
                logger.info(
                    f"  Pretrain Epoch {epoch:03d} | GCA Loss: {loss:.4f} | "
                    f"time: {t_pre:.2f} s, mem_allocated: {mem_pre_alloc:.2f} MB, "
                    f"mem_reserved: {mem_pre_res:.2f} MB"
                )
        del pretrain_optimizer
        if cfg.gca.reset_prediction_head:
            reset_prediction_head(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
    criterion = get_criterion(cfg.train.loss_type).to(cfg.device)
    key_metric = dataset_info['key_metric']
    scheduler, scheduler_name = make_scheduler(cfg, optimizer, key_metric)
    higher_better = is_higher_better_metric(key_metric)
    best_val_metric = -float('inf') if higher_better else float('inf')
    best_val_perf = {}
    best_model_state = None
    patience_count = 0
    finetune_epoch_count = 0
    best_epoch = 0
    finetune_cycle_times = []

    logger.info(f"Start Session {session_id} | Key Metric: {key_metric}")
    logger.info(
        f"GCA pretrain_epochs={pretrain_epochs}, drop_scheme={cfg.gca.drop_scheme}, "
        f"pe=({cfg.gca.pe1}, {cfg.gca.pe2}), pf=({cfg.gca.pf1}, {cfg.gca.pf2}), "
        f"temperature={cfg.gca.temperature}"
    )

    for epoch in range(1, cfg.train.epochs + 1):
        if patience_count >= cfg.train.patience:
            logger.info(f"Session {session_id}: Early stopping at fine-tune epoch {epoch}.")
            break

        finetune_epoch_count += 1
        train_loss, t_train, mem_tr_alloc, mem_tr_res = supervised_train_epoch(
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
        finetune_cycle_times.append(t_train + t_val)
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
                f"  Fine-tune Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                f"Val {key_metric}: {current_val:.4f} (Best: {best_val_metric:.4f}) | "
                f"Train | time: {t_train:.2f} s, mem_allocated: {mem_tr_alloc:.2f} MB, mem_reserved: {mem_tr_res:.2f} MB | "
                f"Val-infer | time: {t_val:.2f} s, mem_allocated: {mem_val_alloc:.2f} MB, mem_reserved: {mem_val_res:.2f} MB"
            )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(
            f"Session {session_id}: Loading best fine-tuned model for "
            f"{'final testing' if evaluate_test else 'validation-only trial completion'}..."
        )
    else:
        logger.warning(f"Session {session_id}: No best model saved! Using last fine-tune epoch model.")

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

    warm = int(cfg.efficiency.timed_train_warmup_epochs) if efficiency_enabled(cfg) else 1
    steady_pre = pretrain_times[warm:]
    steady_fine = finetune_cycle_times[warm:]
    training_efficiency = {
        'pretrain_cycle_times_s': pretrain_times,
        'pretrain_steady_median_s': float(np.median(steady_pre)) if steady_pre else None,
        'pretrain_epochs_measured': pretrain_epochs,
        'finetune_cycle_times_s': finetune_cycle_times,
        'finetune_steady_cycle_median_s': float(np.median(steady_fine)) if steady_fine else None,
        'steady_cycle_median_s': float(np.median(steady_fine)) if steady_fine else None,
        'finetune_epochs_executed': finetune_epoch_count,
        'best_finetune_epoch': best_epoch,
        'train_peak_allocated_mb': max(max_mem_train, mem_val_alloc),
        'train_peak_reserved_mb': max(max_mem_train_res, mem_val_res),
    }
    strict_result = None
    if evaluate_test and efficiency_enabled(cfg, 'strict'):
        def strict_forward(current_model, data):
            x, edge_attr = get_batch_inputs(data, dataset_info['type'])
            return current_model(
                x, data.edge_index, data.batch, edge_attr,
                getattr(data, 'fingerprint', None),
            )

        # Centrality scores are training-only augmentation metadata.  They are
        # charged to preprocessing, but must not be transferred to the GPU or
        # counted as deployment input during inference.
        clean_test = []
        for original in test_loader.dataset:
            graph = original.clone()
            for attr in ('node_c_log', 'edge_c_log'):
                if hasattr(graph, attr):
                    del graph[attr]
            clean_test.append(graph)
        clean_loader = DataLoader(
            clean_test, cfg.train.batch_size, shuffle=False,
            num_workers=0,
        )

        strict_result = benchmark_session_if_enabled(
            cfg, model, clean_test, clean_test,
            strict_forward,
            session_id=session_id,
            throughput_loader=clean_loader,
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

    total_epoch_count = pretrain_epochs + finetune_epoch_count
    test_log = (
        f"  Test Metrics: {dict_to_str(test_perf)}\n"
        f"  Test-Infer | Time: {t_test:.2f} s, Mem Alloc: {mem_test_alloc:.2f} MB, "
        f"Res: {mem_test_res:.2f} MB\n"
        if evaluate_test else
        "  Test Metrics: skipped (validation-only tuning trial)\n"
    )
    logger.info(
        f"{test_log}"
        f"  Best Val Metrics: {dict_to_str(best_val_perf)}\n"
        f"  Train | Total Time: {total_train_time:.2f}s, Epochs: {total_epoch_count}, "
        f"Pretrain Epochs: {pretrain_epochs}, Fine-tune Epochs: {finetune_epoch_count}, "
        f"Mem Alloc: {max_mem_train:.2f} MB, Res: {max_mem_train_res:.2f} MB\n"
    )

    return {
        'test_perf': test_perf,
        'best_val_perf': best_val_perf,
        'train_time': total_train_time,
        'epoch_count': total_epoch_count,
        'best_epoch': best_epoch,
        'test_time': t_test,
        'train_mem_alloc': max_mem_train,
        'train_mem_res': max_mem_train_res,
        'test_mem_alloc': mem_test_alloc,
        'test_mem_res': mem_test_res,
        'strict_efficiency': strict_result,
        'training_efficiency': training_efficiency,
        'checkpoint_path': checkpoint_path,
    }


def run_gca(cfg=None, key_metric_override=None, config_string=None,
            prune_callback=None, evaluate_test=True):
    args = None
    if cfg is None:
        args = parse_args()
        cfg = init_cfg()
        cfg = update_cfg(cfg, args=args)

    validate_pooling_cfg(cfg)
    dataset_type = supported_dataset_type(cfg.dataset)
    set_loss_for_dataset(cfg, dataset_type)
    cfg.gca.drop_scheme = normalize_drop_scheme(cfg.gca.drop_scheme)

    if cfg.model.use_rwpe:
        print("GCA changes graph topology; disabling cfg.model.use_rwpe.")
        cfg.model.use_rwpe = False

    if config_string is None:
        config_string = f"{cfg.dataset}_GCA_{cfg.model.backbone}_{int(time.time())}"
    logger = setup_logger(cfg, config_string)
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
        use_fingerprints=cfg.model.use_fingerprints,
        fingerprint_dim=cfg.model.fingerprint_dim,
        use_rwpe=False,
        rwpe_dim=0,
        augm_gca=True,
        aug_centrality=cfg.gca.drop_scheme,
        fold_k=cfg.cv.fold_k,
        fold_seed=cfg.cv.seed,
        gnn_type=cfg.model.backbone,
        **dataset_loader_kwargs(cfg),
        )
    # Precompute GCA centrality for all official G-CARE folds; select the
    # requested fold only after preprocessing has completed.
    dataset_info = build_dataset_info(cfg, dataset, max_degree, dataset_type)
    if key_metric_override is not None:
        dataset_info['key_metric'] = key_metric_override

    if efficiency_enabled(cfg, 'prep'):
        result = finish_preprocessing(
            cfg, prep_started, folds_data,
            extra={'transform': f'GCA centrality={cfg.gca.drop_scheme}'},
        )
        save_prepared_input(cfg, {
            'dataset_metadata': dataset_metadata(dataset),
            'folds_data': folds_data,
            'max_degree': max_degree,
        })
        return result

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

        for key in ['train_time', 'epoch_count', 'best_epoch', 'test_time', 'train_mem_alloc', 'train_mem_res', 'test_mem_alloc', 'test_mem_res']:
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
    return run_gca()


if __name__ == '__main__':
    main()
