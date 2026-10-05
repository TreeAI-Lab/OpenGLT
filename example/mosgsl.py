import copy
import datetime
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.loader import DataLoader

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
    dataset_cache_tag,
    dataset_loader_kwargs,
)
from engine import evaluate, get_criterion, seed_everything
from example.main_baselines import evaluate_cardinality_outputs, is_cuda_device, setup_logger, stable_sigmoid
from models.mosgsl import MOSGSL, build_mosgsl_fold
from models.mosgsl.data import _append_edge_context, _sanitize_feature, prepare_parsed_data
from models.good_encoders import GOOD_ATOM_FEATURE_DIMS
from efficiency.runtime import (
    benchmark_session_if_enabled,
    efficiency_enabled,
    finish_preprocessing,
    save_efficiency_checkpoint_if_enabled,
)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
MOSGSL_CACHE_DIR = DATA_DIR / 'mosgsl_cache'


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
    if is_good_dataset_name(dataset):
        return good_model_dataset_type(dataset)
    if dataset.startswith('Count-'):
        return 'counting'
    raise ValueError(
        "MOSGSL runner supports TU, OGB molecular datasets, Count-* datasets, "
        "zinc, aqsol, peptides-func, and peptides-struct."
    )


def set_loss_for_dataset(cfg, dataset_type):
    dataset = base_dataset_name(cfg.dataset)
    dataset_lower = dataset.lower()
    if is_good_dataset_name(cfg.dataset):
        cfg.train.loss_type = good_loss_type(cfg.dataset)
    elif dataset in {'ogbg-molhiv', 'ogbg-molbace', 'IMDB-BINARY', 'REDDIT-BINARY', 'MUTAG', 'NCI1', 'DD', 'PROTEINS'}:
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
        raise ValueError(f"Unsupported MOSGSL dataset type: {dataset_type}")


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


def build_mosgsl_conf(cfg, dataset_info):
    hidden = int(cfg.mosgsl.hidden_size)
    num_layers = int(cfg.mosgsl.num_layers)
    dropout = float(cfg.mosgsl.dropout)
    backbone = str(cfg.mosgsl.backbone).lower()
    norm = None if str(cfg.mosgsl.norm).lower() == 'none' else cfg.mosgsl.norm
    act = cfg.mosgsl.act
    pool = cfg.mosgsl.pool
    categorical_feature_dims = (
        GOOD_ATOM_FEATURE_DIMS
        if dataset_info['type'] == 'good_mol' else None
    )
    select_min_score = None if float(cfg.mosgsl.select_min_score) < 0 else float(cfg.mosgsl.select_min_score)

    gnn_encoder = {
        'type': 'gnn',
        'n_hidden': hidden,
        'n_class': hidden,
        'n_layers': num_layers,
        'dropout': dropout,
        'norm': norm,
        'act': act,
        'pool': pool,
        'backbone': backbone,
        'output_layer': False,
        'categorical_feature_dims': categorical_feature_dims,
    }
    return SimpleNamespace(
        use_select=bool(cfg.mosgsl.use_select),
        use_gsl=bool(cfg.mosgsl.use_gsl),
        use_motif=bool(cfg.mosgsl.use_motif),
        use_seperate_encoder=bool(cfg.mosgsl.use_separate_encoder),
        parsing_module={
            'parsing': cfg.mosgsl.parsing,
            'parsing_params': {
                'max_n_sub': int(cfg.mosgsl.max_n_sub),
                'max_sub_size': int(cfg.mosgsl.max_sub_size),
            },
            'reparsing': bool(cfg.mosgsl.force_reload),
        },
        selecting_module={
            'topk': float(cfg.mosgsl.select_topk),
            'min_score': select_min_score,
            'conf_encoder': gnn_encoder,
            'conf_estimator': {
                'type': cfg.mosgsl.select_estimator,
                'act': 'sigmoid',
                'n_hidden': hidden,
                'n_layers': 2,
                'dropout': dropout,
                'use_bn': False,
            },
        },
        gsl_module={
            'conf_encoder': gnn_encoder,
            'metric_type': 'weighted_cosine',
            'conf_metric': {
                'num_pers': int(cfg.mosgsl.metric_heads),
                'weighted': bool(cfg.mosgsl.metric_weighted),
                'normalize': True,
            },
            'epsilon': float(cfg.mosgsl.gsl_epsilon),
            'lamb1': float(cfg.mosgsl.gsl_lambda),
            'gsl_one_by_one': bool(cfg.mosgsl.gsl_one_by_one),
        },
        backbone={
            'n_hidden': hidden,
            'n_layers': num_layers,
            'dropout': dropout,
            'norm': norm,
            'act': act,
            'pool': pool,
            'backbone': backbone,
            'output_layer': True,
            'n_layers_output': int(cfg.mosgsl.output_layers),
            'categorical_feature_dims': categorical_feature_dims,
        },
        classifier={
            'n_hidden': hidden,
            'n_layers': int(cfg.mosgsl.output_layers),
            'dropout': dropout,
            'use_bn': True,
            'activation': act,
        },
        motif={
            'n_motif_per_class': int(cfg.mosgsl.motif_per_class),
            'sim_type': cfg.mosgsl.motif_sim_type,
            'temperature': float(cfg.mosgsl.motif_temperature),
            'update_type': cfg.mosgsl.motif_update_type,
            'k2': float(cfg.mosgsl.motif_k2),
            'tau': float(cfg.mosgsl.motif_tau),
            'weighted_mean': bool(cfg.mosgsl.motif_weighted_mean),
        },
    )


def infer_feature_dim(dataset):
    data = dataset[0]
    if data.x is None:
        return 1
    if data.x.dim() == 1:
        return 1
    return int(data.x.size(-1))


def build_model(cfg, dataset_info, train_dataset, graphs2subgraphs):
    conf = build_mosgsl_conf(cfg, dataset_info)
    motif_n_class = None
    if dataset_info['task'] == 'b':
        motif_n_class = 2
    elif dataset_info['task'] == 'mc':
        motif_n_class = dataset_info['output_dim']
    if motif_n_class is None:
        conf.use_motif = False

    model = MOSGSL(
        conf=conf,
        n_feat=infer_feature_dim(train_dataset),
        n_class=dataset_info['output_dim'],
        motif_n_class=motif_n_class,
    ).to(cfg.device)
    model.parsing.graphs2subgraphs = graphs2subgraphs
    model.reset_parameters()
    return model


def compute_supervised_loss(out, y, criterion, loss_type):
    if loss_type in ('L1', 'MSE'):
        y_target = y.float()
        out_for_loss = out
        if y_target.shape != out_for_loss.shape:
            y_target = y_target.view_as(out_for_loss)
        finite_mask = torch.isfinite(y_target)
        if finite_mask.sum() == 0:
            return None, y.size(0)
        return criterion(out_for_loss[finite_mask], y_target[finite_mask]), y.size(0)

    if loss_type == 'CE':
        y_target = y.squeeze().long()
        return criterion(out, y_target), y.size(0)

    y_target = y.float()
    mask = ~torch.isnan(y_target)
    if mask.sum() == 0:
        return None, y.size(0)
    if out.size(-1) == 1:
        mask_1d = mask.squeeze(-1) if mask.dim() > 1 else mask
        return criterion(out.squeeze(-1)[mask_1d], y_target.squeeze(-1)[mask_1d]), y.size(0)
    return criterion(out[mask], y_target[mask]), y.size(0)


def train_epoch(train_loader, model, criterion, optimizer, device, cfg,
                dataset_type, use_motif_loss=True):
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
        out, loss_con = model(data)
        if not torch.isfinite(out).all():
            raise FloatingPointError("Non-finite MOSGSL output during training.")
        loss, num_graphs = compute_supervised_loss(out, data.y, criterion, cfg.train.loss_type)
        if loss is None:
            continue
        if use_motif_loss and model.use_motif:
            loss = loss + float(cfg.mosgsl.motif_loss_weight) * loss_con
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite MOSGSL loss: {loss.item()}")
        loss.backward()
        if dataset_type != 'counting':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item()) * num_graphs
        total_graphs += num_graphs

    t_end = time.time()
    mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if track_cuda else 0
    mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2) if track_cuda else 0
    return total_loss / max(total_graphs, 1), t_end - t_start, mem_allocated, mem_reserved


def pretrain_epoch(train_loader, model, criterion, optimizer, device, cfg,
                   dataset_type):
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
        out, _ = model.forward_wo_gsl(
            data,
            only_backbone=bool(cfg.mosgsl.pretrain_backbone_only),
        )
        loss, num_graphs = compute_supervised_loss(out, data.y, criterion, cfg.train.loss_type)
        if loss is None:
            continue
        loss.backward()
        if dataset_type != 'counting':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item()) * num_graphs
        total_graphs += num_graphs

    t_end = time.time()
    mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if track_cuda else 0
    mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2) if track_cuda else 0
    return total_loss / max(total_graphs, 1), t_end - t_start, mem_allocated, mem_reserved


@torch.no_grad()
def eval_epoch_wo_gsl(loader, model, device, dataset_task='b',
                      only_backbone=False):
    model.eval()
    y_preds = []
    y_trues = []
    y_cards = []
    y_means = []
    y_stds = []
    for data in loader:
        data = data.to(device)
        out, _ = model.forward_wo_gsl(data, only_backbone=only_backbone)
        if not torch.isfinite(out).all():
            raise FloatingPointError("Non-finite MOSGSL pretrain output during evaluation.")
        y_preds.append(out.detach().cpu())
        y_trues.append(data.y.detach().cpu())
        if dataset_task == 'card':
            y_cards.append(data.y_card.detach().cpu())
            y_means.append(data.y_mean.detach().cpu())
            y_stds.append(data.y_std.detach().cpu())

    y_preds = torch.cat(y_preds, dim=0).numpy()
    y_trues = torch.cat(y_trues, dim=0).numpy()
    if dataset_task == 'reg':
        if y_preds.ndim > 1 and y_preds.shape[1] == 1:
            y_preds = y_preds.squeeze(1)
        if y_trues.ndim > 1 and y_trues.shape[1] == 1:
            y_trues = y_trues.squeeze(1)
        return evaluate(y_trues, y_preds, task='reg')
    if dataset_task == 'card':
        y_cards = torch.cat(y_cards, dim=0).numpy()
        y_means = torch.cat(y_means, dim=0).numpy()
        y_stds = torch.cat(y_stds, dim=0).numpy()
        return evaluate_cardinality_outputs(y_preds, y_cards, y_means, y_stds)
    if dataset_task == 'mc':
        return evaluate(y_trues, torch.softmax(torch.tensor(y_preds), dim=1).numpy(), task='mc')
    if y_preds.ndim > 1 and y_preds.shape[1] == 1:
        y_preds = y_preds.squeeze(1)
    return evaluate(y_trues, stable_sigmoid(y_preds), task=dataset_task)


@torch.no_grad()
def eval_epoch(loader, model, device, dataset_task='b'):
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
        out, _ = model(data)
        if not torch.isfinite(out).all():
            raise FloatingPointError("Non-finite MOSGSL output during evaluation.")
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
        raise FloatingPointError("Non-finite MOSGSL raw predictions during evaluation.")

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


def graph_confidence(out):
    if out.size(-1) == 1:
        prob = torch.sigmoid(out.view(-1))
        return torch.maximum(prob, 1.0 - prob)
    return torch.softmax(out, dim=-1).max(dim=1).values


@torch.no_grad()
def collect_subgraph_embeddings(loader, model, device, use_gsl=False,
                                filter_confident=False, keep_ratio=1.0):
    model.eval()
    zs = []
    ys = []
    for data in loader:
        data = data.to(device)
        if use_gsl:
            out, _, z_sub, selected_belong, importance = model(data, return_sub=True)
        else:
            out, _, z_sub, selected_belong, importance = model.forward_wo_gsl(data, return_sub=True)
        y_sub = data.y[selected_belong].view(-1)
        valid = torch.isfinite(y_sub.float())
        if valid.sum() == 0:
            continue
        z_sub = z_sub[valid]
        y_sub = y_sub[valid]
        if filter_confident and z_sub.size(0) > 0:
            confidence = graph_confidence(out)[selected_belong][valid]
            score = (importance[valid] + confidence) / 2.0
            n_keep = max(1, int(score.numel() * float(keep_ratio)))
            keep = score.topk(n_keep).indices
            z_sub = z_sub[keep]
            y_sub = y_sub[keep]
        zs.append(z_sub.detach())
        ys.append(y_sub.long().detach())
    if not zs:
        return None, None
    return torch.cat(zs, dim=0), torch.cat(ys, dim=0)


def init_motif(model, train_loader, device, logger, cfg, pretrained=False):
    if not model.use_motif:
        return
    z, y = collect_subgraph_embeddings(
        train_loader,
        model,
        device,
        use_gsl=False,
        filter_confident=pretrained,
        keep_ratio=float(cfg.mosgsl.motif_k1),
    )
    if z is None:
        logger.warning("MOSGSL motif initialization skipped: no valid single-label subgraphs.")
        model.use_motif = False
        return
    model.motif.init_motif(z, y)


def update_motif(model, train_loader, device, logger, cfg):
    if not model.use_motif:
        return
    z, y = collect_subgraph_embeddings(
        train_loader,
        model,
        device,
        use_gsl=True,
        filter_confident=True,
        keep_ratio=float(cfg.mosgsl.motif_k1),
    )
    if z is None:
        logger.warning("MOSGSL motif update skipped: no valid single-label subgraphs.")
        return
    model.motif.update_motif(z, y)


def run_pretrain(train_loader, val_loader, model, criterion, optimizer, device,
                 cfg, dataset_info, logger):
    n_pretrain = int(cfg.mosgsl.pretrain_epochs)
    if n_pretrain <= 0 or not bool(cfg.mosgsl.use_gsl):
        return 0.0, 0.0, 0.0, False, []
    logger.info(f"MOSGSL pretrain for {n_pretrain} epochs.")
    total_time = 0.0
    max_mem = 0.0
    max_mem_res = 0.0
    key_metric = dataset_info['key_metric']
    higher_better = metric_higher_better(key_metric)
    best_val_metric = -float('inf') if higher_better else float('inf')
    best_model_state = None
    patience_count = 0
    pretrain_times = []
    for epoch in range(1, n_pretrain + 1):
        if patience_count >= cfg.train.patience:
            logger.info(f"MOSGSL pretrain early stopping at epoch {epoch}.")
            break
        loss, t_train, mem_alloc, mem_res = pretrain_epoch(
            train_loader, model, criterion, optimizer, device, cfg, dataset_info['type']
        )
        total_time += t_train
        max_mem = max(max_mem, mem_alloc)
        max_mem_res = max(max_mem_res, mem_res)
        if is_cuda_device(device):
            torch.cuda.synchronize(device)
        val_started = time.perf_counter()
        val_perf = eval_epoch_wo_gsl(
            val_loader,
            model,
            device,
            dataset_task=dataset_info['task'],
            only_backbone=bool(cfg.mosgsl.pretrain_backbone_only),
        )
        if is_cuda_device(device):
            torch.cuda.synchronize(device)
        pretrain_times.append(float(t_train + time.perf_counter() - val_started))
        current_val = val_perf[key_metric]
        improved = current_val > best_val_metric if higher_better else current_val < best_val_metric
        if improved:
            best_val_metric = current_val
            best_model_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
        if epoch == 1 or epoch % 50 == 0 or epoch == n_pretrain:
            logger.info(
                f"  Pretrain {epoch:03d} | Loss: {loss:.4f} | "
                f"Val {key_metric}: {current_val:.4f} (Best: {best_val_metric:.4f}) | "
                f"time: {t_train:.2f}s"
            )
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("MOSGSL pretrain: loaded best pretrain checkpoint.")
    return total_time, max_mem, max_mem_res, True, pretrain_times


def run_single_session(session_id, train_loader, val_loader, test_loader,
                       dataset_info, cfg, logger, graphs2subgraphs,
                       prune_callback=None, prune_step_offset=0,
                       evaluate_test=True, efficiency_raw_test=None):
    model = build_model(cfg, dataset_info, train_loader.dataset, graphs2subgraphs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
    criterion = get_criterion(cfg.train.loss_type).to(cfg.device)
    key_metric = dataset_info['key_metric']
    higher_better = metric_higher_better(key_metric)
    best_val_metric = -float('inf') if higher_better else float('inf')
    best_val_perf = {}
    best_model_state = None
    patience_count = 0
    epoch_count = 0
    best_epoch = 0
    cycle_times = []
    total_train_time = 0.0
    max_mem_train = 0.0
    max_mem_train_res = 0.0

    logger.info(f"Start Session {session_id} | Key Metric: {key_metric}")
    pre_time, pre_mem, pre_mem_res, pretrained, pretrain_times = run_pretrain(
        train_loader, val_loader, model, criterion, optimizer, cfg.device, cfg, dataset_info, logger
    )
    total_train_time += pre_time
    max_mem_train = max(max_mem_train, pre_mem)
    max_mem_train_res = max(max_mem_train_res, pre_mem_res)

    # Fine-tuning starts from the selected pretrain weights, but must not keep
    # Adam moments from the last (possibly non-best) pretrain epoch.  Keeping
    # those stale moments was particularly unstable for GNCE log-cardinality
    # regression.  A fresh optimizer also matches independent re-training of
    # the full SGSL model from the selected initialization.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd
    )
    scheduler, scheduler_name = make_scheduler(cfg, optimizer, key_metric)
    init_motif(model, train_loader, cfg.device, logger, cfg, pretrained=pretrained)

    for epoch in range(1, cfg.train.epochs + 1):
        if patience_count >= cfg.train.patience:
            logger.info(f"Session {session_id}: Early stopping at epoch {epoch}.")
            break

        if (
            model.use_motif
            and epoch >= int(cfg.mosgsl.motif_update_min_epoch)
            and int(cfg.mosgsl.motif_update_each) > 0
            and epoch % int(cfg.mosgsl.motif_update_each) == 0
        ):
            update_motif(model, train_loader, cfg.device, logger, cfg)

        epoch_count += 1
        train_loss, t_train, mem_tr_alloc, mem_tr_res = train_epoch(
            train_loader, model, criterion, optimizer, cfg.device, cfg, dataset_info['type']
        )
        total_train_time += t_train
        max_mem_train = max(max_mem_train, mem_tr_alloc)
        max_mem_train_res = max(max_mem_train_res, mem_tr_res)

        val_perf, t_val, mem_val_alloc, mem_val_res = eval_epoch(
            val_loader, model, cfg.device, dataset_task=dataset_info['task']
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

    test_perf = {}
    t_test = 0.0
    mem_test_alloc = 0.0
    mem_test_res = 0.0
    if evaluate_test:
        test_perf, t_test, mem_test_alloc, mem_test_res = eval_epoch(
            test_loader, model, cfg.device, dataset_task=dataset_info['task']
        )

    warm = int(cfg.efficiency.timed_train_warmup_epochs) if efficiency_enabled(cfg) else 1
    steady_pre = pretrain_times[warm:]
    steady_cycles = cycle_times[warm:]
    training_efficiency = {
        'pretrain_seconds': pre_time,
        'pretrain_cycle_seconds': float(sum(pretrain_times)),
        'pretrain_cycle_times_s': pretrain_times,
        'pretrain_steady_median_s': float(np.median(steady_pre)) if steady_pre else None,
        'pretrain_epochs_measured': int(cfg.mosgsl.pretrain_epochs),
        'finetune_cycle_times_s': cycle_times,
        'finetune_steady_cycle_median_s': float(np.median(steady_cycles)) if steady_cycles else None,
        'steady_cycle_median_s': float(np.median(steady_cycles)) if steady_cycles else None,
        'finetune_epochs_executed': epoch_count,
        'best_finetune_epoch': best_epoch,
        'train_peak_allocated_mb': max(max_mem_train, mem_val_alloc),
        'train_peak_reserved_mb': max(max_mem_train_res, mem_val_res),
    }
    strict_result = None
    if evaluate_test and efficiency_enabled(cfg, 'strict'):
        parsing_params = {
            'max_n_sub': int(cfg.mosgsl.max_n_sub),
            'max_sub_size': int(cfg.mosgsl.max_sub_size),
        }
        require_full_edge = not bool(cfg.mosgsl.gsl_one_by_one)

        use_edge_context = cfg.dataset.lower().startswith('gnce-')

        def strict_transform(raw_item):
            raw_graph, global_idx = raw_item
            graph = _sanitize_feature(raw_graph)
            if use_edge_context:
                graph = _append_edge_context(graph)
            graph.idx = torch.tensor([int(global_idx)], dtype=torch.long)
            parsed = prepare_parsed_data(
                [graph],
                parsing=str(cfg.mosgsl.parsing),
                parsing_params=parsing_params,
                require_full_edge=require_full_edge,
                # No progress-bar/logging is allowed inside a timed region.
                desc=None,
            )
            model.parsing.graphs2subgraphs.update(parsed)
            return graph

        strict_result = benchmark_session_if_enabled(
            cfg, model, test_loader.dataset, efficiency_raw_test,
            lambda current_model, data: current_model(data),
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

    def dict_to_str(d):
        return ", ".join([f"{k}: {v:.4f}" for k, v in d.items()])

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
        f"  Train | Total Time: {total_train_time:.2f}s, Epochs: {epoch_count}, "
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


def prepare_mosgsl_folds(folds_data, cfg, logger, include_test=True):
    parsing_params = {
        'max_n_sub': int(cfg.mosgsl.max_n_sub),
        'max_sub_size': int(cfg.mosgsl.max_sub_size),
    }
    require_full_edge = not bool(cfg.mosgsl.gsl_one_by_one)
    use_edge_context = cfg.dataset.lower().startswith('gnce-')
    cache_dir = str(MOSGSL_CACHE_DIR)
    if efficiency_enabled(cfg) and str(getattr(cfg.efficiency, 'cache_root', '')):
        cache_dir = str(Path(str(cfg.efficiency.cache_root)) / 'mosgsl')
    prepared = []
    logger.info(
        f"MOSGSL preprocessing: parsing={cfg.mosgsl.parsing}, "
        f"max_n_sub={parsing_params['max_n_sub']}, "
        f"max_sub_size={parsing_params['max_sub_size']}, "
        f"require_full_edge={require_full_edge}, "
        f"gnce_edge_context={use_edge_context}"
    )
    for fold_id, (train_d, val_d, test_d) in enumerate(folds_data):
        train_m, val_m, test_m, graphs2subgraphs = build_mosgsl_fold(
            train_d,
            val_d,
            test_d,
            dataset_name=cfg.dataset,
            fold_name=(
                f"fold{fold_id}_{dataset_cache_tag(cfg)}_full"
                if include_test
                else f"fold{fold_id}_{dataset_cache_tag(cfg)}_validation_only"
            ),
            cache_dir=cache_dir,
            parsing=cfg.mosgsl.parsing,
            parsing_params=parsing_params,
            require_full_edge=require_full_edge,
            force_reload=bool(cfg.mosgsl.force_reload),
            include_test=include_test,
            use_edge_context=use_edge_context,
        )
        prepared.append((train_m, val_m, test_m, graphs2subgraphs))
    return prepared


def apply_mosgsl_training_defaults(cfg, opts=None, config_path=""):
    opts = opts or []
    if config_path:
        return
    overridden = set(opts[0::2])
    if 'train.lr' not in overridden and cfg.train.lr == 1e-2:
        cfg.train.lr = 1e-3
    if 'train.wd' not in overridden and cfg.train.wd == 0.0:
        cfg.train.wd = 5e-4
    if 'train.batch_size' not in overridden and cfg.train.batch_size == 128:
        cfg.train.batch_size = 64


def run_mosgsl(cfg=None, key_metric_override=None, config_string=None,
               prune_callback=None, evaluate_test=True):
    args = None
    created_cfg_from_cli = cfg is None
    if cfg is None:
        args = parse_args()
        cfg = init_cfg()
        cfg = update_cfg(cfg, args=args)

    dataset_type = supported_dataset_type(cfg.dataset)
    set_loss_for_dataset(cfg, dataset_type)
    if created_cfg_from_cli:
        apply_mosgsl_training_defaults(
            cfg,
            args.opts if args is not None else None,
            args.config if args is not None else "",
        )

    if config_string is None:
        config_string = f"{cfg.dataset}_MOSGSL_{int(time.time())}"
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
    # Preserve all official G-CARE fold IDs while building MOSGSL caches.
    # The requested fold is selected only for the training session below.
    dataset_info = build_dataset_info(dataset, dataset_type)
    if key_metric_override is not None:
        dataset_info['key_metric'] = key_metric_override
        logger.info(f"Using key_metric override: {key_metric_override}")
    else:
        configured_metric = str(
            getattr(cfg.train, 'key_metric', 'auto')
        ).lower()
        if configured_metric != 'auto':
            dataset_info['key_metric'] = configured_metric
            logger.info(f"Using configured key_metric: {configured_metric}")

    folds_data = prepare_mosgsl_folds(
        folds_data, cfg, logger,
        include_test=(evaluate_test or efficiency_enabled(cfg)),
    )

    if efficiency_enabled(cfg, 'prep'):
        stripped_folds = [(train, val, test) for train, val, test, _ in folds_data]
        parsed_subgraphs = sum(
            len(items)
            for _, _, _, mapping in folds_data
            for items in mapping.values()
        )
        parsed_nodes = sum(
            int(subgraph.num_nodes)
            for _, _, _, mapping in folds_data
            for items in mapping.values()
            for subgraph in items
        )
        raw_nodes = sum(
            int(graph.num_nodes)
            for train, val, test in raw_folds_data
            for split in (train, val, test)
            for graph in split
        )
        result = finish_preprocessing(
            cfg, prep_started, stripped_folds, raw_folds_data,
            extra={
                'transform': 'MOSGSL parsing and subgraph enumeration',
                'parsed_subgraph_count': parsed_subgraphs,
                'parsed_subgraph_total_nodes': parsed_nodes,
                'node_inflation_factor': parsed_nodes / max(raw_nodes, 1),
                'cache_root': str(getattr(cfg.efficiency, 'cache_root', '')),
            },
        )
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
            train_d, val_d, test_d, graphs2subgraphs = folds_data[session_id]
            current_seed = cfg.seed
        else:
            train_d, val_d, test_d, graphs2subgraphs = folds_data[0]
            current_seed = cfg.seed + session_id
        raw_fold = raw_folds_data[session_id if use_fold_sessions else 0]
        raw_train, raw_val, raw_test = raw_fold
        raw_test_with_global_idx = [
            (raw_test[index], len(raw_train) + len(raw_val) + index)
            for index in range(len(raw_test))
        ]
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
                num_workers=cfg.num_workers,
            )
            if evaluate_test else None
        )

        results_run = run_single_session(
            session_id + 1,
            train_loader,
            val_loader,
            test_loader,
            dataset_info,
            cfg,
            logger,
            graphs2subgraphs,
            prune_callback=prune_callback,
            prune_step_offset=session_id * cfg.train.epochs,
            evaluate_test=evaluate_test,
            efficiency_raw_test=raw_test_with_global_idx,
        )
        stats_collector['training_efficiency'].append(results_run['training_efficiency'])

        for key in ['train_time', 'epoch_count', 'best_epoch', 'test_time', 'train_mem_alloc', 'train_mem_res', 'test_mem_alloc', 'test_mem_res']:
            stats_collector[key].append(results_run[key])
        for metric, value in results_run['test_perf'].items():
            stats_collector['test_perf'].setdefault(metric, []).append(value)
        for metric, value in results_run['best_val_perf'].items():
            stats_collector['best_val_perf'].setdefault(metric, []).append(value)

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
        for metric, stat in metrics.items():
            logger.info(f"{metric.upper()}: {stat['mean']:.4f} +/- {stat['std']:.4f}")
    logger.info("[Efficiency]")
    for metric, stat in aggregate['efficiency'].items():
        logger.info(f"{metric}: {stat['mean']:.4f} +/- {stat['std']:.4f}")
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
    return run_mosgsl()


if __name__ == '__main__':
    main()
