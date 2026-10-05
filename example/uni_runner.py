import copy
import datetime
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from efficiency.runtime import (
    benchmark_session_if_enabled,
    default_collate as efficiency_default_collate,
    efficiency_enabled,
    finish_preprocessing,
    load_prepared_input,
    restore_dataset_metadata,
    save_prepared_input,
    dataset_metadata,
    save_efficiency_checkpoint_if_enabled,
)

from configs import init_cfg, parse_args, update_cfg
from datasets import (
    load_dataset,
    base_dataset_name,
    dataset_loader_kwargs,
)
from engine import evaluate, get_criterion, seed_everything
from example.main_baselines import (
    evaluate_cardinality_outputs,
    is_cuda_device,
    setup_logger,
    stable_sigmoid,
)


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / 'configs'
DATA_DIR = BASE_DIR / 'data'


def supported_dataset_type(dataset):
    dataset = base_dataset_name(dataset)
    dataset_lower = dataset.lower()
    if dataset in {'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molpcba', 'ogbg-moltox21'}:
        return 'ogb'
    if dataset in {
        'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB',
        'MUTAG', 'NCI1', 'DD', 'ENZYMES', 'PROTEINS',
    }:
        return 'tu'
    if dataset_lower in {'peptides-func', 'peptides-struct'}:
        return 'lrgb'
    if dataset_lower in {'zinc', 'aqsol'}:
        return 'pyg_mol'
    if dataset_lower in {'gcare-aids', 'gcare-yago'} or dataset_lower.startswith('gnce-'):
        return 'tu'
    if dataset.startswith('Count-'):
        return 'counting'
    raise ValueError(f"Dataset {dataset} is not supported by this runner.")


def set_loss_for_dataset(cfg, dataset_type):
    dataset = base_dataset_name(cfg.dataset)
    dataset_lower = dataset.lower()
    if dataset in {'ogbg-molhiv', 'ogbg-molbace', 'IMDB-BINARY', 'REDDIT-BINARY', 'MUTAG', 'NCI1', 'DD', 'PROTEINS'}:
        cfg.train.loss_type = 'BCE'
    elif dataset in {'IMDB-MULTI', 'ENZYMES', 'COLLAB'}:
        cfg.train.loss_type = 'CE'
    elif dataset in {'ogbg-molpcba', 'ogbg-moltox21'} or dataset_lower == 'peptides-func':
        cfg.train.loss_type = 'BCE'
    elif dataset_lower in {'peptides-struct', 'zinc', 'aqsol', 'gcare-aids', 'gcare-yago'} or dataset_lower.startswith('gnce-') or dataset_type == 'counting':
        cfg.train.loss_type = 'L1'
    else:
        raise ValueError(f"Cannot infer loss_type for dataset={cfg.dataset}.")


def make_scheduler(cfg, optimizer, key_metric):
    scheduler_name = str(cfg.train.scheduler).lower() if cfg.train.scheduler is not None else 'none'
    if scheduler_name == 'steplr':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.train.lr_patience, gamma=cfg.train.lr_decay
        ), scheduler_name
    if scheduler_name == 'reducelr':
        mode = 'max' if is_higher_better_metric(key_metric) else 'min'
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


def supervised_loss(out, y, criterion, loss_type):
    if loss_type in ('L1', 'MSE'):
        y_target = y.float()
        out_for_loss = out
        if y_target.shape != out_for_loss.shape:
            y_target = y_target.squeeze(-1)
            out_for_loss = out_for_loss.squeeze(-1)
        finite_mask = torch.isfinite(y_target)
        if finite_mask.sum() == 0:
            return None, y.size(0)
        return criterion(out_for_loss[finite_mask], y_target[finite_mask]), y.size(0)

    if loss_type == 'CE':
        return criterion(out, y.squeeze().long()), y.size(0)

    y_target = y.float()
    mask = ~torch.isnan(y_target)
    if mask.sum() == 0:
        return None, y.size(0)
    if out.size(-1) == 1:
        mask_1d = mask.squeeze(-1) if mask.dim() > 1 else mask
        return criterion(out.squeeze(-1)[mask_1d], y_target.squeeze(-1)[mask_1d]), y.size(0)
    return criterion(out[mask], y_target[mask]), y.size(0)


def model_forward(model, data, dataset_type):
    if dataset_type in ('ogb', 'lrgb', 'pyg_mol'):
        x = data.x.long() if data.x.dtype == torch.float else data.x
        edge_attr = data.edge_attr
        if edge_attr is not None and edge_attr.dtype == torch.float:
            edge_attr = edge_attr.long()
    else:
        x = data.x
        edge_attr = data.edge_attr

    out, emb = model(
        x,
        data.edge_index,
        data.batch,
        edge_attr=edge_attr,
        fingerprint=getattr(data, 'fingerprint', None),
        rwpe=getattr(data, 'random_walk_pe', None),
    )
    return out, emb


def train_epoch(train_loader, model, criterion, optimizer, device, cfg, dataset_type,
                model_forward_fn=None, extra_loss_fn=None, model_name='model'):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    forward_fn = model_forward_fn or model_forward
    track_cuda = is_cuda_device(device)
    if track_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.time()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, emb = forward_fn(model, data, dataset_type)
        if not torch.isfinite(out).all():
            raise FloatingPointError(f"Non-finite {model_name} output during training.")
        loss, num_graphs = supervised_loss(out, data.y, criterion, cfg.train.loss_type)
        if loss is None:
            continue
        if extra_loss_fn is not None:
            extra_loss = extra_loss_fn(model, data, out, emb, loss, cfg, dataset_type)
            if extra_loss is not None:
                loss = loss + extra_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {model_name} loss: {loss.item()}")
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
def eval_epoch(loader, model, device, dataset_task='b', dataset_type='ogb',
               model_forward_fn=None, model_name='model'):
    model.eval()
    y_preds = []
    y_trues = []
    y_cards = []
    y_means = []
    y_stds = []
    t_forward_sum = 0.0
    track_cuda = is_cuda_device(device)
    forward_fn = model_forward_fn or model_forward
    if track_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    for data in loader:
        data = data.to(device)
        if track_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        out, _ = forward_fn(model, data, dataset_type)
        if not torch.isfinite(out).all():
            raise FloatingPointError(f"Non-finite {model_name} output during evaluation.")
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
        raise FloatingPointError(f"Non-finite raw {model_name} predictions during evaluation.")

    if dataset_task == 'reg':
        if y_preds.ndim > 1 and y_preds.shape[1] == 1:
            y_preds = y_preds.squeeze(1)
        if y_trues.ndim > 1 and y_trues.shape[1] == 1:
            y_trues = y_trues.squeeze(1)
        results = evaluate(y_trues, y_preds, task='reg')
    elif dataset_task == 'card':
        results = evaluate_cardinality_outputs(
            y_preds,
            torch.cat(y_cards, dim=0).numpy(),
            torch.cat(y_means, dim=0).numpy(),
            torch.cat(y_stds, dim=0).numpy(),
        )
    elif dataset_task == 'mc':
        y_preds_prob = torch.softmax(torch.tensor(y_preds), dim=1).numpy()
        results = evaluate(y_trues, y_preds_prob, task='mc')
    else:
        if y_preds.ndim > 1 and y_preds.shape[1] == 1:
            y_preds = y_preds.squeeze(1)
        results = evaluate(y_trues, stable_sigmoid(y_preds), task=dataset_task)
    return results, t_forward_sum, mem_allocated, mem_reserved


def build_dataset_info(dataset, dataset_type, max_degree):
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


def use_fold_sessions(dataset, dataset_type):
    return dataset_type == 'tu' and not dataset.lower().startswith('gnce-')


def run_single_session(session_id, train_loader, val_loader, test_loader,
                       dataset_info, cfg, logger, build_model_fn,
                       prune_callback=None, prune_step_offset=0,
                       model_forward_fn=None, extra_loss_fn=None,
                       model_name='model', evaluate_test=True,
                       efficiency_raw_test=None, efficiency_transform_fn=None,
                       efficiency_collate_fn=efficiency_default_collate):
    model = build_model_fn(cfg, dataset_info).to(cfg.device)
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
    criterion = get_criterion(cfg.train.loss_type).to(cfg.device)
    key_metric = dataset_info['key_metric']
    scheduler, scheduler_name = make_scheduler(cfg, optimizer, key_metric)
    higher_better = is_higher_better_metric(key_metric)
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
            train_loader, model, criterion, optimizer, cfg.device, cfg, dataset_info['type'],
            model_forward_fn=model_forward_fn,
            extra_loss_fn=extra_loss_fn,
            model_name=model_name,
        )
        total_train_time += t_train
        max_mem_train = max(max_mem_train, mem_tr_alloc)
        max_mem_train_res = max(max_mem_train_res, mem_tr_res)

        val_perf, t_val, mem_val_alloc, mem_val_res = eval_epoch(
            val_loader, model, cfg.device,
            dataset_task=dataset_info['task'],
            dataset_type=dataset_info['type'],
            model_forward_fn=model_forward_fn,
            model_name=model_name,
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
    else:
        logger.warning(f"Session {session_id}: No best model saved! Using last epoch model.")

    if evaluate_test:
        test_perf, t_test, mem_test_alloc, mem_test_res = eval_epoch(
            test_loader, model, cfg.device,
            dataset_task=dataset_info['task'],
            dataset_type=dataset_info['type'],
            model_forward_fn=model_forward_fn,
            model_name=model_name,
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
        session_forward_fn = model_forward_fn or model_forward

        def strict_forward(current_model, data):
            return session_forward_fn(current_model, data, dataset_info['type'])

        strict_result = benchmark_session_if_enabled(
            cfg,
            model,
            test_loader.dataset,
            efficiency_raw_test,
            strict_forward,
            session_id=session_id,
            transform_fn=efficiency_transform_fn,
            collate_fn=efficiency_collate_fn,
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
        'strict_efficiency': strict_result,
        'training_efficiency': training_efficiency,
        'checkpoint_path': checkpoint_path,
    }


def run_uni_baseline(model_name, build_model_fn, cfg=None, key_metric_override=None,
                     config_string=None, prune_callback=None,
                     model_forward_fn=None, extra_loss_fn=None,
                     preprocess_folds_fn=None, loader_cls=DataLoader,
                     efficiency_transform_fn=None,
                     efficiency_collate_fn=efficiency_default_collate,
                     evaluate_test=True):
    args = None
    if cfg is None:
        args = parse_args()
        cfg = init_cfg()
        cfg = update_cfg(cfg, args=args)

    if cfg.load_best_params:
        config_dataset = base_dataset_name(cfg.dataset)
        config_path = CONFIG_DIR / f'{config_dataset}_{model_name}.yaml'
        if not config_path.exists():
            raise FileNotFoundError(
                f"load_best_params=True but best-param config not found: {config_path}"
            )
        cfg.merge_from_file(str(config_path))
        if args is not None:
            cfg.merge_from_list(args.opts)

    dataset_type = supported_dataset_type(cfg.dataset)
    set_loss_for_dataset(cfg, dataset_type)
    cfg.model.use_fingerprints = False
    cfg.model.use_rwpe = False

    if config_string is None:
        config_string = f"{cfg.dataset}_{model_name}_{int(time.time())}"
    logger = setup_logger(cfg, config_string)
    logger.info(f"Config:\n{cfg}\n")
    logger.info(f'Start Time: {datetime.datetime.now().strftime("%Y/%m/%d - %H:%M")}\n')

    prep_started = time.perf_counter()
    efficiency_data_root = str(getattr(cfg.efficiency, 'data_root', '')) if efficiency_enabled(cfg) else ''
    prepared = load_prepared_input(cfg)
    if prepared is not None:
        dataset = restore_dataset_metadata(prepared['dataset_metadata'])
        folds_data = prepared['folds_data']
        raw_folds_data = prepared['raw_folds_data']
        max_degree = prepared['max_degree']
    else:
        dataset, folds_data, max_degree = load_dataset(
            dataset_name=cfg.dataset,
            root=efficiency_data_root or str(DATA_DIR),
            use_fingerprints=False,
            fingerprint_dim=0,
            use_rwpe=False,
            rwpe_dim=0,
            augm_gca=False,
            fold_k=cfg.cv.fold_k,
            fold_seed=cfg.cv.seed,
            gnn_type=None,
            **dataset_loader_kwargs(cfg),
        )
        raw_folds_data = folds_data
    dataset_info = build_dataset_info(dataset, dataset_type, max_degree)
    if key_metric_override is not None:
        dataset_info['key_metric'] = key_metric_override
        logger.info(f"Using key_metric override: {key_metric_override}")
    else:
        configured_metric = str(getattr(cfg.train, 'key_metric', 'auto')).lower()
        if configured_metric not in ('auto', 'none', ''):
            dataset_info['key_metric'] = configured_metric
            logger.info(f"Using configured key_metric: {configured_metric}")

    if prepared is None and preprocess_folds_fn is not None:
        folds_data = preprocess_folds_fn(
            folds_data, cfg, dataset_type, logger,
            include_test=evaluate_test,
        )

    if efficiency_enabled(cfg, 'prep'):
        result = finish_preprocessing(cfg, prep_started, folds_data, raw_folds_data)
        if bool(getattr(cfg.efficiency, 'require_prepared_input', False)):
            save_prepared_input(cfg, {
                'dataset_metadata': dataset_metadata(dataset),
                'folds_data': folds_data,
                'raw_folds_data': raw_folds_data,
                'max_degree': max_degree,
            })
        return result

    fold_sessions = use_fold_sessions(cfg.dataset, dataset_type)
    num_sessions = len(folds_data) if fold_sessions else cfg.train.runs
    session_indices = list(range(num_sessions))
    tuning_fold_limit = int(getattr(cfg.cv, 'tuning_fold_limit', 0))
    if fold_sessions and not evaluate_test and tuning_fold_limit > 0:
        session_indices = session_indices[:min(tuning_fold_limit, num_sessions)]
        logger.info(
            f"[TU tuning fold limit] validation-only trial uses folds "
            f"{session_indices}; final evaluation still uses all {num_sessions} folds."
        )
    if efficiency_enabled(cfg):
        selected = int(cfg.efficiency.fold_index) if fold_sessions else 0
        if selected < 0 or selected >= num_sessions:
            raise IndexError(
                f'efficiency.fold_index={selected} outside 0..{num_sessions - 1}'
            )
        session_indices = [selected]
    cv_dataset = 'G-CARE' if cfg.dataset.lower().startswith('gcare-') else 'TU'
    logger.info(
        f"{cv_dataset} Dataset (K-Fold = {num_sessions})"
        if fold_sessions
        else f"{dataset_type.upper()} Dataset (Repeated Runs = {num_sessions})"
    )

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
        if fold_sessions:
            train_d, val_d, test_d = folds_data[session_id]
            current_seed = cfg.seed
        else:
            train_d, val_d, test_d = folds_data[0]
            current_seed = cfg.seed + session_id

        seed_everything(current_seed, deterministic=cfg.deterministic, warn_only=cfg.deterministic_warn_only)
        train_loader = loader_cls(train_d, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False)
        val_loader = loader_cls(val_d, cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)
        test_loader = (
            loader_cls(test_d, cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)
            if evaluate_test else None
        )

        results_run = run_single_session(
            session_id + 1, train_loader, val_loader, test_loader,
            dataset_info, cfg, logger, build_model_fn,
            prune_callback=prune_callback,
            prune_step_offset=session_id * cfg.train.epochs,
            model_forward_fn=model_forward_fn,
            extra_loss_fn=extra_loss_fn,
            model_name=model_name,
            evaluate_test=evaluate_test,
            efficiency_raw_test=raw_folds_data[session_id if fold_sessions else 0][2],
            efficiency_transform_fn=efficiency_transform_fn,
            efficiency_collate_fn=efficiency_collate_fn,
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

    for section, metrics in [('Test Effectiveness', aggregate['test_perf']), ('Best Val', aggregate['best_val_perf'])]:
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
        'num_sessions': len(session_indices),
        'raw': stats_collector,
        'aggregate': aggregate,
        'best_val_perf': aggregate['best_val_perf'],
        'test_perf': aggregate['test_perf'],
        'efficiency': aggregate['efficiency'],
    }
