import os
import sys
import time
import datetime
import copy
import logging
from pathlib import Path
import numpy as np
import torch
import torch_geometric.transforms as T
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
from engine import adaptive_aug, compute_nt_xent_loss, evaluate, get_criterion, seed_everything
from efficiency.runtime import (
    benchmark_session_if_enabled,
    deterministic_indices,
    efficiency_enabled,
    environment_snapshot,
    finish_preprocessing,
    load_prepared_input,
    restore_dataset_metadata,
    save_prepared_input,
    dataset_metadata,
    summarize_latency,
    write_result,
)
from models import (
    GLGNN,
    GraphFeatureMLP,
    MLP_BASELINE_BACKBONES,
    TABULAR_BASELINE_BACKBONES,
    build_graph_feature_batch,
    graph_feature_dim,
    gnn_type_supports_edgepool,
)

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / 'configs'
DATA_DIR = BASE_DIR / 'data'
LOG_DIR = BASE_DIR / 'logs'


def validate_pooling_cfg(cfg):
    if cfg.model.backbone in TABULAR_BASELINE_BACKBONES:
        cfg.model.pooling_method = 'none'
        cfg.model.use_virtual_node = False
        return

    valid_methods = {'none', 'topk', 'edgepool', 'gmt'}
    if cfg.model.pooling_method not in valid_methods:
        raise ValueError(f"Unknown pooling_method: {cfg.model.pooling_method}. Supported: {sorted(valid_methods)}")

    valid_readouts = {'mean', 'sum', 'add', 'max', 'attn'}
    if cfg.model.readout_type not in valid_readouts:
        raise ValueError(f"Unknown readout_type: {cfg.model.readout_type}. Supported: {sorted(valid_readouts)}")

    if cfg.model.pooling_method == 'edgepool' and not gnn_type_supports_edgepool(cfg.model.backbone):
        raise ValueError(
            "pooling_method='edgepool' only supports node-only backbones "
            f"GCN / GIN / SAGE, but got model.backbone={cfg.model.backbone}."
        )
    
    if cfg.model.pooling_method != 'none':
        cfg.model.use_virtual_node = False
        print(f"Disabling virtual node when pooling_method={cfg.model.pooling_method} is used.")


def setup_logger(cfg, config_string):
    log_dir = LOG_DIR / cfg.dataset
    os.makedirs(log_dir, exist_ok=True)
    log_path = log_dir / f'{config_string}.txt'

    logger = logging.getLogger(config_string)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def is_cuda_device(device):
    return torch.device(device).type == 'cuda' and torch.cuda.is_available()


def stable_sigmoid(x):
    out = np.empty_like(x, dtype=np.float64)
    pos_mask = x >= 0
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[~pos_mask])
    out[~pos_mask] = exp_x / (1.0 + exp_x)
    return out


def evaluate_cardinality_outputs(y_preds, y_cards, y_means, y_stds):
    if y_preds.ndim > 1 and y_preds.shape[1] == 1:
        y_preds = y_preds.squeeze(1)
    y_cards = y_cards.squeeze()
    y_means = y_means.squeeze()
    y_stds = y_stds.squeeze()
    pred_log = y_preds * y_stds + y_means
    pred_log = np.clip(pred_log, -50, 50)
    pred_card = np.exp(pred_log)
    return evaluate(y_cards, pred_card, task='card')


def is_mlp_feature_baseline(backbone):
    return backbone in MLP_BASELINE_BACKBONES


def is_lightgbm_baseline(backbone):
    return backbone == 'LightGBM'


def tabular_include_structure(cfg):
    if cfg.model.backbone == 'LightGBM':
        return True
    if cfg.model.backbone in {'MLPFeatureStruct', 'MLPStruct'}:
        return True
    if cfg.model.backbone in {'MLPFeature', 'MLP'}:
        return False
    return str(cfg.baseline.feature_mode).lower() in {'feature_struct', 'struct', 'features_struct'}


def tabular_include_sum(dataset_info):
    return not bool(dataset_info.get('has_blind_tabular_features', False))


def tabular_include_topology(dataset_info, cfg):
    return bool(tabular_include_structure(cfg) and dataset_info.get('has_topology_size', False))


def infer_tabular_input_dim(dataset_info, cfg):
    return graph_feature_dim(
        dataset_info.get('tabular_node_feature_dim', dataset_info['node_feature_dim']),
        dataset_info.get('tabular_edge_feature_dim', dataset_info['edge_feature_dim']),
        include_structure=tabular_include_structure(cfg),
        include_sum=tabular_include_sum(dataset_info),
        include_topology=tabular_include_topology(dataset_info, cfg),
    )


@torch.no_grad()
def extract_tabular_arrays(dataset, batch_size, include_structure,
                           include_sum=True, include_topology=False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    xs, ys, y_cards, y_means, y_stds = [], [], [], [], []
    for data in loader:
        x = getattr(data, 'x_blind', data.x)
        edge_attr = getattr(data, 'edge_attr_blind', getattr(data, 'edge_attr', None))
        feat = build_graph_feature_batch(
            x,
            data.edge_index,
            data.batch,
            edge_attr=edge_attr,
            include_structure=include_structure,
            include_sum=include_sum,
            topology_size=getattr(data, 'topology_size', None),
            include_topology=include_topology,
        )
        xs.append(feat.cpu())
        ys.append(data.y.view(data.num_graphs, -1).float().cpu())
        if hasattr(data, 'y_card'):
            y_cards.append(data.y_card.view(data.num_graphs, -1).float().cpu())
            y_means.append(data.y_mean.view(data.num_graphs, -1).float().cpu())
            y_stds.append(data.y_std.view(data.num_graphs, -1).float().cpu())
    out = {
        'x': torch.cat(xs, dim=0).numpy(),
        'y': torch.cat(ys, dim=0).numpy(),
    }
    if y_cards:
        out['y_card'] = torch.cat(y_cards, dim=0).numpy()
        out['y_mean'] = torch.cat(y_means, dim=0).numpy()
        out['y_std'] = torch.cat(y_stds, dim=0).numpy()
    return out


def evaluate_tabular_predictions(pred, arrays, dataset_task):
    if dataset_task == 'reg':
        y_true = arrays['y']
        if pred.ndim > 1 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if y_true.ndim > 1 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)
        return evaluate(y_true, pred, task='reg')
    if dataset_task == 'card':
        return evaluate_cardinality_outputs(
            np.asarray(pred),
            arrays['y_card'],
            arrays['y_mean'],
            arrays['y_std'],
        )
    if dataset_task == 'mc':
        return evaluate(arrays['y'], np.asarray(pred), task='mc')
    if dataset_task == 'b':
        pred = np.asarray(pred)
        if pred.ndim == 2 and pred.shape[1] == 2:
            pred = pred[:, 1]
        return evaluate(arrays['y'], pred, task='b')
    return evaluate(arrays['y'], np.asarray(pred), task=dataset_task)


def make_lightgbm_model(cfg, dataset_task, seed):
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as exc:
        raise ImportError(
            "LightGBM baseline requires the 'lightgbm' package. "
            "Install it with `pip install lightgbm` in the benchmark environment."
        ) from exc

    params = {
        'n_estimators': int(cfg.baseline.lightgbm_n_estimators),
        'learning_rate': float(cfg.baseline.lightgbm_learning_rate),
        'num_leaves': int(cfg.baseline.lightgbm_num_leaves),
        'max_depth': int(cfg.baseline.lightgbm_max_depth),
        'min_child_samples': int(cfg.baseline.lightgbm_min_child_samples),
        'subsample': float(cfg.baseline.lightgbm_subsample),
        'colsample_bytree': float(cfg.baseline.lightgbm_colsample_bytree),
        'reg_alpha': float(cfg.baseline.lightgbm_reg_alpha),
        'reg_lambda': float(cfg.baseline.lightgbm_reg_lambda),
        'random_state': int(seed),
        'n_jobs': int(getattr(cfg.efficiency, 'cpu_threads', 10)) if efficiency_enabled(cfg) else -1,
        'verbose': -1,
    }
    if dataset_task in ('reg', 'card'):
        return LGBMRegressor(objective='regression', **params)
    if dataset_task == 'mc':
        return LGBMClassifier(objective='multiclass', **params)
    if dataset_task == 'b':
        return LGBMClassifier(objective='binary', **params)
    raise ValueError("LightGBM baseline currently supports graph-level b/mc/reg/card tasks.")


def run_lightgbm_session(session_id, train_d, val_d, test_d, dataset_info, cfg, logger,
                         seed, prune_callback=None, prune_step_offset=0,
                         evaluate_test=True):
    include_structure = tabular_include_structure(cfg)
    include_sum = tabular_include_sum(dataset_info)
    include_topology = tabular_include_topology(dataset_info, cfg)
    logger.info(
        f"Start Session {session_id} | LightGBM | "
        f"features={'feature+structure' if include_structure else 'feature'} | "
        f"Key Metric: {dataset_info['key_metric']}"
    )

    t0 = time.time()
    train_arrays = extract_tabular_arrays(
        train_d, cfg.train.batch_size, include_structure,
        include_sum=include_sum, include_topology=include_topology,
    )
    val_arrays = extract_tabular_arrays(
        val_d, cfg.train.batch_size, include_structure,
        include_sum=include_sum, include_topology=include_topology,
    )
    trainval_feature_seconds = time.time() - t0
    model = make_lightgbm_model(cfg, dataset_info['task'], seed)

    y_train = train_arrays['y']
    if y_train.ndim > 1 and y_train.shape[1] == 1:
        y_train_fit = y_train.squeeze(1)
    else:
        y_train_fit = y_train
    if dataset_info['task'] in ('b', 'mc'):
        y_train_fit = y_train_fit.squeeze().astype(int)

    if efficiency_enabled(cfg, 'strict'):
        # The efficiency protocol enables validation-based early stopping.
        try:
            from lightgbm import early_stopping, log_evaluation

            rounds = int(getattr(cfg.efficiency, 'lgbm_early_stopping_rounds', 0))
            if rounds > 0:
                callbacks = [
                    log_evaluation(period=0),
                    early_stopping(
                        stopping_rounds=rounds, first_metric_only=True,
                        verbose=False,
                    ),
                ]
            model = make_lightgbm_model(cfg, dataset_info['task'], seed)
            t0 = time.time()
            if rounds > 0:
                val_target = val_arrays['y'].squeeze()
                if dataset_info['task'] in ('b', 'mc'):
                    val_target = val_target.astype(int)
                elif dataset_info['task'] == 'card' and 'y_card' in val_arrays:
                    # `y` is already the normalized log-cardinality used by
                    # the training loss; y_card/y_mean/y_std are retained for
                    # metrics.
                    val_target = val_arrays['y'].squeeze()
                model.fit(
                    train_arrays['x'], y_train_fit,
                    eval_set=[(val_arrays['x'], val_target)],
                    callbacks=callbacks,
                )
            else:
                # Reproduce fixed-round GNCE/G-CARE LGBM exactly: validation
                # is evaluated after fitting, not once per boosting round.
                model.fit(train_arrays['x'], y_train_fit)
            fit_seconds = time.time() - t0
            total_train_time = trainval_feature_seconds + fit_seconds
        except TypeError as exc:
            raise RuntimeError(
                'Strict LightGBM validation early stopping is unsupported by '
                'the installed LightGBM API.'
            ) from exc
    else:
        model.fit(train_arrays['x'], y_train_fit)
        total_train_time = time.time() - t0
        fit_seconds = max(0.0, total_train_time - trainval_feature_seconds)

    # Test feature materialization is inference/preprocessing, never training.
    test_arrays = None
    if evaluate_test:
        test_arrays = extract_tabular_arrays(
            test_d, cfg.train.batch_size, include_structure,
            include_sum=include_sum, include_topology=include_topology,
        )

    t_val0 = time.time()
    if dataset_info['task'] in ('b', 'mc'):
        val_pred = model.predict_proba(val_arrays['x'])
    else:
        val_pred = model.predict(val_arrays['x'])
    val_time = time.time() - t_val0
    val_perf = evaluate_tabular_predictions(val_pred, val_arrays, dataset_info['task'])
    if prune_callback is not None:
        prune_callback(val_perf[dataset_info['key_metric']], prune_step_offset + 1)

    test_perf = {}
    test_time = 0.0
    if evaluate_test:
        t_test0 = time.time()
        if dataset_info['task'] in ('b', 'mc'):
            test_pred = model.predict_proba(test_arrays['x'])
        else:
            test_pred = model.predict(test_arrays['x'])
        test_time = time.time() - t_test0
        test_perf = evaluate_tabular_predictions(
            test_pred, test_arrays, dataset_info['task']
        )

    try:
        import resource
        rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mb = float(rss_raw) / 1024.0
    except (ImportError, AttributeError):
        peak_rss_mb = None
    best_iteration = int(
        getattr(model, 'best_iteration_', 0) or cfg.baseline.lightgbm_n_estimators
    )
    training_efficiency = {
        'training_mode': str(getattr(cfg.efficiency, 'training_mode', 'direct_full_fit')),
        'fit_seconds': float(fit_seconds),
        'trainval_feature_seconds': float(trainval_feature_seconds),
        'fit_plus_feature_seconds': float(total_train_time),
        'best_iteration': best_iteration,
        'peak_rss_mb': peak_rss_mb,
        'train_threads': int(getattr(cfg.efficiency, 'cpu_threads', 10)),
    }

    strict_result = None
    if evaluate_test and efficiency_enabled(cfg, 'strict'):
        # Strict single-query deployment uses one CPU thread. Training above
        # uses the explicit physical-core allocation assigned to this job.
        model.set_params(n_jobs=1)
        indices = deterministic_indices(
            len(test_d), int(cfg.efficiency.sample_count),
            int(cfg.efficiency.sample_seed),
        )
        warm = indices[:min(int(cfg.efficiency.warmup_count), len(indices))]
        predict_fn = model.predict_proba if dataset_info['task'] in ('b', 'mc') else model.predict
        for idx in warm:
            predict_fn(test_arrays['x'][idx:idx + 1])

        fwd_rounds = []
        e2e_rounds = []
        for _ in range(int(cfg.efficiency.latency_repeats)):
            values = []
            for idx in indices:
                start = time.perf_counter()
                predict_fn(test_arrays['x'][idx:idx + 1])
                values.append((time.perf_counter() - start) * 1000.0)
            fwd_rounds.append(values)

            values = []
            for idx in indices:
                start = time.perf_counter()
                one = extract_tabular_arrays(
                    [test_d[idx]], 1, include_structure,
                    include_sum=include_sum, include_topology=include_topology,
                )
                predict_fn(one['x'])
                values.append((time.perf_counter() - start) * 1000.0)
            e2e_rounds.append(values)

        booster_text = model.booster_.model_to_string()
        booster_dump = model.booster_.dump_model()
        total_leaves = sum(
            int(tree.get('num_leaves', 0))
            for tree in booster_dump.get('tree_info', [])
        )
        throughput_start = time.perf_counter()
        predict_fn(test_arrays['x'])
        throughput_seconds = time.perf_counter() - throughput_start
        strict_result = {
            'schema_version': 1,
            'entry_id': str(cfg.efficiency.entry_id),
            'session_id': int(session_id),
            'environment': environment_snapshot(),
            'device': 'CPU',
            'model': {
                'parameters': None,
                'state_dict_size_mb': len(booster_text.encode('utf-8')) / 2**20,
                'num_trees': int(model.booster_.num_trees()),
                'total_leaves': total_leaves,
            },
            'training': training_efficiency,
            'inference': {
                'sample_indices': indices,
                'sample_seed': int(cfg.efficiency.sample_seed),
                'threads': 1,
                'forward': summarize_latency(fwd_rounds),
                'end_to_end': summarize_latency(e2e_rounds),
                'peak_rss_mb': peak_rss_mb,
            },
            'throughput': {
                'graphs': len(test_d),
                'seconds': throughput_seconds,
                'graphs_per_second': len(test_d) / max(throughput_seconds, 1e-12),
                'num_threads': 1,
            },
        }
        strict_result['environment_after'] = environment_snapshot()
        safe_id = str(cfg.efficiency.entry_id).replace('/', '_').replace('::', '__')
        strict_path = Path(str(cfg.efficiency.output_dir)) / 'strict' / f'{safe_id}__session{session_id}.json'
        write_result(strict_path, strict_result)
        logger.info(f'[efficiency strict] wrote {strict_path}')

    checkpoint_path = None
    if efficiency_enabled(cfg) and bool(getattr(cfg.efficiency, 'save_checkpoint', False)):
        safe_id = str(cfg.efficiency.entry_id).replace('/', '_').replace('::', '__')
        output_session_id = int(getattr(cfg.efficiency, 'output_session_id', session_id))
        checkpoint_path = (
            Path(str(cfg.efficiency.output_dir)) / 'checkpoints' /
            str(cfg.efficiency.phase) /
            f'{safe_id}__session{output_session_id}.pt'
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'entry_id': str(cfg.efficiency.entry_id),
            'session_id': output_session_id,
            'seed': int(seed),
            'best_iteration': best_iteration,
            'booster_model_string': model.booster_.model_to_string(num_iteration=best_iteration),
        }, checkpoint_path)
        logger.info(f'[efficiency checkpoint] wrote {checkpoint_path}')

    def dict_to_str(d):
        return ", ".join([f"{k}: {v:.4f}" for k, v in d.items()])

    test_log = (
        f"  Test Metrics: {dict_to_str(test_perf)}\n"
        f"  Test-Infer | Time: {test_time:.2f} s, Mem Alloc: 0.00 MB, Res: 0.00 MB\n"
        if evaluate_test else
        "  Test Metrics: skipped (validation-only tuning trial)\n"
    )
    logger.info(
        f"  Val Metrics: {dict_to_str(val_perf)}\n"
        f"{test_log}"
        f"  Train | Total Time: {total_train_time:.2f}s, Epochs: 1, Mem Alloc: 0.00 MB, Res: 0.00 MB\n"
    )

    return {
        'test_perf': test_perf,
        'best_val_perf': val_perf,
        'train_time': total_train_time,
        'epoch_count': 1,
        'best_epoch': best_iteration,
        'test_time': test_time,
        'train_mem_alloc': 0.0,
        'train_mem_res': 0.0,
        'test_mem_alloc': 0.0,
        'test_mem_res': 0.0,
        'strict_efficiency': strict_result,
        'training_efficiency': training_efficiency,
        'checkpoint_path': str(checkpoint_path) if checkpoint_path else None,
    }


def train_epoch(train_loader, model, criterion, optimizer, device, cfg, dataset_type='ogb'):
    model.train()

    use_augm_gca = cfg.use_augm_gca

    total_loss = 0
    track_cuda = is_cuda_device(device)
    if track_cuda:
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
        y = data.y

        mask = ~torch.isnan(y)
        if mask.sum() == 0: continue

        model_x = x
        model_edge_attr = edge_attr
        if isinstance(model, GraphFeatureMLP):
            model_x = getattr(data, 'x_blind', x)
            model_edge_attr = getattr(data, 'edge_attr_blind', edge_attr)

        out, _ = model(
            model_x, data.edge_index, data.batch, model_edge_attr, fingerprint,
            rwpe=rwpe, topology_size=getattr(data, 'topology_size', None),
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
        elif cfg.train.loss_type == 'CE': # Multi-class
            y_target = y.squeeze().long()
            loss_label = criterion(out, y_target)
        else: # BCE, Focal, Asymmetric (Binary/Multi-label)
            y_target = y.float()
            if out.size(-1) == 1:
                # loss_label = criterion(out.squeeze(-1)[mask], y_target[mask])
                mask_1d = mask.squeeze(-1) if mask.dim() > 1 else mask
                loss_label = criterion(out.squeeze(-1)[mask_1d], y_target.squeeze(-1)[mask_1d])
            else:
                loss_label = criterion(out[mask], y_target[mask])

        loss = loss_label
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite supervised loss during training: {loss.item()}")

        if use_augm_gca:
            x_emb_origin = model.input_encoder(x)
            ei1, ea1, x_emb1 = adaptive_aug(data, x_emb_origin, cfg.augm_gca.pe1, cfg.augm_gca.pf1, cfg.augm_gca.p_tau, p_corr='min')
            ei2, ea2, x_emb2 = adaptive_aug(data, x_emb_origin, cfg.augm_gca.pe2, cfg.augm_gca.pf2, cfg.augm_gca.p_tau, p_corr='min')

            _, emb1 = model(None, ei1, data.batch, ea1, fingerprint, rwpe, input_embedding=x_emb1)
            _, emb2 = model(None, ei2, data.batch, ea2, fingerprint, rwpe, input_embedding=x_emb2)

            loss_cl = compute_nt_xent_loss(emb1, emb2, temperature=cfg.train.temperature) * cfg.train.contrast_weight
            if not torch.isfinite(loss_cl):
                raise FloatingPointError(f"Non-finite contrastive loss during training: {loss_cl.item()}")
            loss = loss + loss_cl

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite total loss during training: {loss.item()}")
        loss.backward()
        batch_loss = loss.item()

        if dataset_type != 'counting':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += batch_loss

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

        if track_cuda: torch.cuda.synchronize()
        t0 = time.time()

        model_x = x
        model_edge_attr = edge_attr
        if isinstance(model, GraphFeatureMLP):
            model_x = getattr(data, 'x_blind', x)
            model_edge_attr = getattr(data, 'edge_attr_blind', edge_attr)

        out, _ = model(
            model_x, data.edge_index, data.batch, model_edge_attr, fingerprint,
            rwpe=rwpe, topology_size=getattr(data, 'topology_size', None),
        )
        if not torch.isfinite(out).all():
            raise FloatingPointError("Non-finite model output during evaluation.")
        
        if track_cuda: torch.cuda.synchronize()
        t_forward_sum += (time.time() - t0)
        
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

    if dataset_task == 'reg': # Regression
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
    elif dataset_task == 'mc': # Multi-class
        y_preds_prob = torch.softmax(torch.tensor(y_preds), dim=1).numpy()
        if not np.isfinite(y_preds_prob).all():
            raise FloatingPointError("Non-finite prediction probabilities during evaluation.")
        results = evaluate(y_trues, y_preds_prob, task=dataset_task)
    else: # Binary or Multi-label
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
    max_degree = dataset_info['max_degree']
    deg = torch.zeros(max_degree + 1, dtype=torch.long)

    _dataset_iter = train_loader.dataset
    if isinstance(_dataset_iter, list):
        for data in _dataset_iter:
            if data.edge_index.numel() > 0:
                num_nodes_safe = max(data.num_nodes, int(data.edge_index.max().item()) + 1)
                d = degree(data.edge_index[1], num_nodes=num_nodes_safe, dtype=torch.long)
                max_d = int(d.max())
                if max_d >= deg.size(0):
                    deg = torch.cat([deg, torch.zeros(max_d - deg.size(0) + 1, dtype=torch.long)])
                deg += torch.bincount(d, minlength=deg.size(0))
    else:
        for data in _dataset_iter:
            num_nodes_safe = max(data.num_nodes, int(data.edge_index.max().item()) + 1) if data.edge_index.numel() > 0 else data.num_nodes
            d = degree(data.edge_index[1], num_nodes=num_nodes_safe, dtype=torch.long)
            deg += torch.bincount(d, minlength=max_degree + 1)

    deg = deg.to(cfg.device)

    if is_mlp_feature_baseline(cfg.model.backbone):
        model = GraphFeatureMLP(
            in_dim=infer_tabular_input_dim(dataset_info, cfg),
            out_dim=dataset_info['output_dim'],
            hidden_size=cfg.baseline.hidden_size,
            num_layers=cfg.baseline.num_layers,
            dropout=cfg.baseline.dropout,
            norm_type=cfg.baseline.norm_type,
            act_type=cfg.baseline.act_type,
            include_structure=tabular_include_structure(cfg),
            include_sum=tabular_include_sum(dataset_info),
            include_topology=tabular_include_topology(dataset_info, cfg),
        ).to(cfg.device)
    else:
        extra_model_kwargs = {
            'n_heads_gat': 4,
            'train_eps_gin': cfg.model.train_eps_gin,
            'aggr_sage': cfg.model.aggr_sage,
            'towers_pna': cfg.model.towers_pna,
            'pre_layers_pna': cfg.model.pre_layers_pna,
            'post_layers_pna': cfg.model.post_layers_pna,
            'divide_input_pna': cfg.model.divide_input_pna,
        }

        model = GLGNN(
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
            use_rwpe=cfg.model.use_rwpe,
            rwpe_dim=cfg.model.rwpe_dim,
            norm_type=cfg.model.norm_type,
            act_type=cfg.model.act_type,
            use_ffn=cfg.model.use_ffn,
            ffn_expansion=cfg.model.ffn_expansion,
            **extra_model_kwargs
        ).to(cfg.device)
    
    model.reset_parameters()
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

    key_metric = dataset_info['key_metric']
    logger.info(f"Start Session {session_id} | Key Metric: {key_metric}")

    scheduler_name = str(cfg.train.scheduler).lower() if cfg.train.scheduler is not None else 'none'
    scheduler = None
    if scheduler_name == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train.lr_patience, gamma=cfg.train.lr_decay)
    elif scheduler_name == 'reducelr':
        if key_metric in ('acc', 'auc', 'auc_macro', 'auc_micro', 'ap', 'ap_macro', 'ap_micro', 'r2'):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=cfg.train.factor_R, patience=cfg.train.lr_patience_R, min_lr=1e-6)
        elif key_metric in ('mae', 'mre', 'mse', 'rmse', 'medq', 'p90q', 'p95q', 'p99q', 'meanq', 'maxq', 'log_mae'):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.train.factor_R, patience=cfg.train.lr_patience_R, min_lr=1e-6)
        else:
            raise ValueError(f"Check key_metric for mode='max|min' in ReduceLROnPlateau scheduler!")
    elif scheduler_name == 'cosrestart':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.train.t_0, T_mult=cfg.train.t_mult, eta_min=cfg.train.eta_min)
    elif scheduler_name in ('none', 'null', ''):
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {cfg.train.scheduler}. Supported: None, StepLR, ReduceLR, CosRestart.")

    criterion = get_criterion(cfg.train.loss_type).to(cfg.device)

    if key_metric in ('acc', 'auc', 'auc_macro', 'auc_micro', 'ap', 'ap_macro', 'ap_micro', 'r2'):
        best_val_metric = -float('inf')
        is_higher_better = True
    elif key_metric in ('mae', 'mre', 'mse', 'rmse', 'medq', 'p90q', 'p95q', 'p99q', 'meanq', 'maxq', 'log_mae'):
        best_val_metric = float('inf')
        is_higher_better = False
    else:
        raise ValueError(f"Check key_metric: {key_metric}!")
    best_val_perf = {}
    best_model_state = None
    patience_count = 0
    best_epoch = 0
    cycle_times = []
    epoch_history = []
    
    total_train_time = 0
    epoch_count = 0
    max_mem_train = 0
    max_mem_train_res = 0
    max_mem_val = 0
    max_mem_val_res = 0

    for epoch in range(1, cfg.train.epochs + 1):
        if patience_count >= cfg.train.patience:
            logger.info(f"Session {session_id}: Early stopping at epoch {epoch}.")
            break
        
        epoch_count += 1

        # Train + validation.  If a late epoch becomes non-finite after a valid
        # checkpoint has already been selected, treat it as numerical early
        # stopping and retain that checkpoint.  This avoids discarding an
        # otherwise valid fold while still failing loudly when no finite model
        # has ever been observed.
        try:
            if is_cuda_device(cfg.device):
                torch.cuda.synchronize(cfg.device)
            cycle_started = time.perf_counter()
            train_loss, t_train, mem_tr_alloc, mem_tr_res = train_epoch(train_loader, model, criterion, optimizer, cfg.device, cfg, dataset_info['type'])
            total_train_time += t_train
            max_mem_train = max(max_mem_train, mem_tr_alloc)
            max_mem_train_res = max(max_mem_train_res, mem_tr_res)

            val_perf, t_val, mem_val_alloc, mem_val_res = eval_epoch(val_loader, model, cfg.device, dataset_task=dataset_info['task'], dataset_type=dataset_info['type'])
        except FloatingPointError as exc:
            if best_model_state is None:
                raise
            logger.warning(
                f"Session {session_id}: numerical early stop at epoch {epoch} "
                f"({exc}); retaining finite best checkpoint from epoch {best_epoch}."
            )
            break
        if is_cuda_device(cfg.device):
            torch.cuda.synchronize(cfg.device)
        cycle_elapsed = time.perf_counter() - cycle_started
        max_mem_val = max(max_mem_val, mem_val_alloc)
        max_mem_val_res = max(max_mem_val_res, mem_val_res)
        cycle_times.append(cycle_elapsed)
        epoch_history.append({
            'epoch': int(epoch),
            'train_loss': float(train_loss),
            'val_metrics': {key: float(value) for key, value in val_perf.items()},
            'train_time_s': float(t_train),
            'val_time_s': float(t_val),
            'cycle_time_s': float(cycle_elapsed),
            'train_mem_allocated_mb': float(mem_tr_alloc),
            'train_mem_reserved_mb': float(mem_tr_res),
            'val_mem_allocated_mb': float(mem_val_alloc),
            'val_mem_reserved_mb': float(mem_val_res),
        })

        if key_metric not in val_perf:
            raise KeyError(f"Key metric '{key_metric}' not found in validation metrics. Available metrics: {sorted(val_perf)}")

        if scheduler_name == 'reducelr':
            scheduler.step(val_perf[key_metric])
        elif scheduler is not None:
            scheduler.step()

        current_val = val_perf[key_metric]
        if (current_val > best_val_metric and is_higher_better) or (current_val < best_val_metric and not is_higher_better):
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
            logger.info(f"  Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val {key_metric}: {current_val:.4f} (Best: {best_val_metric:.4f}) | "
                        f"Train | time: {t_train:.2f} s, mem_allocated: {mem_tr_alloc:.2f} MB, mem_reserved: {mem_tr_res:.2f} MB | "
                        f"Val-infer | time: {t_val:.2f} s, mem_allocated: {mem_val_alloc:.2f} MB, mem_reserved: {mem_val_res:.2f} MB")

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
            test_loader, model, cfg.device,
            dataset_task=dataset_info['task'],
            dataset_type=dataset_info['type'],
        )

    strict_result = None
    training_efficiency = {
        'cycle_definition': 'one full training epoch + one full validation pass',
        'cycle_times_s': cycle_times,
        'steady_cycle_median_s': (
            float(np.median(cycle_times[int(cfg.efficiency.timed_train_warmup_epochs):]))
            if efficiency_enabled(cfg) and cycle_times[int(cfg.efficiency.timed_train_warmup_epochs):]
            else None
        ),
        'epochs_executed': epoch_count,
        'best_epoch': best_epoch,
        'train_peak_allocated_mb': max(max_mem_train, max_mem_val),
        'train_peak_reserved_mb': max(max_mem_train_res, max_mem_val_res),
    }
    if (
            evaluate_test and efficiency_enabled(cfg, 'strict')
            and bool(getattr(cfg.efficiency, 'measure_inference', True))):
        def strict_forward(current_model, data):
            if dataset_info['type'] in ('ogb', 'lrgb', 'pyg_mol', 'good_mol'):
                x = data.x.long() if data.x.dtype == torch.float else data.x
                edge_attr = data.edge_attr
                if edge_attr is not None and edge_attr.dtype == torch.float:
                    edge_attr = edge_attr.long()
            else:
                x, edge_attr = data.x, data.edge_attr
            if isinstance(current_model, GraphFeatureMLP):
                x = getattr(data, 'x_blind', x)
                edge_attr = getattr(data, 'edge_attr_blind', edge_attr)
            return current_model(
                x, data.edge_index, data.batch, edge_attr,
                getattr(data, 'fingerprint', None),
                rwpe=getattr(data, 'random_walk_pe', None),
                topology_size=getattr(data, 'topology_size', None),
            )

        rw_transform = (
            T.AddRandomWalkPE(
                walk_length=int(cfg.model.rwpe_dim),
                attr_name='random_walk_pe',
            ) if bool(cfg.model.use_rwpe) else None
        )

        def strict_transform(data):
            data = data.clone()
            if rw_transform is not None:
                if hasattr(data, 'random_walk_pe'):
                    del data.random_walk_pe
                data = rw_transform(data)
            return data

        steady_cycles = cycle_times[int(cfg.efficiency.timed_train_warmup_epochs):]
        strict_result = benchmark_session_if_enabled(
            cfg, model, test_loader.dataset, test_loader.dataset,
            strict_forward,
            session_id=int(getattr(cfg.efficiency, 'output_session_id', session_id)),
            transform_fn=strict_transform,
            throughput_loader=test_loader,
            extra={'training': training_efficiency},
        )

    checkpoint_path = None
    if efficiency_enabled(cfg) and bool(getattr(cfg.efficiency, 'save_checkpoint', False)):
        safe_id = str(cfg.efficiency.entry_id).replace('/', '_').replace('::', '__')
        output_session_id = int(getattr(cfg.efficiency, 'output_session_id', session_id))
        checkpoint_path = (
            Path(str(cfg.efficiency.output_dir)) / 'checkpoints' /
            str(cfg.efficiency.phase) /
            f'{safe_id}__session{output_session_id}.pt'
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'entry_id': str(cfg.efficiency.entry_id),
            'session_id': output_session_id,
            'seed': int(cfg.seed),
            'best_epoch_within_ten': int(best_epoch),
            'state_dict': best_model_state,
        }, checkpoint_path)
        logger.info(f'[efficiency checkpoint] wrote {checkpoint_path}')
    
    del model, optimizer, scheduler, criterion, best_model_state
    torch.cuda.empty_cache()
    
    def dict_to_str(d):
        return ", ".join([f"{k}: {v:.4f}" for k, v in d.items()])

    test_log = (
        f"  Test Metrics: {dict_to_str(test_perf)}\n"
        f"  Test-Infer | Time: {t_test:.2f} s, Mem Alloc: {mem_test_alloc:.2f} MB, Res: {mem_test_res:.2f} MB\n"
        if evaluate_test else
        "  Test Metrics: skipped (validation-only tuning trial)\n"
    )
    logger.info(
        f"{test_log}"
        f"  Best Val Metrics: {dict_to_str(best_val_perf)}\n"
        f"  Train | Total Time: {total_train_time:.2f}s, Epochs: {epoch_count}, Mem Alloc: {max_mem_train:.2f} MB, Res: {max_mem_train_res:.2f} MB\n"
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
        'epoch_history': epoch_history,
        'checkpoint_path': str(checkpoint_path) if checkpoint_path else None,
    }


def run_baselines(cfg=None, key_metric_override=None, config_string=None,
                  prune_callback=None, evaluate_test=True):
    args = None
    if cfg is None:
        args = parse_args()
        cfg = init_cfg()
        cfg = update_cfg(cfg, args=args)

    if cfg.load_best_params:
        config_dataset = base_dataset_name(cfg.dataset)
        config_path = CONFIG_DIR / f'{config_dataset}_{cfg.model.backbone}.yaml'
        if not config_path.exists():
            raise FileNotFoundError(
                f"load_best_params=True but best-param config not found: {config_path}"
            )
        cfg.merge_from_file(str(config_path))
        if args is not None:
            cfg.merge_from_list(args.opts)
    validate_pooling_cfg(cfg)
    
    if config_string is None:
        config_string = f"{cfg.dataset}_{cfg.model.backbone}_{int(time.time())}"
    logger = setup_logger(cfg, config_string)
    sys.stdout.flush()

    supported_ogb = ['ogbg-molhiv', 'ogbg-molbace', 'ogbg-molpcba', 'ogbg-moltox21']
    supported_tu = ['IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB', 'MUTAG', 'NCI1', 'DD', 'ENZYMES', 'PROTEINS']
    supported_lrgb = ['peptides-func', 'peptides-struct']
    supported_pyg_mol = ['zinc', 'aqsol']
    supported_gcare = ['gcare-aids', 'gcare-yago']
    base_dataset = base_dataset_name(cfg.dataset)
    dataset_lower = base_dataset.lower()
    is_gnce = dataset_lower.startswith('gnce-')
    is_good = is_good_dataset_name(cfg.dataset)
    
    dataset_info = {}
    
    if base_dataset in supported_ogb:
        dataset_info['type'] = 'ogb'
    elif base_dataset in supported_tu:
        dataset_info['type'] = 'tu'
    elif dataset_lower in supported_lrgb:
        dataset_info['type'] = 'lrgb'
    elif dataset_lower in supported_pyg_mol:
        dataset_info['type'] = 'pyg_mol'
    elif dataset_lower in supported_gcare or is_gnce:
        dataset_info['type'] = 'tu'
    elif is_good:
        dataset_info['type'] = good_model_dataset_type(cfg.dataset)
    elif base_dataset.startswith('Count-'):
        dataset_info['type'] = 'counting'
    else:
        raise ValueError(f"Dataset {cfg.dataset} not supported.")

    if is_good:
        cfg.train.loss_type = good_loss_type(cfg.dataset)
    elif base_dataset in ['ogbg-molhiv', 'ogbg-molbace', 'IMDB-BINARY', 'REDDIT-BINARY', 'MUTAG', 'NCI1', 'DD', 'PROTEINS']: # single-label
        cfg.train.loss_type = 'BCE'
    elif base_dataset in ['IMDB-MULTI', 'ENZYMES', 'COLLAB']: # single-label multi-classes
        cfg.train.loss_type = 'CE'
    elif base_dataset in ['ogbg-molpcba', 'ogbg-moltox21'] or dataset_lower == 'peptides-func': # multi-labels
        cfg.train.loss_type = 'BCE'
    elif dataset_lower == 'peptides-struct': # multi-target regression
        cfg.train.loss_type = 'L1'
    elif dataset_lower in supported_pyg_mol: # single-target molecular regression
        cfg.train.loss_type = 'L1'
    elif dataset_lower in supported_gcare or is_gnce: # cardinality regression
        cfg.train.loss_type = 'L1'
    elif dataset_info['type'] == 'counting':
        cfg.train.loss_type = 'L1'
    else:
        raise ValueError(f"Dataset {cfg.dataset} not supported.")

    if cfg.use_augm_gca and cfg.model.use_rwpe:
        logger.warning("augm_gca=True: disabling use_rwpe because GCA changes graph topology.")
        cfg.model.use_rwpe = False
    if (
            dataset_info['type'] in ('lrgb', 'pyg_mol', 'good_mol')
            and cfg.model.use_fingerprints):
        logger.warning(f"{cfg.dataset} does not provide SMILES fingerprints; disabling model.use_fingerprints.")
        cfg.model.use_fingerprints = False

    logger.info(f"Config:\n{cfg}\n")
    logger.info(f'Start Time: {datetime.datetime.now().strftime("%Y/%m/%d - %H:%M")}\n')

    # Load Data
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
            use_rwpe=cfg.model.use_rwpe,
            rwpe_dim=cfg.model.rwpe_dim,
            augm_gca=cfg.use_augm_gca,
            aug_centrality=cfg.augm_gca.centrality,
            fold_k=cfg.cv.fold_k,
            fold_seed=cfg.cv.seed,
            gnn_type=cfg.model.backbone,
            **dataset_loader_kwargs(cfg),
        )
    # Keep all G-CARE folds through preprocessing.  The efficiency session is
    # selected below only after fold-specific artifacts have retained their
    # official fold identities.

    dataset_info.update({
        'max_degree': max_degree,
        'output_dim': dataset.output_dim,
        'task': getattr(dataset, 'task', 'b')
    })
    sample_data = folds_data[0][0][0]
    dataset_info['node_feature_dim'] = int(sample_data.x.size(-1)) if sample_data.x is not None else 1
    dataset_info['edge_feature_dim'] = (
        int(sample_data.edge_attr.size(-1))
        if getattr(sample_data, 'edge_attr', None) is not None and sample_data.edge_attr.dim() > 1
        else 1
    )
    if getattr(sample_data, 'x_blind', None) is not None:
        dataset_info['has_blind_tabular_features'] = True
        dataset_info['tabular_node_feature_dim'] = int(sample_data.x_blind.size(-1))
        dataset_info['tabular_edge_feature_dim'] = (
            int(sample_data.edge_attr_blind.size(-1))
            if getattr(sample_data, 'edge_attr_blind', None) is not None and sample_data.edge_attr_blind.dim() > 1
            else 1
        )
    else:
        dataset_info['has_blind_tabular_features'] = False
        dataset_info['tabular_node_feature_dim'] = dataset_info['node_feature_dim']
        dataset_info['tabular_edge_feature_dim'] = dataset_info['edge_feature_dim']
    dataset_info['has_topology_size'] = getattr(sample_data, 'topology_size', None) is not None

    if efficiency_enabled(cfg, 'prep'):
        extra = {'transform': 'dataset loading and graph feature construction'}
        if is_lightgbm_baseline(cfg.model.backbone):
            include_structure = tabular_include_structure(cfg)
            include_sum = tabular_include_sum(dataset_info)
            include_topology = tabular_include_topology(dataset_info, cfg)
            for train_d, val_d, test_d in folds_data:
                extract_tabular_arrays(
                    train_d, cfg.train.batch_size, include_structure,
                    include_sum=include_sum,
                    include_topology=include_topology,
                )
                extract_tabular_arrays(
                    val_d, cfg.train.batch_size, include_structure,
                    include_sum=include_sum,
                    include_topology=include_topology,
                )
                if test_d is not None:
                    extract_tabular_arrays(
                        test_d, cfg.train.batch_size, include_structure,
                        include_sum=include_sum,
                        include_topology=include_topology,
                    )
            extra.update({
                'tabular_features_materialized': True,
                'tabular_include_structure': bool(include_structure),
                'tabular_include_sum': bool(include_sum),
                'tabular_include_topology': bool(include_topology),
            })
        prep_result = finish_preprocessing(
            cfg, prep_started, folds_data, extra=extra,
        )
        if bool(getattr(cfg.efficiency, 'require_prepared_input', False)):
            save_prepared_input(cfg, {
                'dataset_metadata': dataset_metadata(dataset),
                'folds_data': folds_data,
                'max_degree': max_degree,
            })
        return prep_result
    
    use_fold_sessions = (
        dataset_info['type'] == 'tu' and not is_gnce and not is_good
    )

    if use_fold_sessions:
        dataset_info.update({
            'dim_node_deg': dataset.dim_node_deg,
            'dim_node_label': dataset.dim_node_label,
            'dim_node_attr': dataset.dim_node_attr,
            'num_edge_classes': dataset.num_edge_classes,
            'key_metric': getattr(dataset, 'key_metric', 'acc')
        })
        num_sessions = len(folds_data)
        cv_dataset = 'G-CARE' if cfg.dataset.lower().startswith('gcare-') else 'TU'
        logger.info(f"{cv_dataset} Dataset (K-Fold = {num_sessions})")
    elif is_gnce:
        dataset_info.update({
            'dim_node_deg': dataset.dim_node_deg,
            'dim_node_label': dataset.dim_node_label,
            'dim_node_attr': dataset.dim_node_attr,
            'num_edge_classes': dataset.num_edge_classes,
            'key_metric': getattr(dataset, 'key_metric', 'medq')
        })
        num_sessions = cfg.train.runs
        logger.info(f"GNCE Dataset (Repeated Runs = {num_sessions})")
    elif is_good:
        if dataset_info['type'] == 'tu':
            dataset_info.update({
                'dim_node_deg': dataset.dim_node_deg,
                'dim_node_label': dataset.dim_node_label,
                'dim_node_attr': dataset.dim_node_attr,
                'num_edge_classes': dataset.num_edge_classes,
            })
        dataset_info['key_metric'] = getattr(dataset, 'key_metric', 'acc')
        num_sessions = cfg.train.runs
        logger.info(
            f"GOOD official fixed split "
            f"({getattr(dataset, 'good_name', cfg.dataset)}; "
            f"Repeated Runs = {num_sessions}; val/test=OOD for shifts)"
        )
    elif dataset_info['type'] == 'counting':
        dataset_info.update({
            'dim_node_deg': dataset.dim_node_deg,
            'dim_node_label': dataset.dim_node_label,
            'dim_node_attr': dataset.dim_node_attr,
            'num_edge_classes': dataset.num_edge_classes,
            'key_metric': getattr(dataset, 'key_metric', 'mae')
        })
        num_sessions = cfg.train.runs
        logger.info(f"Counting Dataset (Repeated Runs = {num_sessions})")
    elif dataset_info['type'] == 'lrgb':
        dataset_info.update({
            'key_metric': getattr(dataset, 'key_metric', 'acc')
        })
        num_sessions = cfg.train.runs
        logger.info(f"LRGB Dataset (Repeated Runs = {num_sessions})")
    elif dataset_info['type'] == 'pyg_mol':
        dataset_info.update({
            'key_metric': getattr(dataset, 'key_metric', 'mae')
        })
        num_sessions = cfg.train.runs
        logger.info(f"PyG Molecular Regression Dataset (Repeated Runs = {num_sessions})")
    else: # ogb
        dataset_info.update({
            'key_metric': getattr(dataset, 'key_metric', 'acc')
        })
        num_sessions = cfg.train.runs
        logger.info(f"OGB Dataset (Repeated Runs = {num_sessions})")

    session_indices = list(range(num_sessions))
    tuning_fold_limit = int(getattr(cfg.cv, 'tuning_fold_limit', 0))
    if use_fold_sessions and not evaluate_test and tuning_fold_limit > 0:
        session_indices = session_indices[:min(tuning_fold_limit, num_sessions)]
        logger.info(
            f"[TU tuning fold limit] validation-only trial uses folds "
            f"{session_indices}; final evaluation still uses all {num_sessions} folds."
        )
    if efficiency_enabled(cfg):
        selected = int(cfg.efficiency.fold_index) if use_fold_sessions else 0
        if selected < 0 or selected >= num_sessions:
            raise IndexError(f'efficiency.fold_index={selected} outside 0..{num_sessions - 1}')
        session_indices = [selected]

    if key_metric_override is not None:
        dataset_info['key_metric'] = key_metric_override
        logger.info(f"Using key_metric override: {key_metric_override}")

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
        'epoch_history': [],
        'checkpoint_path': [],
    }

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
        train_loader = DataLoader(train_d, cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False)
        val_loader = DataLoader(val_d, cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)
        test_loader = (
            DataLoader(
                test_d, cfg.train.batch_size, shuffle=False,
                num_workers=cfg.num_workers,
            )
            if evaluate_test else None
        )

        if is_lightgbm_baseline(cfg.model.backbone):
            results_run = run_lightgbm_session(
                session_id + 1,
                train_d,
                val_d,
                test_d,
                dataset_info,
                cfg,
                logger,
                current_seed,
                prune_callback=prune_callback,
                prune_step_offset=session_id * cfg.train.epochs,
                evaluate_test=evaluate_test,
            )
        else:
            # Single Run
            results_run = run_single_session(
                session_id+1, train_loader, val_loader, test_loader, dataset_info, cfg, logger,
                prune_callback=prune_callback,
                prune_step_offset=session_id * cfg.train.epochs,
                evaluate_test=evaluate_test,
            )
        
        # Collect Stats
        for key in [
            'train_time', 'epoch_count', 'test_time',
            'best_epoch',
            'train_mem_alloc', 'train_mem_res',
            'test_mem_alloc', 'test_mem_res',
            'num_parameters', 'num_trainable_parameters',
        ]:
            # Tree baselines do not expose torch parameter counts.
            stats_collector[key].append(results_run.get(key, 0))
        stats_collector['training_efficiency'].append(
            results_run.get('training_efficiency', {})
        )
        stats_collector['epoch_history'].append(
            results_run.get('epoch_history', [])
        )
        stats_collector['checkpoint_path'].append(
            results_run.get('checkpoint_path')
        )
        
        for m, v in results_run['test_perf'].items():
            if m not in stats_collector['test_perf']: stats_collector['test_perf'][m] = []
            stats_collector['test_perf'][m].append(v)
            
        for m, v in results_run['best_val_perf'].items():
            if m not in stats_collector['best_val_perf']: stats_collector['best_val_perf'][m] = []
            stats_collector['best_val_perf'][m].append(v)

    logger.info("\n" + "=" * 60)
    measured_sessions = len(session_indices)
    logger.info(f"Final Aggregated Results ({measured_sessions} sessions)")
    logger.info("=" * 60)
    
    def log_agg(name, values, unit=""):
        v = torch.tensor(values).float()
        logger.info(f"{name}: {v.mean():.4f} +/- {v.std(unbiased=False):.4f} {unit}")

    # Metrics
    if evaluate_test:
        logger.info("[Test Effectiveness]")
        for m in stats_collector['test_perf']:
            log_agg(m.upper(), stats_collector['test_perf'][m])
    else:
        logger.info("[Test Effectiveness] skipped (validation-only tuning trial)")
        
    logger.info("\n[Best Val]")
    for m in stats_collector['best_val_perf']:
        log_agg(m.upper(), stats_collector['best_val_perf'][m])
        
    # Efficiency
    logger.info("\n[Efficiency]")
    log_agg("Total Train Time", stats_collector['train_time'], "s")
    log_agg("Epochs", stats_collector['epoch_count'], "")
    log_agg("Test Infer Time", stats_collector['test_time'], "s")
    log_agg("Train Mem Alloc", stats_collector['train_mem_alloc'], "MB")
    log_agg("Train Mem Res", stats_collector['train_mem_res'], "MB")
    log_agg("Test Mem Alloc", stats_collector['test_mem_alloc'], "MB")
    log_agg("Test Mem Res", stats_collector['test_mem_res'], "MB")
    log_agg("Model Parameters", stats_collector['num_parameters'], "")
    log_agg("Trainable Parameters", stats_collector['num_trainable_parameters'], "")
    
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
    return run_baselines()


if __name__ == '__main__':
    main()
