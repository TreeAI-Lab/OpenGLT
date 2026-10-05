import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    r2_score
)
from sklearn.preprocessing import label_binarize


def _safe_auc(y_true, y_score):
    """Return NaN instead of aborting when a split contains one class."""
    return (
        roc_auc_score(y_true, y_score)
        if len(np.unique(y_true)) >= 2 else float('nan')
    )


def _eval_binary(y_true, y_pred_prob):
    if y_pred_prob.ndim > 1:
        y_pred_prob = y_pred_prob.squeeze()
    if y_true.ndim > 1:
        y_true = y_true.squeeze()

    y_pred_label = (y_pred_prob > 0.5).astype(int)

    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred_prob = y_pred_prob[mask]
    y_pred_label = y_pred_label[mask]

    return {
        'acc': accuracy_score(y_true, y_pred_label),
        'auc': _safe_auc(y_true, y_pred_prob),
        'ap': average_precision_score(y_true, y_pred_prob),
        'precision': precision_score(y_true, y_pred_label, zero_division=0),
        'recall': recall_score(y_true, y_pred_label, zero_division=0),
        'f1': f1_score(y_true, y_pred_label, zero_division=0)
    }


def _eval_multi_label(y_true, y_pred_prob):
    y_pred_label = (y_pred_prob > 0.5).astype(int)

    mask = ~np.isnan(y_true)

    y_true_safe = np.nan_to_num(y_true, nan=-1) 
    correct_exact = np.all((y_pred_label == y_true_safe) | ~mask, axis=1)
    acc_exact = np.mean(correct_exact)

    y_true_flat = y_true[mask].ravel()
    y_pred_prob_flat = y_pred_prob[mask].ravel()
    y_pred_label_flat = y_pred_label[mask].ravel()

    auc_list, ap_list = [], []
    prec_list, recall_list, f1_list = [], [], []

    num_tasks = y_true.shape[1]
    for i in range(num_tasks):
        valid_mask_col = mask[:, i]
        if np.sum(valid_mask_col) == 0:
            continue
            
        y_t = y_true[valid_mask_col, i]
        y_p_prob = y_pred_prob[valid_mask_col, i]
        y_p_label = y_pred_label[valid_mask_col, i]

        if len(np.unique(y_t)) >= 2:
            auc_list.append(roc_auc_score(y_t, y_p_prob))
        # GOOD-PCBA's official evaluator computes AP for every task with at
        # least one observed target, including single-class split/task pairs.
        ap_list.append(average_precision_score(y_t, y_p_prob))
        prec_list.append(precision_score(y_t, y_p_label, zero_division=0))
        recall_list.append(recall_score(y_t, y_p_label, zero_division=0))
        f1_list.append(f1_score(y_t, y_p_label, zero_division=0))

    def safe_mean(values):
        return np.mean(values) if values else float('nan')

    return {
        'acc': acc_exact,
        'auc_micro': _safe_auc(y_true_flat, y_pred_prob_flat),
        'ap_micro': average_precision_score(y_true_flat, y_pred_prob_flat),
        'precision_micro': precision_score(y_true_flat, y_pred_label_flat, zero_division=0),
        'recall_micro': recall_score(y_true_flat, y_pred_label_flat, zero_division=0),
        'f1_micro': f1_score(y_true_flat, y_pred_label_flat, zero_division=0),
        'auc_macro': safe_mean(auc_list),
        'ap_macro': safe_mean(ap_list),
        'precision_macro': safe_mean(prec_list),
        'recall_macro': safe_mean(recall_list),
        'f1_macro': safe_mean(f1_list)
    }


def _eval_multi_class(y_true, y_pred_prob):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred_label = np.argmax(y_pred_prob, axis=1)

    num_classes = y_pred_prob.shape[1]
    classes = np.arange(num_classes)
    y_true_bin = label_binarize(y_true, classes=classes)
    auc_per_class = [
        _safe_auc(y_true_bin[:, class_id], y_pred_prob[:, class_id])
        for class_id in range(num_classes)
    ]
    valid_auc = [value for value in auc_per_class if np.isfinite(value)]

    return {
        'acc': accuracy_score(y_true, y_pred_label),
        'auc_macro': (
            float(np.mean(valid_auc)) if valid_auc else float('nan')
        ),
        'ap_macro': average_precision_score(y_true_bin, y_pred_prob, average="macro"),
        'precision_macro': precision_score(y_true, y_pred_label, average="macro", zero_division=0),
        'recall_macro': recall_score(y_true, y_pred_label, average="macro", zero_division=0),
        'f1_macro': f1_score(y_true, y_pred_label, average="macro", zero_division=0),
        'auc_micro': _safe_auc(y_true_bin.ravel(), y_pred_prob.ravel()),
        'ap_micro': average_precision_score(y_true_bin, y_pred_prob, average="micro"),
        'precision_micro': precision_score(y_true, y_pred_label, average="micro", zero_division=0),
        'recall_micro': recall_score(y_true, y_pred_label, average="micro", zero_division=0),
        'f1_micro': f1_score(y_true, y_pred_label, average="micro", zero_division=0)
    }


def _eval_regression(y_true, y_pred):
    if y_pred.ndim > 1:
        y_pred = y_pred.squeeze()
    if y_true.ndim > 1:
        y_true = y_true.squeeze()

    mae = np.mean(np.abs(y_true - y_pred))
    mre = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        'mae': mae,
        'mre': mre,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
    }


def _eval_cardinality(y_true, y_pred):
    if y_pred.ndim > 1:
        y_pred = y_pred.squeeze()
    if y_true.ndim > 1:
        y_true = y_true.squeeze()

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    y_true = y_true[mask]
    y_pred = np.maximum(y_pred[mask], 1e-12)
    if y_true.size == 0:
        return {
            'medq': float('nan'),
            'p90q': float('nan'),
            'p95q': float('nan'),
            'p99q': float('nan'),
            'meanq': float('nan'),
            'maxq': float('nan'),
            'log_mae': float('nan'),
        }

    qerror = np.maximum(y_pred / y_true, y_true / y_pred)
    log_mae = np.mean(np.abs(np.log(y_pred) - np.log(y_true)))
    return {
        'medq': float(np.percentile(qerror, 50)),
        'p90q': float(np.percentile(qerror, 90)),
        'p95q': float(np.percentile(qerror, 95)),
        'p99q': float(np.percentile(qerror, 99)),
        'meanq': float(np.mean(qerror)),
        'maxq': float(np.max(qerror)),
        'log_mae': float(log_mae),
    }


def evaluate(y_trues, y_preds, task='b'):
    y_trues = np.array(y_trues) if not isinstance(y_trues, np.ndarray) else y_trues
    y_preds = np.array(y_preds) if not isinstance(y_preds, np.ndarray) else y_preds

    if task == 'b':
        return _eval_binary(y_trues, y_preds)
    elif task == 'ml': # multi-label
        return _eval_multi_label(y_trues, y_preds)
    elif task == 'mc': # multi-class (single-label)
        return _eval_multi_class(y_trues, y_preds)
    elif task == 'reg':
        return _eval_regression(y_trues, y_preds)
    elif task == 'card':
        return _eval_cardinality(y_trues, y_preds)
    else:
        raise ValueError(f"Unknown mode: {task}.")
