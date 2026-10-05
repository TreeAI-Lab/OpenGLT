from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn

from .config import SEED


def utility_from_ranks(ranks: pd.DataFrame) -> pd.DataFrame:
    n_models = len(ranks.columns)
    return (n_models - ranks.astype(float)) / float(n_models - 1)


def feature_matrix(
    features: pd.DataFrame,
    train_idx: list[str],
    test_idx: list[str],
    scaling: str,
    bounded: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    train = features.loc[train_idx].to_numpy(dtype=float)
    test = features.loc[test_idx].to_numpy(dtype=float)
    scaler = MinMaxScaler() if scaling == "minmax" else StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    if bounded:
        variable = np.ptp(train, axis=0) > 1e-12
        train_scaled[:, ~variable] = 0.0
        test_scaled[:, ~variable] = 0.0
        lower, upper = (0.0, 1.0) if scaling == "minmax" else (-5.0, 5.0)
        train_scaled = np.clip(train_scaled, lower, upper)
        test_scaled = np.clip(test_scaled, lower, upper)
    return train_scaled, test_scaled


def feature_weights(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    weights = np.zeros(x_train.shape[1], dtype=float)
    for feature_id in range(x_train.shape[1]):
        feature = x_train[:, feature_id]
        if np.std(feature) < 1e-12:
            continue
        associations = []
        feature_rank = rankdata(feature)
        for model_id in range(y_train.shape[1]):
            target = y_train[:, model_id]
            if np.std(target) < 1e-12:
                continue
            correlation = np.corrcoef(feature_rank, rankdata(target))[0, 1]
            if np.isfinite(correlation):
                associations.append(abs(float(correlation)))
        if associations:
            weights[feature_id] = float(np.mean(associations))
    if weights.sum() <= 1e-12:
        weights[:] = 1.0
    return weights / weights.sum()


def cosine_scores(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    relevance = feature_weights(x_train, y_train)
    weighted_train = x_train * np.sqrt(relevance).reshape(1, -1)
    rows = []
    for query in x_test:
        weighted_query = query * np.sqrt(relevance)
        denominator = np.linalg.norm(weighted_train, axis=1) * max(np.linalg.norm(weighted_query), 1e-12)
        similarity = (weighted_train @ weighted_query) / np.maximum(denominator, 1e-12)
        similarity = (np.nan_to_num(similarity) + 1.0) / 2.0
        indices = np.argsort(-similarity, kind="mergesort")[: min(3, len(similarity))]
        weights = similarity[indices]
        if weights.sum() <= 1e-12:
            weights = np.ones(len(indices), dtype=float)
        rows.append((weights / weights.sum()) @ y_train[indices])
    return np.vstack(rows)


def graphgym_scores(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    train_norm = np.linalg.norm(x_train, axis=1)
    rows = []
    for query in x_test:
        denominator = train_norm * max(float(np.linalg.norm(query)), 1e-12)
        similarity = (x_train @ query) / np.maximum(denominator, 1e-12)
        source = int(np.argsort(-similarity, kind="mergesort")[0])
        rows.append(y_train[source])
    return np.vstack(rows)


def kgnas_scores(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    relevance = feature_weights(x_train, y_train)
    rows = []
    for query in x_test:
        distance = np.minimum(np.abs(x_train - query.reshape(1, -1)), 1.0) @ relevance
        similarity = np.clip(1.0 - distance, 0.0, 1.0)
        indices = np.argsort(-similarity, kind="mergesort")[: min(3, len(similarity))]
        weights = similarity[indices]
        if weights.sum() <= 1e-12:
            weights = np.ones(len(indices), dtype=float)
        rows.append((weights / weights.sum()) @ y_train[indices])
    return np.vstack(rows)


def knn_scores(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    rows = []
    for query in x_test:
        distance = np.linalg.norm(x_train - query.reshape(1, -1), axis=1)
        indices = np.argsort(distance, kind="mergesort")[: min(3, len(distance))]
        weights = 1.0 / (1.0 + distance[indices])
        rows.append((weights / weights.sum()) @ y_train[indices])
    return np.vstack(rows)


def rf_scores(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    model = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def glemos_mlp_scores(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        alpha=1e-4,
        max_iter=500,
        random_state=SEED,
    )
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def _masked_nmf(
    matrix: np.ndarray,
    n_components: int,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    mask = np.isfinite(values)
    observed = np.where(mask, values, 0.0)
    rows, columns = observed.shape
    components = max(1, min(int(n_components), rows, columns))
    rng = np.random.RandomState(SEED)
    eps = 1e-8
    u = np.maximum(rng.uniform(0.0, 1.0, size=(rows, components)), eps)
    v = np.maximum(np.linalg.lstsq(u, observed, rcond=None)[0], eps)
    previous_error = np.inf
    for _ in range(max_iter):
        estimate = u @ v
        weighted_estimate = mask * estimate
        u *= ((observed * mask) @ v.T) / (weighted_estimate @ v.T + eps)
        u = np.maximum(u, eps)
        estimate = u @ v
        weighted_estimate = mask * estimate
        v *= (u.T @ (observed * mask)) / (u.T @ weighted_estimate + eps)
        v = np.maximum(v, eps)
        residual = float(np.linalg.norm(mask * (observed - u @ v), ord="fro"))
        if abs(previous_error - residual) <= tol * max(1.0, previous_error):
            break
        previous_error = residual
    return u, v


def _mlp_regression(
    x_train: np.ndarray,
    x_test: np.ndarray,
    targets: np.ndarray,
    hidden_layer_sizes: tuple[int, ...],
) -> np.ndarray:
    n_rows = len(x_train)
    use_validation = n_rows >= 5
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        alpha=1e-4,
        max_iter=500,
        random_state=SEED,
        early_stopping=use_validation,
        validation_fraction=0.2,
        n_iter_no_change=20,
        batch_size="auto",
    )
    model.fit(x_train, targets)
    return np.asarray(model.predict(x_test), dtype=float)


def alors_scores(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    latent_dim = min(32, y_train.shape[1], y_train.shape[0])
    latent_train, model_factors = _masked_nmf(y_train, latent_dim)
    latent_test = _mlp_regression(
        x_train,
        x_test,
        latent_train,
        hidden_layer_sizes=(128, 128),
    )
    scores = np.asarray(latent_test @ model_factors, dtype=float)
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)


def model_features(model_names: list[str]) -> np.ndarray:
    family = {
        "GINE": "node",
        "PNA": "node",
        "GPS": "node",
        "GMT": "pooling",
        "AK+": "subgraph",
        "MO": "graph-learning",
        "GCA": "self-supervised",
    }
    families = [family[name] for name in model_names]
    family_names = sorted(set(families))
    family_id = {name: index for index, name in enumerate(family_names)}
    one_hot = np.zeros((len(model_names), len(family_names)), dtype=float)
    for model_id, name in enumerate(families):
        one_hot[model_id, family_id[name]] = 1.0
    return np.hstack([np.eye(len(model_names), dtype=float), one_hot])


def s2_scores(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    target_mask = np.isfinite(y_train)
    y_filled = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)

    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    hidden = 128
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], hidden),
        nn.ReLU(),
        nn.Linear(hidden, y_train.shape[1]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_tensor = torch.from_numpy(x_train).to(device)
    y_tensor = torch.from_numpy(y_filled).to(device)
    mask_tensor = torch.from_numpy(target_mask).to(device)

    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(x_train))
    if len(order) >= 5:
        n_val = max(1, int(round(len(order) * 0.2)))
        n_val = min(n_val, len(order) - 2)
        val_indices = order[:n_val]
        fit_indices = order[n_val:]
    else:
        fit_indices = order
        val_indices = np.asarray([], dtype=int)
    fit_idx = torch.as_tensor(fit_indices, dtype=torch.long, device=device)
    val_idx = torch.as_tensor(val_indices, dtype=torch.long, device=device)

    def masked_loss(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = mask.bool()
        return torch.mean((pred[selected] - truth[selected]) ** 2)

    best_state = copy.deepcopy(model.state_dict())
    best_value = np.inf
    epochs_without_improvement = 0
    for _ in range(300):
        model.train()
        optimizer.zero_grad()
        prediction = model(x_tensor[fit_idx])
        loss = masked_loss(prediction, y_tensor[fit_idx], mask_tensor[fit_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if len(val_indices):
                validation = masked_loss(model(x_tensor[val_idx]), y_tensor[val_idx], mask_tensor[val_idx])
            else:
                validation = loss.detach()
        value = float(validation.detach().cpu().item())
        if value + 1e-9 < best_value:
            best_value = value
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 20:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(x_test).to(device)).cpu().numpy()
    return np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)


def simplets_scores(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    n_clusters = min(5, len(x_train))
    clusterer = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    labels = clusterer.fit_predict(y_train)
    unique = np.unique(labels)
    if len(unique) == 1:
        return np.tile(np.mean(y_train, axis=0), (len(x_test), 1))
    centroids = {label: np.mean(y_train[labels == label], axis=0) for label in unique}
    classifier = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)
    classifier.fit(x_train, labels)
    probabilities = classifier.predict_proba(x_test)
    rows = np.zeros((len(x_test), y_train.shape[1]), dtype=float)
    for column, label in enumerate(classifier.classes_.astype(int)):
        rows += probabilities[:, column:column + 1] * centroids[label]
    return rows


def modsnet_scores(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    model_names: list[str],
) -> np.ndarray:
    n_models = y_train.shape[1]
    side = model_features(model_names)
    pair_x = np.hstack([np.repeat(x_train, n_models, axis=0), np.tile(side, (len(x_train), 1))])
    threshold = np.median(y_train, axis=1, keepdims=True)
    pair_y = (y_train >= threshold).reshape(-1).astype(int)
    classifier = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=SEED,
        n_jobs=1,
    )
    classifier.fit(pair_x, pair_y)
    rows = []
    positive_column = int(np.where(classifier.classes_ == 1)[0][0])
    for query in x_test:
        query_x = np.hstack([np.repeat(query.reshape(1, -1), n_models, axis=0), side])
        rows.append(classifier.predict_proba(query_x)[:, positive_column])
    return np.vstack(rows)


def aggregate_global_scores(ranks: pd.DataFrame, train_idx: list[str]) -> np.ndarray:
    utility = utility_from_ranks(ranks.loc[train_idx])
    records = []
    for model in utility.columns:
        values = utility[model]
        records.append((-float(values.mean()), -float(values.median()), -float(values.min()), model))
    ordered = [record[-1] for record in sorted(records)]
    scores = np.empty(len(ordered), dtype=float)
    for position, model in enumerate(ordered):
        scores[utility.columns.get_loc(model)] = float(len(ordered) - position)
    return scores


def score_selector(
    method: str,
    features: pd.DataFrame,
    ranks: pd.DataFrame,
    train_idx: list[str],
    test_idx: list[str],
) -> np.ndarray:
    if method == "Fixed-global":
        score = aggregate_global_scores(ranks, train_idx)
        return np.tile(score, (len(test_idx), 1))
    scaling = "minmax" if method == "KGNAS" else "standard"
    bounded = method in {"Cosine", "GraphGym", "KGNAS"}
    x_train, x_test = feature_matrix(features, train_idx, test_idx, scaling, bounded)
    y_train = utility_from_ranks(ranks.loc[train_idx]).to_numpy(dtype=float)
    if method == "Cosine":
        return cosine_scores(x_train, x_test, y_train)
    if method == "GraphGym":
        return graphgym_scores(x_train, x_test, y_train)
    if method == "KGNAS":
        return kgnas_scores(x_train, x_test, y_train)
    if method == "KNN":
        return knn_scores(x_train, x_test, y_train)
    if method == "RF":
        return rf_scores(x_train, x_test, y_train)
    if method == "GLEMoS-MLP":
        return glemos_mlp_scores(x_train, x_test, y_train)
    if method == "ALORS":
        return alors_scores(x_train, x_test, y_train)
    if method == "S2":
        return s2_scores(x_train, x_test, y_train)
    if method == "SimpleTS":
        return simplets_scores(x_train, x_test, y_train)
    if method == "ModsNet":
        return modsnet_scores(x_train, x_test, y_train, list(ranks.columns))
    raise ValueError(f"Unknown selector: {method}")
