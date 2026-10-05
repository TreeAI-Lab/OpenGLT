from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, roc_auc_score

from .config import DOMAIN_ORDER, METRICS


def rank_matrix(values: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame:
    ranks = pd.DataFrame(index=values.index, columns=values.columns, dtype=float)
    worst = float(len(values.columns))
    for dataset, row in values.iterrows():
        numeric = pd.to_numeric(row, errors="coerce")
        observed = numeric.dropna()
        ranks.loc[dataset, observed.index] = observed.rank(
            ascending=not higher_is_better,
            method="min",
        )
        ranks.loc[dataset, numeric.index[numeric.isna()]] = worst
    return ranks.astype(float)


def instance_metrics(true_ranks: np.ndarray, pred_scores: np.ndarray) -> dict[str, float]:
    true_ranks = np.asarray(true_ranks, dtype=float)
    pred_scores = np.asarray(pred_scores, dtype=float)
    order = np.argsort(-pred_scores, kind="mergesort")
    best_rank = float(true_ranks.min())
    worst_rank = float(true_ranks.max())
    best_models = set(np.flatnonzero(true_ranks == best_rank))
    first_best = next(i for i, model in enumerate(order, start=1) if model in best_models)
    relevance = worst_rank - true_ranks + 1.0
    positives = (true_ranks <= 3).astype(int)
    auc = (
        float(roc_auc_score(positives, pred_scores))
        if len(np.unique(positives)) == 2 and np.std(pred_scores) > 1e-12
        else 0.5
    )
    return {
        "HR@3": float(bool(best_models & set(order[:3]))),
        "nDCG@3": float(ndcg_score(relevance[None, :], pred_scores[None, :], k=3)),
        "MRR": 1.0 / first_best,
        "AUC@3": auc,
        "Regret": float(
            (true_ranks[order[0]] - best_rank) / max(worst_rank - best_rank, 1.0)
        ),
    }


def random_policy_metrics(
    ranks: pd.DataFrame,
    repetitions: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    rng = np.random.RandomState(seed)
    output = {}
    for dataset, row in ranks.iterrows():
        true_ranks = row.to_numpy(dtype=float)
        scores = rng.uniform(size=(repetitions, len(true_ranks)))
        order = np.argsort(-scores, axis=1, kind="mergesort")
        best_rank = float(true_ranks.min())
        worst_rank = float(true_ranks.max())
        best = true_ranks == best_rank
        best_by_position = best[order]
        relevance = worst_rank - true_ranks + 1.0
        discount = 1.0 / np.log2(np.arange(3) + 2)
        ideal = np.argsort(-relevance, kind="mergesort")[:3]
        dcg = (relevance[order[:, :3]] * discount).sum(axis=1)
        idcg = float((relevance[ideal] * discount).sum())
        first_best = np.argmax(best_by_position, axis=1) + 1.0
        top1 = order[:, 0]
        output[str(dataset)] = {
            "HR@3": float(np.mean(best_by_position[:, :3].any(axis=1))),
            "nDCG@3": float(np.mean(dcg / idcg)),
            "MRR": float(np.mean(1.0 / first_best)),
            "AUC@3": 0.5,
            "Regret": float(
                np.mean(
                    (true_ranks[top1] - best_rank)
                    / max(worst_rank - best_rank, 1.0)
                )
            ),
        }
    return output


LOGO_TEST_FOLDS = (
    (
        "Peptides-func", "Peptides-struct", "PROTEINS",
        "Count-7-Cycle", "Count-6-Cycle", "Count-4-Cycle",
        "BJ-Accidents", "ST-Accidents", "BJ-Usfs-fia", "ST-Usfs-fia",
        "BJ-Stack", "ST-Stack", "BJ-Airline", "ST-Airline",
        "SWDF-path", "SWDF-star",
    ),
    (
        "DD", "IMDB-BINARY", "OGB-HIV", "OGB-TOX21",
        "Count-5-Cycle", "Count-8-Cycle",
        "BJ-Geo-map", "ST-Geo-map", "BJ-Covid19", "ST-Covid19",
        "BJ-Idc-v10", "ST-Idc-v10", "BJ-Blockchain", "ST-Blockchain",
        "YAGO-path", "YAGO-star",
    ),
    (
        "ENZYMES", "NCI1", "AQSOL", "REDDIT-BINARY",
        "Count-5-Path", "Count-Tri-Rect",
        "BJ-Uspto", "ST-Uspto", "BJ-Tpch-10G", "ST-Tpch-10G",
        "BJ-Patient", "ST-Patient", "BJ-Bitcoin", "ST-Bitcoin",
        "LUBM-path", "LUBM-star",
    ),
    (
        "MUTAG", "OGB-PCBA", "IMDB-MULTI",
        "Count-Tailed-Tri", "Count-3-Cycle", "Count-6-Path",
        "BJ-Github", "ST-Github", "BJ-Wikipedia", "ST-Wikipedia",
        "BJ-Consumer", "ST-Consumer", "BJ-Movielens", "ST-Movielens",
        "Wikidata-path", "Wikidata-star",
    ),
    (
        "COLLAB", "ZINC", "OGB-BACE",
        "Count-Chordal-Cyc", "Count-4-Path", "Count-4-Clique",
        "BJ-Genetics", "ST-Genetics", "BJ-Human", "ST-Human",
        "BJ-Samples", "ST-Samples", "BJ-Employee", "ST-Employee",
        "AIDS", "YAGO",
    ),
)

YAGO_DATASETS = {"YAGO", "YAGO-path", "YAGO-star"}


def dependency_group(dataset: str) -> str:
    if dataset.startswith("Count-"):
        return "Counting"
    if dataset in {"Peptides-func", "Peptides-struct"}:
        return "Peptides"
    if dataset.startswith(("ST-", "BJ-")):
        return f"CardBench:{dataset.split('-', 1)[1]}"
    if dataset.endswith(("-path", "-star")):
        return f"GNCE:{dataset.rsplit('-', 1)[0]}"
    if dataset in {"AIDS", "YAGO"}:
        return "G-CARE"
    return dataset


def logo_folds(domains: pd.Series) -> list[tuple[str, list[str], list[str]]]:
    datasets = list(domains.index)
    roster = [list(fold) for fold in LOGO_TEST_FOLDS]
    folds = []
    for fold_id, test in enumerate(roster, start=1):
        test_set = set(test)
        includes_yago = bool(test_set & YAGO_DATASETS)
        train = [
            dataset
            for dataset in datasets
            if dataset not in test_set
            and not dataset.startswith("Count-")
            and not (includes_yago and dataset in YAGO_DATASETS)
        ]
        folds.append((f"fold-{fold_id}", train, test))
    return folds


def lodo_folds(domains: pd.Series) -> list[tuple[str, list[str], list[str]]]:
    folds = []
    for domain in DOMAIN_ORDER:
        test = list(domains[domains.eq(domain)].index)
        test_set = set(test)
        train = [dataset for dataset in domains.index if dataset not in test_set]
        folds.append((domain, train, test))
    return folds


def paper_summary(instances: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped_folds = folds.groupby(["setting", "method"], sort=False)
    for (setting, method), group in instances.groupby(["setting", "method"], sort=False):
        fold_group = grouped_folds.get_group((setting, method))
        row = {
            "setting": setting,
            "method": method,
            "datasets": int(group["dataset"].nunique()),
            "folds": int(fold_group["fold"].nunique()),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(fold_group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def domain_summary(instances: pd.DataFrame) -> pd.DataFrame:
    subset = instances[instances["setting"].eq("LODO")]
    rows = []
    for (domain, method), group in subset.groupby(["domain", "method"], sort=False):
        row = {
            "domain": domain,
            "method": method,
            "datasets": int(group["dataset"].nunique()),
        }
        row.update({metric: float(group[metric].mean()) for metric in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)
