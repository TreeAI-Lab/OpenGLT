from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import BenchmarkData
from .config import (
    CE_LABEL_COLUMNS,
    COMMON_MODELS,
    GENERAL_DATASETS,
    GENERAL_DOMAINS,
    GENERAL_LABEL_COLUMNS,
    METHODS,
    METRICS,
    SEED,
    TOPOLOGY_COLUMNS,
)
from .metrics import (
    domain_summary,
    instance_metrics,
    lodo_folds,
    logo_folds,
    paper_summary,
    random_policy_metrics,
    rank_matrix,
)
from .selectors import score_selector


@dataclass(frozen=True)
class SelectionTarget:
    name: str
    ranks: pd.DataFrame
    features: pd.DataFrame
    domains: pd.Series


@dataclass(frozen=True)
class TargetResults:
    target: SelectionTarget
    folds: pd.DataFrame
    summary: pd.DataFrame
    domain_summary: pd.DataFrame


def _pivot(frame: pd.DataFrame, value: str, index: list[str]) -> pd.DataFrame:
    return (
        frame.set_index(["dataset", "model"])[value]
        .unstack("model")
        .reindex(index=index, columns=COMMON_MODELS)
    )


def build_target(bundle: BenchmarkData, ce_target: str) -> SelectionTarget:
    general_names = list(GENERAL_DATASETS)
    ce_names = sorted(bundle.ce_performance["dataset"].unique())

    general_values = _pivot(bundle.general_performance, "value", general_names)
    general_ranks = rank_matrix(general_values, higher_is_better=True)

    ce_records = bundle.ce_performance[bundle.ce_performance["metric"].eq(ce_target)]
    ce_values = _pivot(ce_records, "value", ce_names)
    ce_ranks = rank_matrix(ce_values, higher_is_better=False)

    dataset_order = general_names + ce_names
    ranks = pd.concat([general_ranks, ce_ranks]).reindex(dataset_order)

    topology = pd.concat([
        bundle.general_topology.reindex(general_names),
        bundle.ce_topology.reindex(ce_names),
    ]).reindex(dataset_order)
    topology.columns = [f"topo_{column}" for column in TOPOLOGY_COLUMNS]

    general_columns = [f"general_{column}" for column in GENERAL_LABEL_COLUMNS]
    ce_columns = [f"ce_{column}" for column in CE_LABEL_COLUMNS]
    label_columns = general_columns + ce_columns + ["task_ce"]
    general_labels = pd.DataFrame(0.0, index=general_names, columns=label_columns)
    general_labels.loc[:, general_columns] = bundle.general_features.reindex(general_names).to_numpy(dtype=float)
    ce_labels = pd.DataFrame(0.0, index=ce_names, columns=label_columns)
    ce_labels.loc[:, ce_columns] = bundle.ce_features.reindex(ce_names).to_numpy(dtype=float)
    ce_labels["task_ce"] = 1.0
    labels = pd.concat([general_labels, ce_labels]).reindex(dataset_order)
    features = pd.concat([topology, labels], axis=1)

    domains = pd.Series(
        {**GENERAL_DOMAINS, **{dataset: "database" for dataset in ce_names}},
        name="domain",
    ).reindex(dataset_order)
    return SelectionTarget(
        name=ce_target,
        ranks=ranks,
        features=features,
        domains=domains,
    )


def _run_folds(target: SelectionTarget, folds_by_setting: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    random_metrics = random_policy_metrics(target.ranks, seed=SEED)
    instances = []
    fold_rows = []
    for setting, fold_specs in folds_by_setting.items():
        for fold, train, test in fold_specs:
            for method in METHODS:
                scores = None if method == "Random" else score_selector(
                    method,
                    target.features,
                    target.ranks,
                    train,
                    test,
                )
                fold_instances = []
                for row_id, dataset in enumerate(test):
                    if method == "Random":
                        record = dict(random_metrics[dataset])
                    else:
                        predicted = scores[row_id]
                        record = instance_metrics(target.ranks.loc[dataset].to_numpy(dtype=float), predicted)
                    record.update({
                        "ce_target": target.name,
                        "setting": setting,
                        "fold": fold,
                        "method": method,
                        "dataset": dataset,
                        "domain": target.domains.loc[dataset],
                    })
                    instances.append(record)
                    fold_instances.append(record)

                fold_row = {
                    "ce_target": target.name,
                    "setting": setting,
                    "fold": fold,
                    "method": method,
                    "n_test": len(test),
                }
                for metric in METRICS:
                    fold_row[metric] = float(np.mean([row[metric] for row in fold_instances]))
                fold_rows.append(fold_row)
    return pd.DataFrame(instances), pd.DataFrame(fold_rows)


def run_target(target: SelectionTarget) -> TargetResults:
    folds_by_setting = {
        "LOGO": logo_folds(target.domains),
        "LODO": lodo_folds(target.domains),
    }
    instances, folds = _run_folds(target, folds_by_setting)
    summary = paper_summary(instances, folds)
    return TargetResults(
        target=target,
        folds=folds,
        summary=summary,
        domain_summary=domain_summary(instances),
    )


def _write_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    frame.to_csv(path, index=index, line_terminator="\n")


def _tag(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    tagged = frame.drop(columns="ce_target", errors="ignore").copy()
    tagged.insert(0, "target", target)
    return tagged


def write_results(results: list[TargetResults], output: Path) -> None:
    main = pd.concat([_tag(result.summary, result.target.name) for result in results], ignore_index=True)
    folds = pd.concat([_tag(result.folds, result.target.name) for result in results], ignore_index=True)
    domains = pd.concat(
        [_tag(result.domain_summary, result.target.name) for result in results],
        ignore_index=True,
    )
    _write_csv(main, output / "main_results.csv")
    _write_csv(folds, output / "fold_results.csv")
    _write_csv(domains, output / "domain_results.csv")
