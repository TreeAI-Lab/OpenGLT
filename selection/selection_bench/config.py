from __future__ import annotations

SEED = 42
COMMON_MODELS = ("GINE", "PNA", "GPS", "GMT", "AK+", "MO", "GCA")

METHODS = (
    "Random",
    "Fixed-global",
    "Cosine",
    "GraphGym",
    "KGNAS",
    "KNN",
    "RF",
    "GLEMoS-MLP",
    "ALORS",
    "S2",
    "SimpleTS",
    "ModsNet",
)

GENERAL_DOMAINS = {
    "IMDB-BINARY": "social",
    "IMDB-MULTI": "social",
    "REDDIT-BINARY": "social",
    "COLLAB": "social",
    "PROTEINS": "biology",
    "DD": "biology",
    "ENZYMES": "biology",
    "OGB-BACE": "chemistry",
    "OGB-HIV": "chemistry",
    "OGB-TOX21": "chemistry",
    "OGB-PCBA": "chemistry",
    "NCI1": "chemistry",
    "Peptides-func": "chemistry",
    "Peptides-struct": "chemistry",
    "AQSOL": "chemistry",
    "ZINC": "chemistry",
    "MUTAG": "chemistry",
    "Count-3-Cycle": "motif",
    "Count-4-Cycle": "motif",
    "Count-5-Cycle": "motif",
    "Count-6-Cycle": "motif",
    "Count-7-Cycle": "motif",
    "Count-8-Cycle": "motif",
    "Count-4-Path": "motif",
    "Count-5-Path": "motif",
    "Count-6-Path": "motif",
    "Count-4-Clique": "motif",
    "Count-Tailed-Tri": "motif",
    "Count-Chordal-Cyc": "motif",
    "Count-Tri-Rect": "motif",
}

GENERAL_DATASETS = tuple(GENERAL_DOMAINS)
DOMAIN_ORDER = ("social", "biology", "chemistry", "motif", "database")

TOPOLOGY_COLUMNS = (
    "avg_degree",
    "density",
    "clustering_coefficient",
    "assortativity",
    "avg_shortest_path_length",
    "diameter",
    "modularity",
    "triangles_mean",
    "triangles_std",
    "betweenness_mean",
    "betweenness_std",
    "pagerank_mean",
    "pagerank_std",
    "kcore_mean",
    "kcore_std",
)

GENERAL_LABEL_COLUMNS = (
    "task_classification",
    "task_regression",
    "task_counting",
    "num_classes",
    "num_targets",
    "motif_size",
    "class_imbalance",
    "label_missing_fraction",
    "signed_log_label_std",
    "signed_log_label_kurtosis",
    "signed_log_label_tail_fraction",
)

CE_LABEL_COLUMNS = (
    "log_std",
    "log_kurt",
    "p95_card",
    "max_card",
    "max_median",
    "tail_gt_1e6",
    "log_max_p95",
)

METRICS = ("HR@3", "nDCG@3", "MRR", "AUC@3", "Regret")
