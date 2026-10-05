import os

from models.hymn import HyMN
from example.uni_runner import run_uni_baseline

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def build_model(cfg, dataset_info):
    predictor_multiplier = int(cfg.hymn.predictor_multiplier)
    if predictor_multiplier <= 0:
        predictor_multiplier = int(cfg.hymn.hidden_size)
    return HyMN(
        nhid=int(cfg.hymn.hidden_size),
        nout=int(dataset_info['output_dim']),
        num_layers=int(cfg.hymn.num_layers),
        num_samples=int(cfg.hymn.num_samples),
        num_colours=int(cfg.hymn.num_colours),
        dataset_type=dataset_info['type'],
        dim_node_deg=dataset_info.get('dim_node_deg', 0),
        dim_node_label=dataset_info.get('dim_node_label', 0),
        dim_node_attr=dataset_info.get('dim_node_attr', 0),
        dropout=float(cfg.hymn.dropout),
        pooling=str(cfg.hymn.pooling),
        residual=bool(cfg.hymn.residual),
        inject_colours=bool(cfg.hymn.inject_colours),
        multiplier=int(cfg.hymn.multiplier),
        predictor_multiplier=predictor_multiplier,
        bn=bool(cfg.hymn.bn),
        colouring=str(cfg.hymn.colouring),
        counting_input=str(cfg.hymn.counting_input),
        predictor=str(cfg.hymn.predictor),
    )


def run_hymn(cfg=None, key_metric_override=None, config_string=None,
             prune_callback=None, evaluate_test=True):
    return run_uni_baseline(
        'HyMN',
        build_model,
        cfg=cfg,
        key_metric_override=key_metric_override,
        config_string=config_string,
        prune_callback=prune_callback,
        evaluate_test=evaluate_test,
    )


def main():
    return run_hymn()


if __name__ == '__main__':
    main()
