import argparse
import os

from yacs.config import CfgNode as CN


def set_cfg(cfg):
    cfg.dataset = 'MUTAG'
    cfg.load_best_params = False

    cfg.num_workers = 0
    cfg.device = 'cuda:0'
    cfg.seed = 42
    cfg.deterministic = True
    cfg.deterministic_warn_only = True

    cfg.efficiency = CN(new_allowed=True)
    cfg.efficiency.enabled = False
    cfg.efficiency.phase = 'none'       # none / metadata / strict / prep
    cfg.efficiency.entry_id = ''
    cfg.efficiency.output_dir = 'efficiency_results'
    cfg.efficiency.fold_index = 0
    cfg.efficiency.sample_seed = 2024
    cfg.efficiency.sample_count = 200
    cfg.efficiency.warmup_count = 20
    cfg.efficiency.latency_repeats = 3
    cfg.efficiency.timed_train_epochs = 10
    cfg.efficiency.timed_train_warmup_epochs = 1
    cfg.efficiency.training_mode = 'short_cycle_extrapolation'
    cfg.efficiency.lgbm_early_stopping_rounds = 0
    cfg.efficiency.cpu_threads = 10
    cfg.efficiency.require_exclusive = True
    cfg.efficiency.execution_mode = 'server-exclusive'
    cfg.efficiency.require_cache_miss = False
    cfg.efficiency.cache_root = ''
    cfg.efficiency.data_root = ''
    cfg.efficiency.prepared_cache_key = 'default'
    cfg.efficiency.require_prepared_input = False
    cfg.efficiency.prep_repeat = 1
    cfg.efficiency.save_checkpoint = False
    cfg.efficiency.output_session_id = 1
    cfg.efficiency.measure_inference = True

    cfg.cv = CN()
    cfg.cv.fold_k = 0
    cfg.cv.seed = 2024
    cfg.cv.tuning_fold_limit = 0

    cfg.model = CN()
    cfg.model.backbone = 'GCN'
    cfg.model.hidden_size = 128
    cfg.model.num_layers = 4
    cfg.model.readout_type = 'mean'          # mean / sum / max / attn
    cfg.model.pooling_method = 'none'        # none / topk / edgepool / gmt
    cfg.model.pool_ratio = 0.9
    cfg.model.num_pool_layers = 1
    cfg.model.pool_readout = 'mean'          # mean / mean_max
    cfg.model.edgepool_dropout = 0.2
    cfg.model.edge_score_method = 'softmax'
    cfg.model.gmt_k = 8
    cfg.model.gmt_heads = 1
    cfg.model.gmt_blocks = 1
    cfg.model.gmt_layer_norm = False
    cfg.model.dropout = 0.3
    cfg.model.res = True
    cfg.model.norm_type = 'batch_norm'
    cfg.model.act_type = 'relu'
    cfg.model.use_ffn = False
    cfg.model.ffn_expansion = 2
    cfg.model.aggr_sage = 'mean'
    cfg.model.train_eps_gin = False
    cfg.model.towers_pna = 1
    cfg.model.pre_layers_pna = 1
    cfg.model.post_layers_pna = 1
    cfg.model.divide_input_pna = False
    cfg.model.use_virtual_node = False
    cfg.model.use_fingerprints = False
    cfg.model.fingerprint_dim = 2048
    cfg.model.use_rwpe = False
    cfg.model.rwpe_dim = 20

    cfg.baseline = CN()
    cfg.baseline.feature_mode = 'feature_struct'  # feature / feature_struct
    cfg.baseline.hidden_size = 128
    cfg.baseline.num_layers = 3
    cfg.baseline.dropout = 0.1
    cfg.baseline.norm_type = 'batch_norm'
    cfg.baseline.act_type = 'relu'
    cfg.baseline.lightgbm_n_estimators = 500
    cfg.baseline.lightgbm_learning_rate = 0.05
    cfg.baseline.lightgbm_num_leaves = 31
    cfg.baseline.lightgbm_max_depth = -1
    cfg.baseline.lightgbm_min_child_samples = 20
    cfg.baseline.lightgbm_subsample = 1.0
    cfg.baseline.lightgbm_colsample_bytree = 1.0
    cfg.baseline.lightgbm_reg_alpha = 0.0
    cfg.baseline.lightgbm_reg_lambda = 0.0

    cfg.train = CN()
    cfg.train.batch_size = 128
    cfg.train.runs = 3
    cfg.train.epochs = 500
    cfg.train.patience = 50
    cfg.train.lr = 1e-2
    cfg.train.wd = 0.

    cfg.train.scheduler = 'StepLR'  # None / StepLR / ReduceLR / CosRestart
    cfg.train.loss_type = 'BCE'

    cfg.train.lr_patience = 20
    cfg.train.lr_decay = 0.8
    cfg.train.lr_patience_R = 10
    cfg.train.factor_R = 0.5

    cfg.train.t_0 = 50
    cfg.train.t_mult = 1
    cfg.train.eta_min = 1e-5
    cfg.train.num_warmup_epochs = 5
    cfg.train.clip_grad_norm = 0.0
    cfg.train.key_metric = 'auto'

    cfg.train.contrast_weight = 0.1
    cfg.train.temperature = 0.5

    cfg.use_augm_gca = False
    cfg.augm_gca = CN()
    cfg.augm_gca.pe1 = 0.2
    cfg.augm_gca.pf1 = 0.1
    cfg.augm_gca.pe2 = 0.1
    cfg.augm_gca.pf2 = 0.2
    cfg.augm_gca.centrality = 'degree'
    cfg.augm_gca.p_tau = 0.7

    cfg.graphgps = CN(new_allowed=True)
    # Standalone GraphGPS defaults. Per-dataset manifests may override them.
    cfg.graphgps.hidden_size = 128
    cfg.graphgps.num_layers = 4
    cfg.graphgps.local_gnn_type = 'GINE'
    cfg.graphgps.global_model_type = 'Transformer'
    cfg.graphgps.num_heads = 4
    cfg.graphgps.dropout = 0.1
    cfg.graphgps.attn_dropout = 0.1
    cfg.graphgps.act_type = 'relu'
    cfg.graphgps.layer_norm = False
    cfg.graphgps.batch_norm = True
    cfg.graphgps.readout_type = 'mean'
    cfg.graphgps.head_layers = 2
    cfg.graphgps.use_rwpe = False
    cfg.graphgps.rwpe_dim = 20
    cfg.graphgps.rwpe_encoder_type = 'graphgps'
    cfg.graphgps.rwpe_embed_dim = 20
    cfg.graphgps.rwpe_model = 'linear'
    cfg.graphgps.rwpe_layers = 1
    cfg.graphgps.rwpe_raw_norm_type = 'BatchNorm'
    cfg.graphgps.use_lappe = True
    cfg.graphgps.lappe_dim = 8
    cfg.graphgps.lappe_max_freqs = 8
    cfg.graphgps.lappe_eigvec_norm = 'L2'
    cfg.graphgps.lappe_laplacian_norm = 'none'
    cfg.graphgps.lappe_model = 'DeepSet'
    cfg.graphgps.lappe_layers = 2
    cfg.graphgps.lappe_heads = 4
    cfg.graphgps.lappe_raw_norm_type = 'none'

    cfg.gca = CN()
    cfg.gca.pretrain_epochs = 100
    cfg.gca.pretrain_lr = 1e-3
    cfg.gca.pretrain_wd = 1e-5
    cfg.gca.temperature = 0.5
    cfg.gca.drop_scheme = 'degree'  # degree / pr / evc
    cfg.gca.pe1 = 0.2
    cfg.gca.pe2 = 0.3
    cfg.gca.pf1 = 0.1
    cfg.gca.pf2 = 0.2
    cfg.gca.p_tau = 0.7
    cfg.gca.p_corr = 'min'
    cfg.gca.reset_prediction_head = True

    cfg.i2gnn = CN()
    cfg.i2gnn.h = 0
    cfg.i2gnn.node_label = 'spd'
    cfg.i2gnn.max_nodes_per_hop = 0
    cfg.i2gnn.use_rd_auto = True
    cfg.i2gnn.use_rd = False
    cfg.i2gnn.self_loop = False
    cfg.i2gnn.force_reload = False
    cfg.i2gnn.gnn_type = 'gin'
    cfg.i2gnn.virtual_node = False
    cfg.i2gnn.residual = True
    cfg.i2gnn.drop_ratio = 0.35
    cfg.i2gnn.subgraph_pooling = 'mean'
    cfg.i2gnn.subgraph2_pooling = 'mean-center-side'
    cfg.i2gnn.graph_pooling = 'mean'
    cfg.i2gnn.use_pooling_nn = True
    cfg.i2gnn.double_pooling = True
    cfg.i2gnn.gate = True
    cfg.i2gnn.rni = False

    cfg.mosgsl = CN()
    cfg.mosgsl.hidden_size = 128
    cfg.mosgsl.num_layers = 2
    cfg.mosgsl.dropout = 0.5
    cfg.mosgsl.backbone = 'gcn'          # gcn / gin / sage
    cfg.mosgsl.norm = 'BatchNorm'        # BatchNorm / LayerNorm / none
    cfg.mosgsl.pool = 'mean'             # mean / add / max
    cfg.mosgsl.act = 'relu'
    cfg.mosgsl.output_layers = 2
    cfg.mosgsl.use_select = False
    cfg.mosgsl.use_gsl = True
    cfg.mosgsl.use_motif = True
    cfg.mosgsl.use_separate_encoder = True
    cfg.mosgsl.parsing = 'parse_bfs'     # parse_bfs / original
    cfg.mosgsl.max_n_sub = 2
    cfg.mosgsl.max_sub_size = 10
    cfg.mosgsl.force_reload = False
    cfg.mosgsl.select_topk = 0.8
    cfg.mosgsl.select_min_score = -1.0
    cfg.mosgsl.select_estimator = 'projection'
    cfg.mosgsl.gsl_epsilon = 0.8
    cfg.mosgsl.gsl_lambda = 0.5
    cfg.mosgsl.gsl_one_by_one = False
    cfg.mosgsl.metric_heads = 16
    cfg.mosgsl.metric_weighted = True
    cfg.mosgsl.pretrain_epochs = 200
    cfg.mosgsl.pretrain_backbone_only = False
    cfg.mosgsl.motif_per_class = 2
    cfg.mosgsl.motif_sim_type = 'cosine'
    cfg.mosgsl.motif_temperature = 0.7
    cfg.mosgsl.motif_update_type = 'kmeans'
    cfg.mosgsl.motif_k1 = 0.6
    cfg.mosgsl.motif_k2 = 0.2
    cfg.mosgsl.motif_tau = 0.7
    cfg.mosgsl.motif_weighted_mean = False
    cfg.mosgsl.motif_update_min_epoch = 10
    cfg.mosgsl.motif_update_each = 5
    cfg.mosgsl.motif_loss_weight = 0.1

    cfg.gnnak = CN()
    cfg.gnnak.gnn_type = 'GINEConv'       # GINEConv / GCNConv / GATConv / SAGEConv / SimplifiedPNAConv
    cfg.gnnak.hidden_size = 128
    cfg.gnnak.num_layers = 4              # outer GNNAK iterations
    cfg.gnnak.mini_layers = 1             # inner GNN layers; >0 enables GNNAK(+)
    cfg.gnnak.dropout = 0.3
    cfg.gnnak.pool = 'mean'               # mean / add / max
    cfg.gnnak.residual = True
    cfg.gnnak.virtual_node = False
    cfg.gnnak.hops_dim = 16
    cfg.gnnak.embs = [0, 1, 2]            # 0 centroid, 1 subgraph, 2 context
    cfg.gnnak.embs_combine_mode = 'concat'
    cfg.gnnak.mlp_layers = 1
    cfg.gnnak.use_normal_gnn = False
    cfg.gnnak.subgraph_hops = 3
    cfg.gnnak.walk_length = 0
    cfg.gnnak.walk_p = 1.0
    cfg.gnnak.walk_q = 1.0
    cfg.gnnak.walk_repeat = 5
    cfg.gnnak.online = True
    cfg.gnnak.sampling_mode = 'none'      # none / shortest_path / random / min_set_cover
    cfg.gnnak.sampling_redundancy = 0
    cfg.gnnak.sampling_stride = 2
    cfg.gnnak.sampling_random_rate = 0.5
    cfg.gnnak.sampling_batch_factor = 1
    cfg.gnnak.force_reload = False

    cfg.hubgt = CN()
    cfg.hubgt.hidden_size = 128
    cfg.hubgt.num_layers = 4
    cfg.hubgt.num_heads = 4
    cfg.hubgt.dropout = 0.3
    cfg.hubgt.ffn_dim = 256
    cfg.hubgt.dp_attn = 0.1
    cfg.hubgt.dp_bias = 0.0
    cfg.hubgt.pooling = 'mean'          # mean / sum / max

    cfg.nagphormer = CN()
    cfg.nagphormer.hidden_size = 128
    cfg.nagphormer.num_layers = 2
    cfg.nagphormer.num_heads = 4
    cfg.nagphormer.hops = 3
    cfg.nagphormer.dropout = 0.3
    cfg.nagphormer.attention_dropout = 0.1
    cfg.nagphormer.ffn_dim = 0          # 0 -> 2 * hidden_size
    cfg.nagphormer.pooling = 'mean'     # mean / sum / max

    cfg.hymn = CN()
    cfg.hymn.hidden_size = 128
    cfg.hymn.num_layers = 4
    cfg.hymn.num_samples = 3
    cfg.hymn.num_colours = 2
    cfg.hymn.dropout = 0.0
    cfg.hymn.pooling = 'sum'            # mean / sum / max
    cfg.hymn.residual = True
    cfg.hymn.inject_colours = True
    cfg.hymn.multiplier = 1
    cfg.hymn.predictor_multiplier = 0   # 0 -> hidden_size
    cfg.hymn.bn = True
    cfg.hymn.colouring = 'uniform'      # uniform / individualize
    cfg.hymn.counting_input = 'degree'  # degree / constant
    cfg.hymn.predictor = 'mlp'          # mlp / linear

    cfg.ecs = CN()
    cfg.ecs.hidden_size = 300
    cfg.ecs.num_layers = 5
    cfg.ecs.dropout = 0.5
    cfg.ecs.pooling = 'mean'            # mean / sum / max
    cfg.ecs.residual = False
    cfg.ecs.virtual_node = True
    cfg.ecs.h = 3
    cfg.ecs.node_label = 'hop'
    cfg.ecs.use_rd = True
    cfg.ecs.self_loop = True
    cfg.ecs.max_nodes_per_hop = 0       # 0 disables sampling

    cfg.hgpsl = CN()
    cfg.hgpsl.hidden_size = 128
    cfg.hgpsl.pooling_ratio = 0.5
    cfg.hgpsl.dropout = 0.0
    cfg.hgpsl.sample_neighbor = True
    cfg.hgpsl.sparse_attention = True
    cfg.hgpsl.structure_learning = True
    cfg.hgpsl.lamb = 1.0

    cfg.vibgsl = CN()
    cfg.vibgsl.nhid = 64
    cfg.vibgsl.backbone = 'GCN'         # GCN / GIN / GAT
    cfg.vibgsl.graph_type = 'prob'      # prob / KNN / epsilonNN
    cfg.vibgsl.graph_metric_type = 'mlp'
    cfg.vibgsl.repar = True
    cfg.vibgsl.hidden_dim = 16
    cfg.vibgsl.ib_size = 16
    cfg.vibgsl.num_per = 16
    cfg.vibgsl.feature_denoise = False
    cfg.vibgsl.top_k = 10
    cfg.vibgsl.epsilon = 0.3
    cfg.vibgsl.graph_skip_conn = 0.0
    cfg.vibgsl.graph_include_self = True
    cfg.vibgsl.beta = 1e-5
    cfg.vibgsl.dropout = 0.5

    cfg.mvgrl = CN()
    cfg.mvgrl.hidden_size = 512
    cfg.mvgrl.num_layers = 4
    cfg.mvgrl.dropout = 0.0
    cfg.mvgrl.alpha = 0.2

    cfg.rgc = CN()
    cfg.rgc.nhid = 128
    cfg.rgc.backbone = 'gcn'            # gcn / gat / sage
    cfg.rgc.num_layers = 2
    cfg.rgc.hidden_features = 64
    cfg.rgc.embed_features = 32
    cfg.rgc.n_heads = 4
    cfg.rgc.drop_edge = 0.0
    cfg.rgc.drop_node = 0.0
    cfg.rgc.struct_dim = 16
    cfg.rgc.temperature = 0.2
    cfg.rgc.pooling = 'mean'
    cfg.rgc.aux_weight = 0.1

    return cfg


def parse_args(args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", metavar="FILE", help="Path to config file")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line")
    return parser.parse_args(args_str.split() if args_str else None)


def update_cfg(cfg, args_str=None, *, args=None):
    if args is None:
        args = parse_args(args_str)

    cfg = cfg.clone()

    if args.config and os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    cfg.merge_from_list(args.opts)
    return cfg


def init_cfg():
    return set_cfg(CN())


cfg = set_cfg(CN())
