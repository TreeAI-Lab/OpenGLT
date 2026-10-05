import torch
from sklearn.model_selection import StratifiedKFold
def k_fold(dataset, folds=10):
    skf = StratifiedKFold(folds, shuffle=True, random_state=2024)

    train_indices, test_indices = [], []
    ys = dataset.data.y

    for train, test in skf.split(torch.zeros(len(dataset)), ys):
        train_indices.append(torch.from_numpy(train).to(torch.long))
        test_indices.append(torch.from_numpy(test).to(torch.long))

    return train_indices, test_indices


import os
import random
import numpy as np
def seed_everything(seed=2024, deterministic=True, warn_only=True):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    except:
        torch.use_deterministic_algorithms(False)


import networkx as nx
from torch_geometric.utils import to_networkx
from tqdm import tqdm
from torch_geometric.utils import degree
from torch_geometric.data import InMemoryDataset
def precompute_centrality(dataset, type='degree'):
    print(f"Computing {type} centrality for all graphs...")
    data_list = []
    
    for data in tqdm(dataset):
        num_nodes = data.num_nodes

        if type == 'degree':
            # 使用入度+出度 (对于无向图就是度)
            c = degree(data.edge_index[0], num_nodes).float()
            # 防止 log(0)
            c = c + 1e-6
        elif type == 'eigenvector':
            try:
                G = to_networkx(data, to_undirected=True)
                c_dict = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
                c = torch.tensor([c_dict.get(i, 0) for i in range(num_nodes)], dtype=torch.float) + 1e-6
            except:
                # 某些图可能无法收敛，回退到 degree
                c = degree(data.edge_index[0], num_nodes).float() + 1e-6
        elif type == 'pagerank':
            try:
                G = to_networkx(data)
                c_dict = nx.pagerank(G)
                c = torch.tensor([c_dict.get(i, 0) for i in range(num_nodes)], dtype=torch.float) + 1e-6
            except:
                c = degree(data.edge_index[0], num_nodes).float() + 1e-6
        else:
            raise ValueError(f"Unknown centrality: {type}")

        # 边 = 两端点平均
        row, col = data.edge_index
        c_edge = (c[row] + c[col]) / 2.0

        data.node_c_log = torch.log(c)
        data.edge_c_log = torch.log(c_edge)

        data_list.append(data)
    
    # 重新打包数据
    new_data, new_slices = InMemoryDataset.collate(data_list)
    dataset.data = new_data
    dataset.slices = new_slices
    
    return dataset


from torch_scatter import scatter_max, scatter_mean, scatter_min
def adaptive_aug(data, x_emb, p_e, p_f, p_tau=0.7, p_corr='min'):
    """
    p_corr (str): 'mean' / 'min'. 
        'mean': use (max - mean) as denominator
        'min': use (max - min) as denominator
    """
    assert p_corr in ['mean', 'min']

    device = x_emb.device
    node_batch = data.batch # node ∈ graph_i: [0, 0, 0, 1, 1, 2, ...]
    
    # Edge Dropping
    if p_e > 0 and data.edge_index.numel() > 0:
        s_e = data.edge_c_log.to(device)
        
        # belong to which G
        edge_batch = node_batch[data.edge_index[0]] 
        
        s_max_per_graph, max_idxs = scatter_max(s_e, edge_batch) # s_max_per_graph: [batch_size]
        if p_corr == 'mean':
            s_base_per_graph = scatter_mean(s_e, edge_batch)
        else:
            s_base_per_graph, _ = scatter_min(s_e, edge_batch)

        s_max_expanded = s_max_per_graph[edge_batch] # [num_node_in_whole_batch]
        s_mean_expanded = s_base_per_graph[edge_batch]

        denom = s_max_expanded - s_mean_expanded
        denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
        
        # 计算每一条边的 drop 概率 = (该图最大值 - 该边值) / 该图极差 * p_e
        prob_drop = (s_max_expanded - s_e) / denom * p_e
        prob_drop = prob_drop.clamp(0, p_tau)

        drop_mask = torch.bernoulli(prob_drop).to(torch.bool)
        keep_mask = ~drop_mask
        
        new_edge_index = data.edge_index[:, keep_mask]
        new_edge_attr = data.edge_attr[keep_mask] if data.edge_attr is not None else None
    else:
        new_edge_index = data.edge_index
        new_edge_attr = data.edge_attr

    # Feature Masking
    if p_f > 0:
        s_n = data.node_c_log.to(device)

        n_max_per_graph, _ = scatter_max(s_n, node_batch)
        if p_corr == 'mean':
            n_base_per_graph = scatter_mean(s_n, node_batch)
        else:
            n_base_per_graph, _ = scatter_min(s_n, node_batch)
        
        n_max_expanded = n_max_per_graph[node_batch]
        n_mean_expanded = n_base_per_graph[node_batch]
        
        denom = n_max_expanded - n_mean_expanded
        denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
            
        prob_mask = (n_max_expanded - s_n) / denom * p_f
        prob_mask = prob_mask.clamp(0, p_tau)
        
        mask_indices = torch.bernoulli(prob_mask).to(torch.bool)
        
        new_x_emb = x_emb.clone()
        new_x_emb[mask_indices] = 0.0
    else:
        new_x_emb = x_emb

    return new_edge_index, new_edge_attr, new_x_emb


import torch.nn.functional as F
def compute_nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.shape[0]
    device = z1.device

    # L2 归一化
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    out = torch.cat([z1, z2], dim=0)
    
    # 全相似度矩阵: [2*B, 2*B]
    # sim_matrix[i, j]: i-th样本和j-th样本的sim
    logits = torch.mm(out, out.t()) / temperature

    # 屏蔽对角线(和自己的sim)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    logits.masked_fill_(mask, -1e9)

    labels = torch.cat([
        torch.arange(batch_size, device=device) + batch_size, # z1 -> z2
        torch.arange(batch_size, device=device)               # z2 -> z1
    ], dim=0)

    loss = F.cross_entropy(logits, labels)

    return loss


from torch_geometric.data import InMemoryDataset, Data
import random
import os
import scipy.io as sio
from engine.batch import Batch
from collections import defaultdict
class GraphCountDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphCountDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        a=sio.loadmat(self.raw_paths[0])
        self.train_idx = torch.from_numpy(a['train_idx'][0])
        self.val_idx = torch.from_numpy(a['val_idx'][0]) 
        self.test_idx = torch.from_numpy(a['test_idx'][0]) 

    @property
    def raw_file_names(self):
        return ["data.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) # 'subgraphcount/data.mat'
        # list of adjacency matrix
        A=a['A'][0]
        # list of output
        Y=a['Y']

        data_list = []
        for i in range(len(A)):
            a=A[i]
            expy=Y[i]

            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.ones(A[i].shape[0],1).long() # change to category
          
            data_list.append(Data(edge_index=edge_index, x=x, y=expy))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class dataset_random_graph(InMemoryDataset):
    def __init__(self, url=None, dataname='SubstructureCounting', root='data', processed_name='processed', split='train',
                 transform=None, pre_transform=None, pre_filter=None):
        self.url = url
        self.root = root
        self.dataname = dataname
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.raw = os.path.join(root, dataname)
        self.processed = os.path.join(root, dataname, processed_name)
        super(dataset_random_graph, self).__init__(root=root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        split_id = 0 if split == 'train' else 1 if split == 'val' else 2
        self.data, self.slices = torch.load(self.processed_paths[split_id])
        self.y_dim = self.data.y.size(-1)

    @property
    def raw_dir(self):
        name = 'raw'
        return os.path.join(self.root, self.dataname, name)

    @property
    def processed_dir(self):
        return self.processed
    @property
    def raw_file_names(self):
        names = ["data"]
        return ['{}.mat'.format(name) for name in names]

    @property
    def processed_file_names(self):
        return ['data_tr.pt', 'data_val.pt', 'data_te.pt']

    def adj2data(self, A, y):
        begin, end = np.where(A == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        num_nodes = A.shape[0]
        return Data(edge_index=edge_index, y=torch.tensor(y), num_nodes=torch.tensor([num_nodes]))

    @staticmethod
    def wrap2data(d):
        x, A, y = d['x'], d['A'], d['y']
        x = torch.tensor(x)
        begin, end = np.where(np.sum(A, axis=0) == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        y = torch.tensor(y[-1:])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def process(self):
        # process npy data to PyG data
        print('Processing data from ' + self.raw_dir + '...')
        raw_data = sio.loadmat(self.raw_paths[0])
        data_list_all = [[self.adj2data(raw_data['A'][0][i], raw_data['Y'][i]) for i in idx]
                          for idx in [raw_data['train_idx'][0], raw_data['val_idx'][0], raw_data['test_idx'][0]]]
        for save_path, data_list in zip(self.processed_paths, data_list_all):
            print('pre-transforming for data at'+save_path)
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                temp = []
                for i, data in enumerate(data_list):
                    if i % 100 == 0:
                        print('Pre-processing %d/%d' % (i, len(data_list)))
                    temp.append(self.pre_transform(data))
                data_list = temp
            data, slices = self.collate(data_list)
            torch.save((data, slices), save_path)


def create_subgraph(data, nei_h=2, h=1, sample_ratio=1.0, max_nodes_per_hop=None, node_label='hop', use_rd=False, subgraph_pretransform=None, center_idx=True, self_loop=False):
    if type(h) == int:
        h = [h]
    assert (isinstance(data, Data))
    x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes

    new_data_multi_hop = {}
    relabels = []
    for h_ in h:
        subgraphs = []
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_, z_, relabel = k_hop_subgraph(
                ind, h_, edge_index, True, num_nodes, node_label=node_label,
                max_nodes_per_hop=max_nodes_per_hop
            )
            x_ = None
            edge_attr_ = None
            pos_ = None
            if x is not None:
                x_ = x[nodes_]
            else:
                x_ = None

            if 'node_type' in data:
                node_type_ = data.node_type[nodes_]

            if data.edge_attr is not None:
                edge_attr_ = data.edge_attr[edge_mask_]
            if data.pos is not None:
                pos_ = data.pos[nodes_]
            data_ = data.__class__(x_, edge_index_, edge_attr_, None, pos_, z=z_)
            data_.num_nodes = nodes_.shape[0]

            if 'node_type' in data:
                data_.node_type = node_type_

            if nei_h == 2:
                data_ = subgraph_to_subgraph_2h(data_)
            elif nei_h == 3:
                data_ = subgraph_to_subgraph_3h(data_)

            # node_to_original_node
            deg = torch.sum(edge_index_[0] == 0).item()
            if deg == 0:
                deg = 1
            relabels = relabels + list(np.array(relabel)) * deg

            subgraphs.append(data_)

        # new_data is treated as a big disconnected graph of the batch of subgraphs
        new_data = Batch.from_data_list(subgraphs)
        new_data.num_nodes = sum(data_.num_nodes for data_ in subgraphs)
        new_data.num_subgraphs = len(subgraphs)

        # node_to_original_node
        new_data.node_to_original_node = torch.tensor(relabels)
        new_data.num_original_nodes = data.num_nodes

        del new_data.batch
        if 'batch_2' in new_data:
            new_data.assignment2_to_subgraph = new_data.batch_2
            del new_data.batch_2
        if 'batch_3' in new_data:
            new_data.assignment3_to_subgraph = new_data.batch_3
            del new_data.batch_3

        # create a subgraph_to_graph assignment vector (all zero)
        new_data.subgraph_to_graph = torch.zeros(len(subgraphs), dtype=torch.long)
        new_data.subgraph2_to_graph = torch.zeros(torch.sum(new_data.num_subgraphs2), dtype=torch.long)
        del new_data.num_subgraphs2

        # copy remaining graph attributes
        for k, v in data:
            if k not in ['x', 'edge_index', 'edge_attr', 'pos', 'num_nodes', 'batch',
                         'z', 'rd', 'node_type']:
                new_data[k] = v

        if len(h) == 1:
            return new_data
        else:
            new_data_multi_hop[h_] = new_data

    return new_data_multi_hop


def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target', node_label='hop', 
                   max_nodes_per_hop=None):

    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    subsets = [torch.tensor([node_idx], device=row.device).flatten()]
    visited = set(subsets[-1].tolist())
    label = defaultdict(list)
    for node in subsets[-1].tolist():
        label[node].append(1)
    if node_label == 'hop':
        hops = [torch.LongTensor([0], device=row.device).flatten()]
    for h in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        new_nodes = col[edge_mask]
        tmp = []
        for node in new_nodes.tolist():
            if node in visited:
                continue
            tmp.append(node)
            label[node].append(h+2)
        if len(tmp) == 0:
            break
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(tmp):
                tmp = random.sample(tmp, max_nodes_per_hop)
        new_nodes = set(tmp)
        visited = visited.union(new_nodes)
        new_nodes = torch.tensor(list(new_nodes), device=row.device)
        subsets.append(new_nodes)
        if node_label == 'hop':
            hops.append(torch.LongTensor([h+1] * len(new_nodes), device=row.device))
    subset = torch.cat(subsets)
    inverse_map = torch.tensor(range(subset.shape[0]))
    if node_label == 'hop':
        hop = torch.cat(hops)
    # Add `node_idx` to the beginning of `subset`.
    subset = subset[subset != node_idx]
    subset = torch.cat([torch.tensor([node_idx], device=row.device), subset])

    z = None
    if node_label == 'hop':
        hop = hop[hop != 0]
        hop = torch.cat([torch.LongTensor([0], device=row.device), hop])
        z = hop.unsqueeze(1)
        z = torch.zeros_like(z)
        z[0, 0] = 1.
    elif node_label.startswith('spd') or node_label == 'drnl' or node_label.startswith('dspd'):
        if node_label.startswith('spd'):
            # keep top k shortest-path distances
            num_spd = int(node_label[3:]) if len(node_label) > 3 else 2
            z = torch.zeros(
                [subset.size(0), num_spd], dtype=torch.long, device=row.device
            )
        elif node_label.startswith('dspd'):
            # keep top k shortest-path distances
            num_spd = int(node_label[4:]) if len(node_label) > 4 else 2
            z = torch.zeros(
                [subset.size(0), num_spd], dtype=torch.long, device=row.device
            )
        elif node_label == 'drnl':
            # see "Link Prediction Based on Graph Neural Networks", a special
            # case of spd2
            num_spd = 2
            z = torch.zeros([subset.size(0), 1], dtype=torch.long, device=row.device)

        for i, node in enumerate(subset.tolist()):
            dists = label[node][:num_spd]  # keep top num_spd distances
            if node_label == 'spd' or node_label == 'dspd':
                z[i][:min(num_spd, len(dists))] = torch.tensor(dists)
            elif node_label == 'drnl':
                dist1 = dists[0]
                dist2 = dists[1] if len(dists) == 2 else 0
                if dist2 == 0:
                    dist = dist1
                else:
                    dist = dist1 * (num_hops + 1) + dist2
                z[i][0] = dist
        
    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, edge_mask, z, subset


def subgraph_to_subgraph_2h(d, center=0):
    neighbors = d.edge_index[1][torch.where(d.edge_index[0] == center)[0]] # 1st neighbors

    num_neighbors = len(neighbors)

    if num_neighbors == 0:
        d.node_to_subgraph2 = torch.zeros([d.num_nodes]).long()
        d.subgraph2_to_subgraph = torch.zeros([1]).long()
        d.num_subgraphs2 = 1

        d.center_idx = torch.tensor([[0], [0]]).long().T
        return d

    z = torch.zeros_like(d.z)

    z = torch.tile(z, [num_neighbors, 1])
    edge_index = torch.zeros([2, d.num_edges * num_neighbors]).long()

    # tile edge attr if it exists
    if d.edge_attr is not None:
        edge_attr = torch.zeros([d.num_edges * num_neighbors, d.edge_attr.size(1)], dtype=d.edge_attr.dtype)
    else:
        edge_attr = None

    node_to_subgraph2 = torch.zeros([d.num_nodes * num_neighbors]).long()

    center_idx = [[], []]
    for i, n in enumerate(neighbors):
        z_n = d.z.clone()
        z_n[n] = 2
        z[i * d.num_nodes : (i + 1) * d.num_nodes] = z_n

        temp = d.edge_index + i * d.num_nodes
        edge_index[:, i * d.num_edges : (i + 1) * d.num_edges] = temp

        # Copy edge_attr for each neighbor subgraph
        if edge_attr is not None:
            edge_attr[i * d.num_edges : (i + 1) * d.num_edges] = d.edge_attr

        node_to_subgraph2[i * d.num_nodes : (i + 1) * d.num_nodes] = i * torch.ones([d.num_nodes])

        center_idx[0].append(0 + d.num_nodes * i)
        center_idx[1].append(n.item() + d.num_nodes * i) # center node id and side node id

    subgraphs_2 = Data(edge_index=edge_index, z=z, node_to_subgraph2=node_to_subgraph2,
                       subgraph2_to_subgraph=torch.zeros([num_neighbors]).long(), num_subgraphs2=num_neighbors,
                       center_idx=torch.tensor(center_idx).long().T)

    subgraphs_2.num_nodes = d.num_nodes * num_neighbors

    if d.x != None:
        subgraphs_2.x = d.x

    if edge_attr is not None:
        subgraphs_2.edge_attr = edge_attr

    return subgraphs_2


def subgraph_to_subgraph_3h(d, center=0):
    neighbors = d.edge_index[1][torch.where(d.edge_index[0] == center)[0]] # 1st neighbors

    num_neighbors = len(neighbors)

    if num_neighbors == 0:
        d.node_to_subgraph2 = torch.zeros([d.num_nodes]).long()
        d.subgraph2_to_subgraph = torch.zeros([1]).long()
        d.num_subgraphs2 = 1

        d.center_idx = torch.tensor([[0], [0]]).long().T
        return d
    
    neighbors_2stage = {}
    for neighbor in neighbors:
        second_stage_neighbors = d.edge_index[1][torch.where(d.edge_index[0] == neighbor)[0]]
        second_stage_neighbors = second_stage_neighbors[second_stage_neighbors != center] # 2nd neighbors

        if second_stage_neighbors.numel() == 0:
            second_stage_neighbors = torch.tensor([-1])

        neighbors_2stage[neighbor.item()] = second_stage_neighbors

    total_neighbors = sum(len(v) for v in neighbors_2stage.values())

    z = torch.zeros_like(d.z)

    z = torch.tile(z, [total_neighbors, 1])

    edge_index = torch.zeros([2, d.num_edges * total_neighbors]).long()

    if d.edge_attr is not None:
        edge_attr = torch.zeros([d.num_edges * total_neighbors, d.edge_attr.size(1)], dtype=d.edge_attr.dtype)
    else:
        edge_attr = None

    node_to_subgraph2 = torch.zeros([d.num_nodes * total_neighbors]).long()

    center_idx = [[], []]

    idx = 0
    for i, (n1, neighbors) in enumerate(neighbors_2stage.items()):
        for n2 in neighbors:
            # print(f"{idx}, {n1}, {n2.item()}")
            if n2.item() == -1:
                z_n = d.z.clone()
                z_n[n1] = 2
                z[idx * d.num_nodes : (idx + 1) * d.num_nodes] = z_n

                temp = d.edge_index + idx * d.num_nodes
                edge_index[:, idx * d.num_edges : (idx + 1) * d.num_edges] = temp

                if edge_attr is not None:
                    edge_attr[idx * d.num_edges : (idx + 1) * d.num_edges] = d.edge_attr

                node_to_subgraph2[idx * d.num_nodes : (idx + 1) * d.num_nodes] = idx * torch.ones([d.num_nodes])

                center_idx[0].append(0 + d.num_nodes * idx)
                center_idx[1].append(n1 + d.num_nodes * idx)

                idx += 1

            else:
                z_n = d.z.clone()
                z_n[n1] = 2
                z_n[n2.item()] = 3
                z[idx * d.num_nodes : (idx + 1) * d.num_nodes] = z_n

                temp = d.edge_index + idx * d.num_nodes
                edge_index[:, idx * d.num_edges : (idx + 1) * d.num_edges] = temp

                if edge_attr is not None:
                    edge_attr[idx * d.num_edges : (idx + 1) * d.num_edges] = d.edge_attr

                node_to_subgraph2[idx * d.num_nodes : (idx + 1) * d.num_nodes] = idx * torch.ones([d.num_nodes])

                center_idx[0].append(0 + d.num_nodes * idx)
                center_idx[1].append(n1 + d.num_nodes * idx)

                idx += 1

    subgraphs_2 = Data(edge_index=edge_index, z=z, node_to_subgraph2=node_to_subgraph2,
                       subgraph2_to_subgraph=torch.zeros([total_neighbors]).long(), num_subgraphs2=total_neighbors,
                       center_idx=torch.tensor(center_idx).long().T)

    subgraphs_2.num_nodes = d.num_nodes * total_neighbors

    if d.x != None:
        subgraphs_2.x = d.x

    if edge_attr is not None:
        subgraphs_2.edge_attr = edge_attr

    return subgraphs_2


def preprocess_subgraph_dataset(data_list, path_k, subgraph_range):
    processed = []
    for i, data in enumerate(data_list):
        if i % 100 == 0:
            print(f'Subgraph pre-processing {i}/{len(data_list)} (path_k={path_k}, h={subgraph_range})')
        new_data = create_subgraph(data, nei_h=path_k, h=subgraph_range)
        processed.append(new_data)
    return processed


def subgraph_collate_fn(batch):
    return Batch.from_data_list(batch)
