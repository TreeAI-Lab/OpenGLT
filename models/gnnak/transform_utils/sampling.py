import networkx as nx
import numpy as np
import torch


def check_values_in_set(x, values, approach=1):
    assert min(x.shape) > 0
    values = torch.as_tensor(values, device=x.device)
    assert min(values.shape) > 0
    if approach == 0:
        mask = sum(x == i for i in values)
    else:
        mapper = torch.zeros(max(x.max() + 1, values.max() + 1),
                             dtype=torch.bool, device=x.device)
        mapper[values] = True
        mask = mapper[x]
    return mask


def random_sampling(subgraphs_nodes, rate=0.5, minimum_redundancy=0,
                    num_nodes=None):
    if num_nodes is None:
        num_nodes = int(subgraphs_nodes[1].max()) + 1
    while True:
        selected = np.random.choice(num_nodes, max(1, int(num_nodes * rate)),
                                    replace=False)
        selected_t = torch.as_tensor(selected, device=subgraphs_nodes.device)
        node_selected_times = torch.bincount(
            subgraphs_nodes[1][check_values_in_set(subgraphs_nodes[0], selected_t)],
            minlength=num_nodes,
        )
        if node_selected_times.min() >= minimum_redundancy:
            break
    return selected, node_selected_times


def shortest_path_sampling(edge_index, subgraphs_nodes, stride=2,
                           minimum_redundancy=0, random_init=False,
                           num_nodes=None):
    if num_nodes is None:
        num_nodes = int(subgraphs_nodes[1].max()) + 1
    edge_index_cpu = edge_index.detach().cpu()
    subgraphs_nodes_cpu = subgraphs_nodes.detach().cpu()
    graph = nx.from_edgelist(edge_index_cpu.t().numpy())
    graph.add_nodes_from(range(num_nodes))
    if random_init:
        farthest = np.random.choice(num_nodes)
    else:
        subgraph_size = torch.bincount(subgraphs_nodes_cpu[0], minlength=num_nodes)
        farthest = subgraph_size.argmax().item()

    distance = np.ones(num_nodes) * 1e10
    selected = []
    node_selected_times = torch.zeros(num_nodes)
    for _ in range(num_nodes):
        selected.append(farthest)
        node_selected_times[subgraphs_nodes_cpu[1][subgraphs_nodes_cpu[0] == farthest]] += 1
        length_dict = nx.single_source_shortest_path_length(graph, farthest)
        length = np.ones(num_nodes) * 1e10
        length[list(length_dict.keys())] = list(length_dict.values())
        mask = length < distance
        distance[mask] = length[mask]
        if distance.max() < stride and node_selected_times.min() >= minimum_redundancy:
            break
        farthest = np.argmax(distance)
    return selected, node_selected_times.to(edge_index.device)


def min_set_cover_sampling(edge_index, subgraphs_nodes_mask, random_init=False,
                           minimum_redundancy=2):
    num_nodes = subgraphs_nodes_mask.size(0)
    if random_init:
        selected = np.random.choice(num_nodes)
    else:
        selected = subgraphs_nodes_mask.sum(-1).argmax().item()

    node_selected_times = torch.zeros(num_nodes, device=subgraphs_nodes_mask.device)
    selected_all = []
    for _ in range(num_nodes):
        selected_all.append(selected)
        node_selected_times[subgraphs_nodes_mask[selected]] += 1
        if node_selected_times.min() >= minimum_redundancy:
            break
        unused_nodes = ~((node_selected_times - node_selected_times.min()).bool())
        scores = (subgraphs_nodes_mask & unused_nodes).sum(-1)
        scores[selected_all] = 0
        selected = scores.argmax().item()
    return selected_all, node_selected_times
