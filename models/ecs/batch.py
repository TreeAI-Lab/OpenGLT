import collections.abc as container_abcs

import torch
import torch.utils.data
from torch_geometric.data import Data


int_classes = int
string_classes = (str, bytes)


def data_keys(data):
    keys = data.keys
    return list(keys()) if callable(keys) else list(keys)


class ECSBatch(Data):
    """PyG batch with ESC structural-encoding offsets.

    The official ESC code stores the sparse structural vector for each edge as
    (pos_index, pos_enc, pos_batch). pos_index indexes a fixed embedding table
    and must not be shifted during batching, while pos_batch is shifted by the
    number of encoded edges in previous graphs.
    """

    def __init__(self, batch=None, **kwargs):
        super().__init__(**kwargs)
        self.batch = batch
        self.__data_class__ = Data
        self.__slices__ = None

    @staticmethod
    def from_data_list(data_list, follow_batch=None):
        follow_batch = follow_batch or []
        keys = [set(data_keys(data)) for data in data_list]
        keys = list(set.union(*keys))
        if 'batch' in keys:
            keys.remove('batch')

        batch = ECSBatch()
        batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key: [0] for key in keys}
        for key in keys:
            batch[key] = []
        for key in follow_batch:
            batch[f'{key}_batch'] = []

        cumsum = {key: 0 for key in keys}
        batch.batch = []
        for graph_id, data in enumerate(data_list):
            for key in data_keys(data):
                if key == 'batch':
                    continue
                item = data[key]
                if torch.is_tensor(item) and item.dtype != torch.bool:
                    item = item + cumsum[key]
                if torch.is_tensor(item):
                    cat_dim = data.__cat_dim__(key, data[key])
                    size = item.size(cat_dim)
                else:
                    size = 1
                batch.__slices__[key].append(size + batch.__slices__[key][-1])

                if key == 'pos_batch':
                    if data['pos_batch'].numel() > 0:
                        cumsum[key] = cumsum[key] + int(data['pos_batch'].max().item()) + 1
                elif key in {'pos_enc', 'pos_index', 'edge_pos'}:
                    cumsum[key] = cumsum[key]
                else:
                    inc = 0 if item is None else data.__inc__(key, item)
                    cumsum[key] = cumsum[key] + inc
                batch[key].append(item)

                if key in follow_batch:
                    item_batch = torch.full((size,), graph_id, dtype=torch.long)
                    batch[f'{key}_batch'].append(item_batch)

            num_nodes = data.num_nodes
            if isinstance(num_nodes, torch.Tensor):
                num_nodes = int(num_nodes.item())
            if num_nodes is not None:
                batch.batch.append(torch.full((num_nodes,), graph_id, dtype=torch.long))

        if batch.batch:
            batch.batch = torch.cat(batch.batch, dim=0)
        else:
            batch.batch = None

        for key in data_keys(batch):
            item = batch[key]
            if isinstance(item, list) and item and torch.is_tensor(item[0]):
                batch[key] = torch.cat(item, dim=data_list[0].__cat_dim__(key, item[0]))
            elif isinstance(item, list) and item and all(v is None for v in item):
                batch[key] = None
            elif isinstance(item, list) and item and all(isinstance(v, (int, float)) for v in item):
                batch[key] = torch.tensor(item)
            elif isinstance(item, list) and len(item) == 1:
                batch[key] = item[0]

        if batch.batch is not None:
            batch.num_nodes = int(batch.batch.numel())

        return batch.contiguous()


class ECSDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=None, **kwargs):
        follow_batch = follow_batch or []

        def collate(batch):
            elem = batch[0]
            if isinstance(elem, Data):
                return ECSBatch.from_data_list(batch, follow_batch)
            if isinstance(elem, float):
                return torch.tensor(batch, dtype=torch.float)
            if isinstance(elem, int_classes):
                return torch.tensor(batch)
            if isinstance(elem, string_classes):
                return batch
            if isinstance(elem, container_abcs.Mapping):
                return {key: collate([d[key] for d in batch]) for key in elem}
            if isinstance(elem, tuple) and hasattr(elem, '_fields'):
                return type(elem)(*(collate(samples) for samples in zip(*batch)))
            if isinstance(elem, container_abcs.Sequence):
                return [collate(samples) for samples in zip(*batch)]
            raise TypeError(f'DataLoader found invalid type: {type(elem)}')

        super().__init__(dataset, batch_size, shuffle, collate_fn=collate, **kwargs)
