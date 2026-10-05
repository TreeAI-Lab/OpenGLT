import os
import os.path as osp
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch_geometric.data import InMemoryDataset, extract_zip

GOOD_ATOM_FEATURE_DIMS = (119, 4, 11, 12, 9, 5, 8, 2, 2)
GOOD_BOND_FEATURE_DIMS = (5, 6, 2)


@dataclass(frozen=True)
class GOODDatasetSpec:
    key: str
    official_name: str
    domains: Tuple[str, ...]
    canonical_no_shift_domain: str
    url: str
    task: str
    key_metric: str
    output_dim: int
    model_dataset_type: str


GOOD_DATASET_SPECS: Dict[str, GOODDatasetSpec] = {
    'hiv': GOODDatasetSpec(
        key='hiv',
        official_name='GOODHIV',
        domains=('scaffold', 'size'),
        canonical_no_shift_domain='scaffold',
        url='https://drive.google.com/file/d/1CoOqYCuLObnG5M0D8a2P2NyL61WjbCzo/view?usp=sharing',
        task='b',
        key_metric='auc',
        output_dim=1,
        model_dataset_type='good_mol',
    ),
    'zinc': GOODDatasetSpec(
        key='zinc',
        official_name='GOODZINC',
        domains=('scaffold', 'size'),
        canonical_no_shift_domain='scaffold',
        url='https://drive.google.com/file/d/1CHR0I1JcNoBqrqFicAZVKU3213hbsEPZ/view?usp=sharing',
        task='reg',
        key_metric='mae',
        output_dim=1,
        model_dataset_type='good_mol',
    ),
}

GOOD_SHIFTS = ('no_shift', 'covariate', 'concept')
GOOD_PROCESSED_FILE_NAMES = (
    'no_shift_train.pt',
    'no_shift_val.pt',
    'no_shift_test.pt',
    'covariate_train.pt',
    'covariate_val.pt',
    'covariate_test.pt',
    'covariate_id_val.pt',
    'covariate_id_test.pt',
    'concept_train.pt',
    'concept_val.pt',
    'concept_test.pt',
    'concept_id_val.pt',
    'concept_id_test.pt',
)


def _normalise_good_name(name: str) -> str:
    value = str(name).strip().lower().replace('_', '-')
    while '--' in value:
        value = value.replace('--', '-')
    return value


def is_good_dataset_name(name: str) -> bool:
    value = _normalise_good_name(name)
    return value.startswith('good-') and any(
        value == f'good-{key}' or value.startswith(f'good-{key}-')
        for key in GOOD_DATASET_SPECS
    )


def parse_good_dataset_name(name: str):
    """Return ``(spec, domain, shift, canonical_name)`` with strict checks."""
    value = _normalise_good_name(name)
    if not value.startswith('good-'):
        raise ValueError(
            f"GOOD dataset names must start with 'good-', got {name!r}."
        )
    tail = value[len('good-'):]
    dataset_key = next(
        (
            key for key in sorted(GOOD_DATASET_SPECS, key=len, reverse=True)
            if tail == key or tail.startswith(f'{key}-')
        ),
        None,
    )
    if dataset_key is None:
        raise ValueError(
            f"Unknown GOOD graph dataset {name!r}. Supported datasets: "
            f"{sorted(GOOD_DATASET_SPECS)}."
        )

    spec = GOOD_DATASET_SPECS[dataset_key]
    suffix = tail[len(dataset_key):].lstrip('-')
    suffix = suffix.replace('no-shift', 'no_shift')

    if suffix in ('', 'no_shift'):
        domain = spec.canonical_no_shift_domain
        shift = 'no_shift'
    else:
        domain = next(
            (
                candidate
                for candidate in sorted(spec.domains, key=len, reverse=True)
                if suffix == candidate or suffix.startswith(f'{candidate}-')
            ),
            None,
        )
        if domain is None:
            raise ValueError(
                f"{spec.official_name} supports domains={list(spec.domains)}, "
                f"but dataset name {name!r} does not specify one of them."
            )
        shift = suffix[len(domain):].lstrip('-').replace('-', '_')
        if not shift:
            raise ValueError(
                f"Missing GOOD shift in {name!r}; choose covariate, concept, "
                "or no-shift."
            )
        if shift not in GOOD_SHIFTS:
            raise ValueError(
                f"Unknown GOOD shift {shift!r}; supported={list(GOOD_SHIFTS)}."
            )
        if shift == 'no_shift' and domain != spec.canonical_no_shift_domain:
            raise ValueError(
                f"{spec.official_name} has one benchmark no-shift setting: "
                f"good-{dataset_key}-{spec.canonical_no_shift_domain}-no-shift "
                f"(or good-{dataset_key}-no-shift)."
            )

    shift_token = 'no-shift' if shift == 'no_shift' else shift
    canonical_name = f'good-{dataset_key}-{domain}-{shift_token}'
    return spec, domain, shift, canonical_name


def good_model_dataset_type(name: str) -> str:
    spec, _, _, _ = parse_good_dataset_name(name)
    return spec.model_dataset_type


def good_loss_type(name: str) -> str:
    spec, _, _, _ = parse_good_dataset_name(name)
    if spec.task == 'mc':
        return 'CE'
    if spec.task in ('b', 'ml'):
        return 'BCE'
    if spec.task == 'reg':
        return 'L1'
    raise ValueError(f"Unsupported GOOD task: {spec.task}")


def good_key_metric(name: str) -> str:
    spec, _, _, _ = parse_good_dataset_name(name)
    return spec.key_metric


def iter_good_benchmark_names(
        include_code_extensions: bool = False) -> Iterable[str]:
    """Yield the non-duplicated graph-level benchmark settings.

    By default this follows the domains reported in the GOOD paper.  The
    released repository additionally contains a CMNIST ``background`` domain;
    it remains loadable by name and can be included explicitly, but is kept
    out of the paper-comparable core suite.
    """
    for key, spec in GOOD_DATASET_SPECS.items():
        yield f'good-{key}-{spec.canonical_no_shift_domain}-no-shift'
        domains = spec.domains
        if key == 'cmnist' and not include_code_extensions:
            domains = ('color',)
        for domain in domains:
            yield f'good-{key}-{domain}-covariate'
            yield f'good-{key}-{domain}-concept'


def _torch_load_pair(path):
    try:
        payload = torch.load(path, weights_only=False)
    except TypeError:
        payload = torch.load(path)
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError(
            f"Expected GOOD's official (data, slices) payload in {path}, "
            f"got {type(payload).__name__}."
        )
    return payload[0], payload[1]


class GOODOfficialSplitDataset(InMemoryDataset):
    """One split backed directly by a GOOD official processed artifact."""

    def __init__(
            self, root: str, spec: GOODDatasetSpec, domain: str,
            shift: str, subset: str):
        self.good_spec = spec
        self.name = spec.official_name
        self.domain = domain
        self.shift = shift
        self.subset = subset
        self.url = spec.url
        super().__init__(root)
        path = osp.join(self.processed_dir, self._split_file_name)
        self.data, self.slices = _torch_load_pair(path)
        if hasattr(self, '_data_list'):
            self._data_list = None

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, self.domain, 'processed')

    @property
    def processed_file_names(self):
        return list(GOOD_PROCESSED_FILE_NAMES)

    @property
    def _split_file_name(self):
        return f'{self.shift}_{self.subset}.pt'

    def _download(self):
        target = osp.join(self.processed_dir, self._split_file_name)
        if osp.exists(target):
            return
        os.makedirs(self.root, exist_ok=True)
        try:
            import gdown
        except ImportError as exc:
            raise ImportError(
                f"{target} is missing. Install gdown (`pip install gdown`) so "
                "OpenGLT can download GOOD's official released split archive, "
                "or copy the official GOOD data directory under data/GOOD/."
            ) from exc

        archive = osp.join(self.root, f'{self.name}.zip')
        print(
            f"[GOOD download] fetching the official {self.name} release "
            f"from {self.url}"
        )
        downloaded = gdown.download(self.url, output=archive, fuzzy=True)
        if not downloaded or not osp.exists(archive):
            raise RuntimeError(f"Failed to download GOOD archive from {self.url}")
        extract_zip(archive, self.root)
        os.unlink(archive)
        if not osp.exists(target):
            raise FileNotFoundError(
                f"GOOD archive was extracted, but expected split is missing: "
                f"{target}"
            )

    def download(self):
        self._download()

    def process(self):
        raise FileNotFoundError(
            "OpenGLT does not regenerate GOOD splits because that could change "
            "benchmark membership. Use GOOD's published processed split files."
        )


def load_official_good_splits(
        name: str, root: str):
    """Load the exact released GOOD split objects and their setting metadata."""
    spec, domain, shift, canonical_name = parse_good_dataset_name(name)
    subsets = ('train', 'val', 'test')
    datasets = {
        subset: GOODOfficialSplitDataset(root, spec, domain, shift, subset)
        for subset in subsets
    }
    if shift != 'no_shift':
        datasets['id_val'] = GOODOfficialSplitDataset(
            root, spec, domain, shift, 'id_val'
        )
        datasets['id_test'] = GOODOfficialSplitDataset(
            root, spec, domain, shift, 'id_test'
        )
    else:
        datasets['id_val'] = None
        datasets['id_test'] = None
    return datasets, spec, domain, shift, canonical_name
