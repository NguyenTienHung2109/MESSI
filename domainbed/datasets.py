# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

import numpy as np
import torch
import torchvision.datasets.folder
from PIL import Image, ImageFile
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

def _resolve_image_size_from_hparams(hparams, default=224):
    raw_resolution = hparams.get("resolution", None)
    raw_image_size = hparams.get("image_size", None)
    if raw_resolution is not None and raw_resolution != "auto":
        return int(raw_resolution)
    if raw_image_size is not None:
        return int(raw_image_size)
    return int(default)


DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "ColoredMNIST_E",
    "ColoredMNIST_K",
    "RotatedMNIST",
    # Big images
    "CUB",
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    "WILDSIWildCam",
    "WILDSIWildCamERM",
    "MetaShift_K",
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 8  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )


class Debug28(Debug):
    N_WORKERS = 0
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']


class Debug224(Debug):
    N_WORKERS = 0
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                           self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
                                                         1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class ColoredMNIST_E(MultipleEnvironmentMNIST):
    """ColoredMNIST with a configurable number of training environments E.

    Protocol from Wang et al., "Lost Domain Generalization Is a Natural
    Consequence of Lack of Training Domains", AAAI 2024.
    """
    # Class-level placeholder of length default_E + 1 = 9, so that
    # datasets.num_environments("ColoredMNIST_E") works before the dataset
    # is instantiated (e.g. in test_datasets.py). The real, p_e-annotated
    # names are written to self.ENVIRONMENTS in __init__.
    ENVIRONMENTS = [f'env_{i}' for i in range(9)]

    def __init__(self, root, test_envs, hparams):
        E = int(hparams.get('num_environments', 8))
        if E < 2:
            raise ValueError(f"num_environments must be >= 2, got {E}")

        candidates = np.linspace(1.0 / (E + 2), (E + 1) / (E + 2), E + 1)
        drop_idx = int(np.argmin(np.abs(candidates - 0.5)))
        train_p = np.delete(candidates, drop_idx).tolist()

        environments = train_p + [0.5]
        self.ENVIRONMENTS = [f'p={p:.3f}' for p in environments]

        super().__init__(root, environments, self.color_dataset,
                         (2, 28, 28,), 2)
        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        labels = (labels < 5).float()
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))),
               (1 - colors).long(), :, :] *= 0
        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class ColoredMNIST_K(MultipleDomainDataset):
    """ColoredMNIST with K source domains + 1 fixed test domain (p=0.5).

    Sample budget (fixed across K):
      - test env: exactly `test_size` samples (default 10000), p_color = 0.5
      - source envs: remaining (70000 - test_size) split equally across K
        (any remainder < K is dropped)

    Source p_color values: linspace(1/(K+2), (K+1)/(K+2), K+1) with the value
    closest to 0.5 dropped. Test env appended at index K (last).
    """
    ENVIRONMENTS = [f'env_{i}' for i in range(10)]

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        K = int(hparams.get('num_source_domains', 9))
        if K < 2:
            raise ValueError(f"num_source_domains must be >= 2, got {K}")
        test_size = int(hparams.get('test_size', 10000))

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)
        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))
        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        N_total = len(original_images)
        if test_size >= N_total:
            raise ValueError(
                f"test_size ({test_size}) must be < total MNIST ({N_total})")
        N_train = N_total - test_size
        per_env = N_train // K
        if per_env < 1:
            raise ValueError(
                f"K={K} too large for source pool of {N_train} samples")

        shuffle = torch.randperm(N_total)
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        train_imgs = original_images[: K * per_env]
        train_lbls = original_labels[: K * per_env]
        test_imgs = original_images[N_train: N_train + test_size]
        test_lbls = original_labels[N_train: N_train + test_size]

        candidates = np.linspace(1.0 / (K + 2), (K + 1) / (K + 2), K + 1)
        drop_idx = int(np.argmin(np.abs(candidates - 0.5)))
        train_p = np.delete(candidates, drop_idx).tolist()
        self.ENVIRONMENTS = [f'p={p:.3f}' for p in train_p] + ['p=0.500']

        self.datasets = []
        for i, p in enumerate(train_p):
            ei = train_imgs[i * per_env: (i + 1) * per_env]
            el = train_lbls[i * per_env: (i + 1) * per_env]
            self.datasets.append(self.color_dataset(ei, el, p))
        self.datasets.append(self.color_dataset(test_imgs, test_lbls, 0.5))

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        labels = (labels < 5).float()
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))),
               (1 - colors).long(), :, :] *= 0
        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                                               interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path, transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


class CUB(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["Candy", "Mosaic", "Natural", "Udnie"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "CUB_DG/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 500
    N_STEPS = 15001
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class TerraIncognita(MultipleEnvironmentImageFolder):
    # may need larger weight decay
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        # Some WILDS datasets (e.g. Camelyon17) return numpy arrays; others
        # (e.g. iWildCam, FMoW) return PIL.Image (incl. JpegImageFile subclass).
        if not isinstance(x, Image.Image):
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def get_labels(self):
        """1-D LongTensor of labels in env-local index order. Cheap: indexes
        wilds_dataset.y_array directly with no image loading."""
        return self.dataset.y_array[self.indices]

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)

    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = ["hospital_0", "hospital_1", "hospital_2", "hospital_3",
                    "hospital_4"]

    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = ["region_0", "region_1", "region_2", "region_3",
                    "region_4", "region_5"]

    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)


def _build_uint8_cache(wilds_dataset, row_indices, target_size=224, num_workers=4):
    """Pre-decode each row to a uint8 tensor (3, H, W) and stack into (N, 3, H, W).

    Uses a temp DataLoader with workers to parallelise PIL decode + resize.
    Workers are torn down once the cache is built so they don't persist
    alongside training/eval loaders.
    """
    class _PreDecodeDS(torch.utils.data.Dataset):
        def __init__(self, ds, idxs, sz):
            self.ds = ds
            self.idxs = np.asarray(idxs, dtype=np.int64)
            self.sz = sz

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, i):
            row = int(self.idxs[i])
            img = self.ds.get_input(row)
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert('RGB').resize((self.sz, self.sz), Image.BILINEAR)
            arr = np.array(img, dtype=np.uint8)          # writable copy
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    n = len(row_indices)
    out = torch.empty(n, 3, target_size, target_size, dtype=torch.uint8)
    loader = torch.utils.data.DataLoader(
        _PreDecodeDS(wilds_dataset, row_indices, target_size),
        batch_size=128, num_workers=num_workers, shuffle=False,
        pin_memory=False)
    ptr = 0
    for batch in loader:
        b = batch.size(0)
        out[ptr:ptr + b] = batch
        ptr += b
    return out


class _WILDSIWildCamSplitEnv(torch.utils.data.Dataset):
    """One env of WILDSIWildCam, defined by row indices into IWildCamDataset.

    Exposes pre_split (class-stratified 80/20) so misc.split_dataset honors
    the partition rather than re-splitting randomly. Both source and test
    envs use 80/20 so env_*_in_acc and env_*_out_acc are well-defined.

    When `cache_in_ram=True`, all images are pre-decoded to a uint8 tensor
    stack at construction time (~150KB/image at 224x224). __getitem__ then
    bypasses PIL decode + Resize and only converts uint8→float32 + normalizes,
    giving a ~5-10x speedup on eval-heavy workloads. Memory cost: ~11GB for
    the 4 WILDS eval envs combined; do NOT enable for the 243 train envs
    (~19GB extra would push past available RAM).
    """

    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, wilds_dataset, row_indices, transform,
                 global_in_mask=None, cache_in_ram=False,
                 cache_target_size=224,
                 shared_cache=None, shared_cache_row_to_idx=None):
        """Args:
            global_in_mask: optional boolean array indexed by *row id in
                the underlying WILDS dataset*. When provided, in/out are
                derived from it (a row is `in` iff global_in_mask[row]) so
                that aggregating in_splits across many env instances yields
                the same global partition as a single class-stratified shuffle
                over their union — matches the legacy single-env split.
                When None, falls back to a per-env class-stratified shuffle
                (used for the 4 eval envs where there is no shared partition).
            shared_cache: optional uint8 tensor (N_total, 3, H, W) holding
                pre-decoded images for a SUPERSET of this env (e.g. all 130k
                train rows). Combined with `shared_cache_row_to_idx` (mapping
                global WILDS row id → cache index), this env reads from the
                shared tensor instead of building its own — avoids 243× cache
                build overhead and memory duplication.
        """
        self.dataset = wilds_dataset
        self.indices = np.asarray(row_indices, dtype=np.int64)
        self.transform = transform
        self._cache = None
        self._shared_cache = None
        self._cache_idx_for_local = None
        if shared_cache is not None:
            self._shared_cache = shared_cache
            self._cache_idx_for_local = np.asarray(
                [shared_cache_row_to_idx[int(r)] for r in self.indices],
                dtype=np.int64)
        elif cache_in_ram:
            self._cache = _build_uint8_cache(
                wilds_dataset, self.indices, target_size=cache_target_size)

        if global_in_mask is not None:
            local_in = [i for i, row in enumerate(self.indices)
                        if bool(global_in_mask[int(row)])]
            local_out = [i for i in range(len(self.indices))
                         if i not in set(local_in)]
            # If a tiny env happens to land entirely in `out` (no `in`
            # samples), keep at least one in `in` so the train loader is
            # non-empty.
            if not local_in and local_out:
                local_in = [local_out[0]]
                local_out = local_out[1:]
            self.pre_split = {"in": sorted(local_in),
                              "out": sorted(local_out)}
            return

        from collections import defaultdict
        labels = wilds_dataset.y_array[self.indices].tolist()
        by_cls: dict[int, list[int]] = defaultdict(list)
        for i, c in enumerate(labels):
            by_cls[int(c)].append(i)
        rng = np.random.RandomState(0)
        in_idx, out_idx = [], []
        for _, idxs in by_cls.items():
            idxs = list(idxs)
            rng.shuffle(idxs)
            if len(idxs) <= 1:
                # Tiny class — keep the sole sample in `in` so the train
                # loader is non-empty; `out` may be empty for this class.
                in_idx.extend(idxs)
                continue
            k_out = max(1, int(round(0.2 * len(idxs))))
            # Guarantee in_idx gets at least one sample of every class with ≥2.
            k_out = min(k_out, len(idxs) - 1)
            out_idx.extend(idxs[:k_out])
            in_idx.extend(idxs[k_out:])
        self.pre_split = {"in": sorted(in_idx), "out": sorted(out_idx)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = int(self.indices[idx])
        y = self.dataset.y_array[row]
        if self._shared_cache is not None:
            cache_i = int(self._cache_idx_for_local[idx])
            x = self._shared_cache[cache_i].float() / 255.0
            x = (x - self._IMAGENET_MEAN) / self._IMAGENET_STD
            return x, y
        if self._cache is not None:
            x = self._cache[idx].float() / 255.0
            x = (x - self._IMAGENET_MEAN) / self._IMAGENET_STD
            return x, y
        x = self.dataset.get_input(row)
        if not isinstance(x, Image.Image):
            x = Image.fromarray(x)
        return self.transform(x), y

    def get_labels(self):
        return self.dataset.y_array[self.indices]


class WILDSIWildCam(MultipleDomainDataset):
    """iWildCam (WILDS v2.0) — 5-env official-split layout.

    Envs:
        env_0 = train pool   (~129,809 imgs from 243 source camera locations)
        env_1 = val_ood       (32 unseen locations,  14,961 imgs)
        env_2 = test_ood      (48 unseen locations,  42,791 imgs)
        env_3 = id_val        (held-out from train locations,  7,314 imgs)
        env_4 = id_test       (held-out from train locations,  8,154 imgs)

    Default usage: --test_envs 1 2 3 4 (env_0 is the only training env).

    Multi-domain training for invariance losses (e.g. GMOE_InvMMD): env_0
    exposes `location_per_idx_in` so train.py can build a
    LocationGroupedBatchSampler that emits batches grouped by source camera
    location. `GROUP_SAMPLER_K` random locations per batch × `GROUP_SAMPLER_K_BATCH`
    samples each gives algorithms a multi-domain view despite there being
    only one source env.
    """
    IMG_SIZE = 224
    INPUT_SHAPE = (3, IMG_SIZE, IMG_SIZE)
    ENVIRONMENTS = ["train", "val_ood", "test_ood", "id_val", "id_test"]
    N_STEPS = 7501
    CHECKPOINT_FREQ = 500
    # Single train env + 4 eval envs = 5 InfiniteDataLoader; safe to fork.
    N_WORKERS = 0
    # Eval has 5 envs × 2 splits = 10 loaders; safe to fork.
    EVAL_N_WORKERS = 4
    # When set, train.py uses LocationGroupedBatchSampler on env_0 to make
    # each batch contain K random source locations × K_BATCH samples each.
    # K=4 → 6 invariance pairs per batch (sufficient L_inv signal).
    GROUP_SAMPLER_K = 4
    GROUP_SAMPLER_K_BATCH = 8  # samples per location per batch (total bs=K*K_BATCH=32)

    def __init__(self, root, test_envs, hparams):
        from wilds.datasets.iwildcam_dataset import IWildCamDataset
        super().__init__()
        dataset = IWildCamDataset(root_dir=root)
        img_size = _resolve_image_size_from_hparams(hparams, self.IMG_SIZE)

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        augment_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        split_array = np.asarray(dataset._split_array)
        # location_remapped is column 0 of metadata_array (see WILDS source).
        loc_array = dataset.metadata_array[:, 0].numpy()

        # Build env_0 = train pool (single source env), with location metadata
        # attached so a LocationGroupedBatchSampler can sample by group.
        train_row_ids = np.where(split_array == 0)[0]
        use_aug = bool(hparams.get('data_augmentation'))
        train_env = _WILDSIWildCamSplitEnv(
            dataset, train_row_ids,
            augment_transform if use_aug else transform)
        # location_per_idx_in: location_remapped for each pre_split['in'] index.
        # train.py uses this (after _SplitDataset.keys remapping) to feed the
        # grouped batch sampler.
        train_in_local = np.asarray(train_env.pre_split['in'], dtype=np.int64)
        train_env.location_per_idx_in = loc_array[train_row_ids[train_in_local]].astype(np.int64)

        self.datasets = [train_env]

        # 4 evaluation envs (always no-aug; cache in RAM so eval skips PIL).
        eval_split_to_name = [(1, "val_ood"), (2, "test_ood"),
                              (3, "id_val"), (4, "id_test")]
        cache_eval = bool(hparams.get('cache_eval_in_ram', True))
        for split_id, name in eval_split_to_name:
            row_idx = np.where(split_array == split_id)[0]
            print(f"[WILDSIWildCam] caching eval env {name} "
                  f"({len(row_idx)} imgs) in RAM..." if cache_eval else
                  f"[WILDSIWildCam] eval env {name}: {len(row_idx)} imgs (no cache)",
                  flush=True)
            self.datasets.append(
                _WILDSIWildCamSplitEnv(
                    dataset, row_idx, transform, cache_in_ram=cache_eval,
                    cache_target_size=img_size))

        self.input_shape = (3, img_size, img_size)
        self.num_classes = dataset.n_classes


class WILDSIWildCamERM(WILDSIWildCam):
    """Plain-ERM iWildCam wrapper using the official WILDS 5-way split.

    This keeps the same env layout and data transforms as WILDSIWildCam, but
    disables the location-grouped sampler so ERM trains with an ordinary
    random loader over the train pool.
    """
    GROUP_SAMPLER_K = None
    GROUP_SAMPLER_K_BATCH = None


class _MetaShiftCSVEnv(torch.utils.data.Dataset):
    """One environment of MetaShift_K, backed by a CSV slice. Resolves images
    via vg_image_id under <data_dir>/metashift/raw/images/<id>.jpg.

    Exposes a `pre_split` attribute with class-stratified in/out indices so
    that misc.split_dataset honors the manifest split rather than re-doing a
    uniform random shuffle. For test envs the manifest only has 'test' rows;
    those are emitted as 'out' (validation-position) so train.py evaluates them
    via the env*_out_acc metric, with 'in' empty.
    """

    def __init__(self, df_slice, image_root, transform):
        df_slice = df_slice.reset_index(drop=True)
        self.image_paths = [os.path.join(image_root, f"{int(v)}.jpg") for v in df_slice.vg_image_id.tolist()]
        self.labels = df_slice.class_idx.astype(int).tolist()
        self.transform = transform
        splits = df_slice.split.tolist()
        in_idx = [i for i, s in enumerate(splits) if s == "in"]
        out_idx = [i for i, s in enumerate(splits) if s == "out"]
        test_idx = [i for i, s in enumerate(splits) if s in ("test", "test_near", "test_far")]
        if in_idx or out_idx:
            self.pre_split = {"in": in_idx, "out": out_idx}
        else:
            # Pure test env: stratify the test rows 80/20 by class so train.py's
            # env_in_acc and env_out_acc are both well-defined (otherwise an
            # empty in-slot would crash evaluation).
            from collections import defaultdict
            by_cls: dict[int, list[int]] = defaultdict(list)
            for i in test_idx:
                by_cls[self.labels[i]].append(i)
            in_t, out_t = [], []
            for cls, idxs in by_cls.items():
                k_out = max(1, int(round(0.2 * len(idxs))))
                out_t.extend(idxs[:k_out])
                in_t.extend(idxs[k_out:])
            self.pre_split = {"in": sorted(in_t), "out": sorted(out_t)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


class MetaShift_K(MultipleDomainDataset):
    """K-domain MetaShift via shared-context split.

    Reads a CSV manifest produced by scripts/metashift_build_splits.py with columns:
        image_path, class_idx, class_name, context, domain_idx, domain_name, split, vg_image_id

    The first K environments are training contexts; the last 1 (single_test mode)
    or 2 (test_near + test_far) are held-out contexts. test_envs should be the
    indices of those held-out envs.

    hparams keys:
        class_set       (str)  e.g. "cat_dog_horse_elephant_bird"
        K               (int)
        split_seed      (int)
        total_per_class (int, default 240)  -- only used to locate the CSV file
        single_test     (bool, default True) -- match the manifest naming
    """
    CHECKPOINT_FREQ = 300
    N_STEPS = 5001
    N_WORKERS = 2  # smaller than the default 8: VG images are large, K envs * 8 workers OOMs at K>=4
    ENVIRONMENTS = []  # populated in __init__

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        import pandas as pd  # lazy: pandas may not be needed for other datasets
        class_set = hparams['class_set']
        K = int(hparams['K'])
        split_seed = int(hparams['split_seed'])
        total = int(hparams.get('total_per_class', 240))
        single_test = bool(hparams.get('single_test', True))
        if single_test:
            csv_name = f"K{K}_N{total}_seed{split_seed}.csv"
        else:
            proto = hparams.get('protocol', 'B')
            csv_name = f"K{K}_proto{proto}_seed{split_seed}.csv"
        manifest = os.path.join(root, "metashift", "splits", class_set, csv_name)
        image_root = os.path.join(root, "metashift", "raw", "images")
        if not os.path.exists(manifest):
            raise FileNotFoundError(f"MetaShift_K manifest not found: {manifest}. "
                                    f"Run scripts/metashift_build_splits.py first.")
        df = pd.read_csv(manifest)

        # Augment training, but not test.
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        plain = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
        augment = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(), norm,
        ])

        # Order envs: training (by domain_idx) then test (test or test_near/test_far).
        train_domains = sorted(df.query("split in ['in','out']").domain_idx.unique().tolist())
        test_split_names = ["test"] if single_test else ["test_near", "test_far"]
        test_domains = sorted(df.query("split in @test_split_names").domain_idx.unique().tolist())
        env_order = train_domains + test_domains

        self.ENVIRONMENTS = []
        self.datasets = []
        do_augment = bool(hparams.get('data_augmentation', True))
        for i, d_idx in enumerate(env_order):
            env_df = df.query("domain_idx == @d_idx")
            ctx_name = env_df.domain_name.iloc[0] if len(env_df) else f"env_{d_idx}"
            self.ENVIRONMENTS.append(ctx_name)
            tfm = augment if (do_augment and i not in test_envs) else plain
            self.datasets.append(_MetaShiftCSVEnv(env_df, image_root, tfm))

        self.input_shape = (3, 224, 224)
        self.num_classes = int(df.class_idx.max()) + 1
