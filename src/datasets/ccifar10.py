"""Corrupted CIFAR-10 (cCIFAR10) spurious correlation benchmark.

Each of 10 CIFAR-10 classes is paired with a specific corruption type.
Class-corruption mapping from LfF (Nam et al., 2020, "Learning from Failure"):
    Airplane->Snow, Automobile->Frost, Bird->Fog, Cat->Brightness, Deer->Contrast,
    Dog->Spatter, Frog->Elastic, Horse->JPEG, Ship->Pixelate, Truck->Saturate

Corruption severity: 5 (maximum), matching the DFA pre-built dataset
(Lee et al., 2021, "Learning Debiased Representation via Disentangled
Feature Augmentation").

Requires the ``imagecorruptions`` package for generating corrupted images:
    pip install imagecorruptions
Corrupted images are cached to disk after first generation.
"""

import os
import warnings

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

# Class -> corruption mapping (Type 0 from LfF code)
CORRUPTION_NAMES = [
    "snow",               # 0: Airplane
    "frost",              # 1: Automobile
    "fog",                # 2: Bird
    "brightness",         # 3: Cat
    "contrast",           # 4: Deer
    "spatter",            # 5: Dog
    "elastic_transform",  # 6: Frog
    "jpeg_compression",   # 7: Horse
    "pixelate",           # 8: Ship
    "saturate",           # 9: Truck
]

SEVERITY = 5


def _generate_corruptions(images_uint8, cache_dir, corruption_names, severity):
    """Generate corrupted image arrays using imagecorruptions, cache to disk.

    Args:
        images_uint8: (N, 32, 32, 3) uint8 array of clean images.
        cache_dir: directory to cache .npy files.
        corruption_names: list of corruption type names.
        severity: corruption severity (1-5).

    Returns:
        dict mapping corruption_name -> (N, 32, 32, 3) uint8 array.
    """
    try:
        from imagecorruptions import corrupt
    except ImportError:
        raise ImportError(
            "cCIFAR10 requires the 'imagecorruptions' package to generate "
            "corrupted images. Install with: pip install imagecorruptions\n"
            "Alternatively, place pre-generated .npy files in: " + cache_dir
        )

    os.makedirs(cache_dir, exist_ok=True)
    result = {}
    n = len(images_uint8)

    for cname in corruption_names:
        npy_path = os.path.join(cache_dir, f"{cname}.npy")
        if os.path.exists(npy_path):
            result[cname] = np.load(npy_path)
            continue

        print(f"  Generating {cname} corruptions (n={n}, severity={severity})...")
        corrupted = np.empty_like(images_uint8)
        for i in range(n):
            corrupted[i] = corrupt(images_uint8[i], corruption_name=cname,
                                   severity=severity)
        np.save(npy_path, corrupted)
        result[cname] = corrupted

    return result


class CCIFAR10(Dataset):
    """Corrupted CIFAR-10 spurious correlation benchmark.

    Args:
        root: data root directory.
        train: if True load training set, else test set.
        p_corr: P(corruption = paired(class)). beta = 1 - p_corr.
        seed: random seed for bias assignment.
        split: "train", "val", or "test". Overrides train if provided.
            Val is a stratified ~10% split from the biased training set.
    """

    VAL_FRAC = 0.1

    def __init__(self, root="./data", train=True, p_corr=0.9, seed=0, split=None):
        super().__init__()

        if split is not None:
            if split not in ("train", "val", "test"):
                raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
            use_cifar_train = split in ("train", "val")
        else:
            use_cifar_train = bool(train)
            split = "train" if use_cifar_train else "test"

        self.rng = np.random.RandomState(seed)
        self.split = split

        # Load CIFAR-10 base
        base = CIFAR10(root=root, train=use_cifar_train, download=True)
        images_uint8 = base.data  # (N, 32, 32, 3) uint8
        labels = np.array(base.targets, dtype=np.int32)

        # Generate/load corruption arrays
        split_name = "train" if use_cifar_train else "test"
        cache_dir = os.path.join(root, "ccifar10", split_name)
        corruption_arrays = _generate_corruptions(
            images_uint8, cache_dir, CORRUPTION_NAMES, SEVERITY,
        )

        n = len(labels)
        y = labels

        if split in ("train", "val"):
            # Biased: P(corruption=paired(y)) = p_corr, else uniform over other 9
            aligned = self.rng.rand(n) < float(p_corr)
            paired = y  # class k is paired with corruption k
            shift = 1 + self.rng.randint(0, 9, size=n).astype(np.int32)
            other = (paired + shift) % 10
            corruption_idx = np.where(aligned, paired, other).astype(np.int32)
        else:
            # Unbiased test: uniform random corruption
            corruption_idx = self.rng.randint(0, 10, size=n).astype(np.int32)

        # Apply assigned corruptions
        corrupted_images = np.empty_like(images_uint8, dtype=np.float32)
        for i in range(n):
            cname = CORRUPTION_NAMES[corruption_idx[i]]
            corrupted_images[i] = corruption_arrays[cname][i].astype(np.float32) / 255.0

        group = (y * 10 + corruption_idx).astype(np.int32)

        # Stratified train/val split
        if split in ("train", "val"):
            split_rng = np.random.RandomState(seed + 7777)
            val_mask = np.zeros(n, dtype=bool)
            for g in np.unique(group):
                g_idx = np.where(group == g)[0]
                n_val = max(1, int(len(g_idx) * self.VAL_FRAC))
                perm = split_rng.permutation(len(g_idx))
                val_mask[g_idx[perm[:n_val]]] = True

            keep = val_mask if split == "val" else ~val_mask
            corrupted_images = corrupted_images[keep]
            y = y[keep]
            corruption_idx = corruption_idx[keep]
            group = group[keep]

        # Store as CHW float32 for consistency with other datasets
        self.images = np.transpose(corrupted_images, (0, 3, 1, 2))  # NHWC -> NCHW
        self.y = y
        self.colors = corruption_idx  # alias for compatibility
        self.group = group
        self.y_bin = self.y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.images[idx], self.y[idx]
