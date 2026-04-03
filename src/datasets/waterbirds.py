"""Waterbirds spurious correlation benchmark.

Binary classification: waterbird (1) vs landbird (0).
Spurious attribute: water vs land background.
Training set: 95% of waterbirds on water, 95% of landbirds on land.
Val/test: group-balanced.

Dataset from Sagawa et al., 2020, "Distributionally Robust Neural Network
Training", using composited images from CUB-200-2011 and Places.

Expects the Waterbirds dataset directory at:
    {root}/waterbirds/waterbird_complete95_forest2water2/

Download from: https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
"""

import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Waterbirds(Dataset):
    """Waterbirds spurious correlation dataset.

    Args:
        root: data root directory.
        split: "train", "val", or "test".
        image_size: resize images to this size (square). Default 256 for
            subsequent 224 crop in augmentation pipeline.
    """

    DATASET_DIR = "waterbirds/waterbird_complete95_forest2water2"

    def __init__(self, root="./data", split="train", image_size=256):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        base_dir = os.path.join(root, self.DATASET_DIR)
        metadata_path = os.path.join(base_dir, "metadata.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Waterbirds metadata not found at {metadata_path}. "
                "Download from: https://nlp.stanford.edu/data/dro/"
                "waterbird_complete95_forest2water2.tar.gz\n"
                f"Extract to: {os.path.join(root, 'waterbirds/')}"
            )

        # Parse metadata.csv
        # Columns: img_id, img_filename, y, split, place, place_filename
        # split: 0=train, 1=val, 2=test
        split_map = {"train": 0, "val": 1, "test": 2}
        split_id = split_map[split]

        lines = open(metadata_path).readlines()
        header = lines[0].strip().split(",")
        col_idx = {name: i for i, name in enumerate(header)}

        filenames, labels, places = [], [], []
        for line in lines[1:]:
            parts = line.strip().split(",")
            if int(parts[col_idx["split"]]) != split_id:
                continue
            filenames.append(parts[col_idx["img_filename"]])
            labels.append(int(parts[col_idx["y"]]))
            places.append(int(parts[col_idx["place"]]))

        self.base_dir = base_dir
        self.filenames = filenames
        self.y = np.array(labels, dtype=np.int32)
        self.places = np.array(places, dtype=np.int32)
        # Groups: (bird_type * 2 + background_type) -> 4 groups
        self.group = (self.y * 2 + self.places).astype(np.int32)
        self.image_size = image_size
        self.y_bin = self.y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_dir, self.filenames[idx])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        x = np.array(img, dtype=np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        return x, self.y[idx]
