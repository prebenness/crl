"""CelebA spurious correlation benchmark.

Binary classification: Blond Hair (1) vs Not Blond Hair (0).
Spurious attribute: Male (natural correlation, not synthetic).
Four groups: (blond/not-blond) x (male/female).

Uses torchvision.datasets.CelebA with standard predefined splits.
Dataset auto-downloads via torchvision, or can be placed manually at:
    {root}/celeba/

Note: CelebA is large (~1.3GB images). This dataset is intended for use
with the streaming data loader rather than full materialization.
"""

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CelebA as TVCelebA


BLOND_HAIR_IDX = 9   # attr index for Blond_Hair
MALE_IDX = 20        # attr index for Male


class CelebA(Dataset):
    """CelebA hair color classification dataset.

    Args:
        root: data root directory.
        split: "train", "val", or "test".
        image_size: resize images to this size (square). Default 256.
    """

    def __init__(self, root="./data", split="train", image_size=256):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        tv_split = "valid" if split == "val" else split
        self.base = TVCelebA(
            root=root,
            split=tv_split,
            target_type="attr",
            download=False,  # large download, user should do manually
        )
        self.image_size = image_size

        # Extract labels and groups from attributes
        attrs = self.base.attr.numpy()  # (N, 40), values {-1, 1}
        blond = ((attrs[:, BLOND_HAIR_IDX] + 1) // 2).astype(np.int32)  # {0, 1}
        male = ((attrs[:, MALE_IDX] + 1) // 2).astype(np.int32)  # {0, 1}

        self.y = blond
        self.male = male
        # Groups: (blond * 2 + male) -> 4 groups
        # 0: not-blond female, 1: not-blond male, 2: blond female, 3: blond male
        self.group = (blond * 2 + male).astype(np.int32)
        self.y_bin = self.y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img, _ = self.base[idx]  # PIL Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        x = np.array(img, dtype=np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        return x, self.y[idx]
