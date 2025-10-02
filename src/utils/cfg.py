from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Mapping


class DatasetEnum(Enum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    FASHION_MNIST = "fashion_mnist"


# Central registry of dataset specs (channels-first)
DATASET_SPECS: Dict[DatasetEnum, Dict[str, Any]] = {
    DatasetEnum.MNIST:   {"input_shape": (1, 28, 28), "num_classes": 10},
    DatasetEnum.CIFAR10: {"input_shape": (3, 32, 32), "num_classes": 10},
    DatasetEnum.CIFAR100: {"input_shape": (3, 32, 32), "num_classes": 100},
    DatasetEnum.FASHION_MNIST: {"input_shape": (1, 28, 28), "num_classes": 10},
}


@dataclass
class Config:
    # General global settings
    seed: int = 0

    # Data and dataloaders
    num_workers: int = 4  # Increase for better parallel data loading
    dataset: DatasetEnum = DatasetEnum.MNIST
    data_root: str = "./input_data"

    # Model
    hidden_sizes: tuple[int, ...] = (512, 256)

    # Training
    batch_size: int = 256  # Increase batch size for better GPU utilization
    epochs: int = 3
    lr: float = 1e-3
    wd: float = 1e-4

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """(C, H, W) for the selected dataset."""
        return DATASET_SPECS[self.dataset]["input_shape"]

    @property
    def input_dim(self) -> int:
        """Flattened input dimension = C * H * W."""
        c, h, w = self.input_shape
        return c * h * w

    @property
    def num_classes(self) -> int:
        """Number of classes for the selected dataset."""
        return DATASET_SPECS[self.dataset]["num_classes"]


def update_cfg(overrides: Mapping[str, str] | None) -> None:
    """Mutate the global CFG with key=value strings from .env."""
    if not overrides:
        return

    def _cast(cur, val: str):
        if isinstance(cur, bool):
            return val.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(cur, int):
            return int(val)
        if isinstance(cur, float):
            return float(val)
        if isinstance(cur, tuple):
            # assume comma-separated ints
            return tuple(int(x) for x in val.split(",") if x.strip() != "")
        if isinstance(cur, DatasetEnum):
            return DatasetEnum(val.strip().lower())

        raise TypeError(f'Unsupported type: {type(cur)}')

    for k, v in overrides.items():
        name = k.lower()
        if hasattr(CFG, name):
            cur = getattr(CFG, name)
            setattr(CFG, name, _cast(cur, v))

CFG = Config()
