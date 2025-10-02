from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Tuple


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

CFG = Config()
