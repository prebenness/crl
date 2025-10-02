# cfg.py
from dataclasses import dataclass

@dataclass
class Config:
    batch_size: int = 128
    epochs: int = 5
    lr: float = 1e-3
    wd: float = 1e-4
    seed: int = 0
    num_workers: int = 2
    hidden_sizes: tuple[int, ...] = (512, 256)

CFG = Config()
