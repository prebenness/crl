"""Experiment configuration: YAML loading + typed ExperimentConfig."""

import sys
from dataclasses import dataclass, field

import yaml
import jax.numpy as jnp


@dataclass
class WandbConfig:
    entity: str = "prebenness-crl"
    project: str = "colored-mnist-vib"


@dataclass
class DatasetConfig:
    name: str = "colored_mnist"
    p_train: float = 0.9
    p_test: float = 0.1


@dataclass
class DataLoaderConfig:
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2


@dataclass
class ModelConfig:
    mode: str = "pair"
    inner: str = "ula_mlp_var"
    outer: str = "ula_mlp"
    num_classes: int = 10
    bottleneck_width: int = 16
    outer_rep_dim: int = 100


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    weight_decay_inner: float = 0.0
    weight_decay_outer: float = 0.0
    epochs: int = 50
    batch_size: int = 128
    seed: int = 0
    alpha: float = 0.01


@dataclass
class ControllerConfig:
    beta_min: float = 0.0
    beta_max: float = 1.0
    ctrl_ki: float = 1.0


@dataclass
class MCSamplesConfig:
    train: int = 2
    eval: int = 8


@dataclass
class HSICConfig:
    weight: float = 0.50


@dataclass
class MDLConfig:
    n_max: int = 5
    m_max: int = 5
    tau_start: float = 2.0
    tau_end: float = 0.1
    n_samples: int = 1
    shared_lambda2: float = 100.0
    shared_epsilon: float = 1e-6
    mode_forward: bool = False
    init_cl_scale: float = 0.0


@dataclass
class SweepConfig:
    lambda_min_exp: float = -3.0
    lambda_max_exp: float = 3.0
    lambda_steps: int = 10
    log_sweep: bool = True


@dataclass
class ExperimentConfig:
    wandb: WandbConfig = field(default_factory=WandbConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    mc_samples: MCSamplesConfig = field(default_factory=MCSamplesConfig)
    hsic: HSICConfig = field(default_factory=HSICConfig)
    mdl: MDLConfig = field(default_factory=MDLConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)

    @property
    def lambdas(self) -> jnp.ndarray:
        args = (self.sweep.lambda_min_exp, self.sweep.lambda_max_exp,
                self.sweep.lambda_steps)
        if self.sweep.log_sweep:
            return jnp.logspace(*args)
        return jnp.linspace(*args)


def load_config(yaml_path: str) -> ExperimentConfig:
    """Load YAML file and return a populated ExperimentConfig."""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    cfg = ExperimentConfig()

    section_map = {
        "wandb": WandbConfig,
        "dataset": DatasetConfig,
        "dataloader": DataLoaderConfig,
        "model": ModelConfig,
        "training": TrainingConfig,
        "controller": ControllerConfig,
        "mc_samples": MCSamplesConfig,
        "hsic": HSICConfig,
        "mdl": MDLConfig,
        "sweep": SweepConfig,
    }

    for section_name, section_cls in section_map.items():
        if section_name in raw:
            section_obj = getattr(cfg, section_name)
            for k, v in raw[section_name].items():
                if not hasattr(section_obj, k):
                    print(f"Warning: unknown config key "
                          f"'{section_name}.{k}', ignoring.",
                          file=sys.stderr)
                    continue
                setattr(section_obj, k, v)

    return cfg
