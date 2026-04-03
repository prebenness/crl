"""Experiment configuration: YAML loading + typed ExperimentConfig.

CLI override syntax (applied after YAML load):
    python script.py config.yaml training.epochs=20 hsic.weight=0.3
"""

import sys
from dataclasses import dataclass, field, fields

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
    beta: float = 0.0  # bias-conflicting ratio; when > 0, p_train = 1 - beta


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
    outer_loss: str = "hsic"  # "hsic" or "mmd"
    num_classes: int = 10
    bottleneck_width: int = 16
    outer_rep_dim: int = 100
    oracle_checkpoint: str = ""


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
class MMDConfig:
    weight: float = 1.0          # lambda for MMD penalty
    w_max: float = 500.0         # importance weight clip
    smoothing_eps: float = 1e-6  # Laplace smoothing for p(s|y)


@dataclass
class CheckpointingConfig:
    early_stopping_patience: int = 0    # 0 = disabled
    restart_patience: int = 0           # 0 = disabled
    resume_from: str = ""               # path to run_dir to resume from
    ckpt_select: str = "auto"           # "auto", "best", or "final"


@dataclass
class CBAOMConfig:
    embed_dim: int = 16
    num_colors: int = 10


@dataclass
class IPSNConfig:
    c_dim: int = 50
    b_dim: int = 16
    num_colors: int = 10
    embed_dim: int = 16
    decoder_hidden: int = 64
    grad_rev_scale: float = 1.0
    lambda_color: float = 1.0
    gamma_adv: float = 0.5
    rho_recon: float = 0.1
    nu_cycle: float = 0.1


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
    mmd: MMDConfig = field(default_factory=MMDConfig)
    mdl: MDLConfig = field(default_factory=MDLConfig)
    cba_om: CBAOMConfig = field(default_factory=CBAOMConfig)
    ipsn: IPSNConfig = field(default_factory=IPSNConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
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
        "mmd": MMDConfig,
        "mdl": MDLConfig,
        "cba_om": CBAOMConfig,
        "ipsn": IPSNConfig,
        "checkpointing": CheckpointingConfig,
        "sweep": SweepConfig,
    }

    for section_name in section_map:
        if section_name in raw:
            section_obj = getattr(cfg, section_name)
            for k, v in raw[section_name].items():
                if not hasattr(section_obj, k):
                    print(f"Warning: unknown config key "
                          f"'{section_name}.{k}', ignoring.",
                          file=sys.stderr)
                    continue
                setattr(section_obj, k, v)

    ck = cfg.checkpointing
    if ck.early_stopping_patience > 0 and ck.restart_patience > 0:
        raise ValueError(
            "early_stopping_patience and restart_patience are mutually "
            "exclusive — set at most one to a positive value"
        )

    return cfg


def _cast_value(value_str: str, target_type: type):
    """Cast a CLI string to the type of the target field."""
    if target_type is bool:
        if value_str.lower() in ("true", "1", "yes"):
            return True
        if value_str.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"Cannot parse {value_str!r} as bool")
    return target_type(value_str)


def apply_overrides(cfg: ExperimentConfig, overrides: list[str]):
    """Apply dotted key=value overrides to a loaded config.

    Each override must be ``section.key=value``, e.g.
    ``training.epochs=20`` or ``hsic.weight=0.3``.

    Type casting is inferred from the dataclass field's default type.
    Raises on unknown sections or keys so typos fail fast.
    """
    for token in overrides:
        if "=" not in token:
            raise ValueError(
                f"Override {token!r} is not in section.key=value format"
            )
        path, value_str = token.split("=", 1)
        parts = path.split(".")
        if len(parts) != 2:
            raise ValueError(
                f"Override path {path!r} must be section.key "
                f"(got {len(parts)} parts)"
            )
        section_name, key = parts

        if not hasattr(cfg, section_name):
            raise ValueError(f"Unknown config section {section_name!r}")
        section_obj = getattr(cfg, section_name)

        if not hasattr(section_obj, key):
            raise ValueError(
                f"Unknown key {key!r} in section {section_name!r}. "
                f"Available: {[f.name for f in fields(section_obj)]}"
            )

        # Infer target type from the dataclass field
        field_type = type(getattr(section_obj, key))
        try:
            setattr(section_obj, key, _cast_value(value_str, field_type))
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot set {path}={value_str!r} "
                f"(expected {field_type.__name__}): {e}"
            ) from None

    return cfg
