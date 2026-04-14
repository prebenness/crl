#!/usr/bin/env python3
"""VICReg-OT: Counterfactual Optimal Transport with VICReg Collapse Prevention.

Decoder-free framework for learning unbiased classifiers under extreme spurious
correlations. A single online encoder processes both anchor and bias-conflicting
samples; VICReg regularization (variance + covariance) replaces the EMA target
encoder for collapse prevention. An interventional predictor with FiLM
conditioning performs counterfactual latent translation, aligned via class-
conditional Sinkhorn OT. The classifier trains on a 50/50 mix of hallucinated
(debiased) and raw anchor representations.

References:
  - Bardes, Ponce & LeCun 2022, "VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning", ICLR (arXiv:2105.04906)
  - Feydy et al. 2019, "Interpolating between Optimal Transport and MMD
    using Sinkhorn Divergences", AISTATS (arXiv:1810.08278)

Usage:
    python jepa_ot.py config/colored_mnist/jepa_ot.yaml [overrides...]

Example:
    python jepa_ot.py config/colored_mnist/jepa_ot.yaml jepa_ot.lambda_inv=0.5
"""

import argparse
import json
import os
os.environ["WANDB_SILENT"] = "true"
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random as jrandom
import flax.linen as nn
import optax
import wandb

from src.config import load_config, apply_overrides
from src.datasets.datasets import (
    build_dataset, dataset_to_jax_arrays, make_epoch_batches, make_eval_batches,
)
from src.datasets.augmentations import ccifar10_train_augment, ccifar10_eval_transform
from src.models.resnet import ResNet
from src.utils.checkpointing import (
    save_config, save_checkpoint, save_checkpoint_meta,
    checkpoint_path, make_experiment_dir, utc_timestamp,
)


# ============================================================
# Config
# ============================================================

@dataclass
class JEPAOTConfig:
    encoder: str = "mlp"
    num_colors: int = 10
    z_dim: int = 32
    predictor_hidden_dim: int = 128
    lambda_inv: float = 0.7
    lambda_var: float = 0.0
    lambda_cov: float = 0.0
    vicreg_gamma: float = 1.0
    sinkhorn_eps: float = 1.0
    sinkhorn_iters: int = 10
    epochs: int = 200


def parse_jepa_ot_config(yaml_path, overrides):
    """Parse JEPA-OT-specific config from YAML + CLI overrides."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = JEPAOTConfig()
    if "jepa_ot" in raw:
        for k, v in raw["jepa_ot"].items():
            if hasattr(cfg, k):
                field_type = type(getattr(cfg, k))
                if field_type is bool:
                    val = (v if isinstance(v, bool)
                           else str(v).lower() in ("true", "1", "yes"))
                    setattr(cfg, k, val)
                else:
                    setattr(cfg, k, field_type(v))

    for token in overrides:
        if "=" not in token:
            continue
        path, value_str = token.split("=", 1)
        parts = path.split(".")
        if len(parts) == 2 and parts[0] == "jepa_ot":
            key = parts[1]
            if hasattr(cfg, key):
                field_type = type(getattr(cfg, key))
                if field_type is bool:
                    setattr(cfg, key,
                            value_str.lower() in ("true", "1", "yes"))
                else:
                    setattr(cfg, key, field_type(value_str))

    return cfg


# ============================================================
# Models
# ============================================================

class Encoder(nn.Module):
    """Feature encoder: 2-layer MLP (matches uLA protocol trunk).

    Flatten -> Dense(100) -> ReLU -> Dense(z_dim) -> ReLU -> z
    """
    z_dim: int = 100

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        z = nn.Dense(self.z_dim)(x)
        z = nn.relu(z)
        return z


class ResNetEncoder(nn.Module):
    """ResNet-18 backbone returning 512-d GAP features.

    Wraps ResNet with cifar_mode stem (3x3, no maxpool) for 32x32 inputs.
    Contains BatchNorm — requires mutable='batch_stats' during training.
    """
    num_classes: int = 10
    cifar_mode: bool = True

    @nn.compact
    def __call__(self, x, train=True):
        _, aux = ResNet(
            num_classes=self.num_classes,
            stage_sizes=(2, 2, 2, 2),
            stage_channels=(64, 128, 256, 512),
            bottleneck=False,
            cifar_mode=self.cifar_mode,
        )(x, train=train)
        return aux["z"]


class FiLMConditioner(nn.Module):
    """FiLM conditioning network: s_onehot -> (gamma, beta).

    Produces per-dimension scale and shift vectors from the spurious
    attribute.  gamma gates which latent dimensions to suppress/preserve,
    beta injects target-attribute information.

    Ref: Perez et al. 2018, "FiLM: Visual Reasoning with a General
    Conditioning Layer" (arXiv:1709.07871).
    """
    z_dim: int = 100
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, s_onehot):
        h = nn.Dense(self.hidden_dim)(s_onehot)
        h = nn.relu(h)
        gamma = nn.Dense(self.z_dim)(h)  # scale
        beta = nn.Dense(self.z_dim)(h)   # shift
        # Initialize gamma near 1 (identity) so the modulation starts
        # close to pass-through and learns deviations from there.
        return 1.0 + gamma, beta


class InterventionalPredictor(nn.Module):
    """Residual FiLM predictor for counterfactual translation.

    Combines two extensions:
    1. FiLM (Feature-wise Linear Modulation): multiplicative gating via
       gamma(s) * z + beta(s).  gamma can zero out colour-correlated
       dimensions while beta injects target-colour information.
    2. Residual connection: z_hat = z + delta.  The MLP only learns the
       geometric delta needed to traverse the spurious manifold, not the
       identity mapping for semantic preservation.

    Architecture:
       film_z = gamma(s) * z + beta(s)         # FiLM modulation
       delta  = MLP([film_z, s])               # residual correction
       z_hat  = z + delta                      # skip connection
    """
    z_dim: int = 100
    num_colors: int = 10
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, z, s_onehot):
        # FiLM: dimension-wise modulation conditioned on target attribute
        gamma, beta = FiLMConditioner(
            z_dim=self.z_dim,
            hidden_dim=self.hidden_dim // 2,
        )(s_onehot)
        film_z = gamma * z + beta

        # Residual MLP: learns the delta from FiLM-modulated features
        h = jnp.concatenate([film_z, s_onehot], axis=-1)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)
        delta = nn.Dense(self.z_dim)(h)

        # Skip connection: preserve semantics, add learned correction
        return z + delta


class ClassifierHead(nn.Module):
    """Linear classifier on latent z."""
    num_classes: int = 10

    @nn.compact
    def __call__(self, z):
        return nn.Dense(self.num_classes)(z)


# ============================================================
# Sinkhorn Divergence (Feydy et al. 2019, arXiv:1810.08278)
# ============================================================

def _cost_matrix(x, y):
    """Squared Euclidean cost between point clouds x [n,d] and y [m,d]."""
    x2 = jnp.sum(x ** 2, axis=1, keepdims=True)
    y2 = jnp.sum(y ** 2, axis=1, keepdims=True)
    C = x2 + y2.T - 2.0 * (x @ y.T)
    return jnp.maximum(C, 0.0)


def _sinkhorn_cost(C, eps, n_iters):
    """Entropic OT cost for cost matrix C with uniform marginals.

    Log-domain Sinkhorn iterations for numerical stability.
    """
    n, m = C.shape
    log_a = jnp.full((n,), -jnp.log(jnp.float32(n)))
    log_b = jnp.full((m,), -jnp.log(jnp.float32(m)))
    log_K = -C / eps

    f = jnp.zeros(n)
    g = jnp.zeros(m)

    def step(carry, _):
        f, g = carry
        f = log_a - jax.nn.logsumexp(log_K + g[None, :], axis=1)
        g = log_b - jax.nn.logsumexp(log_K + f[:, None], axis=0)
        return (f, g), None

    (f, g), _ = lax.scan(step, (f, g), None, length=n_iters)

    log_P = f[:, None] + log_K + g[None, :]
    return jnp.sum(jnp.exp(log_P) * C)


def _sinkhorn_cost_weighted(C, log_a, log_b, eps, n_iters):
    """Entropic OT cost with custom log-marginals.

    Same as _sinkhorn_cost but accepts arbitrary (log) marginal distributions
    instead of assuming uniform.  Used by class_cond_sinkhorn_divergence to
    zero-out padding positions in fixed-size per-class buffers.
    """
    log_K = -C / eps
    f = jnp.zeros_like(log_a)
    g = jnp.zeros_like(log_b)

    def step(carry, _):
        f, g = carry
        f = log_a - jax.nn.logsumexp(log_K + g[None, :], axis=1)
        g = log_b - jax.nn.logsumexp(log_K + f[:, None], axis=0)
        return (f, g), None

    (f, g), _ = lax.scan(step, (f, g), None, length=n_iters)
    log_P = f[:, None] + log_K + g[None, :]
    return jnp.sum(jnp.exp(log_P) * C)


def _sinkhorn_debiased_weighted(C_xy, C_xx, C_yy, log_a, log_b, eps, n_iters):
    """Debiased Sinkhorn (OT_xy - 0.5*OT_xx - 0.5*OT_yy) with the three OT
    problems solved in parallel via vmap over a stacked leading axis.

    All three cost matrices must share shape [n, m].  Used by
    class_cond_sinkhorn_divergence to batch the three OT solves per class
    — exposes 3x parallelism to the GPU instead of sequencing them.
    """
    Cs = jnp.stack([C_xy, C_xx, C_yy], axis=0)  # [3, n, m]
    ots = jax.vmap(
        lambda C: _sinkhorn_cost_weighted(C, log_a, log_b, eps, n_iters),
    )(Cs)
    return ots[0] - 0.5 * ots[1] - 0.5 * ots[2]


def sinkhorn_divergence(x, y, eps=1.0, n_iters=50):
    """Debiased Sinkhorn divergence with L2-normalized point clouds.

    S_eps(x, y) = OT_eps(x, y) - 0.5 * OT_eps(x, x) - 0.5 * OT_eps(y, y)

    Both point clouds are projected onto the unit sphere before computing
    costs. This bounds ||z_i - z_j||^2 to [0, 4], making eps meaningful:
      - eps=1.0: typical kernel ~ exp(-2) ~ 0.14   (smooth, good gradients)
      - eps=0.5: typical kernel ~ exp(-4) ~ 0.018   (sharper matching)
      - eps=0.1: typical kernel ~ exp(-20) ~ 2e-9   (too sharp for 100-d)
    Without normalization, raw 100-d features give ||z_i-z_j||^2 ~ O(100),
    and eps=0.05 produces kernel entries ~ exp(-2000) ~ 0 (Genevay & Chizat
    2019, arXiv:1810.02733 — the epsilon scaling catastrophe).
    """
    x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    y = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)

    ot_xy = _sinkhorn_cost(_cost_matrix(x, y), eps, n_iters)
    ot_xx = _sinkhorn_cost(_cost_matrix(x, x), eps, n_iters)
    ot_yy = _sinkhorn_cost(_cost_matrix(y, y), eps, n_iters)
    return ot_xy - 0.5 * ot_xx - 0.5 * ot_yy


def class_cond_sinkhorn_divergence(z_pred, z_target, y, num_classes,
                                   eps=1.0, n_iters=10):
    """Class-conditional debiased Sinkhorn divergence.

    Computes per-class Sinkhorn divergences and averages over classes
    present in the batch.  This prevents the transport plan from matching
    samples across classes (e.g. digit 3 anchors to digit 7 targets),
    which produces incoherent gradients.  Standard approach for
    class-conditional distribution matching — Courty et al. 2017,
    "Optimal Transport for Domain Adaptation" (arXiv:1507.00504).

    Both point clouds are L2-normalized before computing costs (same
    epsilon-scaling rationale as sinkhorn_divergence).

    Uses fixed-size per-class buffers (jnp.nonzero with size=) for JIT
    compatibility.  Buffer size auto-scales with batch size (2x expected
    per-class count).  Padding positions receive negligible marginal mass
    (exp(-30) ~ 1e-13) and do not affect the transport plan.

    Args:
        z_pred: [B, d] predicted point cloud (predictor output).
        z_target: [B, d] target point cloud (EMA encoder output).
        y: [B] integer class labels (same for both clouds — BC samples
           are class-matched to anchors).
        num_classes: number of classes (Python int, unrolled by JIT).
        eps: entropic regularization.
        n_iters: Sinkhorn iterations per cost computation.
    """
    # Auto-scale buffer from batch size (static at JIT trace time)
    max_per_class = max(y.shape[0] // num_classes * 2, 32)
    z_pred = z_pred / (jnp.linalg.norm(z_pred, axis=-1, keepdims=True) + 1e-8)
    z_target = z_target / (jnp.linalg.norm(z_target, axis=-1, keepdims=True) + 1e-8)

    total_ot = 0.0
    total_weight = 0.0

    for c in range(num_classes):  # unrolled by JIT
        mask = (y == c)
        n_c = mask.sum()
        n_used = jnp.minimum(n_c, max_per_class)

        # Gather class-c indices into a fixed buffer; pads with index 0
        indices = jnp.nonzero(mask, size=max_per_class, fill_value=0)[0]

        z_p = z_pred[indices]    # [max_per_class, d]
        z_t = z_target[indices]  # [max_per_class, d]

        # Marginals: uniform 1/n_used over real samples, ~0 for padding
        valid = jnp.arange(max_per_class) < n_used
        log_w = jnp.where(
            valid,
            -jnp.log(jnp.maximum(n_used, 1).astype(jnp.float32)),
            -30.0,
        )

        # Debiased Sinkhorn on this class: vmap the three OT solves
        # (xy, xx, yy) to expose parallelism to the GPU.
        s_c = _sinkhorn_debiased_weighted(
            _cost_matrix(z_p, z_t),
            _cost_matrix(z_p, z_p),
            _cost_matrix(z_t, z_t),
            log_w, log_w, eps, n_iters,
        )

        # Skip empty classes (shouldn't happen with balanced MNIST)
        w_c = (n_c > 0).astype(jnp.float32)
        total_ot += s_c * w_c
        total_weight += w_c

    return total_ot / jnp.maximum(total_weight, 1.0)


# ============================================================
# VICReg Regularization (Bardes et al. 2022, arXiv:2105.04906)
# ============================================================

def vicreg_variance_loss(z, gamma=1.0, eps=1e-4):
    """VICReg variance hinge: penalizes per-dimension std below gamma.

    Prevents mode collapse by requiring each latent dimension to maintain
    at least gamma standard deviation across the batch.
    """
    std = jnp.sqrt(jnp.var(z, axis=0) + eps)
    return jnp.mean(jnp.maximum(0.0, gamma - std))


def vicreg_covariance_loss(z):
    """VICReg covariance: penalizes off-diagonal correlations.

    Decorrelates latent dimensions by driving the off-diagonal entries
    of the batch covariance matrix toward zero.
    """
    z_c = z - jnp.mean(z, axis=0)
    n = z.shape[0]
    cov = (z_c.T @ z_c) / (n - 1)
    D = cov.shape[0]
    off_diag = cov - jnp.diag(jnp.diag(cov))
    return jnp.sum(off_diag ** 2) / D


# ============================================================
# Bias-Conflicting Sample Pool
# ============================================================

class BiasConflictingPool:
    """Static pool of bias-conflicting samples, indexed by class.

    For CMNISTuLA: bias-conflicting = samples where color != digit
    (identity pairing).
    """

    def __init__(self, x, y, s, num_classes):
        y_np = np.asarray(y)
        s_np = np.asarray(s)

        self.per_class = {}
        n_bc_total = 0
        for c in range(num_classes):
            bc_mask = (y_np == c) & (s_np != c)
            indices = np.where(bc_mask)[0]
            if len(indices) == 0:
                indices = np.where(y_np == c)[0]
                print(f"  WARNING: class {c} has 0 bias-conflicting samples, "
                      f"using all {len(indices)} samples as fallback")
            self.per_class[c] = indices
            n_bc_total += len(indices)

        self.x = x
        self.s = s
        self.num_classes = num_classes
        print(f"  BC pool: {n_bc_total} total bias-conflicting samples")
        for c in range(num_classes):
            print(f"    class {c}: {len(self.per_class[c])} samples")

    def sample_epoch(self, yb, np_rng):
        """Pre-sample BC indices for an entire epoch of batches.

        Args:
            yb: [n_batches, batch_size] integer class labels.
            np_rng: numpy RandomState.

        Returns (x_c, s_c) with shape [n_batches, batch_size, ...].
        """
        n_batches, batch_size = yb.shape
        y_flat = np.asarray(yb).ravel()

        # Vectorized: draw a random float per sample, map to pool index
        rand_vals = np_rng.rand(len(y_flat))
        indices = np.empty(len(y_flat), dtype=np.int64)
        for c in range(self.num_classes):
            mask = y_flat == c
            pool_idx = self.per_class[c]
            slot = (rand_vals[mask] * len(pool_idx)).astype(np.int64)
            np.clip(slot, 0, len(pool_idx) - 1, out=slot)
            indices[mask] = pool_idx[slot]

        x_c = self.x[indices].reshape((n_batches, batch_size) + self.x.shape[1:])
        s_c = self.s[indices].reshape((n_batches, batch_size))
        return x_c, s_c


# ============================================================
# Evaluation
# ============================================================

def make_eval_fn(encoder, predictor, classifier, num_colors, num_classes,
                 eval_transform_fn=None):
    """Build JIT-compiled eval function with test-time marginalization.

    Args:
        eval_transform_fn: optional deterministic transform applied before
            the encoder (e.g. CIFAR-10 normalization).  Captured in the
            JIT closure — None is resolved at trace time.
    """

    @jax.jit
    def _predict(params, batch_stats, x_batch):
        if eval_transform_fn is not None:
            x_batch = eval_transform_fn(x_batch)
        z = encoder.apply(
            {"params": params["encoder"], "batch_stats": batch_stats},
            x_batch, train=False,
        )
        logits_sum = jnp.zeros((x_batch.shape[0], num_classes))
        for c in range(num_colors):
            s_oh = jax.nn.one_hot(
                jnp.full(x_batch.shape[0], c), num_colors,
            )
            z_hat = predictor.apply(
                {"params": params["predictor"]}, z, s_oh,
            )
            logits = classifier.apply(
                {"params": params["classifier"]}, z_hat,
            )
            logits_sum = logits_sum + logits
        return jnp.argmax(logits_sum, -1)

    def eval_accuracy(params, batch_stats, x, y, batch_size):
        xb, yb, counts = make_eval_batches(x, y, batch_size)
        total_correct = 0
        total = 0
        for i in range(xb.shape[0]):
            preds = _predict(params, batch_stats, xb[i])
            n = int(counts[i])
            total_correct += int((preds[:n] == yb[i][:n]).sum())
            total += n
        return total_correct / total

    return eval_accuracy


# ============================================================
# Main
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="JEPA-OT: Action-Conditioned JEPA with "
                    "Latent Counterfactuals via Sinkhorn Alignment",
    )
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides (section.key=value)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ---- Load configs ----
    cfg = load_config(args.config)
    std_overrides = [o for o in args.overrides if not o.startswith("jepa_ot.")]
    if std_overrides:
        apply_overrides(cfg, std_overrides)
    jot_cfg = parse_jepa_ot_config(args.config, args.overrides)

    print("JAX devices:", jax.devices())
    jax.config.update("jax_default_matmul_precision", "high")

    num_classes = cfg.model.num_classes
    num_colors = jot_cfg.num_colors
    z_dim = jot_cfg.z_dim
    batch_size = cfg.training.batch_size

    # ---- Run directory ----
    timestamp = utc_timestamp()
    run_name = f"{timestamp}_jepa_ot_{cfg.dataset.name}"
    run_dir = make_experiment_dir("jepa_ot", run_name)

    save_config(run_dir, {
        "experiment": "jepa_ot",
        "run_name": run_name,
        "config_path": str(Path(args.config).resolve()),
        "dataset": cfg.dataset.name,
        "seed": cfg.training.seed,
        "training": {
            "lr": cfg.training.lr,
            "weight_decay": cfg.training.weight_decay_inner,
            "batch_size": batch_size,
        },
        "jepa_ot": {k: getattr(jot_cfg, k) for k in vars(jot_cfg)},
    })

    # ---- W&B ----
    wandb_run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=run_name,
        config={
            "experiment": "jepa_ot",
            "dataset": cfg.dataset.name,
            "seed": cfg.training.seed,
            "training.lr": cfg.training.lr,
            "training.weight_decay": cfg.training.weight_decay_inner,
            "training.batch_size": batch_size,
            **{f"jepa_ot.{k}": getattr(jot_cfg, k) for k in vars(jot_cfg)},
        },
    )

    # ---- Load data ----
    print("\nLoading datasets...")
    t0 = time.time()
    train_ds = build_dataset(
        cfg.dataset.name, train=True,
        p_corr=cfg.dataset.p_train, seed=cfg.training.seed,
    )
    test_ds = build_dataset(
        cfg.dataset.name, train=False,
        p_corr=cfg.dataset.p_test, seed=cfg.training.seed + 1,
    )
    x_train, y_train = dataset_to_jax_arrays(
        train_ds, batch_size=batch_size,
    )
    x_test, y_test = dataset_to_jax_arrays(
        test_ds, batch_size=batch_size,
    )
    s_train = jnp.array(train_ds.colors, dtype=jnp.int32)

    print(f"  x_train {x_train.shape}, x_test {x_test.shape}")
    print(f"  Colour labels: {len(set(train_ds.colors.tolist()))} unique")
    print(f"  Data ready in {time.time()-t0:.2f}s")

    # ---- Build BC pool (ground-truth colours) ----
    print("\nBuilding bias-conflicting sample pool (oracle colours)...")
    pool = BiasConflictingPool(x_train, y_train, s_train, num_classes)

    # ---- Augmentation ----
    augment_fn = None
    eval_transform_fn = None
    if cfg.dataset.name == "ccifar10":
        augment_fn = ccifar10_train_augment
        eval_transform_fn = ccifar10_eval_transform

    # ---- Initialize models ----
    print("\nInitializing models...")
    if jot_cfg.encoder == "resnet18":
        encoder = ResNetEncoder(num_classes=num_classes, cifar_mode=True)
        z_dim = 512  # ResNet-18 GAP output
    else:
        encoder = Encoder(z_dim=z_dim)
    predictor = InterventionalPredictor(
        z_dim=z_dim, num_colors=num_colors,
        hidden_dim=jot_cfg.predictor_hidden_dim,
    )
    classifier = ClassifierHead(num_classes=num_classes)

    rng = jrandom.PRNGKey(cfg.training.seed)
    rng, rng_enc, rng_pred, rng_cls = jrandom.split(rng, 4)

    input_shape = (batch_size,) + x_train.shape[1:]
    enc_init = encoder.init(rng_enc, jnp.ones(input_shape), train=True)
    enc_params = enc_init["params"]
    batch_stats = enc_init.get("batch_stats", {})

    dummy_z = jnp.ones((batch_size, z_dim))
    dummy_s_oh = jnp.ones((batch_size, num_colors))
    pred_params = predictor.init(rng_pred, dummy_z, dummy_s_oh)["params"]
    cls_params = classifier.init(rng_cls, dummy_z)["params"]

    params = {
        "encoder": enc_params,
        "predictor": pred_params,
        "classifier": cls_params,
    }

    tx = optax.adamw(cfg.training.lr, cfg.training.weight_decay_inner)
    opt_state = tx.init(params)

    n_enc = sum(p.size for p in jax.tree.leaves(enc_params))
    n_pred = sum(p.size for p in jax.tree.leaves(pred_params))
    n_cls = sum(p.size for p in jax.tree.leaves(cls_params))
    print(f"  Encoder:  {jot_cfg.encoder}  z_dim={z_dim}")
    print(f"  Encoder params:    {n_enc:,}")
    print(f"  Predictor params:  {n_pred:,}")
    print(f"  Classifier params: {n_cls:,}")
    print(f"  Total trainable:   {n_enc + n_pred + n_cls:,}")

    # ---- Build JIT-compiled functions ----
    eps = jot_cfg.sinkhorn_eps
    n_iters = jot_cfg.sinkhorn_iters
    lambda_inv = jot_cfg.lambda_inv
    lambda_var = jot_cfg.lambda_var
    lambda_cov = jot_cfg.lambda_cov
    vicreg_gamma = jot_cfg.vicreg_gamma

    def train_step(params, batch_stats, opt_state, x_a, y_a, x_c, s_c, rng):
        rng, rng_s = jrandom.split(rng)
        s_rand = jrandom.randint(rng_s, y_a.shape, 0, num_colors)

        # Augmentation (no-op when augment_fn is None — resolved at trace time)
        if augment_fn is not None:
            rng, rng_aug_a, rng_aug_c = jrandom.split(rng, 3)
            x_a = augment_fn(rng_aug_a, x_a)
            x_c = augment_fn(rng_aug_c, x_c)

        def loss_fn(params):
            # Single shared encoder: concatenate anchors + BC samples for
            # correct BN statistics, then split.  Both z_a and z_c carry
            # gradients — no EMA target, no stop-gradient.
            x_both = jnp.concatenate([x_a, x_c], axis=0)
            z_both, new_enc_vars = encoder.apply(
                {"params": params["encoder"], "batch_stats": batch_stats},
                x_both, train=True, mutable=["batch_stats"],
            )
            new_bs = new_enc_vars.get("batch_stats", batch_stats)
            z_a, z_c = jnp.split(z_both, 2, axis=0)

            # -- Geometry: OT invariance + VICReg --

            # Batch the two predictor calls:
            # [z_hat_trans; z_hat_u] = predictor([z_a; z_a], [s_c; s_rand])
            # s_c drives OT alignment, s_rand drives the hallucinated CE.
            s_c_oh = jax.nn.one_hot(s_c, num_colors)
            s_rand_oh = jax.nn.one_hot(s_rand, num_colors)
            z_a_2x = jnp.concatenate([z_a, z_a], axis=0)
            s_both = jnp.concatenate([s_c_oh, s_rand_oh], axis=0)
            z_hat_both = predictor.apply(
                {"params": params["predictor"]}, z_a_2x, s_both,
            )
            z_hat_trans, z_hat_u = jnp.split(z_hat_both, 2, axis=0)

            # Sinkhorn alignment (class-conditional, L2-normalized inside)
            l_inv = class_cond_sinkhorn_divergence(
                z_hat_trans, z_c, y_a, num_classes, eps, n_iters,
            )

            # VICReg: prevent collapse and decorrelate dimensions
            l_var = (vicreg_variance_loss(z_hat_trans, gamma=vicreg_gamma)
                     + vicreg_variance_loss(z_c, gamma=vicreg_gamma))
            l_cov = (vicreg_covariance_loss(z_hat_trans)
                     + vicreg_covariance_loss(z_c))

            l_geom = lambda_inv * l_inv + lambda_var * l_var + lambda_cov * l_cov

            # -- Task: mixed CE on hallucinated + raw representations --

            # Hallucinated CE (debiased: random color)
            logits_u = classifier.apply(
                {"params": params["classifier"]}, z_hat_u,
            )
            l_ce_u = optax.softmax_cross_entropy_with_integer_labels(
                logits_u, y_a,
            ).mean()

            # Direct anchor CE (raw encoder output)
            logits_a = classifier.apply(
                {"params": params["classifier"]}, z_a,
            )
            l_ce_a = optax.softmax_cross_entropy_with_integer_labels(
                logits_a, y_a,
            ).mean()

            l_task = 0.5 * l_ce_u + 0.5 * l_ce_a

            total = l_task + l_geom
            acc = (jnp.argmax(logits_u, -1) == y_a).mean()
            return total, {
                "loss_ce_u": l_ce_u, "loss_ce_a": l_ce_a,
                "loss_task": l_task, "loss_inv": l_inv,
                "loss_var": l_var, "loss_cov": l_cov,
                "loss_geom": l_geom, "loss_total": total,
                "accuracy": acc, "new_batch_stats": new_bs,
            }

        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True,
        )(params)

        new_batch_stats = metrics.pop("new_batch_stats")

        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_batch_stats, new_opt_state, metrics

    @jax.jit
    def run_epoch(params, batch_stats, opt_state, xb, yb, xb_c, sb_c,
                  rng_keys):
        """Run one epoch of training inside a single jax.lax.scan.

        Collapses the Python batch loop into one JIT kernel: eliminates
        per-batch dispatch overhead and keeps all metrics on device until
        the end of the epoch (no host sync per batch).
        """
        def step_fn(carry, inputs):
            params, batch_stats, opt_state = carry
            x_a, y_a, x_c, s_c, rng_k = inputs
            new_params, new_bs, new_opt_state, metrics = train_step(
                params, batch_stats, opt_state,
                x_a, y_a, x_c, s_c, rng_k,
            )
            return (new_params, new_bs, new_opt_state), metrics

        (params, batch_stats, opt_state), metrics_stack = jax.lax.scan(
            step_fn, (params, batch_stats, opt_state),
            (xb, yb, xb_c, sb_c, rng_keys),
        )
        # Mean across the batch axis while still on device
        ep_means = {k: v.mean() for k, v in metrics_stack.items()}
        return params, batch_stats, opt_state, ep_means

    eval_accuracy = make_eval_fn(
        encoder, predictor, classifier, num_colors, num_classes,
        eval_transform_fn,
    )

    # ---- Training loop ----
    print("\n" + "=" * 60)
    print("Training VICReg-OT")
    print(f"  epochs={jot_cfg.epochs}  lambda_inv={lambda_inv}  "
          f"lambda_var={lambda_var}  lambda_cov={lambda_cov}  "
          f"sinkhorn_eps={eps}")
    print("=" * 60)

    best = {"test_acc": -1.0, "params": None,
            "batch_stats": None, "epoch": 0}
    np_rng = np.random.RandomState(cfg.training.seed + 42)

    for ep in range(jot_cfg.epochs):
        t0 = time.time()
        seed_ep = cfg.training.seed * 10_000 + ep
        xb, yb = make_epoch_batches(x_train, y_train, batch_size, seed_ep)

        # Pre-sample BC matches for the whole epoch (vectorized)
        xb_c, sb_c = pool.sample_epoch(yb, np_rng)

        # Pre-split RNG keys for every batch; run the whole epoch in one
        # JIT kernel (jax.lax.scan) — no per-batch Python dispatch, no
        # host sync for metrics.
        n_batches = xb.shape[0]
        rng, rng_epoch = jrandom.split(rng)
        rng_keys = jrandom.split(rng_epoch, n_batches)
        params, batch_stats, opt_state, ep_means = run_epoch(
            params, batch_stats, opt_state,
            xb, yb, xb_c, sb_c, rng_keys,
        )
        ep_metrics = {k: float(v) for k, v in ep_means.items()}

        te_acc = eval_accuracy(params, batch_stats, x_test, y_test, batch_size)

        if te_acc > best["test_acc"]:
            best["test_acc"] = te_acc
            best["params"] = jax.tree.map(jnp.copy, params)
            best["batch_stats"] = jax.tree.map(jnp.copy, batch_stats)
            best["epoch"] = ep + 1
            save_checkpoint(
                best["params"], checkpoint_path(run_dir, "best.npz"),
            )
            save_checkpoint_meta(
                run_dir, ep + 1, best["test_acc"], best["epoch"],
            )
            print(f"    >>> NEW BEST test_acc={te_acc:.4f} (epoch {ep+1})")

        wandb_run.log({
            "train/loss_ce_u": ep_metrics["loss_ce_u"],
            "train/loss_ce_a": ep_metrics["loss_ce_a"],
            "train/loss_task": ep_metrics["loss_task"],
            "train/loss_inv": ep_metrics["loss_inv"],
            "train/loss_var": ep_metrics["loss_var"],
            "train/loss_cov": ep_metrics["loss_cov"],
            "train/loss_geom": ep_metrics["loss_geom"],
            "train/loss_total": ep_metrics["loss_total"],
            "train/accuracy": ep_metrics["accuracy"],
            "test/accuracy": te_acc,
            "test/best_accuracy": best["test_acc"],
            "epoch": ep + 1,
        })

        print(f"  Epoch {ep+1}/{jot_cfg.epochs}"
              f"  task {ep_metrics['loss_task']:.4f}"
              f"  inv {ep_metrics['loss_inv']:.6f}"
              f"  var {ep_metrics['loss_var']:.4f}"
              f"  cov {ep_metrics['loss_cov']:.4f}"
              f"  train_acc {ep_metrics['accuracy']:.4f}"
              f"  test_acc {te_acc:.4f}"
              f"  ({time.time()-t0:.2f}s)")

    # ---- Save final state ----
    save_checkpoint(params, checkpoint_path(run_dir, "final.npz"))
    save_checkpoint_meta(run_dir, ep + 1, best["test_acc"], best["epoch"])

    summary = {
        "experiment": "jepa_ot",
        "dataset": cfg.dataset.name,
        "seed": cfg.training.seed,
        "best_test_acc": best["test_acc"],
        "best_epoch": best["epoch"],
        "final_epoch": ep + 1,
        "jepa_ot_config": {k: getattr(jot_cfg, k) for k in vars(jot_cfg)},
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    wandb_run.summary["best_test_acc"] = best["test_acc"]
    wandb_run.summary["best_epoch"] = best["epoch"]
    wandb_run.finish()

    print("\n" + "#" * 60)
    print(f"Best test accuracy: {best['test_acc']:.4f} (epoch {best['epoch']})")
    print(f"Results saved to {run_dir}")
    print("#" * 60)


if __name__ == "__main__":
    main()
