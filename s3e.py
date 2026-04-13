#!/usr/bin/env python3
"""S3E: Spectral Spurious Subspace Elimination for Colored MNIST.

4-phase pipeline:
  Phase 1 — Train ERM backbone (3x100 MLP) on biased cMNIST
  Phase 2 — Fit colour probe on frozen features, SVD to find spurious subspace
  Phase 3 — Orthogonal projection to remove top-r spurious directions
  Phase 4 — Retrain linear head on projected features

Inspired by concept erasure (Ravfogel et al., 2020, "Null it out", ACL;
Ravfogel et al., 2022, "Linear adversarial concept erasure", ICML).

Usage:
    python s3e.py config/colored_mnist/s3e.yaml [overrides...]

Example:
    python s3e.py config/colored_mnist/s3e.yaml s3e.phase1_epochs=3
"""

import argparse
import os
os.environ["WANDB_SILENT"] = "true"

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import yaml
import jax
import jax.numpy as jnp
from jax import random as jrandom
import flax.linen as nn
import optax
from flax.training import train_state
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from src.config import load_config, apply_overrides
from src.datasets.datasets import (
    build_dataset, dataset_to_jax_arrays, make_epoch_batches, make_eval_batches,
)
from src.models.classifiers import ULAMLPClassifier
from src.utils.checkpointing import (
    save_config, make_experiment_dir, utc_timestamp,
)


# ============================================================
# S3E Config
# ============================================================

@dataclass
class S3EConfig:
    num_colors: int = 10
    phase1_epochs: int = 20
    probe_lr: float = 1e-3
    probe_epochs: int = 50
    probe_l2: float = 1e-2
    phase4_epochs: int = 50
    phase4_lr: float = 1e-3
    phase4_wd: float = 0.0
    reweight: bool = False
    reweight_w_max: float = 500.0
    reweight_smoothing_eps: float = 1e-6


def parse_s3e_config(yaml_path, overrides):
    """Parse S3E-specific config from YAML + CLI overrides."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    s3e = S3EConfig()
    if "s3e" in raw:
        for k, v in raw["s3e"].items():
            if hasattr(s3e, k):
                field_type = type(getattr(s3e, k))
                if field_type is bool:
                    val = v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
                    setattr(s3e, k, val)
                else:
                    setattr(s3e, k, field_type(v))

    for token in overrides:
        if "=" not in token:
            continue
        path, value_str = token.split("=", 1)
        parts = path.split(".")
        if len(parts) == 2 and parts[0] == "s3e":
            key = parts[1]
            if hasattr(s3e, key):
                field_type = type(getattr(s3e, key))
                if field_type is bool:
                    setattr(s3e, key, value_str.lower() in ("true", "1", "yes"))
                else:
                    setattr(s3e, key, field_type(value_str))

    return s3e


# ============================================================
# Models
# ============================================================

class LinearHead(nn.Module):
    """Single linear classifier on pre-extracted features."""
    num_classes: int

    @nn.compact
    def __call__(self, z, train: bool = True):
        logits = nn.Dense(self.num_classes)(z)
        return logits


# ============================================================
# Utilities
# ============================================================

def eval_accuracy(model, params, data_x, data_y, batch_size):
    """Compute accuracy over all samples, handling padding correctly."""
    xb, yb, counts = make_eval_batches(data_x, data_y, batch_size)

    @jax.jit
    def _predict(params, x_batch):
        out = model.apply({"params": params}, x_batch, train=False)
        logits = out[0] if isinstance(out, tuple) else out
        return jnp.argmax(logits, -1)

    total_correct = 0
    total = 0
    for i in range(xb.shape[0]):
        preds = _predict(params, xb[i])
        n = int(counts[i])
        total_correct += int((preds[:n] == yb[i][:n]).sum())
        total += n
    return total_correct / total


# ============================================================
# Phase 1: ERM Backbone Training
# ============================================================

def train_erm_backbone(x_train, y_train, x_test, y_test, cfg, s3e_cfg,
                       wandb_run, run_dir):
    """Train standard 3x100 MLP with CE on biased cMNIST.

    Returns (best_params, model).
    """
    model = ULAMLPClassifier(rep_dim=100, num_classes=cfg.model.num_classes)
    rng = jrandom.PRNGKey(cfg.training.seed)
    rng, rng_init = jrandom.split(rng)

    input_shape = (cfg.training.batch_size,) + x_train.shape[1:]
    params = model.init(rng_init, jnp.ones(input_shape), train=True)["params"]

    tx = optax.adamw(cfg.training.lr, cfg.training.weight_decay_inner)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
    )

    @jax.jit
    def train_step(state, x, y):
        def loss_fn(params):
            logits, _ = state.apply_fn({"params": params}, x, train=True)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        logits, _ = state.apply_fn({"params": state.params}, x, train=False)
        acc = (jnp.argmax(logits, -1) == y).mean()
        return state, loss, acc

    best = {"test_acc": -1.0, "params": None, "epoch": 0}

    for ep in range(s3e_cfg.phase1_epochs):
        t0 = time.time()
        seed_tr = cfg.training.seed * 10_000 + ep
        xb, yb = make_epoch_batches(
            x_train, y_train, cfg.training.batch_size, seed_tr,
        )

        ep_loss, ep_acc = 0.0, 0.0
        for i in range(xb.shape[0]):
            state, loss, acc = train_step(state, xb[i], yb[i])
            ep_loss += float(loss)
            ep_acc += float(acc)
        ep_loss /= xb.shape[0]
        ep_acc /= xb.shape[0]

        te_acc = eval_accuracy(model, state.params, x_test, y_test,
                               cfg.training.batch_size)

        if te_acc > best["test_acc"]:
            best["test_acc"] = te_acc
            best["params"] = jax.tree.map(jnp.copy, state.params)
            best["epoch"] = ep + 1
            print(f"    >>> NEW BEST test_acc={te_acc:.4f} (epoch {ep+1})")

        wandb_run.log({
            "phase1/train_loss": ep_loss,
            "phase1/train_acc": ep_acc,
            "phase1/test_acc": te_acc,
            "phase1/best_test_acc": best["test_acc"],
            "epoch": ep + 1,
        })
        print(f"  Phase 1 Epoch {ep+1}/{s3e_cfg.phase1_epochs}"
              f"  loss {ep_loss:.4f}  train_acc {ep_acc:.4f}"
              f"  test_acc {te_acc:.4f}  ({time.time()-t0:.2f}s)")

    # Save backbone checkpoint
    if run_dir is not None:
        ckpt_path = run_dir / "phase1_backbone.npz"
        flat, treedef = jax.tree.flatten(best["params"])
        np.savez(str(ckpt_path), *[np.array(a) for a in flat])
        print(f"  Saved Phase 1 backbone: {ckpt_path}")

    return best["params"], model


# ============================================================
# Phase 2: Feature Extraction + Colour Probe + SVD
# ============================================================

def extract_features(model, params, x, batch_size):
    """Extract penultimate features h for all samples."""
    xb, _, counts = make_eval_batches(
        x, jnp.zeros(x.shape[0], dtype=jnp.int32), batch_size,
    )

    @jax.jit
    def _extract(params, x_batch):
        _, aux = model.apply({"params": params}, x_batch, train=False)
        return aux["h"]

    all_h = []
    for i in range(xb.shape[0]):
        h = _extract(params, xb[i])
        all_h.append(h[:int(counts[i])])
    return jnp.concatenate(all_h, axis=0)


def train_colour_probe(h_train, s_train, h_test, s_test, s3e_cfg,
                       batch_size, seed, wandb_run):
    """L2-regularised logistic regression predicting colour from features.

    Returns (probe_params, probe_model).
    """
    probe = LinearHead(num_classes=s3e_cfg.num_colors)
    rng = jrandom.PRNGKey(seed + 7777)
    rng, rng_init = jrandom.split(rng)

    input_shape = (batch_size, h_train.shape[1])
    params = probe.init(rng_init, jnp.ones(input_shape), train=True)["params"]

    tx = optax.adam(s3e_cfg.probe_lr)
    state = train_state.TrainState.create(
        apply_fn=probe.apply, params=params, tx=tx,
    )

    l2_reg = s3e_cfg.probe_l2

    @jax.jit
    def probe_train_step(state, h_batch, s_batch):
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, h_batch, train=True)
            ce = optax.softmax_cross_entropy_with_integer_labels(
                logits, s_batch,
            ).mean()
            w = params["Dense_0"]["kernel"]
            l2 = 0.5 * l2_reg * (w ** 2).sum()
            return ce + l2, logits
        (loss, logits), grads = jax.value_and_grad(
            loss_fn, has_aux=True,
        )(state.params)
        state = state.apply_gradients(grads=grads)
        acc = (jnp.argmax(logits, -1) == s_batch).mean()
        return state, loss, acc

    for ep in range(s3e_cfg.probe_epochs):
        seed_ep = seed * 10_000 + ep + 5000
        hb, sb = make_epoch_batches(h_train, s_train, batch_size, seed_ep)

        ep_loss, ep_acc = 0.0, 0.0
        for i in range(hb.shape[0]):
            state, loss, acc = probe_train_step(state, hb[i], sb[i])
            ep_loss += float(loss)
            ep_acc += float(acc)
        ep_loss /= hb.shape[0]
        ep_acc /= hb.shape[0]

        te_acc = eval_accuracy(probe, state.params, h_test, s_test, batch_size)

        if wandb_run:
            wandb_run.log({
                "phase2/probe_loss": ep_loss,
                "phase2/probe_train_acc": ep_acc,
                "phase2/probe_test_acc": te_acc,
                "epoch": ep + 1,
            })

        if (ep + 1) % 10 == 0 or ep == 0 or ep == s3e_cfg.probe_epochs - 1:
            print(f"  Phase 2 Probe Epoch {ep+1}/{s3e_cfg.probe_epochs}"
                  f"  loss {ep_loss:.4f}  train_acc {ep_acc:.4f}"
                  f"  test_acc {te_acc:.4f}")

    return state.params, probe


def compute_svd(probe_params):
    """SVD of probe weight matrix.

    The Dense kernel has shape (feature_dim, num_colors).
    The probe's "W_s" in the S3E proposal is kernel.T: (num_colors, feature_dim).
    SVD: W_s = U @ diag(sigma) @ V^T.
    V columns are directions in feature space carrying colour information.

    Returns (V, sigma) where V is (feature_dim, num_singular_values).
    """
    kernel = probe_params["Dense_0"]["kernel"]  # (feature_dim, num_colors)
    W_s = kernel.T  # (num_colors, feature_dim)
    U, sigma, Vt = jnp.linalg.svd(W_s, full_matrices=False)
    V = Vt.T  # (feature_dim, num_singular_values)
    return V, sigma


# ============================================================
# Phase 3: Orthogonal Projection
# ============================================================

def project_features(h, V, r):
    """Project features to remove top-r spurious directions.

    P_perp = I - V_r V_r^T  (orthogonal projector)
    h_proj = h @ P_perp = h - (h @ V_r) @ V_r^T
    """
    V_r = V[:, :r]           # (d, r)
    h_V = h @ V_r             # (N, r)
    return h - h_V @ V_r.T    # (N, d)


# ============================================================
# Phase 4: Retrain Linear Head
# ============================================================

def train_linear_head(h_train, y_train, h_test, y_test,
                      cfg, s3e_cfg, r, wandb_run, run_dir,
                      weights=None):
    """Train linear classifier on projected features.

    If weights is not None, uses weighted CE (FG-CCDB reweighting).
    Returns results dict.
    """
    head = LinearHead(num_classes=cfg.model.num_classes)
    rng = jrandom.PRNGKey(cfg.training.seed + r * 100)
    rng, rng_init = jrandom.split(rng)

    input_shape = (cfg.training.batch_size, h_train.shape[1])
    params = head.init(rng_init, jnp.ones(input_shape), train=True)["params"]

    tx = optax.adamw(s3e_cfg.phase4_lr, s3e_cfg.phase4_wd)
    state = train_state.TrainState.create(
        apply_fn=head.apply, params=params, tx=tx,
    )

    use_weights = weights is not None

    @jax.jit
    def train_step(state, h_batch, y_batch):
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, h_batch, train=True)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, y_batch,
            ).mean(), logits
        (loss, logits), grads = jax.value_and_grad(
            loss_fn, has_aux=True,
        )(state.params)
        state = state.apply_gradients(grads=grads)
        acc = (jnp.argmax(logits, -1) == y_batch).mean()
        return state, loss, acc

    @jax.jit
    def train_step_weighted(state, h_batch, y_batch, w_batch):
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, h_batch, train=True)
            per_sample = optax.softmax_cross_entropy_with_integer_labels(
                logits, y_batch,
            )
            loss = (per_sample * w_batch).sum() / w_batch.sum()
            return loss, logits
        (loss, logits), grads = jax.value_and_grad(
            loss_fn, has_aux=True,
        )(state.params)
        state = state.apply_gradients(grads=grads)
        acc = (jnp.argmax(logits, -1) == y_batch).mean()
        return state, loss, acc

    early_patience = cfg.checkpointing.early_stopping_patience
    best = {"test_acc": -1.0, "params": None, "epoch": 0, "train_acc": 0.0}
    epochs_since_best = 0

    for ep in range(s3e_cfg.phase4_epochs):
        t0 = time.time()
        seed_ep = cfg.training.seed * 10_000 + ep + r * 1000

        if use_weights:
            hb, yb, wb = make_epoch_batches(
                h_train, y_train, cfg.training.batch_size, seed_ep, weights,
            )
        else:
            hb, yb = make_epoch_batches(
                h_train, y_train, cfg.training.batch_size, seed_ep,
            )

        ep_loss, ep_acc = 0.0, 0.0
        for i in range(hb.shape[0]):
            if use_weights:
                state, loss, acc = train_step_weighted(
                    state, hb[i], yb[i], wb[i],
                )
            else:
                state, loss, acc = train_step(state, hb[i], yb[i])
            ep_loss += float(loss)
            ep_acc += float(acc)
        ep_loss /= hb.shape[0]
        ep_acc /= hb.shape[0]

        te_acc = eval_accuracy(head, state.params, h_test, y_test,
                               cfg.training.batch_size)

        if te_acc > best["test_acc"]:
            best["test_acc"] = te_acc
            best["params"] = jax.tree.map(jnp.copy, state.params)
            best["epoch"] = ep + 1
            best["train_acc"] = ep_acc
            epochs_since_best = 0
            print(f"    >>> NEW BEST test_acc={te_acc:.4f} (epoch {ep+1})")
        else:
            epochs_since_best += 1

        if wandb_run:
            wandb_run.log({
                "phase4/train_loss": ep_loss,
                "phase4/train_acc": ep_acc,
                "phase4/test_acc": te_acc,
                "phase4/best_test_acc": best["test_acc"],
                "epoch": ep + 1,
            })

        if (ep + 1) % 10 == 0 or ep == 0 or ep == s3e_cfg.phase4_epochs - 1:
            print(f"  Phase 4 Epoch {ep+1}/{s3e_cfg.phase4_epochs}"
                  f"  loss {ep_loss:.4f}  train_acc {ep_acc:.4f}"
                  f"  test_acc {te_acc:.4f}  best {best['test_acc']:.4f}"
                  f"  ({time.time()-t0:.2f}s)")

        if early_patience > 0 and epochs_since_best >= early_patience:
            print(f"  Early stopping at epoch {ep+1} "
                  f"(no improvement for {early_patience} epochs)")
            break

    return {
        "test_acc": best["test_acc"],
        "train_acc": best["train_acc"],
        "best_test_acc": best["test_acc"],
        "best_epoch": best["epoch"],
    }


# ============================================================
# FG-CCDB Importance Reweighting
# ============================================================

def compute_fgccdb_weights(y, s, num_classes, num_colors,
                           smoothing_eps=1e-6, w_max=500.0):
    """Per-sample FG-CCDB importance weights.

    Follows Zhao, Zhang & Li, 2025 (arXiv:2505.06831v1) Eqs. 3-5:
      N_mat[s, y] = count of samples with colour s and label y
      G = N_mat / N
      P[:, j] = G[:, j] / sum(G[:, j])
      q = sum_j G[:, j]
      W[s, y] = q[s] / P[s, y]
      w[s, y] = W[s, y] / N_mat[s, y]
    """
    y_np = np.array(y)
    s_np = np.array(s)
    N = len(y_np)

    N_mat = np.zeros((num_colors, num_classes), dtype=np.float64)
    for i in range(N):
        N_mat[s_np[i], y_np[i]] += 1

    G = N_mat / N
    col_sums = G.sum(axis=0, keepdims=True)
    P = G / (col_sums + smoothing_eps)
    q = G.sum(axis=1)

    # Per-group weight then per-sample weight
    sample_weights = np.zeros(N, dtype=np.float64)
    for idx in range(N):
        si, yi = s_np[idx], y_np[idx]
        if P[si, yi] > smoothing_eps and N_mat[si, yi] > 0:
            W_sy = q[si] / P[si, yi]
            sample_weights[idx] = W_sy / N_mat[si, yi]

    sample_weights = np.clip(sample_weights, 0, w_max)
    sample_weights = sample_weights / sample_weights.sum() * N

    return jnp.array(sample_weights, dtype=jnp.float32)


# ============================================================
# Plotting
# ============================================================

def make_sweep_plot(results, out_path):
    """Plot test accuracy vs r."""
    rs = [r["r"] for r in results]
    test_accs = [r["test_acc"] for r in results]
    train_accs = [r["train_acc"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(rs, test_accs, "o-", label="Test acc (best)")
    plt.plot(rs, train_accs, "s--", alpha=0.5, label="Train acc")
    plt.xlabel("r (spurious directions removed)")
    plt.ylabel("Accuracy")
    plt.title("S3E: accuracy vs spurious subspace rank")
    plt.legend()
    plt.ylim(-0.02, 1.02)
    plt.xticks(rs)

    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved sweep plot: {out_path}")


# ============================================================
# Main
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="S3E: Spectral Spurious Subspace Elimination",
    )
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides (section.key=value)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Load standard config (ignores unknown s3e section)
    cfg = load_config(args.config)
    std_overrides = [o for o in args.overrides if not o.startswith("s3e.")]
    if std_overrides:
        apply_overrides(cfg, std_overrides)

    # Load S3E-specific config
    s3e_cfg = parse_s3e_config(args.config, args.overrides)

    print("JAX devices:", jax.devices())
    jax.config.update("jax_default_matmul_precision", "high")

    timestamp = utc_timestamp()
    sweep_name = f"{timestamp}_s3e_{cfg.dataset.name}"
    sweep_dir = make_experiment_dir("s3e", sweep_name)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    group_id = f"{timestamp}-s3e-sweep"

    save_config(sweep_dir, {
        "experiment": "s3e",
        "sweep_name": sweep_name,
        "config_path": str(Path(args.config).resolve()),
        "dataset": cfg.dataset.name,
        "seed": cfg.training.seed,
        "s3e": {k: getattr(s3e_cfg, k) for k in vars(s3e_cfg)},
        "r_values": [int(v) for v in cfg.lambdas],
    })

    # ---- Load data ----
    print("Loading datasets...")
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
        train_ds, batch_size=cfg.training.batch_size,
    )
    x_test, y_test = dataset_to_jax_arrays(
        test_ds, batch_size=cfg.training.batch_size,
    )
    s_train = jnp.array(train_ds.colors, dtype=jnp.int32)
    s_test = jnp.array(test_ds.colors, dtype=jnp.int32)

    print(f"  x_train {x_train.shape}, x_test {x_test.shape}")
    print(f"  Colour labels: {len(set(train_ds.colors.tolist()))} unique")
    print(f"  Data ready in {time.time()-t0:.2f}s")

    r_values = [int(v) for v in cfg.lambdas]

    # ================================================================
    # PHASE 1: ERM Backbone
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: ERM Backbone Training")
    print("=" * 60)

    phase1_run = wandb.init(
        entity=cfg.wandb.entity, project=cfg.wandb.project,
        group=group_id, name="s3e-phase1-erm",
        config={"phase": 1, "dataset": cfg.dataset.name,
                "epochs": s3e_cfg.phase1_epochs},
        reinit=True,
    )

    backbone_params, backbone_model = train_erm_backbone(
        x_train, y_train, x_test, y_test, cfg, s3e_cfg,
        phase1_run, sweep_dir,
    )
    phase1_run.finish()

    # ================================================================
    # PHASE 2: Feature Extraction + Colour Probe + SVD
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Feature Extraction + Colour Probe + SVD")
    print("=" * 60)

    print("  Extracting features...")
    t0 = time.time()
    h_train = extract_features(
        backbone_model, backbone_params, x_train, cfg.training.batch_size,
    )
    h_test = extract_features(
        backbone_model, backbone_params, x_test, cfg.training.batch_size,
    )
    print(f"  h_train {h_train.shape}, h_test {h_test.shape}"
          f"  ({time.time()-t0:.2f}s)")

    phase2_run = wandb.init(
        entity=cfg.wandb.entity, project=cfg.wandb.project,
        group=group_id, name="s3e-phase2-probe",
        config={"phase": 2, "dataset": cfg.dataset.name,
                "probe_epochs": s3e_cfg.probe_epochs,
                "probe_l2": s3e_cfg.probe_l2},
        reinit=True,
    )

    print("  Training colour probe...")
    probe_params, probe_model = train_colour_probe(
        h_train, s_train, h_test, s_test, s3e_cfg,
        cfg.training.batch_size, cfg.training.seed, phase2_run,
    )

    V, sigma = compute_svd(probe_params)
    var_explained = jnp.cumsum(sigma ** 2) / (sigma ** 2).sum()

    print("\n  SVD Results:")
    print("  r | sigma    | cumulative variance")
    print("  --+----------+--------------------")
    for i in range(len(sigma)):
        print(f"  {i+1} | {float(sigma[i]):8.4f} | {float(var_explained[i]):.4f}")

    phase2_run.summary["singular_values"] = [float(v) for v in sigma]
    phase2_run.summary["variance_explained"] = [float(v) for v in var_explained]
    phase2_run.summary["probe_train_acc"] = eval_accuracy(
        probe_model, probe_params, h_train, s_train, cfg.training.batch_size,
    )
    phase2_run.summary["probe_test_acc"] = eval_accuracy(
        probe_model, probe_params, h_test, s_test, cfg.training.batch_size,
    )
    print(f"\n  Final probe train acc: "
          f"{phase2_run.summary['probe_train_acc']:.4f}")
    print(f"  Final probe test acc:  "
          f"{phase2_run.summary['probe_test_acc']:.4f}")
    phase2_run.finish()

    # ---- Optional FG-CCDB weights ----
    weights = None
    if s3e_cfg.reweight:
        weights = compute_fgccdb_weights(
            y_train, s_train, cfg.model.num_classes, s3e_cfg.num_colors,
            s3e_cfg.reweight_smoothing_eps, s3e_cfg.reweight_w_max,
        )
        print(f"\n  FG-CCDB reweighting enabled:"
              f"  mean={float(weights.mean()):.2f}"
              f"  max={float(weights.max()):.2f}"
              f"  min={float(weights.min()):.4f}")

    # ================================================================
    # PHASE 3+4: Projection + Head Retraining Sweep
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 3+4: Projection + Head Retraining Sweep")
    print("=" * 60)

    all_results = []
    sweep_start = time.time()

    for r in r_values:
        print(f"\n--- r = {r} ---")

        run_dir = sweep_dir / f"r_{r}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Phase 3: Project
        h_train_proj = project_features(h_train, V, r)
        h_test_proj = project_features(h_test, V, r)

        removed_var = float(jnp.sum(sigma[:r] ** 2) / jnp.sum(sigma ** 2))
        print(f"  Removed variance fraction: {removed_var:.4f}")

        # Phase 4: Retrain head
        r_run = wandb.init(
            entity=cfg.wandb.entity, project=cfg.wandb.project,
            group=group_id, name=f"s3e-r_{r}",
            config={"phase": "3+4", "r": r,
                    "dataset": cfg.dataset.name,
                    "removed_variance_frac": removed_var,
                    "reweight": s3e_cfg.reweight},
            reinit=True,
        )

        t_r = time.time()
        res = train_linear_head(
            h_train_proj, y_train, h_test_proj, y_test,
            cfg, s3e_cfg, r, r_run, run_dir, weights=weights,
        )
        res["r"] = r
        res["lambda"] = r
        res["removed_variance_frac"] = removed_var
        res["run_time"] = time.time() - t_r
        res["dataset"] = cfg.dataset.name

        r_run.summary["best_test_acc"] = res["best_test_acc"]
        r_run.summary["best_epoch"] = res["best_epoch"]
        r_run.finish()

        all_results.append(res)

    total_time = time.time() - sweep_start
    print("\n" + "#" * 60)
    print(f"Total sweep wall time: {total_time/60:.2f} minutes")
    print("#" * 60)

    # ---- Summary ----
    print("\nS3E Results:")
    print("r | Test Acc | Train Acc | Best Epoch | Var Removed")
    print("--+----------+-----------+------------+------------")
    for r_res in all_results:
        print(f"{r_res['r']:2d} | {r_res['test_acc']:.4f}   |"
              f" {r_res['train_acc']:.4f}    |"
              f" {r_res['best_epoch']:10d} |"
              f" {r_res['removed_variance_frac']:.4f}")

    # Summary W&B run
    summary_run = wandb.init(
        entity=cfg.wandb.entity, project=cfg.wandb.project,
        group=group_id, name="experiment-summary",
        job_type="summary",
    )
    summary_run.log({
        "raw_data": wandb.Table(dataframe=pd.DataFrame(all_results)),
    })
    summary_run.finish()

    # Plot
    plot_path = sweep_dir / "s3e_sweep_plot.png"
    make_sweep_plot(all_results, plot_path)

    # Save results
    with open(sweep_dir / "summary.json", "w") as f:
        json.dump({
            "experiment": "s3e",
            "sweep_name": sweep_name,
            "dataset": cfg.dataset.name,
            "seed": cfg.training.seed,
            "s3e_config": {k: getattr(s3e_cfg, k) for k in vars(s3e_cfg)},
            "singular_values": [float(v) for v in sigma],
            "variance_explained": [float(v) for v in var_explained],
            "total_time_sec": total_time,
            "results": all_results,
        }, f, indent=2)
    pd.DataFrame(all_results).to_csv(sweep_dir / "summary.csv", index=False)
    print(f"\nResults saved to {sweep_dir}")


if __name__ == "__main__":
    main()
