"""FG-CCDB replication on cMNIST (JAX/Flax).

Faithfully replicates the implementation from the code repository of
Zhao, Zhang & Li, 2025 (arXiv:2505.06831v1).

Key code-vs-paper discrepancies discovered in their repository:
  - Uses 4-layer CNN (MNIST_CNN_fc), not the claimed 3-layer MLP
  - Uses oracle (ground truth) bias labels, not BEO predictions
  - BEO trains on full data initially (not gamma=10% subsample)

CLI flags:
  --ula_mlp   : Use the actual uLA 3x100 MLP instead of their CNN
  --use_beo   : Use BEO predictions instead of oracle labels
"""
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
from flax.training import train_state

from src.datasets.datasets import CMNISTuLA, dataset_to_jax_arrays, make_epoch_batches

log = logging.getLogger(__name__)

NUM_CLASSES = 10
LAMBDA_CTR = 0.001  # contrastive loss weight for BEO enhancement (cMNIST)


# ── Models ──────────────────────────────────────────────────────────────

class MNISTCNN(nn.Module):
    """4-layer CNN matching MNIST_CNN_fc from their code.
    Conv->BN->ReLU x4, GlobalAvgPool, Dense(256, 10).
    """
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (B, 28, 28, 3) NHWC
        x = nn.Conv(64, (3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(128, (3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(256, (3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Conv(256, (3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        fea = x.mean(axis=(1, 2))  # GlobalAvgPool -> (B, 256)
        logits = nn.Dense(self.num_classes)(fea)
        return logits, {"z": fea}


class BNTrainState(train_state.TrainState):
    batch_stats: Any


# ── Contrastive loss ────────────────────────────────────────────────────

def contrastive_loss(z, y, num_classes):
    """Intra-class pairwise distance loss (matches ctr_loss from their code)."""
    mat = jax.nn.one_hot(y, num_classes)
    A2 = (z ** 2).sum(axis=1, keepdims=True)
    B2 = (z ** 2).sum(axis=1)
    D = A2 + B2 - 2 * (z @ z.T)
    mask = mat @ mat.T
    mask = mask / mask.sum()
    return (D * mask).sum()


# ── Data augmentation (Stage 3) ─────────────────────────────────────────
# Matches their cMNIST transform: RandomResizedCrop + RandomRotation + Normalize
# Implemented as batch affine transform for speed (no per-sample Python loop).


def augment_batch(x_nhwc_np, np_rng):
    """Batch augmentation via affine_grid + grid_sample.
    Approximates RandomResizedCrop(28, scale=(0.8,1.1), ratio=(0.75,1.33))
    + RandomRotation(30) + Normalize(0.5,0.5,0.5).
    Input: numpy NHWC [0,1]. Output: jax NHWC [-1,1].
    """
    B = x_nhwc_np.shape[0]
    t = torch.from_numpy(np.asarray(x_nhwc_np)).permute(0, 3, 1, 2).float()

    # Crop scale + aspect ratio (matching RandomResizedCrop params)
    area_scale = np_rng.uniform(0.8, 1.1, size=B).astype(np.float32)
    log_aspect = np_rng.uniform(
        np.log(0.75), np.log(4.0 / 3.0), size=B).astype(np.float32)
    s = np.sqrt(area_scale)
    aspect = np.exp(log_aspect)
    sx = s * np.sqrt(aspect)
    sy = s / np.sqrt(aspect)

    # Random translation (crop position offset)
    tx = np_rng.uniform(-0.15, 0.15, size=B).astype(np.float32)
    ty = np_rng.uniform(-0.15, 0.15, size=B).astype(np.float32)

    # Random rotation in [-30, 30] degrees
    angles = (np_rng.uniform(-30, 30, size=B) * (np.pi / 180)).astype(
        np.float32)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # Build batch of 2x3 affine matrices
    theta = np.zeros((B, 2, 3), dtype=np.float32)
    theta[:, 0, 0] = sx * cos_a
    theta[:, 0, 1] = -sy * sin_a
    theta[:, 1, 0] = sx * sin_a
    theta[:, 1, 1] = sy * cos_a
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    grid = torch.nn.functional.affine_grid(
        torch.from_numpy(theta), t.size(), align_corners=False)
    out = torch.nn.functional.grid_sample(
        t, grid, mode='bilinear', padding_mode='zeros',
        align_corners=False)

    out = out.permute(0, 2, 3, 1).numpy()
    return jnp.array((out - 0.5) / 0.5)


def normalize_batch(x):
    """Normalize [0,1] -> [-1,1] matching Normalize(0.5, 0.5, 0.5)."""
    return 2.0 * x - 1.0


# ── JIT-compiled train/predict functions ────────────────────────────────

def make_fns(use_bn):
    """Build JIT-compiled functions for CNN (use_bn=True) or MLP (False)."""

    if use_bn:
        @jax.jit
        def train_step_ce(state, x, y):
            def loss_fn(params):
                (logits, _), updates = state.apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    x, train=True, mutable=["batch_stats"])
                return optax.softmax_cross_entropy_with_integer_labels(
                    logits, y).mean(), updates
            (loss, updates), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state.replace(batch_stats=updates["batch_stats"]), loss

        @jax.jit
        def train_step_ctr(state, x, y):
            """CE + contrastive loss for BEO enhancement."""
            def loss_fn(params):
                (logits, aux), updates = state.apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    x, train=True, mutable=["batch_stats"])
                ce = optax.softmax_cross_entropy_with_integer_labels(
                    logits, y).mean()
                ctr = contrastive_loss(aux["z"], y, NUM_CLASSES)
                return ce + LAMBDA_CTR * ctr, updates
            (loss, updates), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state.replace(batch_stats=updates["batch_stats"]), loss

        @jax.jit
        def train_step_weighted(state, x, y, w):
            """Per-sample weighted CE for debiased training."""
            def loss_fn(params):
                (logits, _), updates = state.apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    x, train=True, mutable=["batch_stats"])
                ce = optax.softmax_cross_entropy_with_integer_labels(logits, y)
                return (ce * w).mean(), updates
            (loss, updates), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state.replace(batch_stats=updates["batch_stats"]), loss

        @jax.jit
        def predict_batch(state, x):
            (logits, _) = state.apply_fn(
                {"params": state.params, "batch_stats": state.batch_stats},
                x, train=False)
            return logits

    else:
        @jax.jit
        def train_step_ce(state, x, y):
            def loss_fn(params):
                logits, _ = state.apply_fn(
                    {"params": params}, x, train=True)
                return optax.softmax_cross_entropy_with_integer_labels(
                    logits, y).mean()
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            return state.apply_gradients(grads=grads), loss

        @jax.jit
        def train_step_ctr(state, x, y):
            def loss_fn(params):
                logits, aux = state.apply_fn(
                    {"params": params}, x, train=True)
                ce = optax.softmax_cross_entropy_with_integer_labels(
                    logits, y).mean()
                ctr = contrastive_loss(aux["z"], y, NUM_CLASSES)
                return ce + LAMBDA_CTR * ctr
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            return state.apply_gradients(grads=grads), loss

        @jax.jit
        def train_step_weighted(state, x, y, w):
            def loss_fn(params):
                logits, _ = state.apply_fn(
                    {"params": params}, x, train=True)
                ce = optax.softmax_cross_entropy_with_integer_labels(logits, y)
                return (ce * w).mean()
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            return state.apply_gradients(grads=grads), loss

        @jax.jit
        def predict_batch(state, x):
            logits, _ = state.apply_fn(
                {"params": state.params}, x, train=False)
            return logits

    return train_step_ce, train_step_ctr, train_step_weighted, predict_batch


# ── Helpers ─────────────────────────────────────────────────────────────

def create_model(use_mlp):
    if use_mlp:
        from src.models.classifiers import ULAMLPClassifier
        return ULAMLPClassifier(rep_dim=100, num_classes=NUM_CLASSES)
    return MNISTCNN(num_classes=NUM_CLASSES)


def create_state(model, rng, lr, wd, use_bn, schedule_steps=None):
    variables = model.init(rng, jnp.ones((1, 28, 28, 3)), train=True)
    params = variables["params"]
    if schedule_steps:
        schedule = optax.cosine_decay_schedule(
            init_value=lr, decay_steps=schedule_steps)
        tx = optax.adamw(learning_rate=schedule, weight_decay=wd)
    else:
        tx = optax.adamw(learning_rate=lr, weight_decay=wd)
    if use_bn:
        return BNTrainState.create(
            apply_fn=model.apply, params=params, tx=tx,
            batch_stats=variables["batch_stats"],
        )
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
    )


def predict_all_logits(predict_fn, state, x, batch_size=512):
    """Returns logits as numpy array (N, C)."""
    all_logits = []
    for start in range(0, x.shape[0], batch_size):
        logits = predict_fn(state, x[start:start + batch_size])
        all_logits.append(np.asarray(logits))
    return np.concatenate(all_logits)


def eval_accuracy(predict_fn, state, x, y, batch_size=512):
    """Returns (overall_acc, worst_class_acc, per_class_acc)."""
    logits = predict_all_logits(predict_fn, state, x, batch_size)
    preds = logits.argmax(axis=-1)
    labels = np.asarray(y)
    per_class = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() > 0:
            per_class[c] = (preds[mask] == c).mean()
    return (preds == labels).mean(), per_class.min(), per_class


# ── Stage 1: BEO ────────────────────────────────────────────────────────

def select_top_half(logits, labels):
    """Select top 50% per class by true-class logit (matches their code)."""
    selected = []
    for c in range(NUM_CLASSES):
        class_idx = np.where(labels == c)[0]
        conf = logits[class_idx, c]
        k = max(1, int(len(class_idx) * 0.5))
        top_k = np.argsort(conf)[-k:]
        selected.append(class_idx[top_k])
    return np.concatenate(selected)


def stage1_beo(x_train, y_train, rng, model, use_bn, fns, cfg):
    """BEO: 1 initial training + 3 enhancement rounds.

    Matches their code: initial trains on full data (CE only),
    enhancements select top 50% by true-class logit and retrain
    fresh models with CE + contrastive loss.
    Returns bias predictions (numpy, shape N, values 0..9).
    """
    train_step_ce, train_step_ctr, _, predict_fn = fns
    labels = np.asarray(y_train)
    n = x_train.shape[0]
    bs = cfg['batch_size']
    n_epochs = cfg['beo_epochs']
    steps_per_epoch = n // bs

    # Stage 0: full data, CE only, cosine LR
    total_steps = n_epochs * steps_per_epoch
    rng, init_rng = jrandom.split(rng)
    state = create_state(model, init_rng, cfg['beo_lr'], cfg['wd'], use_bn,
                         schedule_steps=total_steps)
    log.info(f"  BEO Stage 0: full data ({n}), {n_epochs} ep, CE only")
    for ep in range(n_epochs):
        xb, yb = make_epoch_batches(x_train, y_train, bs,
                                    cfg['seed'] * 1000 + ep)
        for i in range(xb.shape[0]):
            state, _ = train_step_ce(state, xb[i], yb[i])

    # 3 enhancement rounds: select top 50% -> fresh model with CE + ctr
    for rep in range(cfg['beo_reps']):
        logits = predict_all_logits(predict_fn, state, x_train)
        selected = select_top_half(logits, labels)
        x_sel, y_sel = x_train[selected], y_train[selected]
        n_sel = len(selected)
        sel_steps = (n_sel // bs) * n_epochs

        rng, init_rng = jrandom.split(rng)
        state = create_state(model, init_rng, cfg['beo_lr'], cfg['wd'],
                             use_bn, schedule_steps=sel_steps)
        log.info(f"  BEO round {rep+1}/{cfg['beo_reps']}: "
                 f"{n_sel} samples, CE+ctr")
        for ep in range(n_epochs):
            xb, yb = make_epoch_batches(
                x_sel, y_sel, bs,
                (cfg['seed'] + rep + 1) * 1000 + ep)
            for i in range(xb.shape[0]):
                state, _ = train_step_ctr(state, xb[i], yb[i])

    final_logits = predict_all_logits(predict_fn, state, x_train)
    return final_logits.argmax(axis=-1)


# ── Stage 2: FG-CCDB weights ────────────────────────────────────────────

def stage2_weights_v4(bias_preds, labels, num_classes=NUM_CLASSES):
    """FG-CCDB v4 weight computation with class balancing.
    Matches group_weight_specific_v4 from their code.
    """
    labels = np.asarray(labels)
    bias_preds = np.asarray(bias_preds)
    N = len(labels)

    # Count matrix: group[i,j] = count(bias_pred=i, class=j)
    group = np.zeros((num_classes, num_classes), dtype=np.float64)
    for k in range(N):
        group[bias_preds[k], labels[k]] += 1

    # Class balancing factor: lamd = max_count / per_class_count
    class_counts = group.sum(axis=0, keepdims=True)
    class_counts = np.maximum(class_counts, 1e-12)
    lamd = class_counts.max() / class_counts

    groupB = group * lamd
    group_p = groupB / np.maximum(groupB.sum(), 1e-12)
    p_m = group_p.sum(axis=1, keepdims=True)
    p_c = group_p / np.maximum(group_p.sum(axis=0, keepdims=True), 1e-12)

    prob = p_m / (p_c + 1e-5) * (group > 0)
    weight_g = prob * lamd
    weight = weight_g / (group + 1e-5) * (group > 0)

    sample_w = np.array([weight[bias_preds[k], labels[k]] for k in range(N)])

    # Clamp and normalize (matching their code lines 388-389)
    sample_w = np.maximum(sample_w, 1e-4)
    sample_w = sample_w / sample_w.max()

    log.info(f"  Diagonal fraction: "
             f"{np.diag(group).sum() / max(group.sum(), 1):.4f}")
    log.info(f"  Weight range: [{sample_w.min():.6f}, {sample_w.max():.6f}]")
    return sample_w


# ── Stage 3: Debiased training ──────────────────────────────────────────

def stage3_debiased(x_train, y_train, sample_weights, x_val, y_val,
                    rng, model, use_bn, fns, cfg):
    """Weighted sampling + per-sample weighted CE (matching their code).
    Both resampling and loss weighting are applied (double correction).
    Training batches get augmentation + normalize; eval uses normalize only.
    """
    _, _, train_step_w, predict_fn = fns
    num_iters = cfg['final_iters']
    bs = cfg['batch_size']
    eval_every = cfg['eval_interval']
    np_rng = np.random.RandomState(cfg['seed'] + 2000)

    probs = sample_weights / sample_weights.sum()

    # Pre-normalize val data (Stage 3 model sees [-1,1] inputs)
    x_val_norm = normalize_batch(x_val)

    rng, init_rng = jrandom.split(rng)
    state = create_state(model, init_rng, cfg['final_lr'], cfg['wd'], use_bn)

    best_worst = -1.0
    best_state = None
    best_iter = 0

    # Pre-convert training data to numpy for augmentation
    x_train_np = np.asarray(x_train)

    for it in range(1, num_iters + 1):
        idx = np_rng.choice(len(y_train), size=bs, replace=True, p=probs)
        xb_raw = x_train_np[idx]
        yb = y_train[idx]
        wb = jnp.array(sample_weights[idx], dtype=jnp.float32)

        # Augment + normalize training batch
        xb = augment_batch(xb_raw, np_rng)

        state, loss = train_step_w(state, xb, yb, wb)

        if it % eval_every == 0 or it == num_iters:
            avg, worst, _ = eval_accuracy(
                predict_fn, state, x_val_norm, y_val)
            if worst > best_worst:
                best_worst = worst
                best_state = state
                best_iter = it
            log.info(f"    Iter {it}/{num_iters}: val_avg={avg:.4f}, "
                     f"worst={worst:.4f} (best={best_worst:.4f} @{best_iter})")

    return best_state, best_worst, best_iter


# ── Pipeline ────────────────────────────────────────────────────────────

def run_single(p_corr, seed, model, use_bn, fns, cfg):
    bc_ratio = 1.0 - p_corr
    log.info(f"\n=== bc_ratio={bc_ratio:.1%}, seed={seed} ===")
    t0 = time.time()
    cfg = {**cfg, 'seed': seed}

    train_ds = CMNISTuLA(split='train', p_corr=p_corr, seed=seed)
    val_ds = CMNISTuLA(split='val', p_corr=p_corr, seed=seed)
    test_ds = CMNISTuLA(split='test', p_corr=p_corr, seed=seed)

    x_train, y_train = dataset_to_jax_arrays(train_ds)
    x_val, y_val = dataset_to_jax_arrays(val_ds)
    x_test, y_test = dataset_to_jax_arrays(test_ds)

    train_colors = np.asarray(train_ds.colors)
    labels = np.asarray(y_train)
    log.info(f"  train={x_train.shape[0]}, val={x_val.shape[0]}, "
             f"test={x_test.shape[0]}")

    rng = jrandom.PRNGKey(seed)

    # Stage 1: BEO
    log.info("  Stage 1: BEO")
    rng, beo_rng = jrandom.split(rng)
    bias_preds = stage1_beo(x_train, y_train, beo_rng, model, use_bn,
                            fns, cfg)
    beo_color_acc = (bias_preds == train_colors).mean()
    log.info(f"  BEO predicts color: {beo_color_acc:.4f}")

    # Bias labels for weight computation
    if cfg['oracle']:
        # Match their code `if 1: pseudo_g = gt_g; all_place = gt_g%2`
        # Binary: 0=aligned (color==digit), 1=misaligned
        bias_labels = (train_colors != labels).astype(np.int32)
        log.info("  Using ORACLE alignment labels (binary 0/1)")
    else:
        bias_labels = bias_preds
        log.info("  Using BEO predictions as bias labels (10-way)")

    # Stage 2: Weights
    log.info("  Stage 2: FG-CCDB weights (v4)")
    sample_weights = stage2_weights_v4(bias_labels, labels)

    # Stage 3: Debiased training
    log.info("  Stage 3: Debiased training")
    rng, deb_rng = jrandom.split(rng)
    predict_fn = fns[3]
    best_state, best_worst_val, best_iter = stage3_debiased(
        x_train, y_train, sample_weights, x_val, y_val,
        deb_rng, model, use_bn, fns, cfg)

    # Stage 3 model expects normalized [-1,1] inputs
    x_test_norm = normalize_batch(x_test)
    test_avg, test_worst, test_pc = eval_accuracy(
        predict_fn, best_state, x_test_norm, y_test)
    elapsed = time.time() - t0
    log.info(f"  test_acc={test_avg*100:.2f}%, worst={test_worst*100:.2f}%, "
             f"time={elapsed:.1f}s")

    return {
        'p_corr': p_corr, 'bc_ratio': bc_ratio, 'seed': seed,
        'test_acc': test_avg * 100,
        'test_worst_class_acc': test_worst * 100,
        'test_per_class_acc': (test_pc * 100).tolist(),
        'best_val_worst_class_acc': best_worst_val * 100,
        'best_iter': best_iter,
        'beo_color_acc': float(beo_color_acc),
        'elapsed_s': elapsed,
    }


PAPER_TARGETS = {
    0.005: (89.02, 0.45),
    0.01: (94.93, 0.17),
    0.02: (96.18, 0.19),
    0.05: (98.21, 0.02),
}


def main():
    parser = argparse.ArgumentParser(
        description="FG-CCDB replication on cMNIST "
                    "(Zhao et al. 2025, arXiv:2505.06831v1)")
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4])
    parser.add_argument('--ratios', type=float, nargs='+',
                        default=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument('--ula_mlp', action='store_true',
                        help='Use uLA 3x100 MLP instead of CNN')
    parser.add_argument('--use_beo', action='store_true',
                        help='Use BEO predictions for weights '
                             '(default: oracle labels)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    use_bn = not args.ula_mlp
    model = create_model(args.ula_mlp)
    fns = make_fns(use_bn)
    arch = "MLP (uLA 3x100)" if args.ula_mlp else "CNN (MNIST_CNN_fc)"

    cfg = {
        'beo_lr': 1e-4,       # their code; paper claims 1e-2
        'final_lr': 1e-3,     # their code; paper claims 1e-2
        'wd': 1e-4,
        'batch_size': 256,
        'beo_epochs': 20,
        'beo_reps': 3,        # 3 enhancement rounds after initial
        'final_iters': 2000,  # their code; paper claims 5000
        'eval_interval': 100,
        'oracle': not args.use_beo,
        'use_mlp': args.ula_mlp,
    }

    log.info(f"Architecture: {arch}")
    log.info(f"Oracle labels: {cfg['oracle']}")
    log.info(f"Config: {cfg}")
    log.info(f"JAX devices: {jax.devices()}")

    all_results = []
    for bc_ratio in args.ratios:
        p_corr = 1.0 - bc_ratio
        seed_results = []
        for seed in args.seeds:
            result = run_single(p_corr, seed, model, use_bn, fns, cfg)
            seed_results.append(result)
            all_results.append(result)

        accs = [r['test_acc'] for r in seed_results]
        target = PAPER_TARGETS.get(bc_ratio, (None, None))
        log.info(f"  bc_ratio={bc_ratio:.1%}: "
                 f"{np.mean(accs):.2f} +/- {np.std(accs):.2f}"
                 f" (target: {target[0]} +/- {target[1]})")

    # Summary table
    mode = "BEO" if args.use_beo else "Oracle"
    print(f"\n{'=' * 70}")
    print("FG-CCDB Replication — Zhao et al. 2025 (arXiv:2505.06831v1)")
    print(f"Architecture: {arch} | Bias labels: {mode}")
    print(f"{'=' * 70}")
    print(f"{'bc_ratio':>10} | {'Ours (mean +/- std)':>22} | "
          f"{'Paper Target':>22}")
    print(f"{'-' * 70}")
    for bc_ratio in args.ratios:
        accs = [r['test_acc'] for r in all_results
                if abs(r['bc_ratio'] - bc_ratio) < 1e-6]
        if accs:
            target = PAPER_TARGETS.get(bc_ratio, (None, None))
            t = (f"{target[0]:.2f} +/- {target[1]:.2f}"
                 if target[0] else "N/A")
            print(f"{bc_ratio:>9.1%} | {np.mean(accs):>8.2f} +/- "
                  f"{np.std(accs):<8.2f} | {t:>22}")
    print(f"{'=' * 70}")

    out_dir = Path("results/fg_ccdb_replication")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (f"{'mlp' if args.ula_mlp else 'cnn'}_"
              f"{'beo' if args.use_beo else 'oracle'}")
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"results_{suffix}_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump({'config': cfg, 'arch': arch, 'mode': mode,
                   'results': all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
