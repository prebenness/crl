"""FG-CCDB replication on cMNIST (JAX/Flax).
Zhao, Zhang & Li, 2025 (arXiv:2505.06831v1)
"""
import argparse
import json
import logging
import time
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from flax.training import train_state

from src.datasets.datasets import CMNISTuLA, dataset_to_jax_arrays, make_epoch_batches
from src.models.classifiers import ULAMLPClassifier

log = logging.getLogger(__name__)

NUM_CLASSES = 10


# ── helpers ──────────────────────────────────────────────────────────────

def create_state(rng, lr, wd):
    model = ULAMLPClassifier(rep_dim=100, num_classes=NUM_CLASSES)
    params = model.init(rng, jnp.ones((1, 28, 28, 3)), train=True)["params"]
    tx = optax.adamw(learning_rate=lr, weight_decay=wd)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
    )


@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        logits, _ = state.apply_fn({"params": params}, x, train=True)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@jax.jit
def predict_batch(state, x):
    logits, _ = state.apply_fn({"params": state.params}, x, train=False)
    probs = jax.nn.softmax(logits, axis=-1)
    preds = jnp.argmax(probs, axis=-1)
    confs = jnp.max(probs, axis=-1)
    return preds, confs


def train_erm(state, x, y, batch_size, epochs, seed_base):
    """Train with ERM for given epochs. Returns updated state."""
    for ep in range(epochs):
        xb, yb = make_epoch_batches(x, y, batch_size, seed_base + ep)
        for i in range(xb.shape[0]):
            state, _ = train_step(state, xb[i], yb[i])
    return state


def predict_all(state, x, batch_size=512):
    """Returns (preds, confs) as numpy arrays."""
    n = x.shape[0]
    all_preds, all_confs = [], []
    for start in range(0, n, batch_size):
        xb = x[start:start + batch_size]
        p, c = predict_batch(state, xb)
        all_preds.append(np.asarray(p))
        all_confs.append(np.asarray(c))
    return np.concatenate(all_preds), np.concatenate(all_confs)


def eval_accuracy(state, x, y, batch_size=512):
    """Returns (overall_acc, worst_class_acc, per_class_acc)."""
    preds, _ = predict_all(state, x, batch_size)
    labels = np.asarray(y)
    per_class = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() > 0:
            per_class[c] = (preds[mask] == c).mean()
    return (preds == labels).mean(), per_class.min(), per_class


# ── Stage 1: BEO ────────────────────────────────────────────────────────

def stage1_beo(x_train, y_train, rng, cfg):
    """Bias Exploration via Overfitting. Returns bias pseudo-labels (numpy)."""
    np_rng = np.random.RandomState(cfg['seed'])
    n = x_train.shape[0]
    labels = np.asarray(y_train)

    # Initial bias learning: gamma=10% random subset
    n_sub = int(cfg['gamma'] * n)
    sub_idx = np_rng.choice(n, size=n_sub, replace=False)
    x_sub, y_sub = x_train[sub_idx], y_train[sub_idx]

    rng, init_rng = jrandom.split(rng)
    state = create_state(init_rng, cfg['lr'], cfg['wd'])

    log.info(f"  BEO initial: {n_sub} samples, {cfg['beo_epochs']} epochs")
    state = train_erm(state, x_sub, y_sub, cfg['batch_size'],
                      cfg['beo_epochs'], seed_base=cfg['seed'] * 1000)

    # Bias enhancement (repeat beo_reps times)
    for rep in range(cfg['beo_reps']):
        preds, confs = predict_all(state, x_train)

        # Per class: keep top 50% by confidence
        selected = []
        for c in range(NUM_CLASSES):
            cidx = np.where(labels == c)[0]
            cconf = confs[cidx]
            k = max(1, len(cidx) // 2)
            top = np.argsort(cconf)[::-1][:k]
            selected.append(cidx[top])
        selected = np.concatenate(selected)

        x_sel, y_sel = x_train[selected], y_train[selected]

        rng, init_rng = jrandom.split(rng)
        state = create_state(init_rng, cfg['lr'], cfg['wd'])
        log.info(f"  BEO enhancement {rep+1}/{cfg['beo_reps']}: {len(selected)} samples")
        state = train_erm(state, x_sel, y_sel, cfg['batch_size'],
                          cfg['beo_epochs'],
                          seed_base=(cfg['seed'] + rep + 1) * 1000)

    bias_preds, _ = predict_all(state, x_train)
    return bias_preds


# ── Stage 2: FG-CCDB weights ────────────────────────────────────────────

def stage2_weights(bias_preds, labels):
    """Closed-form FG-CCDB per-sample weights. Returns numpy array (N,)."""
    C = NUM_CLASSES
    labels = np.asarray(labels)
    bias_preds = np.asarray(bias_preds)
    N = len(labels)

    # Count matrix
    N_mat = np.zeros((C, C), dtype=np.float64)
    for i in range(N):
        N_mat[bias_preds[i], labels[i]] += 1

    G = N_mat / N
    col_sums = np.maximum(G.sum(axis=0, keepdims=True), 1e-12)
    P = G / col_sums
    q = G.sum(axis=1)

    # w[i,j] = q[i] / (P[i,j] * N_mat[i,j])
    w = np.zeros((C, C), dtype=np.float64)
    for i in range(C):
        for j in range(C):
            if P[i, j] > 0 and N_mat[i, j] > 0:
                w[i, j] = q[i] / (P[i, j] * N_mat[i, j])

    sample_w = np.array([w[bias_preds[k], labels[k]] for k in range(N)])

    log.info(f"  N_mat diagonal fraction: {np.diag(N_mat).sum() / N_mat.sum():.4f}")
    log.info(f"  Weight range: [{sample_w.min():.6f}, {sample_w.max():.6f}]")
    return sample_w


# ── Stage 3: debiased training ──────────────────────────────────────────

def stage3_debiased(x_train, y_train, sample_weights, x_val, y_val, rng, cfg):
    """Weighted resampling + ERM for 5000 iterations. Returns best state."""
    num_iters = cfg['final_iters']
    bs = cfg['batch_size']
    eval_every = cfg['eval_interval']
    np_rng = np.random.RandomState(cfg['seed'] + 2000)

    # Normalise weights to probabilities for multinomial sampling
    probs = sample_weights / sample_weights.sum()

    rng, init_rng = jrandom.split(rng)
    state = create_state(init_rng, cfg['lr'], cfg['wd'])

    best_worst = -1.0
    best_state = None
    best_iter = 0

    it = 0
    while it < num_iters:
        # Sample a batch of indices proportional to weights
        idx = np_rng.choice(len(y_train), size=bs, replace=True, p=probs)
        xb = x_train[idx]
        yb = y_train[idx]
        state, loss = train_step(state, xb, yb)
        it += 1

        if it % eval_every == 0 or it == num_iters:
            avg, worst, _ = eval_accuracy(state, x_val, y_val)
            if worst > best_worst:
                best_worst = worst
                best_state = state
                best_iter = it
            log.info(f"    Iter {it}/{num_iters}: val_avg={avg:.4f}, "
                     f"worst_class={worst:.4f} (best={best_worst:.4f} @{best_iter})")

    return best_state, best_worst, best_iter


# ── pipeline ─────────────────────────────────────────────────────────────

def run_single(p_corr, seed, cfg):
    bc_ratio = 1.0 - p_corr
    log.info(f"=== bc_ratio={bc_ratio:.1%}, seed={seed} ===")
    t0 = time.time()
    cfg = {**cfg, 'seed': seed}

    # Materialise datasets into JAX arrays (one-shot, fast)
    train_ds = CMNISTuLA(split='train', p_corr=p_corr, seed=seed)
    val_ds   = CMNISTuLA(split='val',   p_corr=p_corr, seed=seed)
    test_ds  = CMNISTuLA(split='test',  p_corr=p_corr, seed=seed)

    x_train, y_train = dataset_to_jax_arrays(train_ds)
    x_val,   y_val   = dataset_to_jax_arrays(val_ds)
    x_test,  y_test  = dataset_to_jax_arrays(test_ds)

    train_colors = np.asarray(train_ds.colors)
    log.info(f"  train={x_train.shape[0]}, val={x_val.shape[0]}, test={x_test.shape[0]}")

    rng = jrandom.PRNGKey(seed)

    # Stage 1
    log.info("  Stage 1: BEO")
    rng, beo_rng = jrandom.split(rng)
    bias_preds = stage1_beo(x_train, y_train, beo_rng, cfg)
    bias_acc = (bias_preds == train_colors).mean()
    log.info(f"  BEO bias prediction accuracy: {bias_acc:.4f}")

    # Stage 2
    log.info("  Stage 2: FG-CCDB weights")
    sample_weights = stage2_weights(bias_preds, y_train)

    # Stage 3
    log.info("  Stage 3: Debiased training")
    rng, deb_rng = jrandom.split(rng)
    best_state, best_worst_val, best_iter = stage3_debiased(
        x_train, y_train, sample_weights, x_val, y_val, deb_rng, cfg
    )

    # Test
    test_avg, test_worst, test_per_class = eval_accuracy(best_state, x_test, y_test)
    elapsed = time.time() - t0
    log.info(f"  test_acc={test_avg*100:.2f}%, worst={test_worst*100:.2f}%, "
             f"time={elapsed:.1f}s")

    return {
        'p_corr': p_corr, 'bc_ratio': bc_ratio, 'seed': seed,
        'test_acc': test_avg * 100,
        'test_worst_class_acc': test_worst * 100,
        'test_per_class_acc': (test_per_class * 100).tolist(),
        'best_val_worst_class_acc': best_worst_val * 100,
        'best_iter': best_iter,
        'elapsed_s': elapsed,
    }


PAPER_TARGETS = {
    0.005: (89.02, 0.45),
    0.01:  (94.93, 0.17),
    0.02:  (96.18, 0.19),
    0.05:  (98.21, 0.02),
}


def main():
    parser = argparse.ArgumentParser(description="FG-CCDB replication on cMNIST")
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--ratios', type=float, nargs='+',
                        default=[0.005, 0.01, 0.02, 0.05])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    cfg = {
        'num_classes': 10,
        'lr': 1e-2,
        'wd': 1e-4,
        'batch_size': 256,
        'gamma': 0.1,
        'beo_epochs': 20,
        'beo_reps': 3,
        'final_iters': 5000,
        'eval_interval': 500,
    }

    log.info(f"Config: {cfg}")
    log.info(f"JAX devices: {jax.devices()}")

    all_results = []
    for bc_ratio in args.ratios:
        p_corr = 1.0 - bc_ratio
        seed_results = []
        for seed in args.seeds:
            result = run_single(p_corr, seed, cfg)
            seed_results.append(result)
            all_results.append(result)

        accs = [r['test_acc'] for r in seed_results]
        target = PAPER_TARGETS.get(bc_ratio, (None, None))
        log.info(f"  bc_ratio={bc_ratio:.1%}: {np.mean(accs):.2f} +/- {np.std(accs):.2f} "
                 f"(target: {target[0]} +/- {target[1]})")

    # Summary
    print("\n" + "=" * 70)
    print("FG-CCDB Replication Results (cMNIST)")
    print("Zhao, Zhang & Li, 2025 (arXiv:2505.06831v1)")
    print("=" * 70)
    print(f"{'bc_ratio':>10} | {'Ours (mean +/- std)':>22} | {'Paper Target':>22}")
    print("-" * 70)
    for bc_ratio in args.ratios:
        accs = [r['test_acc'] for r in all_results
                if abs(r['bc_ratio'] - bc_ratio) < 1e-6]
        if accs:
            target = PAPER_TARGETS.get(bc_ratio, (None, None))
            t = f"{target[0]:.2f} +/- {target[1]:.2f}" if target[0] else "N/A"
            print(f"{bc_ratio:>9.1%} | {np.mean(accs):>8.2f} +/- "
                  f"{np.std(accs):<8.2f} | {t:>22}")
    print("=" * 70)

    out_dir = Path("results/fg_ccdb_replication")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"results_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump({'config': cfg, 'results': all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
