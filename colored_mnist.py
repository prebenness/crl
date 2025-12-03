# Run with: python colored_mnist_masked_fast.py
# Requirements:
#   pip install jax jaxlib flax optax torch torchvision matplotlib tqdm

import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

import wandb
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms

import jax
import jax.numpy as jnp
from jax import random as jrandom
import jax.lax as lax

import flax.linen as nn
from flax.training import train_state
import optax

import matplotlib.pyplot as plt


# ============================================================
# 0) Global performance knobs (safe on RTX 3080)
# ============================================================

print("JAX devices:", jax.devices())
jax.config.update("jax_default_matmul_precision", "high")

@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    batch_size: int = 128
    seed: int = 0
    lambdas = jnp.logspace(-3, -1, 20)


# ============================================================
# 1) PyTorch datasets (unchanged, no TF)
# ============================================================

class ColoredMNIST(Dataset):
    """
    Standard ColoredMNIST:
      - Binary label: y_bin = (digit < 5)
      - Color correlates with y_bin with prob p_corr
      - Returns (x_rgb, y_bin) with x_rgb in CHW.
    """
    def __init__(self, root="./data", train=True, p_corr=0.9, seed=0):
        super().__init__()
        self.rng = np.random.RandomState(seed)

        base = MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
        images = base.data.numpy().astype(np.float32) / 255.0  # [N,28,28]
        labels = base.targets.numpy().astype(np.int32)

        y_bin = (labels < 5).astype(np.int32)

        flip = self.rng.rand(len(y_bin)) > p_corr
        colors = np.where(flip, 1 - y_bin, y_bin).astype(np.int32)

        self.images = images
        self.y_bin = y_bin
        self.colors = colors

    def __len__(self):
        return len(self.y_bin)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.y_bin[idx]
        c = self.colors[idx]

        xr = x if c == 0 else np.zeros_like(x)
        xg = x if c == 1 else np.zeros_like(x)
        xb = np.zeros_like(x)

        x_rgb = np.stack([xr, xg, xb], axis=0)  # [3,28,28] CHW
        return x_rgb, y


class StandardMNISTBin(Dataset):
    """Binary MNIST (digit<5), grayscale CHW."""
    def __init__(self, root="./data", train=True):
        base = MNIST(root=root, train=train, download=True, transform=transforms.ToTensor())
        images = base.data.numpy().astype(np.float32) / 255.0
        labels = base.targets.numpy().astype(np.int32)
        y_bin = (labels < 5).astype(np.int32)

        self.images = images
        self.y_bin = y_bin

    def __len__(self):
        return len(self.y_bin)

    def __getitem__(self, idx):
        x = self.images[idx]
        x = np.expand_dims(x, 0)  # [1,28,28]
        return x, self.y_bin[idx]


def dataset_to_jax_arrays(dataset: Dataset):
    """Load whole PyTorch dataset once, convert to NHWC JAX arrays once."""
    xs, ys = [], []
    for x, y in dataset:
        xs.append(x)
        ys.append(y)
    xs = np.stack(xs, axis=0)  # NCHW
    xs = np.transpose(xs, (0, 2, 3, 1))  # NHWC
    ys = np.array(ys, dtype=np.int32)
    return jnp.array(xs), jnp.array(ys)


# ============================================================
# 2) JAX batching (host shuffle + reshape)
# ============================================================

def make_epoch_batches(x, y, batch_size, seed):
    """
    Fast host-side permutation, device-side gather.
    Avoids jax.random.permutation (slow + sync).
    """
    n = x.shape[0]
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)

    n_trim = (n // batch_size) * batch_size
    perm = perm[:n_trim]

    perm = jnp.asarray(perm)  # move indices to device once

    x_shuf = jnp.take(x, perm, axis=0)
    y_shuf = jnp.take(y, perm, axis=0)

    n_batches = n_trim // batch_size
    xb = x_shuf.reshape((n_batches, batch_size) + x.shape[1:])
    yb = y_shuf.reshape((n_batches, batch_size))
    return xb, yb


# ============================================================
# 3) Regularization Penalties (Spectral / HSIC placeholders)
# ============================================================

def spectral_penalty(z):
    """
    Computes the nuclear norm (sum of singular values) of the batch.
    Encourages the representation to lie on a low-dimensional manifold.
    """
    # Center the batch (conceptually important for covariance rank)
    z_centered = z - jnp.mean(z, axis=0, keepdims=True)
    
    # Compute singular values (SVD)
    # full_matrices=False is critical for performance
    _, s, _ = jnp.linalg.svd(z_centered, full_matrices=False)
    
    # Minimize the sum of singular values (nuclear norm)
    return jnp.sum(s)

# Placeholder for future HSIC experiments
def hsic_penalty(x, z, sigma=1.0):
    # You can implement the kernel logic here later
    return 0.0



# ============================================================
# 4) Deterministic Bottleneck Classifier
# ============================================================

class IBClassifier(nn.Module):
    bottleneck_width: int = 128  # Keep this large; regularization will shrink effective rank
    num_classes: int = 2

    @nn.compact
    def __call__(self, x, train=True):
        # --- Encoder ---
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), (2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), (2, 2))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        # --- The Bottleneck (Z) ---
        # No masking, no quantization. Just a constrained vector space.
        z = nn.Dense(self.bottleneck_width)(x)
        # Tanh is recommended for IB to bound the embedding space [-1, 1]
        z = nn.tanh(z)

        # --- Classifier ---
        # Stop gradient ensures the classifier doesn't try to "expand" the 
        # bottleneck to cheat the penalty, though usually fine without it.
        logits = nn.Dense(self.num_classes)(z)
        
        # Return z so we can penalize it in the loss function
        return logits, z


# ============================================================
# 5) Training / eval (epoch-scanned, donated state)
# ============================================================


def create_state(rng, model, input_shape, cfg):
    params = model.init(
        rng,
        jnp.ones(input_shape, jnp.float32),
        train=True
    )["params"]
    tx = optax.adamw(cfg.lr, cfg.weight_decay)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch, lamb):
    x, y = batch

    def loss_fn(params):
        # 1. Forward pass returns logits AND z
        logits, z = state.apply_fn(
            {"params": params},
            x,
            train=True
        )
        
        # 2. Main Task Loss
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        
        # 3. Information Bottleneck Penalty
        # (For HSIC, you would change this line to hsic_penalty(x, z))
        reg_loss = spectral_penalty(z)
        
        # Total Loss
        total_loss = ce_loss + (lamb * reg_loss)
        
        return total_loss, (logits, ce_loss, reg_loss)

    # Gradient update
    (loss, (logits, ce_loss, reg_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    acc = (jnp.argmax(logits, -1) == y).mean()
    
    # Return metrics for logging
    metrics = {"loss": loss, "ce": ce_loss, "reg": reg_loss, "acc": acc}
    return state, metrics


@jax.jit
def eval_step(state, batch):
    x, y = batch
    # Model now returns (logits, z), we only need logits for eval
    logits, _ = state.apply_fn(
        {"params": state.params},
        x,
        train=False
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    acc = (jnp.argmax(logits, -1) == y).mean()
    return loss, acc


def train_epoch(state, xb, yb, hyper):
    lamb = hyper["lamb"]

    def body(carry, batch):
        st = carry
        x, y = batch
        st, metrics = train_step(st, (x, y), lamb)
        return st, metrics

    state, metrics_history = lax.scan(body, state, (xb, yb))
    
    # Average the metrics over the epoch
    avg_metrics = {k: jnp.mean(v) for k, v in metrics_history.items()}
    return state, avg_metrics

# IMPORTANT: JIT **after** defining function, using wrapper syntax
train_epoch = jax.jit(train_epoch, donate_argnums=(0,))


def eval_epoch(state, xb, yb):
    def body(carry, batch):
        x, y = batch
        # No hyper/lamb needed for pure evaluation anymore
        loss, acc = eval_step(state, (x, y))
        return carry, (loss, acc)

    _, (losses, accs) = lax.scan(body, None, (xb, yb))
    return losses.mean(), accs.mean()

eval_epoch = jax.jit(eval_epoch)


def run_train_eval(x_train, y_train, x_test, y_test, model, cfg, lamb):
    rng = jrandom.PRNGKey(cfg.seed)
    input_shape = (cfg.batch_size,) + x_train.shape[1:]

    state = create_state(rng, model, input_shape, cfg)

    # Initialize placeholders so they exist even if loop fails (safety)
    tr_acc, te_acc, tr_loss, te_loss = 0.0, 0.0, 0.0, 0.0

    for ep in range(cfg.epochs):
        seed_tr = cfg.seed * 10_000 + ep
        seed_te = cfg.seed * 20_000 + ep

        xb, yb = make_epoch_batches(x_train, y_train, cfg.batch_size, seed_tr)
        xt, yt = make_epoch_batches(x_test,  y_test,  cfg.batch_size, seed_te)

        hyper = dict(lamb=lamb)

        state, metrics = train_epoch(state, xb, yb, hyper)
        tr_loss = metrics["loss"]
        tr_acc  = metrics["acc"]

        te_loss, te_acc = eval_epoch(state, xt, yt)

        print(
            f"  ep {ep+1}/{cfg.epochs} "
            f"train_acc={float(tr_acc):.4f} test_acc={float(te_acc):.4f}"
        )

    return dict(
        train_acc=float(tr_acc),
        test_acc=float(te_acc),
        train_loss=float(tr_loss),
        test_loss=float(te_loss),
    )


# ============================================================
# 6) Sweep + plots + timing
# ============================================================

def capacity_proxy(w, b):
    return w * b


def main():
    cfg = TrainConfig()

    # correlations
    p_train = 0.9
    p_test = 0.1

    print("Loading PyTorch datasets and converting to JAX arrays once...")
    t0 = time.time()

    train_col = ColoredMNIST(train=True,  p_corr=p_train, seed=cfg.seed)
    test_col  = ColoredMNIST(train=False, p_corr=p_test,  seed=cfg.seed + 1)
    train_std = StandardMNISTBin(train=True)
    test_std  = StandardMNISTBin(train=False)

    x_train_col, y_train = dataset_to_jax_arrays(train_col)
    x_test_col,  y_test  = dataset_to_jax_arrays(test_col)
    x_train_std, _       = dataset_to_jax_arrays(train_std)
    x_test_std,  _       = dataset_to_jax_arrays(test_std)

    print(f"Data ready in {time.time()-t0:.2f}s")

    model = IBClassifier(bottleneck_width=128, num_classes=2)

    results_colored: List[Dict] = []
    results_mnist: List[Dict] = []

    sweep_start = time.time()

    for lamb in cfg.lambdas:
        print("\n" + "=" * 70)
        print(f"ColoredMNIST run: lamb={lamb:.3E}")
        run_start = time.time()
        res_col = run_train_eval(
            x_train_col, y_train, x_test_col, y_test,
            model, cfg, lamb=lamb
        )
        run_time = time.time() - run_start
        print(f"  run_time_colored = {run_time:.2f}s")
        results_colored.append(dict(lamb=lamb, run_time=run_time, **res_col))

        print("\n" + "-" * 70)
        print(f"Standard MNIST run: lamb={lamb:.3E}")
        run_start = time.time()
        res_std = run_train_eval(
            x_train_std, y_train, x_test_std, y_test,
            model, cfg, lamb=lamb
        )
        run_time = time.time() - run_start
        print(f"  run_time_mnist = {run_time:.2f}s")
        results_mnist.append(dict(lamb=lamb, run_time=run_time, **res_std))

    total_time = time.time() - sweep_start
    print("\n" + "#" * 70)
    print(f"Total sweep wall time: {total_time/60:.2f} minutes")
    print("#" * 70)

    # Prepare plots
    def to_xy(rs):
        x = np.array([r["lamb"] for r in rs])
        tr = np.array([r["train_acc"] for r in rs])
        te = np.array([r["test_acc"] for r in rs])
        return x, tr, te

    xs_c, tr_c, te_c = to_xy(results_colored)
    xs_m, tr_m, te_m = to_xy(results_mnist)

    # Update Plot labels
    plt.figure()
    plt.scatter(xs_c, tr_c, label="ColoredMNIST train", marker="o")
    plt.scatter(xs_c, te_c, label="ColoredMNIST test", marker="x")
    plt.xscale("symlog", linthresh=0.01) # Better for 0.0 values
    plt.xlabel("Spectral Penalty Lambda")
    plt.ylabel("Accuracy")
    plt.title("ColoredMNIST accuracy vs Information Penalty")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("coloredmnist_ib_sweep.png", dpi=300)

    plt.figure()
    plt.scatter(xs_m, tr_m, label="MNIST train", marker="o")
    plt.scatter(xs_m, te_m, label="MNIST test", marker="x")
    plt.xscale("symlog", linthresh=0.01)
    plt.xlabel("Spectral Penalty Lambda")
    plt.ylabel("Accuracy")
    plt.title("Standard MNIST accuracy vs Information Penalty")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mnist_ib_sweep.png", dpi=300)

    print("Saved plots: coloredmnist_ib_sweep.png, mnist_ib_sweep.png")

    # Print a compact timing/accuracy table
    def pretty_print(results, name):
        # Sort by lambda
        results = sorted(results, key=lambda r: r["lamb"])
        print("\n" + name)
        print("Lambda\t\tRun Time\tTrain Acc.\tTest Acc.")
        for r in results:
            print(f"{r['lamb']:.3E}\t\t{r['run_time']:.2f}\t{r['train_acc']:.3f}\t\t{r['test_acc']:.3f}")

    pretty_print(results_colored, "ColoredMNIST results")
    pretty_print(results_mnist, "Standard MNIST results")


if __name__ == "__main__":
    main()
