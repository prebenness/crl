# Run with: python colored_mnist_masked_fast.py
# Requirements:
#   pip install jax jaxlib flax optax torch torchvision matplotlib tqdm

import os
os.environ["WANDB_SILENT"] = "true"     # Stops WandB from cluttering the terminal

import time
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import wandb
import pandas as pd

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
# 0) Global settings (safe on RTX 3080)
# ============================================================

# Generate a unique ID for this entire execution (The "Experiment")
# We use timestamp so all models in this script share it.
WANDB_ENTITY='prebenness-crl'
WANDB_PROJECT='colored-mnist-ib'
LAMBDA_RANGE = (0, 2, 20)      # (a, b, N) -> N lambda values, evenly logspaced in 10**a to 10**b
timestamp = time.strftime("%Y-%m-%d-T%H-%M-%S", time.gmtime())
experiment_group_id = f"sweep-lambda-{LAMBDA_RANGE[0]}-{LAMBDA_RANGE[1]}-{LAMBDA_RANGE[2]}-{timestamp}"

@dataclass
class TrainConfig:
    num_classes = 2
    bottleneck_width = 16
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 50
    batch_size: int = 128
    seed: int = 0
    lambdas = jnp.logspace(*LAMBDA_RANGE)

print("JAX devices:", jax.devices())
jax.config.update("jax_default_matmul_precision", "high")


# ============================================================
# 1) PyTorch datasets
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


class IRMColoredMNIST(Dataset):
    """
    IRM-style ColoredMNIST:
      - Binary label (clean): y_clean = (digit < 5)
      - Noisy label: y = y_clean flipped with prob 0.25
      - Color correlates with *noisy* y with prob p_corr
      - Returns (x_rgb, y) with x_rgb in CHW.
    """
    LABEL_FLIP_PROB = 0.25  # P(y != y_clean)

    def __init__(self, root="./data", train=True, p_corr=0.9, seed=0):
        super().__init__()
        self.rng = np.random.RandomState(seed)

        base = MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
        # [N, 28, 28], float32 in [0, 1]
        images = base.data.numpy().astype(np.float32) / 255.0
        labels = base.targets.numpy().astype(np.int32)

        # 1. Clean binary label from digit
        #    (you can swap <5 / >=5 if you prefer the other convention)
        y_clean = (labels < 5).astype(np.int32)

        # 2. Add label noise: flip with probability LABEL_FLIP_PROB
        flip_y = self.rng.rand(len(y_clean)) < self.LABEL_FLIP_PROB
        y_noisy = np.where(flip_y, 1 - y_clean, y_clean).astype(np.int32)

        # 3. Color correlated with *noisy* label with probability p_corr
        #    (same semantics as your original: p_corr = P(color == y_noisy))
        flip_c = self.rng.rand(len(y_noisy)) > p_corr
        colors = np.where(flip_c, 1 - y_noisy, y_noisy).astype(np.int32)

        self.images = images            # [N, 28, 28], grayscale
        self.y_clean = y_clean          # optional: clean label (for analysis)
        self.y_bin = y_noisy            # noisy label used for training
        self.colors = colors            # color indicator (0=red, 1=green)

    def __len__(self):
        return len(self.y_bin)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.y_bin[idx]
        c = self.colors[idx]

        xr = x if c == 0 else np.zeros_like(x)
        xg = x if c == 1 else np.zeros_like(x)
        xb = np.zeros_like(x)

        x_rgb = np.stack([xr, xg, xb], axis=0)  # [3, 28, 28] CHW
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

def svd_spectral_penalty(z):
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


def frobenius_penalty(z):
    """
    Fast surrogate for nuclear norm:

    - Center z over the batch
    - Compute covariance C = (Zᵀ Z) / N
    - Penalise ||C||_F² = sum_ij C_ij²

    This is cheap (just matmuls) and JAX-friendly,
    but still encourages low-rank / low-variance embeddings.
    """
    # z: [batch, dim]
    z = z.astype(jnp.float32)

    # Center across the batch
    z_centered = z - jnp.mean(z, axis=0, keepdims=True)

    # Covariance (up to a constant factor)
    n = z_centered.shape[0]
    cov = (z_centered.T @ z_centered) / n  # [dim, dim]

    # Matrix-wide average of squared cov values - essentially a normalised frobenius norm
    return jnp.mean(cov ** 2)


def effective_rank_penalty(z, eps=1e-10):
    """
    Scale-invariant, rank-focused penalty for the bottleneck Z.

    z: [batch, dim] representation.

    Computes R_eff = (trace(C)^2 / ||C||_F^2), where C is the
    centered covariance. R_eff is in [1, dim] and behaves like
    an "effective rank": 1 when all variance is in one dimension,
    dim when variance is spread equally.

    Minimizing this encourages low effective rank while being
    invariant to global rescaling of z.
    """
    # Center across batch
    zc = z - jnp.mean(z, axis=0, keepdims=True)

    # Covariance (up to constant factor)
    n = zc.shape[0]
    C = (zc.T @ zc) / n  # [dim, dim]

    # trace and Frobenius norm squared
    trace = jnp.trace(C)
    frob_sq = jnp.sum(C ** 2)

    # Effective-rank-like quantity
    R_eff = (trace ** 2) / (frob_sq + eps)

    return R_eff


def linear_barrier(R_eff, eps=1e-10):
    return R_eff + (1 / (R_eff - 1 + eps))


# Placeholder for future HSIC experiments
def hsic_penalty(x, z, sigma=1.0):
    # You can implement the kernel logic here later
    return 0.0



# ============================================================
# 4) Deterministic Bottleneck Classifier
# ============================================================

class IBClassifier(nn.Module):
    bottleneck_width: int
    num_classes: int
    lamb: float      # interpreted as noise std


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
        z = nn.Dense(self.bottleneck_width)(x)
        # Tanh is recommended for IB to bound the embedding space [-1, 1]
        z = nn.tanh(z)

        # --- Add noise to channel ---
        # Additive Gaussian noise in the bottleneck
        if train:
            noise = jrandom.normal(self.make_rng("noise"), z.shape) * self.lamb
            z = z + noise

        # --- Classifier ---
        logits = nn.Dense(self.num_classes)(z)
        
        # Return z for logging
        return logits, z


# ============================================================
# 5) Training / eval (epoch-scanned, donated state)
# ============================================================


def create_state(rng, model, input_shape, cfg):
    params = model.init(
        rng,
        jnp.ones(input_shape, jnp.float32),
        train=True,
    )["params"]
    tx = optax.adamw(cfg.lr, cfg.weight_decay)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch, rng):
    x, y = batch

    def loss_fn(params):
        # 1. Forward pass returns logits AND z
        logits, z = state.apply_fn(
            {"params": params},
            x,
            train=True,
            rngs={"noise": rng},  # <- key for self.make_rng("noise")
        )
        
        # 2. Main Task Loss
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        
        # Total Loss
        total_loss = ce_loss
        
        return total_loss, (logits, ce_loss)

    # Gradient update
    (loss, (logits, ce_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    acc = (jnp.argmax(logits, -1) == y).mean()
    
    # Return metrics for logging
    metrics = {"loss": loss, "ce": ce_loss, "acc": acc}
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


def train_epoch(state, xb, yb, rng):
    n_batches = xb.shape[0]
    rngs = jrandom.split(rng, n_batches)   # [n_batches, 2]

    def body(carry, inputs):
        st = carry
        x, y, r = inputs                   # unpack one batch + one key
        st, metrics = train_step(st, (x, y), r)
        return st, metrics

    state, metrics_history = lax.scan(
        body,
        state,
        (xb, yb, rngs),                    # scan over x, y, rng in parallel
    )

    avg_metrics = {k: jnp.mean(v) for k, v in metrics_history.items()}
    return state, avg_metrics

# IMPORTANT: JIT **after** defining function, using wrapper syntax
train_epoch = jax.jit(train_epoch, donate_argnums=(0,))


def eval_epoch(state, xb, yb):
    def body(carry, batch):
        x, y = batch
        loss, acc = eval_step(state, (x, y))
        return carry, (loss, acc)

    _, (losses, accs) = lax.scan(body, None, (xb, yb))
    return losses.mean(), accs.mean()

eval_epoch = jax.jit(eval_epoch)


def run_train_eval(x_train, y_train, x_test, y_test, model, cfg, wandb_run):
    rng = jrandom.PRNGKey(cfg.seed)
    input_shape = (cfg.batch_size,) + x_train.shape[1:]
    rng, rng_init = jrandom.split(rng)

    state = create_state(rng_init, model, input_shape, cfg)

    for ep in range(cfg.epochs):
        t0 = time.time()

        seed_tr = cfg.seed * 10_000 + ep
        seed_te = cfg.seed * 20_000 + ep

        # new RNG for this epoch's noise
        rng, rng_epoch = jrandom.split(rng)

        xb, yb = make_epoch_batches(x_train, y_train, cfg.batch_size, seed_tr)
        xt, yt = make_epoch_batches(x_test,  y_test,  cfg.batch_size, seed_te)


        state, metrics = train_epoch(state, xb, yb, rng_epoch)

        te_loss, te_acc = eval_epoch(state, xt, yt)

        # --- WANDB: Log Metrics ---
        results = {
            "epoch": ep + 1,
            "train_loss": float(metrics["loss"]),
            "train_acc": float(metrics["acc"]),
            "test_loss": float(te_loss),
            "test_acc": float(te_acc),
        }
        wandb_run.log(results)

        print(
            f"  Epoch {ep+1}/{cfg.epochs}"
            f"  Acc. Train: {float(metrics['acc']):.4f} Test: {float(te_acc):.4f}"
            f"  Loss Train: {float(metrics['loss']):.4f} Test: {float(te_loss):.4f}"
            f"      Time: {time.time()-t0:.2f}s"
        )

    # Returns results from final epoch
    return results


# ============================================================
# 6) Sweep + plots + timing
# ============================================================


def wandb_summary_plot(all_data, wandb_run):
    """
    Log a summary chart: accuracy vs lambda for
    ColoredMNIST / MNIST, train / test.

    all_data: list of dicts with keys including
              ["dataset", "lambda", "train_acc", "test_acc", ...]
    """

    if not all_data:
        return

    df = pd.DataFrame(all_data)

    df["lambda"] = df["lambda"].map(float)      # Convert from logspace array to floats

    # Map to nicer names
    ds_map = {
        "colored_mnist": "ColoredMNIST",
        "mnist": "MNIST",
    }
    df["ds_name"] = df["dataset"].map(ds_map)

    # Sort lambdas once; we assume each dataset/split has one point per lambda
    lambdas = sorted(df["lambda"].unique())

    # Define the four curves we want
    curves = [
        ("ColoredMNIST – train", "colored_mnist", "train_acc"),
        ("ColoredMNIST – test",  "colored_mnist", "test_acc"),
        ("MNIST – train",        "mnist",         "train_acc"),
        ("MNIST – test",         "mnist",         "test_acc"),
    ]

    xs = lambdas
    ys = []
    keys = []

    for label, ds_code, acc_key in curves:
        keys.append(label)
        series = []
        for lamb in lambdas:
            row = df[(df["dataset"] == ds_code) & (df["lambda"] == lamb)]
            if len(row) == 0:
                # In case some combination is missing; show a gap
                series.append(None)
            else:
                series.append(float(row[acc_key].iloc[0]))
        ys.append(series)

    chart = wandb.plot.line_series(
        xs=xs,
        ys=ys,
        keys=keys,
        title="Accuracy vs λ (ColoredMNIST & MNIST)",
        xname="lambda",
    )

    wandb_run.log({"final_accuracy_plot": chart})


def main():
    # We will collect results here to log a summary table at the very end
    all_summary_data = []

    cfg = TrainConfig()

    # correlations
    p_train = 0.9
    p_test = 0.1

    print("Loading PyTorch datasets and converting to JAX arrays once...")
    t0 = time.time()

    train_col = IRMColoredMNIST(train=True,  p_corr=p_train, seed=cfg.seed)
    test_col  = IRMColoredMNIST(train=False, p_corr=p_test,  seed=cfg.seed + 1)
    train_std = StandardMNISTBin(train=True)
    test_std  = StandardMNISTBin(train=False)

    x_train_col, y_train = dataset_to_jax_arrays(train_col)
    x_test_col,  y_test  = dataset_to_jax_arrays(test_col)
    x_train_std, _       = dataset_to_jax_arrays(train_std)
    x_test_std,  _       = dataset_to_jax_arrays(test_std)

    print(f"Data ready in {time.time()-t0:.2f}s")

    results_colored: List[Dict] = []
    results_mnist: List[Dict] = []

    sweep_start = time.time()

    for lamb in cfg.lambdas:
        model = IBClassifier(
            bottleneck_width=cfg.bottleneck_width, num_classes=cfg.num_classes,
            lamb=lamb
        )

        # ---------------------------------------------------------
        # 1. ColoredMNIST Run
        # ---------------------------------------------------------
        print(f"\n--- Colored Run (lamb={lamb:.1e}) ---")
        run_config = asdict(cfg)
        run_config.update({"lambda": lamb, "dataset": "colored_mnist", "type": "noisy"})

        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            group=experiment_group_id,
            name=f"cmnist-lamb_{lamb:.1e}",
            config=run_config,
            reinit=True
        )

        t_start = time.time()
        res_col = run_train_eval(
            x_train_col, y_train, x_test_col, y_test,
            model, cfg, 
            wandb_run=run  # Passing the specific run object
        )
        res_col["run_time"] = time.time() - t_start
        res_col["dataset"] = "colored_mnist"
        res_col["lambda"] = lamb
        
        # Log summary to WandB
        run.summary["final_test_acc"] = res_col["test_acc"]
        run.summary["final_train_acc"] = res_col["train_acc"]
        run.finish()

        # Append to local lists (for plotting code later)
        results_colored.append(res_col)
        all_summary_data.append(res_col)

        # ---------------------------------------------------------
        # 2. Standard MNIST Run
        # ---------------------------------------------------------
        print(f"--- MNIST Run (lamb={lamb:.1e}) ---")
        run_config = asdict(cfg)
        run_config.update({"lambda": lamb, "dataset": "mnist", "type": "noisy"})

        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            group=experiment_group_id,
            name=f"mnist-lamb_{lamb:.1e}",
            config=run_config,
            reinit=True
        )

        t_start = time.time()
        res_std = run_train_eval(
            x_train_std, y_train, x_test_std, y_test,
            model, cfg, wandb_run=run
        )
        res_std["run_time"] = time.time() - t_start
        res_std["dataset"] = "mnist"
        res_std["lambda"] = lamb

        # Log summary to WandB
        run.summary["final_test_acc"] = res_std["test_acc"]
        run.summary["final_train_acc"] = res_std["train_acc"]
        run.finish()

        # Append to local lists
        results_mnist.append(res_std)
        all_summary_data.append(res_std)

    total_time = time.time() - sweep_start
    print("\n" + "#" * 70)
    print(f"Total sweep wall time: {total_time/60:.2f} minutes")
    print("#" * 70)

    # Prepare plots
    def to_xy(rs):
        x = np.array([r["lambda"] for r in rs])
        tr = np.array([r["train_acc"] for r in rs])
        te = np.array([r["test_acc"] for r in rs])
        return x, tr, te

    xs_c, tr_c, te_c = to_xy(results_colored)
    xs_m, tr_m, te_m = to_xy(results_mnist)


    # ---------------------------------------------------------
    # 3. The "Summary" Run
    # ---------------------------------------------------------
    summary_run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=experiment_group_id,
        name="experiment-summary",
        job_type="summary"
    )

    wandb_summary_plot(all_data=all_summary_data, wandb_run=summary_run)
    
    # Log the raw data table too if you want to inspect values
    summary_run.log({"raw_data": wandb.Table(dataframe=pd.DataFrame(all_summary_data))})


    # Update Plot labels
    plt.figure()
    plt.scatter(xs_c, tr_c, label="ColoredMNIST train", marker="o")
    plt.scatter(xs_c, te_c, label="ColoredMNIST test", marker="x")
    plt.xscale("symlog", linthresh=0.01) # Better for 0.0 values
    plt.xlabel("Noise STD")
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
    plt.xlabel("Noise STD")
    plt.ylabel("Accuracy")
    plt.title("Standard MNIST accuracy vs Information Penalty")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mnist_ib_sweep.png", dpi=300)

    print("Saved plots: coloredmnist_ib_sweep.png, mnist_ib_sweep.png")
    
    summary_run.finish()

    # Print a compact timing/accuracy table
    def pretty_print(results, name):
        # Sort by lambda
        results = sorted(results, key=lambda r: r["lambda"])
        print("\n" + name)
        print("Lambda\t\tRun Time\tTrain Acc.\tTest Acc.")
        for r in results:
            print(f"{r['lambda']:.3E}\t\t{r['run_time']:.2f}\t{r['train_acc']:.3f}\t\t{r['test_acc']:.3f}")

    pretty_print(results_colored, "ColoredMNIST results")
    pretty_print(results_mnist, "Standard MNIST results")


if __name__ == "__main__":
    main()
