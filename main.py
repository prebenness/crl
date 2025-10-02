# main.py
import time
from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad
from flax import linen as nn
from flax.training import train_state
import optax
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils.cfg import CFG, Config


# =========================
# Data pipeline (PyTorch)
# =========================
def make_dataloaders(batch_size=128, num_workers=2, drop_last=True, seed=0):
    """
    Returns (train_loader, test_loader). Uses pinned memory and workers.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    tfm = transforms.ToTensor()  # -> float32 in [0,1], shape [1,28,28]
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), generator=g, drop_last=drop_last
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), drop_last=drop_last
    )
    return train_loader, test_loader


def to_jax_batch(images_t, labels_t):
    """
    Convert CPU torch tensors to JAX device arrays with static shapes.
    images_t: [B,1,28,28] float32 in [0,1]; labels_t: [B] int64
    """
    # DataLoader yields CPU tensors (pinned). .numpy() is zero-copy view on CPU.
    x_np = images_t.numpy().reshape(images_t.shape[0], -1).astype(np.float32)  # [B,784]
    y_np = labels_t.numpy().astype(np.int32)
    return jnp.asarray(x_np), jnp.asarray(y_np)


# =========================
# Model (Flax)
# =========================
class MLP(nn.Module):
    hidden_sizes: tuple[int, ...] = (512, 256)

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: [B, 784]
        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.gelu(x)
        x = nn.Dense(10)(x)  # logits
        return x


# =========================
# Train state (Flax+Optax)
# =========================
def create_train_state(rng, model, learning_rate, weight_decay, batch_size):
    dummy = jnp.ones((batch_size, 784), jnp.float32)
    variables = model.init(rng, dummy)
    params = variables["params"]
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    return train_state.TrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=tx.init(params),
    )


# =========================
# Loss & metrics (pure fns)
# =========================
def cross_entropy_loss(logits, labels):
    onehot = jax.nn.one_hot(labels, num_classes=10)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(onehot * log_probs, axis=-1))


def accuracy(logits, labels):
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean((preds == labels).astype(jnp.float32))


# =========================
# JIT-compiled steps
# =========================
@jax.jit
def train_step(state, batch_x, batch_y):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch_x, train=True)
        loss = cross_entropy_loss(logits, batch_y)
        return loss, logits

    (loss, logits), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params=state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state)
    acc = accuracy(logits, batch_y)
    return new_state, loss, acc


def make_eval_step(apply_fn):
    """
    Factory to avoid passing Python callables into jitted fns.
    Captures apply_fn in a closure; the jitted function only takes arrays.
    """
    @jax.jit
    def eval_step(params, batch_x, batch_y):
        logits = apply_fn({"params": params}, batch_x, train=False)
        return accuracy(logits, batch_y)
    return eval_step


# =========================
# Training loop
# =========================
def train_and_eval():
    print("JAX devices:", jax.devices())
    train_loader, test_loader = make_dataloaders(
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        drop_last=True,
        seed=CFG.seed,
    )

    rng = random.PRNGKey(CFG.seed)
    model = MLP(hidden_sizes=CFG.hidden_sizes)
    state = create_train_state(rng, model, CFG.lr, CFG.wd, CFG.batch_size)

    # Warmup compile to exclude JIT time from epoch stats
    images_t, labels_t = next(iter(train_loader))
    xb, yb = to_jax_batch(images_t, labels_t)
    state, _, _ = train_step(state, xb, yb)

    eval_step = make_eval_step(state.apply_fn)

    for epoch in range(1, CFG.epochs + 1):
        t0 = time.time()
        train_losses, train_accs = [], []

        # ---- Train
        for images_t, labels_t in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            xb, yb = to_jax_batch(images_t, labels_t)
            state, loss, acc = train_step(state, xb, yb)
            train_losses.append(float(loss))
            train_accs.append(float(acc))

        # ---- Eval
        accs = []
        for images_t, labels_t in test_loader:
            xb, yb = to_jax_batch(images_t, labels_t)
            accs.append(float(eval_step(state.params, xb, yb)))

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {np.mean(train_losses):.4f} | "
            f"train acc {np.mean(train_accs):.4f} | "
            f"test acc  {np.mean(accs):.4f} | "
            f"time {dt:.2f}s"
        )

    return state


if __name__ == "__main__":
    _ = train_and_eval()
