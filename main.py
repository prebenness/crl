# main.py
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad
from flax.training import train_state
import optax
from tqdm import tqdm

from src.utils.cfg import CFG
from src.utils.data.load_data import make_dataloaders, benchmark_dataloader, to_jax_batch
from src.models.mlps import SimpleMLP


# =========================
# Train state (Flax+Optax)
# =========================
def create_train_state(rng, model, learning_rate, weight_decay, batch_size, input_dim):
    dummy = jnp.ones((batch_size, input_dim), jnp.float32)
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
    onehot = jax.nn.one_hot(labels, num_classes=CFG.num_classes)
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
    
    # Enable JAX memory optimization
    jax.config.update('jax_platform_name', 'gpu' if jax.devices()[0].platform == 'gpu' else 'cpu')
    
    train_loader, test_loader = make_dataloaders(
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        drop_last=True,
        seed=CFG.seed,
        dataset=CFG.dataset,
    )

    model = SimpleMLP(hidden_sizes=CFG.hidden_sizes, num_classes=CFG.num_classes)
    dummy_input = jnp.ones((CFG.batch_size, CFG.input_dim))
    print(model.tabulate(jax.random.PRNGKey(0), dummy_input, compute_flops=True))
    
    rng = random.PRNGKey(CFG.seed)
    state = create_train_state(rng, model, CFG.lr, CFG.wd, CFG.batch_size, CFG.input_dim)

    # Benchmark dataloader performance
    print("Benchmarking data pipeline...")
    benchmark_dataloader(train_loader, num_batches=5)
    
    # Warmup compile to exclude JIT time from epoch stats
    print("Warming up JAX compilation...")
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

        # ---- Eval (vectorized for better performance)
        eval_accs = []
        for images_t, labels_t in test_loader:
            xb, yb = to_jax_batch(images_t, labels_t)
            eval_accs.append(eval_step(state.params, xb, yb))
        
        # Compute mean accuracy more efficiently
        # test_acc = float(jnp.mean(jnp.array(eval_accs)))

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {np.mean(train_losses):.4f} | "
            f"train acc {np.mean(train_accs):.4f} | "
            f"test acc  {np.mean(eval_accs):.4f} | "
            f"time {dt:.2f}s"
        )

    return state


if __name__ == "__main__":
    _ = train_and_eval()
