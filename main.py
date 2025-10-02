# main.py
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from tqdm import tqdm

from src.models.mlps import SimpleMLP
from src.jitted.train_eval import train_step, create_train_state, make_eval_step
from src.utils.cfg import CFG
from src.utils.data.load_data import make_dataloaders, benchmark_dataloader, to_jax_batch
from src.utils.utils import jax_mean


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

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {jax_mean(train_losses):.4f} | "
            f"train acc {jax_mean(train_accs):.4f} | "
            f"test acc  {jax_mean(eval_accs):.4f} | "
            f"time {dt:.2f}s"
        )

    return state


if __name__ == "__main__":
    _ = train_and_eval()
