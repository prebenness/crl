# main.py
import time
import argparse
from pathlib import Path
import multiprocessing as mp

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
from dotenv import dotenv_values

from src.models.mlps import SimpleMLP
from src.jitted.train_eval import train_step, create_train_state
from src.utils.cfg import CFG, update_cfg
from src.utils.data.load_data import make_dataloaders, benchmark_dataloader, to_jax_batch
from src.utils.train_eval import evaluate_model
from src.utils.utils import jax_mean



def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./cfg.env",
                        help="Path to .env config file")
    args = parser.parse_args()

    # main reads the .env and passes parsed values to CFG
    env_path = Path(args.cfg)
    env_cfg = dotenv_values(env_path) if env_path.exists() else None
    update_cfg(overrides=env_cfg)  # <-- mutates the global CFG in-place

    # Train
    _ = train_and_eval()

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
    _ = benchmark_dataloader(train_loader, num_batches=5)
    
    # Warmup compile to exclude JIT time from epoch stats
    print("Warming up JAX compilation...")
    images_t, labels_t = next(iter(train_loader))
    xb, yb = to_jax_batch(images_t, labels_t)
    state, _ = train_step(state, xb, yb)

    print("Starting training...")
    for epoch in range(1, CFG.epochs + 1):
        t0 = time.time()
        train_losses = []

        # ---- Train
        for images_t, labels_t in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            xb, yb = to_jax_batch(images_t, labels_t)

            state, loss = train_step(state, xb, yb)
            train_losses.append(float(loss))

        # ---- Eval (vectorized for better performance)
        m_test = evaluate_model(state, test_loader, tqdm_str='test')
        I_ry, H_y = m_test['I_ry'], m_test['H_y']
        I_norm = I_ry / H_y

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {jax_mean(train_losses):.4f} | "
            f"test acc  {jax_mean(m_test['acc']):.4f} | "
            f"test I(R;Y)/H(Y) ≈ {I_norm:.4f} H(Y)={H_y:.3f} CE={m_test['loss']:.3f} "
            f"time {dt:.2f}s"
        )

    return state


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    main()
