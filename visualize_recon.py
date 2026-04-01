"""Visualize VIB reconstructions vs originals.

Usage:
    python visualize_recon.py <checkpoint_path> [config_path] [options]

Examples:
    python visualize_recon.py results/.../best.npz config/colored_mnist/vib_pair_ula_best.yaml
    python visualize_recon.py results/.../best.npz config/colored_mnist/vib_pair_ula_best.yaml --n 20 --split test
    python visualize_recon.py results/.../best.npz config/colored_mnist/vib_pair_ula_best.yaml --out recon.png
"""

import argparse
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jrandom
import matplotlib.pyplot as plt

from src.config import load_config, apply_overrides
from src.datasets.datasets import build_dataset, dataset_to_jax_arrays
from src.utils.checkpointing import load_checkpoint


MODELS = {
    "vib_cnn": lambda cfg: __import__(
        "src.models.ib_classifiers", fromlist=["VIBClassifier"]
    ).VIBClassifier(
        bottleneck_width=cfg.model.bottleneck_width,
        num_classes=cfg.model.num_classes,
    ),
    "ula_mlp_var": lambda cfg: __import__(
        "src.models.ib_classifiers", fromlist=["ULAMLPVarClassifier"]
    ).ULAMLPVarClassifier(num_classes=cfg.model.num_classes),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Visualize VIB reconstructions.")
    parser.add_argument("checkpoint", help="Path to .npz checkpoint.")
    parser.add_argument("config", help="Path to YAML config used for training.")
    parser.add_argument("overrides", nargs="*",
                        help="Config overrides in section.key=value format.")
    parser.add_argument("--n", type=int, default=10, help="Number of samples.")
    parser.add_argument("--split", default="test", choices=["train", "test"],
                        help="Which split to sample from.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path (default: show interactively).")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.overrides:
        apply_overrides(cfg, args.overrides)

    model_key = cfg.model.inner
    if model_key not in MODELS:
        print(f"Model '{model_key}' is not a VIB model. "
              f"Available: {list(MODELS.keys())}", file=sys.stderr)
        sys.exit(1)

    model = MODELS[model_key](cfg)
    params = load_checkpoint(args.checkpoint)

    # Load data
    is_train = args.split == "train"
    p_corr = cfg.dataset.p_train if is_train else cfg.dataset.p_test
    ds = build_dataset(cfg.dataset.name, train=is_train,
                       p_corr=p_corr, seed=cfg.training.seed + (0 if is_train else 1))
    x_all, y_all = dataset_to_jax_arrays(ds, batch_size=128, num_workers=0,
                                          pin_memory=False, persistent_workers=False,
                                          prefetch_factor=2)

    # Sample random indices
    rng = np.random.RandomState(args.seed)
    n_total = x_all.shape[0]
    idx = rng.choice(n_total, size=args.n, replace=False)
    x_samples = x_all[idx]  # (N, H, W, C)
    y_samples = y_all[idx]

    # Forward pass (eval mode — uses mu, no sampling)
    rng_jax = jrandom.PRNGKey(args.seed)
    logits, aux = model.apply({"params": params}, x_samples, train=False)
    preds = jnp.argmax(logits, -1)

    # Reconstruct: decoder outputs logits, apply sigmoid for pixel values
    recon = jax.nn.sigmoid(aux["x_recon_logits"])

    # Plot
    n = args.n
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.5))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        orig = np.array(x_samples[i])
        rec = np.array(recon[i])

        # Clamp for display
        orig = np.clip(orig, 0, 1)
        rec = np.clip(rec, 0, 1)

        axes[0, i].imshow(orig)
        axes[0, i].set_title(f"y={int(y_samples[i])}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(rec)
        axes[1, i].set_title(f"pred={int(preds[i])}", fontsize=9)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=11)
    axes[1, 0].set_ylabel("Recon", fontsize=11)
    fig.suptitle(f"VIB Reconstruction ({args.split} split, ckpt: {args.checkpoint.split('/')[-3]})",
                 fontsize=10)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
