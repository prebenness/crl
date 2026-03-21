"""Baseline LSTM experiments for comparison with differentiable MDL.

Trains standard LSTMs with cross-entropy loss and optional L1/L2
regularization, reproducing the baseline from Lan et al. (2024) Section 4.2.

This provides the "standard training" comparison point: networks trained
with common objectives (CE, CE+L1, CE+L2) consistently fail to generalize,
while MDL-trained networks find generalizing solutions.

Usage:
    # Single run
    python3.12 scripts/experiments/baseline_lstm_experiments.py --reg none --seed 42

    # With L1 regularization
    python3.12 scripts/experiments/baseline_lstm_experiments.py --reg l1 --reg_lambda 0.1 --seed 42

    # Full grid search (matching Lan et al.)
    bash scripts/experiments/run_baselines.sh
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jrandom
import optax

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.mdl.data import (
    make_anbn_dataset,
    make_test_set,
    make_validation_set,
    sequences_to_padded_arrays,
    NUM_SYMBOLS,
    SYMBOL_A,
)
from src.mdl.baseline_lstm import (
    BaselineLSTM,
    create_baseline_state,
    make_baseline_train_step,
    compute_baseline_mdl_score,
    flatten_params,
)
from src.mdl.training import evaluate_deterministic_accuracy
from src.mdl.golden import golden_mdl_score
from src.mdl.coding import integer_code_length
from src.mdl.evaluation import (
    compute_grammar_weighted_nll_bits,
    compute_optimal_dh_test,
    compute_optimal_dh_train,
    compute_train_dh,
    compute_delta_pct,
    format_abudy_comparison_table,
)


# ---------------------------------------------------------------------------
# Run management (mirrors differentiable_mdl.py)
# ---------------------------------------------------------------------------

class TeeLogger:
    """Mirrors stdout to both the terminal and a log file."""

    def __init__(self, log_path, mode="w"):
        self._log_path = log_path
        self._mode = mode
        self._file = None
        self._orig = None

    def __enter__(self):
        self._file = open(self._log_path, self._mode)
        self._orig = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._orig
        self._file.close()

    def write(self, data):
        self._orig.write(data)
        self._file.write(data)

    def flush(self):
        self._orig.flush()
        self._file.flush()

    def fileno(self):
        return self._orig.fileno()


def make_run_dir(args):
    """Create a timestamped results directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reg_str = args.reg if args.reg != "none" else "noreg"
    lam_str = f"_rl{args.reg_lambda}" if args.reg != "none" else ""
    drop_str = f"_d{args.dropout}" if args.dropout > 0 else ""
    es_str = f"_es{args.early_stop}" if args.early_stop > 0 else ""
    name = (
        f"{ts}_baseline_{reg_str}{lam_str}{drop_str}{es_str}"
        f"_e{args.epochs}_lr{args.lr}_s{args.seed}"
    )
    run_dir = Path("results") / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    return run_dir


def get_train_max_n(inputs):
    """Find the maximum n in the training set."""
    max_n = 0
    for inp in inputs:
        n = sum(1 for s in inp if s == SYMBOL_A)
        max_n = max(max_n, n)
    return max_n


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(args, model, x_train, y_train, mask_train,
                 val_inputs, val_targets, rng, run_dir=None):
    """Train a baseline LSTM with optional regularization and early stopping.

    Args:
        args: experiment configuration
        model: BaselineLSTM instance
        x_train, y_train, mask_train: padded training arrays
        val_inputs, val_targets: validation data (list of sequences)
        rng: JAX PRNG key
        run_dir: results directory (None = no saving)

    Returns:
        state: final training state
        best_params: params with best validation loss
        best_val_loss: best validation loss achieved
    """
    rng, init_rng = jrandom.split(rng)
    N = x_train.shape[0]
    max_seq_len = x_train.shape[1]
    bs = args.batch_size if args.batch_size > 0 else N

    state = create_baseline_state(
        init_rng, model,
        seq_len=max_seq_len,
        batch_size=min(bs, N),
        lr=args.lr,
    )

    n_params = len(flatten_params(state.params))
    print(f"  Number of parameters: {n_params}")

    train_step = make_baseline_train_step(
        reg_type=args.reg if args.reg != "none" else None,
        reg_lambda=args.reg_lambda,
    )

    print(f"\n  Epochs: {args.epochs}")
    print(f"  lr={args.lr}, reg={args.reg}, reg_lambda={args.reg_lambda}")
    print(f"  dropout={args.dropout}, early_stop={args.early_stop}")
    print(f"  batch_size={bs}")
    print("-" * 70)

    best_val_loss = float("inf")
    best_params = None
    epochs_no_improve = 0

    t0 = time.time()
    for epoch in range(args.epochs):
        # Shuffle and batch
        rng, perm_rng = jrandom.split(rng)
        perm = jrandom.permutation(perm_rng, N)
        n_batches = max(N // bs, 1)

        epoch_loss = 0.0
        epoch_data_nll_bits = 0.0
        epoch_reg = 0.0

        for b in range(n_batches):
            idx = perm[b * bs:(b + 1) * bs] if N >= bs else jnp.arange(N)
            xb, yb, mb = x_train[idx], y_train[idx], mask_train[idx]

            rng, batch_rng = jrandom.split(rng)
            state, loss, aux = train_step(state, xb, yb, mb, batch_rng)

            epoch_loss += float(loss)
            epoch_data_nll_bits += float(aux["data_nll_bits"])
            epoch_reg += float(aux["reg_regularizer"])

        epoch_loss /= n_batches
        epoch_data_nll_bits /= n_batches
        epoch_reg /= n_batches

        # Logging
        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:5d} | "
                f"objective_total_bits={epoch_loss:8.2f}b  "
                f"data_nll_bits={epoch_data_nll_bits:8.1f}b  "
                f"reg_regularizer={epoch_reg:8.4f}"
            )

        # Validation (for early stopping and checkpointing)
        if (epoch + 1) % args.eval_every == 0:
            # Compute validation loss
            val_loss = _compute_val_loss(
                state, val_inputs, val_targets,
            )

            # Also compute deterministic accuracy for monitoring
            val_result = evaluate_deterministic_accuracy(
                state.apply_fn, state.params, None,
                val_inputs, val_targets,
            )
            n_perfect = val_result["n_perfect"]
            n_val = len(val_inputs)
            gen_n = val_result["gen_n"]

            is_best = val_loss < best_val_loss
            best_tag = "  * BEST" if is_best else ""
            print(
                f"              -> val_loss={val_loss:.2f}  "
                f"gen_n={gen_n}  ({n_perfect}/{n_val} perfect){best_tag}"
            )

            if is_best:
                best_val_loss = val_loss
                best_params = jax.tree.map(lambda x: x.copy(), state.params)
                epochs_no_improve = 0
                if run_dir is not None:
                    _save_params(best_params, run_dir / "checkpoint_best.npz")
            else:
                epochs_no_improve += args.eval_every

            # Early stopping
            if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
                print(f"\n  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {epochs_no_improve} epochs)")
                break

    elapsed = time.time() - t0
    print("-" * 70)
    print(f"Training complete in {elapsed:.1f}s ({epoch+1} epochs)")

    return state, best_params, best_val_loss


def _save_params(params, path):
    """Save params to npz file."""
    leaves = jax.tree.leaves(params)
    save_dict = {f"p{i}": np.array(l) for i, l in enumerate(leaves)}
    np.savez(str(path), **save_dict)


def _compute_val_loss(state, val_inputs, val_targets):
    """Compute validation loss (CE only, no regularization)."""
    from src.mdl.data import sequences_to_padded_arrays
    x_val, y_val, mask_val = sequences_to_padded_arrays(val_inputs, val_targets)

    logits, _ = state.apply_fn({"params": state.params}, x_val, train=False)
    ce_nats = optax.softmax_cross_entropy_with_integer_labels(logits, y_val)
    ce_bits = ce_nats / jnp.log(2.0)
    val_loss = float(jnp.sum(ce_bits * mask_val))
    return val_loss


def evaluate_baseline_on_test(state, test_inputs, test_targets, test_max_n):
    """Evaluate a baseline model on the test set."""
    print(f"\nEvaluating on test set (n=1..{test_max_n})...")
    test_result = evaluate_deterministic_accuracy(
        state.apply_fn, state.params, None,
        test_inputs, test_targets, max_n=test_max_n,
    )
    return test_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baseline LSTM experiments")
    # Regularization
    parser.add_argument("--reg", type=str, default="none",
                        choices=["none", "l1", "l2"],
                        help="Regularization type")
    parser.add_argument("--reg_lambda", type=float, default=0.1,
                        help="Regularization coefficient")
    # Architecture
    parser.add_argument("--hidden_size", type=int, default=3,
                        help="LSTM hidden size")
    # Data
    parser.add_argument("--num_train", type=int, default=1000,
                        help="Number of training strings")
    parser.add_argument("--p", type=float, default=0.3,
                        help="PCFG termination probability")
    # Training
    parser.add_argument("--epochs", type=int, default=20000,
                        help="Max training epochs (20000 in Lan et al.)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (0.001 in Lan et al.)")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Batch size (0 = full batch)")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate")
    parser.add_argument("--early_stop", type=int, default=0,
                        help="Early stopping patience in epochs (0 = disabled)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--init", type=str, default="normal",
                        choices=["normal", "uniform"],
                        help="Weight initialization method")
    # Evaluation
    parser.add_argument("--test_max_n", type=int, default=1500,
                        help="Max n for test set")
    parser.add_argument("--eval_every", type=int, default=100,
                        help="Evaluate every N epochs")
    parser.add_argument("--log_every", type=int, default=200,
                        help="Log every N epochs")

    args = parser.parse_args()
    np.random.seed(args.seed)

    run_dir = make_run_dir(args)

    _tee = TeeLogger(run_dir / "train.log")
    _tee.__enter__()
    try:
        _main_inner(args, run_dir)
    finally:
        _tee.__exit__(None, None, None)


def _main_inner(args, run_dir):
    """Inner main logic."""
    print("=" * 60)
    print("Baseline LSTM experiment")
    print(f"  reg={args.reg}, lambda={args.reg_lambda}")
    print(f"  dropout={args.dropout}, early_stop={args.early_stop}")
    print(f"  init={args.init}, seed={args.seed}")
    print(f"  Run directory: {run_dir}")
    print("=" * 60)

    # Golden baseline
    golden_mdl = golden_mdl_score(p=args.p)
    print(f"\nGolden |H| = {golden_mdl['total_bits']} bits")

    # Generate data
    print(f"\nGenerating data (num_train={args.num_train}, p={args.p})...")
    train_inputs, train_targets = make_anbn_dataset(
        num_strings=args.num_train, p=args.p, seed=args.seed,
    )
    train_max_n = get_train_max_n(train_inputs)
    print(f"  Training strings: {len(train_inputs)}, max n={train_max_n}")

    # 95/5 split
    n_train = int(len(train_inputs) * 0.95)
    train_inputs = train_inputs[:n_train]
    train_targets = train_targets[:n_train]
    print(f"  After split: {len(train_inputs)} train")

    # Validation and test sets
    val_inputs, val_targets = make_validation_set(
        train_max_n, val_max_n=71, val_min_n=22,
    )
    print(f"  Validation: {len(val_inputs)} strings (n={train_max_n+1}..71)")

    test_inputs, test_targets = make_test_set(max_n=args.test_max_n)
    print(f"  Test: {len(test_inputs)} strings (n=1..{args.test_max_n})")

    # Pad training data
    x_train, y_train, mask_train = sequences_to_padded_arrays(
        train_inputs, train_targets,
    )
    print(f"  Max sequence length: {x_train.shape[1]}")

    # Create model
    model = BaselineLSTM(
        hidden_size=args.hidden_size,
        input_size=NUM_SYMBOLS,
        output_size=NUM_SYMBOLS,
        dropout_rate=args.dropout,
    )

    rng = jrandom.PRNGKey(args.seed)

    # Train
    state, best_params, best_val_loss = run_training(
        args, model, x_train, y_train, mask_train,
        val_inputs, val_targets, rng, run_dir=run_dir,
    )

    # Evaluate on test set using best params
    if best_params is not None:
        state = state.replace(params=best_params)

    test_result = evaluate_baseline_on_test(
        state, test_inputs, test_targets, args.test_max_n,
    )

    # Compute MDL score of trained network
    mdl_score = compute_baseline_mdl_score(
        state.params, args.hidden_size,
    )

    # Grammar-weighted evaluation metrics (Abudy et al. 2025 convention)
    def _baseline_fwd(x):
        logits, _ = state.apply_fn({"params": state.params}, x, train=False)
        return logits

    print("\nComputing grammar-weighted |D:H| metrics...")
    test_dh = compute_grammar_weighted_nll_bits(
        _baseline_fwd, max_n=args.test_max_n, p=args.p, batch_size=64,
    )
    train_dh = compute_train_dh(_baseline_fwd, train_inputs, train_targets)
    golden_opt = compute_optimal_dh_test(max_n=args.test_max_n, p=args.p)
    golden_train = compute_optimal_dh_train(train_inputs, train_targets, p=args.p)

    test_data_dh = test_dh["data_dh_bits"]
    train_data_dh = train_dh["train_dh_data_bits"]
    delta_test = compute_delta_pct(test_data_dh, golden_opt["data_dh_bits"])
    delta_train = compute_delta_pct(train_data_dh, golden_train["train_dh_data_bits"])

    # Summary
    our_gen_n = test_result["gen_n"]
    our_n_perfect = test_result["n_perfect"]
    first_failure = test_result["first_failure_n"]

    print(f"\n  Perfect strings: {our_n_perfect}/{args.test_max_n}")
    print(f"  Generalization range: n=1..{our_gen_n}")
    if first_failure:
        print(f"  First failure at n={first_failure}")

    print(f"\n  MDL score (rationalized): {mdl_score['total_bits']} bits "
          f"({mdl_score['arch_bits']} arch + {mdl_score['weight_bits']} weights)")
    print(f"  Test |D:H|: {test_data_dh:.4f} bits  (Δ_test = {delta_test:+.1f}%)")
    print(f"  Train |D:H|: {train_data_dh:.2f} bits  (Δ_train = {delta_train:+.1f}%)")

    # Comparison table
    reg_label = f"CE+{args.reg.upper()}" if args.reg != "none" else "CE only"
    if args.early_stop > 0:
        reg_label += f"+ES({args.early_stop})"
    if args.dropout > 0:
        reg_label += f"+DO({args.dropout})"

    abudy_table = format_abudy_comparison_table(
        our_test_data_dh=test_data_dh,
        our_train_data_dh=train_data_dh,
        our_h_bits=mdl_score["total_bits"],
        opt_test_data_dh=golden_opt["data_dh_bits"],
        opt_train_data_dh=golden_train["train_dh_data_bits"],
        golden_h_bits=golden_opt["h_bits"],
    )
    print(f"\n{abudy_table}")

    # Save results
    mode_str = f"baseline_{args.reg}"
    if args.early_stop > 0:
        mode_str += f"_es{args.early_stop}"
    if args.dropout > 0:
        mode_str += f"_do{args.dropout}"

    results = {
        "mode": mode_str,
        "gen_n": int(our_gen_n),
        "n_perfect": int(our_n_perfect),
        "total_mdl_bits": int(mdl_score["total_bits"]),
        "arch_bits": int(mdl_score["arch_bits"]),
        "weight_bits": int(mdl_score["weight_bits"]),
        "first_failure_n": first_failure,
        "mean_det_accuracy": float(test_result["mean_accuracy"]),
        "best_val_loss": float(best_val_loss),
        "reg": args.reg,
        "reg_lambda": args.reg_lambda,
        "test_data_dh_bits": float(test_data_dh),
        "train_data_dh_bits": float(train_data_dh),
        "delta_test_pct": float(delta_test),
        "delta_train_pct": float(delta_train),
        "golden_test_data_dh_bits": float(golden_opt["data_dh_bits"]),
        "golden_train_data_dh_bits": float(golden_train["train_dh_data_bits"]),
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {run_dir}/")


if __name__ == "__main__":
    main()
