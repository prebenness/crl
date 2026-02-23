"""Differentiable MDL for Rational-Weight Networks.

Reproduces the a^n b^n experiment from Lan et al. (2024) using a
differentiable relaxation of the discrete MDL objective via Gumbel-Softmax
straight-through estimation over a categorical weight parameterization.

Lan et al. setup:
    - Language: a^n b^n with PCFG p=0.3
    - Architecture: LSTM hidden_size=3, input/output size=3
    - Training: 1000 strings (950 train, 50 validation)
    - Test: all a^n b^n for 1 <= n <= 1500
    - Metric: deterministic accuracy (correct predictions from first b onward)

Our approach:
    - Each weight parameterized as categorical over finite rational grid S
    - Gumbel-Softmax ST for data term gradients
    - Coding term (expected codelength) computed exactly under categorical dist
    - Entropy bonus annealed via temperature tau = 1/beta

Modes:
    --mode basic   : basic categorical MDL (Sections 2-5 of proposal)
    --mode shared  : shared-weight extension with adaptive prior (Section 8)

Usage:
    python differentiable_mdl.py [--n_max 10] [--m_max 10] [--epochs 5000] ...
"""

import argparse
import time
from fractions import Fraction

import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jrandom

from src.mdl.coding import (
    grid_values_and_codelengths,
    build_rational_grid,
    rational_codelength,
    integer_code_length,
)
from src.mdl.data import (
    make_anbn_dataset,
    make_test_set,
    make_validation_set,
    sequences_to_padded_arrays,
    NUM_SYMBOLS,
    SYMBOL_A,
)
from src.mdl.lstm import GumbelSoftmaxLSTM, decode_weights
from src.mdl.training import (
    create_mdl_state,
    make_train_step,
    evaluate_deterministic_accuracy,
)
from src.mdl.golden import (
    build_golden_network_params,
    golden_forward,
    golden_mdl_score,
    evaluate_golden_network,
)
from src.mdl.shared_weights import (
    compute_p_base,
    create_shared_mdl_state,
    make_shared_train_step,
)


def get_train_max_n(inputs):
    """Find the maximum n in the training set."""
    max_n = 0
    for inp in inputs:
        n = sum(1 for s in inp if s == SYMBOL_A)
        max_n = max(max_n, n)
    return max_n


def anneal_tau(epoch, total_epochs, tau_start, tau_end):
    """Exponential temperature annealing: tau_start -> tau_end over training."""
    progress = epoch / max(total_epochs - 1, 1)
    log_tau = jnp.log(tau_start) + progress * (jnp.log(tau_end) - jnp.log(tau_start))
    return jnp.exp(log_tau)


def compute_discrete_mdl_score(eval_params, grid, grid_values):
    """Compute the discrete MDL hypothesis codelength from trained logits.

    Args:
        eval_params: params dict with "logits" key
        grid: list of Fraction objects (the rational grid)
        grid_values: float array of grid values

    Returns:
        total_hyp_bits: total hypothesis codelength in bits
    """
    logits = eval_params["logits"]
    idx = jnp.argmax(logits, axis=-1)
    total_hyp_bits = 0
    for i in range(len(idx)):
        w_frac = grid[int(idx[i])]
        total_hyp_bits += rational_codelength(w_frac)
    return total_hyp_bits


def evaluate_golden_baseline(test_max_n, p):
    """Run golden network evaluation and MDL scoring."""
    print("\n" + "=" * 60)
    print("GOLDEN NETWORK BASELINE (Lan et al. 2024)")
    print("=" * 60)

    # MDL score of the golden network
    mdl = golden_mdl_score(p=p)
    print(f"  Golden |H| = {mdl['total_bits']} bits "
          f"({mdl['arch_bits']} arch + {mdl['weight_bits']} weights, "
          f"{mdl['n_nonzero']} non-zero)")

    # Accuracy on test set
    print(f"  Evaluating golden network on n=1..{test_max_n}...")
    golden_result = evaluate_golden_network(max_n=test_max_n, p=p)
    golden_acc = golden_result["mean_accuracy"]
    print(f"  Golden det. accuracy: {golden_acc*100:.1f}%")
    if golden_result["all_correct"]:
        print(f"  All correct: YES")
    else:
        print(f"  First failure at n={golden_result['first_failure_n']}")

    return mdl, golden_result


def _compute_discrete_hyp_bits(params, grid, grid_values):
    """Compute discrete |H| from current logits (argmax weights)."""
    logits = params["logits"] if "logits" in params else params
    idx = jnp.argmax(logits, axis=-1)
    total = 0
    for i in range(len(idx)):
        total += rational_codelength(grid[int(idx[i])])
    return total


def _run_epoch(state, x_train, y_train, mask_train, N, bs, rng, train_step):
    """Run one training epoch, return updated state and aggregated metrics."""
    rng, perm_rng = jrandom.split(rng)
    perm = jrandom.permutation(perm_rng, N)
    n_batches = max(N // bs, 1)

    epoch_data_cl = 0.0
    epoch_hyp_cl = 0.0
    epoch_entropy = 0.0

    for b in range(n_batches):
        idx = perm[b * bs:(b + 1) * bs] if N >= bs else jnp.arange(N)
        xb, yb, mb = x_train[idx], y_train[idx], mask_train[idx]

        rng, batch_rng = jrandom.split(rng)
        state, loss, aux = train_step(state, xb, yb, mb, batch_rng)

        epoch_data_cl += float(aux["data_codelength"])
        epoch_hyp_cl += float(aux["hyp_codelength"])
        epoch_entropy += float(aux["entropy"])

    return state, rng, {
        "data_cl": epoch_data_cl / n_batches,
        "hyp_cl": epoch_hyp_cl / n_batches,
        "entropy": epoch_entropy / n_batches,
    }


def run_training_basic(args, model, grid_values, grid_codelengths,
                       x_train, y_train, mask_train,
                       val_inputs, val_targets, rng, grid):
    """Run two-phase training with the basic MDL objective.

    Phase 1 (warmup): Continuous soft relaxation, zero-variance gradients.
    Phase 2 (ST):     Gumbel-Softmax straight-through, multi-sample variance reduction.
    """
    rng, init_rng = jrandom.split(rng)
    N = x_train.shape[0]
    max_seq_len = x_train.shape[1]
    bs = args.batch_size if args.batch_size > 0 else N

    state = create_mdl_state(
        init_rng, model,
        seq_len=max_seq_len,
        batch_size=min(bs, N),
        lr=args.lr,
        tau_init=args.tau_start,
    )

    print(f"  Number of LSTM+output parameters: {state.params['logits'].shape[0]}")
    print(f"  Logit array shape: {state.params['logits'].shape}")

    # Create train steps for both phases
    warmup_train_step = make_train_step(
        args.mdl_lambda, n_train=N, n_samples=1, soft_forward=True,
    )
    st_train_step = make_train_step(
        args.mdl_lambda, n_train=N, n_samples=args.n_samples, soft_forward=False,
    )

    warmup_epochs = args.warmup_epochs
    total_epochs = args.epochs
    st_epochs = total_epochs - warmup_epochs

    # Warmup anneals tau from tau_start down to tau_start (staying constant
    # is useless at high tau). Instead, anneal across the full range during
    # warmup too: tau_start -> tau_end, then ST phase continues from tau_end
    # with Gumbel noise for discretization.
    # Actually: warmup anneals tau_start -> tau_start (same as ST start),
    # then ST anneals tau_start -> tau_end. But the real issue is tau_start
    # is too high. We anneal over BOTH phases: tau_start -> tau_end.
    print(f"\n  Phase 1 (warmup): {warmup_epochs} epochs, soft forward (no Gumbel)")
    print(f"  Phase 2 (ST):     {st_epochs} epochs, {args.n_samples} Gumbel samples")
    print(f"  tau: {args.tau_start} -> {args.tau_end} (annealed across both phases)")
    print(f"  lr={args.lr}, lambda={args.mdl_lambda}, batch_size={bs}")
    print("-" * 70)

    best_val_n_perfect = -1
    best_params = None

    t0 = time.time()
    for epoch in range(total_epochs):
        is_warmup = epoch < warmup_epochs
        phase_name = "warmup" if is_warmup else "ST"

        # Anneal tau across BOTH phases
        tau = anneal_tau(epoch, total_epochs, args.tau_start, args.tau_end)

        if is_warmup:
            train_step = warmup_train_step
        else:
            train_step = st_train_step

        state = state.replace(tau=tau)

        state, rng, metrics = _run_epoch(
            state, x_train, y_train, mask_train, N, bs, rng, train_step,
        )

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            # Compute discrete |H| periodically
            hyp_disc = _compute_discrete_hyp_bits(state.params, grid, grid_values)
            print(
                f"Epoch {epoch+1:5d} | data_cl={metrics['data_cl']:8.1f} | "
                f"|H|={metrics['hyp_cl']:7.1f}(disc:{hyp_disc:4d}) | "
                f"H={metrics['entropy']:6.1f} | "
                f"tau={float(tau):.4f} | {phase_name}"
            )

        if (epoch + 1) % args.eval_every == 0:
            val_result = evaluate_deterministic_accuracy(
                state.apply_fn, state.params, grid_values,
                val_inputs, val_targets,
            )
            n_perfect = val_result["n_perfect"]
            n_val = len(val_inputs)
            gen_n = val_result["gen_n"]
            print(f"  [VAL] perfect={n_perfect}/{n_val} gen_n={gen_n}"
                  f" (first_fail={val_result['first_failure_n']})")
            if n_perfect > best_val_n_perfect:
                best_val_n_perfect = n_perfect
                best_params = jax.tree.map(lambda x: x.copy(), state.params)

    elapsed = time.time() - t0
    print("-" * 70)
    print(f"Training complete in {elapsed:.1f}s")

    return state, best_params, best_val_n_perfect


def _run_epoch_shared(state, x_train, y_train, mask_train, N, bs, rng,
                      train_step, p_base):
    """Run one shared-weight training epoch."""
    rng, perm_rng = jrandom.split(rng)
    perm = jrandom.permutation(perm_rng, N)
    n_batches = max(N // bs, 1)

    epoch_data_cl = 0.0
    epoch_hyp_cl = 0.0
    epoch_entropy = 0.0

    for b in range(n_batches):
        idx = perm[b * bs:(b + 1) * bs] if N >= bs else jnp.arange(N)
        xb, yb, mb = x_train[idx], y_train[idx], mask_train[idx]

        rng, batch_rng = jrandom.split(rng)
        state, loss, aux = train_step(state, xb, yb, mb, batch_rng, p_base)

        epoch_data_cl += float(aux["data_codelength"])
        epoch_hyp_cl += float(aux["hyp_codelength"])
        epoch_entropy += float(aux["entropy"])

    return state, rng, {
        "data_cl": epoch_data_cl / n_batches,
        "hyp_cl": epoch_hyp_cl / n_batches,
        "entropy": epoch_entropy / n_batches,
    }


def run_training_shared(args, model, grid_values, grid_codelengths,
                        x_train, y_train, mask_train,
                        val_inputs, val_targets, rng, grid):
    """Run two-phase training with the shared-weight MDL objective."""
    rng, init_rng = jrandom.split(rng)
    N = x_train.shape[0]
    max_seq_len = x_train.shape[1]
    bs = args.batch_size if args.batch_size > 0 else N

    state = create_shared_mdl_state(
        init_rng, model, grid_values,
        seq_len=max_seq_len,
        batch_size=min(bs, N),
        lr=args.lr,
        tau_init=args.tau_start,
    )

    print(f"  Number of LSTM+output parameters: {state.params['logits'].shape[0]}")
    print(f"  Logit array shape: {state.params['logits'].shape}")
    print(f"  Phi logits shape: {state.params['phi_logits'].shape}")

    p_base = compute_p_base(grid_values)

    warmup_train_step = make_shared_train_step(
        args.lambda1, args.lambda2, args.epsilon, n_train=N,
        n_samples=1, soft_forward=True,
    )
    st_train_step = make_shared_train_step(
        args.lambda1, args.lambda2, args.epsilon, n_train=N,
        n_samples=args.n_samples, soft_forward=False,
    )

    warmup_epochs = args.warmup_epochs
    total_epochs = args.epochs
    st_epochs = total_epochs - warmup_epochs

    print(f"\n  Phase 1 (warmup): {warmup_epochs} epochs, soft forward (no Gumbel)")
    print(f"  Phase 2 (ST):     {st_epochs} epochs, {args.n_samples} Gumbel samples")
    print(f"  tau: {args.tau_start} -> {args.tau_end} (annealed across both phases)")
    print(f"  lr={args.lr}, lambda1={args.lambda1}, lambda2={args.lambda2}, "
          f"eps={args.epsilon}, batch_size={bs}")
    print("-" * 70)

    best_val_n_perfect = -1
    best_params = None

    t0 = time.time()
    for epoch in range(total_epochs):
        is_warmup = epoch < warmup_epochs
        phase_name = "warmup" if is_warmup else "ST"

        # Anneal tau across BOTH phases
        tau = anneal_tau(epoch, total_epochs, args.tau_start, args.tau_end)

        if is_warmup:
            train_step = warmup_train_step
        else:
            train_step = st_train_step

        state = state.replace(tau=tau)

        state, rng, metrics = _run_epoch_shared(
            state, x_train, y_train, mask_train, N, bs, rng,
            train_step, p_base,
        )

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            hyp_disc = _compute_discrete_hyp_bits(state.params, grid, grid_values)
            print(
                f"Epoch {epoch+1:5d} | data_cl={metrics['data_cl']:8.1f} | "
                f"|H|={metrics['hyp_cl']:7.1f}(disc:{hyp_disc:4d}) | "
                f"H={metrics['entropy']:6.1f} | "
                f"tau={float(tau):.4f} | {phase_name}"
            )

        if (epoch + 1) % args.eval_every == 0:
            model_params = {"logits": state.params["logits"]}
            val_result = evaluate_deterministic_accuracy(
                state.apply_fn, model_params, grid_values,
                val_inputs, val_targets,
            )
            n_perfect = val_result["n_perfect"]
            n_val = len(val_inputs)
            gen_n = val_result["gen_n"]
            print(f"  [VAL] perfect={n_perfect}/{n_val} gen_n={gen_n}"
                  f" (first_fail={val_result['first_failure_n']})")
            if n_perfect > best_val_n_perfect:
                best_val_n_perfect = n_perfect
                best_params = jax.tree.map(lambda x: x.copy(), state.params)

    elapsed = time.time() - t0
    print("-" * 70)
    print(f"Training complete in {elapsed:.1f}s")

    return state, best_params, best_val_n_perfect


def main():
    parser = argparse.ArgumentParser(description="Differentiable MDL experiment")
    # Mode
    parser.add_argument("--mode", type=str, default="basic",
                        choices=["basic", "shared"],
                        help="basic = Sections 2-5, shared = Section 8")
    # Grid parameters
    parser.add_argument("--n_max", type=int, default=10,
                        help="Max numerator in rational grid")
    parser.add_argument("--m_max", type=int, default=10,
                        help="Max denominator in rational grid")
    # Architecture
    parser.add_argument("--hidden_size", type=int, default=3,
                        help="LSTM hidden size (3 matches Lan et al.)")
    # Data
    parser.add_argument("--num_train", type=int, default=1000,
                        help="Number of training strings (1000 in Lan et al.)")
    parser.add_argument("--p", type=float, default=0.3,
                        help="PCFG termination probability")
    parser.add_argument("--data_seed", type=int, default=0,
                        help="Seed for data generation")
    # Training
    parser.add_argument("--epochs", type=int, default=5000,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--mdl_lambda", type=float, default=1.0,
                        help="MDL trade-off parameter (basic mode)")
    parser.add_argument("--tau_start", type=float, default=1.0,
                        help="Initial Gumbel-Softmax temperature")
    parser.add_argument("--tau_end", type=float, default=0.01,
                        help="Final Gumbel-Softmax temperature")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size (0 = full batch)")
    parser.add_argument("--n_samples", type=int, default=16,
                        help="Gumbel samples per step in ST phase (variance reduction)")
    parser.add_argument("--warmup_epochs", type=int, default=500,
                        help="Soft warmup epochs before switching to ST")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    # Shared-weight mode parameters
    parser.add_argument("--lambda1", type=float, default=1.0,
                        help="Weight-sharing KL weight (shared mode)")
    parser.add_argument("--lambda2", type=float, default=1.0,
                        help="Dictionary cost KL weight (shared mode)")
    parser.add_argument("--epsilon", type=float, default=1e-6,
                        help="Min probability for adaptive prior (shared mode)")
    # Evaluation
    parser.add_argument("--test_max_n", type=int, default=1500,
                        help="Max n for test set")
    parser.add_argument("--eval_every", type=int, default=100,
                        help="Evaluate every N epochs")
    parser.add_argument("--log_every", type=int, default=50,
                        help="Log training metrics every N epochs")

    args = parser.parse_args()

    print("=" * 60)
    print("Differentiable MDL for a^n b^n")
    print(f"Mode: {args.mode}")
    print("=" * 60)

    # --- Golden network baseline ---
    golden_mdl, golden_result = evaluate_golden_baseline(args.test_max_n, args.p)

    # --- Build rational grid ---
    print(f"\nBuilding rational grid with n_max={args.n_max}, m_max={args.m_max}...")
    grid_values, grid_codelengths = grid_values_and_codelengths(
        args.n_max, args.m_max,
    )
    M = len(grid_values)
    grid = build_rational_grid(args.n_max, args.m_max)
    print(f"  Grid size |S| = {M}")
    print(f"  Grid range: [{grid_values.min():.4f}, {grid_values.max():.4f}]")
    print(f"  Codelength range: [{grid_codelengths.min():.0f}, {grid_codelengths.max():.0f}] bits")

    # --- Generate data ---
    print(f"\nGenerating a^n b^n data (num_train={args.num_train}, p={args.p})...")
    train_inputs, train_targets = make_anbn_dataset(
        num_strings=args.num_train, p=args.p, seed=args.data_seed,
    )
    train_max_n = get_train_max_n(train_inputs)
    print(f"  Training strings: {len(train_inputs)}")
    print(f"  Max n in training: {train_max_n}")

    # Train/val split: 95/5 as in Lan et al.
    n_train = int(len(train_inputs) * 0.95)
    train_inputs = train_inputs[:n_train]
    train_targets = train_targets[:n_train]
    print(f"  After 95/5 split: {len(train_inputs)} train")

    # Structured validation set (all strings with train_max_n < n <= 71)
    val_inputs, val_targets = make_validation_set(train_max_n, val_max_n=71)
    print(f"  Structured val set: {len(val_inputs)} strings (n={train_max_n+1}..71)")

    # Test set
    test_inputs, test_targets = make_test_set(max_n=args.test_max_n)
    print(f"  Test set: {len(test_inputs)} strings (n=1..{args.test_max_n})")

    # Pad training data
    x_train, y_train, mask_train = sequences_to_padded_arrays(
        train_inputs, train_targets,
    )
    max_seq_len = x_train.shape[1]
    print(f"  Max sequence length (training): {max_seq_len}")

    # --- Create model ---
    print(f"\nCreating GumbelSoftmaxLSTM (hidden={args.hidden_size}, grid_size={M})...")
    model = GumbelSoftmaxLSTM(
        hidden_size=args.hidden_size,
        input_size=NUM_SYMBOLS,
        output_size=NUM_SYMBOLS,
        grid_values=grid_values,
        grid_codelengths=grid_codelengths,
    )

    rng = jrandom.PRNGKey(args.seed)

    # --- Training ---
    if args.mode == "basic":
        state, best_params, best_val_n_perfect = run_training_basic(
            args, model, grid_values, grid_codelengths,
            x_train, y_train, mask_train,
            val_inputs, val_targets, rng, grid,
        )
    else:
        state, best_params, best_val_n_perfect = run_training_shared(
            args, model, grid_values, grid_codelengths,
            x_train, y_train, mask_train,
            val_inputs, val_targets, rng, grid,
        )

    # --- Final evaluation ---
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    eval_params = best_params if best_params is not None else state.params

    # For shared mode, extract model-only params for evaluation
    if args.mode == "shared":
        eval_model_params = {"logits": eval_params["logits"]}
    else:
        eval_model_params = eval_params

    # Decode discrete weights
    discrete_weights = decode_weights({"params": eval_model_params}, grid_values)
    n_nonzero = int(jnp.sum(discrete_weights != 0))
    print(f"\nDiscrete weights ({len(discrete_weights)} total, {n_nonzero} non-zero)")

    # Compute discrete MDL score
    total_hyp_bits = compute_discrete_mdl_score(eval_model_params, grid, grid_values)
    arch_bits = integer_code_length(args.hidden_size)
    total_mdl_bits = arch_bits + total_hyp_bits
    print(f"  Discrete |H|: {total_mdl_bits} bits ({arch_bits} arch + {total_hyp_bits} weights)")

    # Test accuracy
    test_max_n = args.test_max_n
    print(f"\nEvaluating on test set (n=1..{test_max_n})...")
    test_result = evaluate_deterministic_accuracy(
        state.apply_fn, eval_model_params, grid_values,
        test_inputs, test_targets, max_n=test_max_n,
    )
    our_n_perfect = test_result["n_perfect"]
    our_gen_n = test_result["gen_n"]
    print(f"  Perfect strings: {our_n_perfect}/{test_max_n}")
    print(f"  Generalisation range: n=1..{our_gen_n}")
    if not test_result["all_correct"]:
        print(f"  First failure at n={test_result['first_failure_n']}")

    # --- Summary comparison table ---
    # Trivial baseline: always predict 'b'. Gets (n-1)/n per string,
    # 0 perfect, and |H|=540 bits (108 zero weights * 5 bits each).
    n_params = len(discrete_weights)
    trivial_hyp = n_params * 5  # all-zero weights, 5 bits each
    trivial_mdl = arch_bits + trivial_hyp

    golden_gen_n = test_max_n if golden_result["all_correct"] else (golden_result["first_failure_n"] - 1)
    golden_n_perfect = test_max_n if golden_result["all_correct"] else 0

    print("\n" + "=" * 70)
    print("COMPARISON TABLE (cf. Lan et al. 2024, Table 1)")
    print("=" * 70)
    print(f"{'Method':<30} {'|H| (bits)':>10} {'gen_n':>8} {'n_perfect':>16}")
    print("-" * 70)
    print(f"{'Lan et al. golden':<30} {golden_mdl['total_bits']:>10d} {golden_gen_n:>8d} "
          f"{golden_n_perfect:>10d}/{test_max_n}")
    print(f"{'Lan et al. backprop (reported)':<30} {'---':>10} {'---':>8} {'---':>16}")
    print(f"{'Trivial (always-b)':<30} {trivial_mdl:>10d} {0:>8d} "
          f"{0:>10d}/{test_max_n}")
    mode_name = "Ours (basic MDL)" if args.mode == "basic" else "Ours (shared MDL)"
    print(f"{mode_name:<30} {total_mdl_bits:>10d} {our_gen_n:>8d} "
          f"{our_n_perfect:>10d}/{test_max_n}")
    print("=" * 70)


if __name__ == "__main__":
    main()
