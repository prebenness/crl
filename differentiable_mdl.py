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
    - Entropy bonus (subtracted): tau * H(pi), annealed via temperature tau = 1/beta

Modes:
    --mode basic   : basic categorical MDL (Sections 2-5 of proposal)
    --mode shared  : shared-weight extension with adaptive prior (Section 8)

Usage:
    python differentiable_mdl.py config/anbn_mdl/basic_train.yaml
    python differentiable_mdl.py config/anbn_mdl/basic_train.yaml --epochs 50000
    python differentiable_mdl.py [--n_max 10] [--m_max 10] [--epochs 5000] ...
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path

# XLA_FLAGS must be set before JAX/XLA initialises - we can't wait for argparse.
# We check sys.argv directly so the --deterministic flag still controls it.
if "--deterministic" in sys.argv:
    _xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_deterministic_ops" not in _xla_flags:
        os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_deterministic_ops=true").strip()

import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jrandom
import yaml

# Persist compiled XLA kernels across runs to avoid redundant JIT compilation.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

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
    make_fused_epoch_fn,
    make_fused_epoch_fn_fixed_tau,
    evaluate_deterministic_accuracy,
    anneal_tau,
    compute_data_nll_bits_smoothed,
)
from src.mdl.golden import (
    build_golden_network_params,
    golden_forward,
    golden_mdl_score,
    evaluate_golden_network,
    estimate_golden_float32_limit,
)
from src.mdl.shared_weights import (
    compute_p_base,
    create_shared_mdl_state,
    make_shared_train_step,
)
from src.mdl.analysis import (
    analyze_model,
    _check_single_n,
    extract_weights,
    evaluate_range_f64,
    find_failure_n,
)
from src.mdl.evaluation import (
    compute_grammar_weighted_nll_bits,
    compute_optimal_dh_test,
    compute_optimal_dh_train,
    compute_train_dh,
    compute_trained_h_bits,
    compute_delta_pct,
    evaluate_golden_under_regularisers,
    format_abudy_comparison_table,
    format_golden_regulariser_table,
)
from src.utils.checkpointing import (
    TeeLogger,
    save_checkpoint,
    load_checkpoint,
    save_results,
    save_config,
    make_experiment_dir,
    checkpoint_path,
    utc_timestamp,
)


# ---------------------------------------------------------------------------
# Run management: directories, logging, checkpointing
# ---------------------------------------------------------------------------


def make_run_dir(args, suffix=""):
    """Create a timestamped results directory and write config.json.

    Name format:
        results/anbn_mdl/YYYYMMDD_HHMMSS_MODE_eEPOCHS_lrLR_lamLAMBDA_gNxM_sSEED/
    """
    ts = utc_timestamp()
    lam = args.mdl_lambda if args.mode == "basic" else args.lambda1
    cfg_tag = f"_cfg{args.config_name}" if args.config_name else ""
    name = (
        f"{ts}_{args.mode}_e{args.epochs}"
        f"_lr{args.lr}_lam{lam}"
        f"_g{args.n_max}x{args.m_max}_s{args.seed}{cfg_tag}{suffix}"
    )
    run_dir = make_experiment_dir("anbn_mdl", name)
    save_config(run_dir, vars(args))
    return run_dir


def load_run_config(run_dir):
    """Load config.json from a run directory into an argparse.Namespace."""
    with open(Path(run_dir) / "config.json") as f:
        return argparse.Namespace(**json.load(f))


def save_checkpoint_meta(run_dir, epoch, best_val_n_perfect, best_checkpoint_epoch=None):
    """Write a small sidecar recording the last completed epoch."""
    meta_path = checkpoint_path(run_dir, "meta.json")
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    meta["last_epoch"] = int(epoch)
    meta["best_val_n_perfect"] = int(best_val_n_perfect)
    if best_checkpoint_epoch is not None:
        meta["best_checkpoint_epoch"] = int(best_checkpoint_epoch)
    elif "best_checkpoint_epoch" in meta:
        meta["best_checkpoint_epoch"] = int(meta["best_checkpoint_epoch"])
    with open(checkpoint_path(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _format_val_summary(val_result, val_inputs):
    """Convert relative validation metrics into absolute n values."""
    gen_n = val_result["gen_n"]
    fail_n = val_result["first_failure_n"]
    n_val = len(val_inputs)
    if not val_inputs:
        val_desc = f"perfect_prefix_len={gen_n}"
        fail_abs_n = None
    else:
        val_start_n = len(val_inputs[0]) // 2
        val_end_n = len(val_inputs[-1]) // 2
        last_perfect_n = val_start_n + gen_n - 1 if gen_n > 0 else val_start_n - 1
        fail_abs_n = val_start_n + fail_n - 1 if fail_n is not None else None
        val_desc = (
            f"contig_ok_through_n={last_perfect_n} "
            f"(prefix_len={gen_n}, val_n={val_start_n}..{val_end_n})"
        )
    return {
        "n_val": n_val,
        "val_desc": val_desc,
        "fail_abs_n": fail_abs_n,
    }


def _zero_adam_moments(opt_state):
    """Zero mu and nu in an Adam optimizer state, preserving count."""
    adam_state, scale_state = opt_state
    zeroed = adam_state._replace(
        mu=jax.tree.map(jnp.zeros_like, adam_state.mu),
        nu=jax.tree.map(jnp.zeros_like, adam_state.nu),
    )
    return (zeroed, scale_state)


def _maybe_restart(state, best, epochs_since_best, restart_patience):
    """Check stall and restart from best checkpoint if patience exceeded.

    Returns (state, epochs_since_best, did_restart).
    """
    if restart_patience <= 0 or best['params'] is None:
        return state, epochs_since_best, False
    if epochs_since_best < restart_patience:
        return state, epochs_since_best, False
    new_opt_state = _zero_adam_moments(best['opt_state'])
    state = state.replace(params=best['params'], opt_state=new_opt_state)
    return state, 0, True


def _should_update_best(n_perfect, gen_n, complexity_bits,
                        best_val_n_perfect, best_val_gen_n,
                        best_complexity_bits):
    """Prefer better validation coverage, then lower complexity."""
    if n_perfect != best_val_n_perfect:
        return n_perfect > best_val_n_perfect
    if gen_n != best_val_gen_n:
        return gen_n > best_val_gen_n
    return complexity_bits < best_complexity_bits


def get_train_max_n(inputs):
    """Find the maximum n in the training set."""
    max_n = 0
    for inp in inputs:
        n = sum(1 for s in inp if s == SYMBOL_A)
        max_n = max(max_n, n)
    return max_n


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
    """Run golden network evaluation and MDL scoring.

    For moderate test_max_n, evaluate exactly with the batched JAX forward.
    For larger ranges, switch to a sparse finite-precision benchmark instead
    of assuming the handcrafted network is perfect for all n.
    """
    print("\n" + "=" * 60)
    print("GOLDEN NETWORK BASELINE (Lan et al. 2024)")
    print("=" * 60)

    # MDL score of the golden network
    mdl = golden_mdl_score(p=p)
    print(f"  Golden |H| = {mdl['total_bits']} bits "
          f"({mdl['arch_bits']} arch + {mdl['weight_bits']} weights, "
          f"{mdl['n_nonzero']} non-zero)")

    if test_max_n <= 1500:
        # Evaluate normally for standard test range
        print(f"  Evaluating golden network on n=1..{test_max_n}...")
        golden_result = evaluate_golden_network(max_n=test_max_n, p=p)
        golden_acc = golden_result["mean_accuracy"]
        print(f"  Golden det. accuracy: {golden_acc*100:.1f}%")
        if golden_result["all_correct"]:
            print(f"  All correct: YES")
        else:
            print(f"  First failure at n={golden_result['first_failure_n']}")
    else:
        # For large n, use the exact float32 counter limit rather than the
        # idealized mathematical claim.
        print(f"  Estimating float32 golden boundary up to n={test_max_n}...")
        golden_result = estimate_golden_float32_limit(max_n=test_max_n, p=p)
        print(f"  Finite-precision golden range: n=1..{golden_result['max_correct_n']}")
        if golden_result["all_correct"]:
            print("  All correct in the requested test range: YES")
        else:
            print(
                "  First finite-precision failure at "
                f"n={golden_result['first_failure_n']}"
            )
        print(f"  Probe trace: {len(golden_result['probes'])} sparse checks")

    return mdl, golden_result


def _compute_discrete_hyp_bits(params, grid_codelengths):
    """Compute discrete |H| from current logits (argmax weights).

    This is the Lan-style codelength: sum_i l(s_{argmax_i}).  In shared mode,
    this does NOT reflect the shared-objective complexity — use
    ``_compute_shared_discrete_complexity_bits`` for that.
    """
    logits = params["logits"] if "logits" in params else params
    idx = jnp.argmax(logits, axis=-1)
    return float(jnp.sum(grid_codelengths[idx]))


def _compute_shared_discrete_complexity_bits(
    params, grid_codelengths, p_base, lambda1, lambda2, epsilon,
):
    """Compute discrete shared-objective complexity at argmax convergence.

    At convergence each pi_i is concentrated on its argmax grid point, so:
        code_ce  = sum_i CE_2(one_hot_i, phi) = sum_i -log2(phi[argmax_i])
        kl_dict  = KL(phi || P_base)
        shared_complexity = lambda1 * code_ce + lambda2 * kl_dict

    Returns a dict with component breakdown.
    """
    from src.mdl.shared_weights import epsilon_bound_simplex, _kl_divergence

    logits = params["logits"]
    idx = jnp.argmax(logits, axis=-1)

    phi = epsilon_bound_simplex(params["phi_logits"], epsilon)
    phi_at_argmax = phi[idx]                                   # (n_params,)
    code_ce = float(-jnp.sum(jnp.log2(phi_at_argmax + 1e-30)))
    kl_dict = float(_kl_divergence(phi, jnp.asarray(p_base)))

    return {
        "shared_complexity": lambda1 * code_ce + lambda2 * kl_dict,
        "code_ce": code_ce,
        "kl_dict": kl_dict,
    }


# ---------------------------------------------------------------------------
# Logging: metric legend and epoch formatting
# ---------------------------------------------------------------------------

_METRIC_LEGEND_BASIC = """\
======================================================================
  METRIC LEGEND  (paper §2 Eq. 3, §3 Eq. 4)
======================================================================

  J  —  Relaxed MDL objective J_β(α), the quantity being minimised.
         In a standard classifier this would be the total loss.
         J = L_D + λ/N · |H|_exp − τ/N · H(π).
         The three terms trade off data fit, model complexity, and
         exploration (entropy bonus).

  L_D  —  Data codelength E[L_D(θ)]: cross-entropy between the
         model's output distribution and the targets, in bits,
         averaged over sequence positions. This is the standard
         training loss — how well the model predicts the next token.
         Analogous to the CE loss in a classifier.

  λ|H|/N  —  Net regularisation = λ/N · |H|_exp − τ/N · H(π).
         Analogous to a weight-decay penalty in a standard model,
         but here it penalises the description length of the weights
         rather than their norm, minus an entropy bonus that
         encourages exploration early in training and vanishes as
         τ → 0.

  |H|_disc  —  Discrete hypothesis codelength Σᵢ ℓ(argmax πᵢ).
         The actual MDL cost you would pay to encode the current
         hard (argmax) weights as rationals. This is the number
         that matters at convergence — the "model size" in bits.
         Analogous to parameter count or L0 norm.

  |H|_exp  —  Expected hypothesis codelength Σᵢ Σₘ πᵢₘ ℓ(sₘ).
         The soft (differentiable) version of |H|_disc: a weighted
         average over the categorical distributions, used during
         training because it has gradients. Converges to |H|_disc
         as the categoricals sharpen (τ → 0).

  H(π)  —  Total categorical entropy Σᵢ H₂(πᵢ) in bits.
         Measures how "spread out" the weight distributions are.
         High early in training (exploring), drops to ~0 at
         convergence (each weight committed to one grid value).

  τ  —  Temperature = 1/β, annealed toward 0 during training.
         Controls the entropy bonus: high τ encourages exploration,
         low τ forces commitment to discrete weights.
======================================================================"""

_METRIC_LEGEND_SHARED = """\
======================================================================
  METRIC LEGEND  (paper §2 Eq. 3, §3 Eq. 4/8)
======================================================================

  J  —  Shared-weight relaxed objective J^shared_β(α, φ).
         J = L_D + λ₁/N·CE(π,φ) + λ₂/N·KL(φ‖P) − τ/N·H(π).

  L_D  —  Data codelength (cross-entropy in bits, position-averaged).
         How well the model predicts the next token. Same as the
         CE loss in a standard classifier.

  λ|H|/N  —  Net regularisation (all non-data terms combined).

  |H|_disc  —  Discrete Lan-style codelength Σᵢ ℓ(argmax πᵢ).
         Cost to encode the hard weights as rationals.

  shared  —  Shared-objective complexity at argmax convergence:
         λ₁ · CE₂(one_hot, φ) + λ₂ · KL(φ ‖ P_base).
         Replaces |H|_disc when comparing shared-mode runs.

  H(π)  —  Total categorical entropy Σᵢ H₂(πᵢ).
         Exploration measure, drops to ~0 at convergence.

  H(φ)  —  Entropy of the shared prior φ.

  τ  —  Temperature = 1/β, annealed toward 0.
======================================================================"""


def _print_metric_legend(mode):
    """Print the metric legend once at the start of training."""
    if mode == "shared":
        print(_METRIC_LEGEND_SHARED)
    else:
        print(_METRIC_LEGEND_BASIC)


def _fmt_epoch_header(epoch, tau):
    """Format the epoch header line (with leading blank line for separation)."""
    return f"\n--- Epoch {epoch:5d}  τ={float(tau):.4f} ---"


def _fmt_train_line(metrics, complexity_argmax_bits):
    """Format training metrics (two lines, vertically aligned columns).

    Three columns with right-aligned labels so = signs and numbers align:
      col1: {label:>8}={number:>8}   (J / |H|_disc / acc_hard)
      col2: {label:>8}={number:>8}   (L_D / |H|_exp / gen_n)
      col3: {label:>6}={number:>8}   (λ|H|/N / H(π) / |H|)
    4-space gaps between columns, 10-char prefix.
    """
    return (
        f"  train   {'J':>8}={float(metrics['objective_total_bits']):8.4f}"
        f"    {'L_D':>8}={float(metrics['data_nll_bits']):8.4f}"
        f"    {'λ|H|/N':>6}={float(metrics['reg_net_bits']):8.4f}\n"
        f"          {'|H|_disc':>8}={int(complexity_argmax_bits):>8}"
        f"    {'|H|_exp':>8}={float(metrics['complexity_expected_bits']):8.2f}"
        f"    {'H(π)':>6}={float(metrics['entropy_weights_bits']):8.2f}"
    )


def _fmt_train_line_shared(metrics, lan_hyp_bits, shared_disc):
    """Format training metrics for shared-weight mode."""
    return (
        f"  train   {'J':>8}={float(metrics['objective_total_bits']):8.4f}"
        f"    {'L_D':>8}={float(metrics['data_nll_bits']):8.4f}"
        f"    {'λ|H|/N':>6}={float(metrics['reg_net_bits']):8.4f}\n"
        f"          {'|H|_disc':>8}={int(lan_hyp_bits):>8}"
        f"    {'shared':>8}={shared_disc['shared_complexity']:8.2f}"
        f"  (CE={shared_disc['code_ce']:.2f}+KL={shared_disc['kl_dict']:.2f})"
        f"    {'H(π)':>6}={float(metrics['entropy_weights_bits']):8.2f}"
        f"    {'H(φ)':>6}={float(metrics['phi_entropy_bits']):8.2f}"
        f"    φ∈[{metrics['phi_min_prob']:.1e},{metrics['phi_max_prob']:.1e}]"
    )


def _fmt_val_line(n_perfect, n_val, gen_n, complexity_total_bits, best_tag=""):
    """Format a validation summary line (same column grid as train)."""
    acc_hard_val = f"{n_perfect}/{n_val}"
    col1 = f"{'acc_hard':>8}={acc_hard_val:>8}"
    col2 = f"{'gen_n':>8}={gen_n:>8}"
    col3 = f"{'|H|':>6}={int(complexity_total_bits):>8}"
    return (
        f"  val     {col1}    {col2}    {col3}"
        f"{best_tag}"
    )


def _eval_and_update_best(state, epoch, hidden_size, grid_values, grid_codelengths,
                          val_inputs, val_targets, best, run_dir):
    """Run validation eval, print summary, update best-tracking dict.

    Args:
        best: dict with keys n_perfect, gen_n,
            complexity_bits, params. Modified in-place if new best found.

    Returns:
        The best dict (same object, possibly modified).
    """
    val_result = evaluate_deterministic_accuracy(
        state.apply_fn, state.params, grid_values,
        val_inputs, val_targets,
    )
    n_perfect = val_result["n_perfect"]
    gen_n = val_result["gen_n"]
    current_complexity_bits = _compute_discrete_hyp_bits(
        state.params, grid_codelengths,
    )

    n_val = len(val_inputs)
    is_best = _should_update_best(
        n_perfect,
        gen_n,
        current_complexity_bits,
        best['n_perfect'],
        best['gen_n'],
        best['complexity_bits'],
    )
    best_tag = "  ★ NEW BEST" if is_best else ""
    print(_fmt_val_line(
        n_perfect, n_val, gen_n,
        current_complexity_bits + integer_code_length(hidden_size),
        best_tag=best_tag,
    ))
    if is_best:
        best['n_perfect'] = n_perfect
        best['gen_n'] = gen_n
        best['complexity_bits'] = current_complexity_bits
        best['params'] = jax.tree.map(lambda x: x.copy(), state.params)
        best['opt_state'] = jax.tree.map(lambda x: x.copy(), state.opt_state)
        best['epoch'] = epoch + 1
        if run_dir is not None:
            save_checkpoint(best['params'], checkpoint_path(run_dir, "best.npz"))
            save_checkpoint_meta(
                run_dir,
                epoch + 1,
                best['n_perfect'],
                best_checkpoint_epoch=epoch + 1,
            )
            print(f"              ↳ [CKPT] checkpoint saved")
    return best, is_best


def _run_epoch(state, x_train, y_train, mask_train, N, bs, rng, train_step):
    """Run one training epoch, return updated state and aggregated metrics."""
    rng, perm_rng = jrandom.split(rng)
    perm = jrandom.permutation(perm_rng, N)
    n_batches = max(N // bs, 1)

    epoch_obj = 0.0
    epoch_data_nll_bits = 0.0
    epoch_complexity_expected_bits = 0.0
    epoch_entropy_weights_bits = 0.0
    epoch_reg_complexity = 0.0
    epoch_reg_entropy_bonus = 0.0
    epoch_reg_net = 0.0

    for b in range(n_batches):
        idx = perm[b * bs:(b + 1) * bs] if N >= bs else jnp.arange(N)
        xb, yb, mb = x_train[idx], y_train[idx], mask_train[idx]

        rng, batch_rng = jrandom.split(rng)
        state, loss, aux = train_step(state, xb, yb, mb, batch_rng)

        epoch_obj += float(aux["objective_total_bits"])
        epoch_data_nll_bits += float(aux["data_nll_bits"])
        epoch_complexity_expected_bits += float(aux["complexity_expected_bits"])
        epoch_entropy_weights_bits += float(aux["entropy_weights_bits"])
        epoch_reg_complexity += float(aux["reg_complexity_weighted_bits"])
        epoch_reg_entropy_bonus += float(aux["reg_entropy_bonus_bits"])
        epoch_reg_net += float(aux["reg_net_bits"])

    return state, rng, {
        "objective_total_bits": epoch_obj / n_batches,
        "data_nll_bits": epoch_data_nll_bits / n_batches,
        "complexity_expected_bits": epoch_complexity_expected_bits / n_batches,
        "entropy_weights_bits": epoch_entropy_weights_bits / n_batches,
        "reg_complexity_weighted_bits": epoch_reg_complexity / n_batches,
        "reg_entropy_bonus_bits": epoch_reg_entropy_bonus / n_batches,
        "reg_net_bits": epoch_reg_net / n_batches,
    }


def run_training_basic(args, model, grid_values, grid_codelengths,
                       x_train, y_train, mask_train,
                       val_inputs, val_targets, rng, grid,
                       run_dir=None, start_epoch=0, init_params=None):
    """Run training with the basic MDL objective.

    Uses Gumbel-Softmax straight-through with exponential tau annealing
    from tau_start to tau_end over all epochs.

    Args:
        run_dir: Path to results directory for checkpointing (None = no saving).
        start_epoch: Epoch to start from (>0 when resuming).
        init_params: If provided, replace state params after init (for resume/eval).
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
    if init_params is not None:
        state = state.replace(params=init_params)
        print(f"  Loaded params from checkpoint (resuming from epoch {start_epoch})")

    print(f"  Number of LSTM+output parameters: {state.params['logits'].shape[0]}")
    print(f"  Logit array shape: {state.params['logits'].shape}")

    total_epochs = args.epochs
    # mode_forward makes the forward pass deterministic; K>1 samples are redundant.
    effective_n_samples = 1 if args.mode_forward else args.n_samples
    train_step = make_train_step(
        args.mdl_lambda, n_train=N, n_samples=effective_n_samples,
        deterministic_st=args.deterministic_st,
    )

    est_name = "deterministic ST" if args.deterministic_st else f"Gumbel ST ({effective_n_samples} samples)"
    print(f"\n  Training: {total_epochs} epochs, {est_name}")
    print(f"  τ: {args.tau_start} → {args.tau_end} (exponential annealing)")
    print(f"  lr={args.lr}, lambda={args.mdl_lambda}, batch_size={bs}")
    _print_metric_legend("basic")

    best = {
        'n_perfect': -1,
        'gen_n': -1,
        'complexity_bits': math.inf,
        'params': None,
        'opt_state': None,
        'epoch': 0,
    }
    epochs_since_best = 0
    n_restarts = 0
    restart_patience = getattr(args, 'restart_patience', 0)

    is_full_batch = bs >= N

    # Full-batch fusion: create non-JIT train step + lax.scan runner
    if is_full_batch:
        step_nojit = make_train_step(
            args.mdl_lambda, n_train=N, n_samples=effective_n_samples,
            deterministic_st=args.deterministic_st, jit=False,
        )
        fused = make_fused_epoch_fn(
            step_nojit, x_train, y_train, mask_train,
            total_epochs=total_epochs,
            tau_start=args.tau_start, tau_end=args.tau_end,
        )
        print(f"  [FUSED] Full-batch mode: epochs fused via lax.scan")

    t0 = time.time()

    if is_full_batch:
        # ----- Fused full-batch path: lax.scan over contiguous epoch runs -----
        epoch = start_epoch
        while epoch < total_epochs:
            # Scan forward to next log/eval boundary.
            run_end = epoch + 1
            for e in range(epoch, total_epochs):
                run_end = e + 1
                needs_log = ((e + 1) % args.log_every == 0) or e == 0
                needs_eval = (e + 1) % args.eval_every == 0
                if needs_log or needs_eval:
                    break

            n_steps = run_end - epoch
            state, rng, metrics = fused(state, rng, epoch, n_steps)

            last_epoch = run_end - 1
            last_tau = anneal_tau(last_epoch, total_epochs, args.tau_start, args.tau_end)

            needs_log = ((last_epoch + 1) % args.log_every == 0) or last_epoch == 0
            needs_eval = (last_epoch + 1) % args.eval_every == 0

            if needs_log or needs_eval:
                print(_fmt_epoch_header(last_epoch + 1, last_tau))

            if needs_log:
                complexity_argmax_bits = _compute_discrete_hyp_bits(
                    state.params, grid_codelengths,
                )
                print(_fmt_train_line(metrics, complexity_argmax_bits))

            if needs_eval:
                best, is_best = _eval_and_update_best(
                    state, last_epoch, args.hidden_size,
                    grid_values, grid_codelengths,
                    val_inputs, val_targets,
                    best, run_dir,
                )
                if is_best:
                    epochs_since_best = 0
                else:
                    epochs_since_best += n_steps
                    state, epochs_since_best, did_restart = _maybe_restart(
                        state, best, epochs_since_best, restart_patience,
                    )
                    if did_restart:
                        n_restarts += 1
                        print(f"  ↻ RESTART #{n_restarts} at epoch {last_epoch + 1}"
                              f" → best checkpoint (epoch {best['epoch']})")
            else:
                epochs_since_best += n_steps

            epoch = run_end

    else:
        # ----- Mini-batch path: per-epoch loop -----
        for epoch in range(start_epoch, total_epochs):
            tau = anneal_tau(epoch, total_epochs, args.tau_start, args.tau_end)
            state = state.replace(tau=tau)

            state, rng, metrics = _run_epoch(
                state, x_train, y_train, mask_train, N, bs, rng, train_step,
            )

            if (epoch + 1) % args.log_every == 0 or epoch == 0:
                complexity_argmax_bits = _compute_discrete_hyp_bits(
                    state.params, grid_codelengths,
                )
                print(_fmt_epoch_header(epoch + 1, tau))
                print(_fmt_train_line(metrics, complexity_argmax_bits))

            if (epoch + 1) % args.eval_every == 0:
                best, is_best = _eval_and_update_best(
                    state, epoch, args.hidden_size,
                    grid_values, grid_codelengths,
                    val_inputs, val_targets,
                    best, run_dir,
                )
                if is_best:
                    epochs_since_best = 0
                else:
                    epochs_since_best += args.eval_every
                    state, epochs_since_best, did_restart = _maybe_restart(
                        state, best, epochs_since_best, restart_patience,
                    )
                    if did_restart:
                        n_restarts += 1
                        print(f"  ↻ RESTART #{n_restarts} at epoch {epoch + 1}"
                              f" → best checkpoint (epoch {best['epoch']})")

    elapsed = time.time() - t0
    print("-" * 70)
    print(f"Training complete in {elapsed:.1f}s"
          f"{f' ({n_restarts} restarts)' if n_restarts else ''}")

    if run_dir is not None:
        save_checkpoint(state.params, checkpoint_path(run_dir, "final.npz"))
        print(f"  [CKPT] Final checkpoint saved")

    return state, best['params'], best['n_perfect'], best['epoch']


def run_training_dst_fixed(args, model, grid_values, grid_codelengths,
                           x_train, y_train, mask_train,
                           val_inputs, val_targets, rng, grid,
                           run_dir=None, start_epoch=0, init_params=None):
    """Run fixed-beta DST training: optimize J_beta at a single fixed tau.

    Uses deterministic straight-through (hard argmax forward, soft grads).
    Tau is set once and never changes. No phase transitions, no optimizer
    resets, no tau annealing.

    Returns:
        (state, best_params, best_n_perfect, best_epoch)
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
        tau_init=args.tau_fixed,
    )
    if init_params is not None:
        state = state.replace(params=init_params)
        print(f"  Loaded params from checkpoint (resuming from epoch {start_epoch})")

    print(f"  Number of LSTM+output parameters: {state.params['logits'].shape[0]}")
    print(f"  Logit array shape: {state.params['logits'].shape}")

    total_epochs = args.epochs
    # DST is deterministic — n_samples=1 always (no Gumbel noise).
    train_step = make_train_step(
        args.mdl_lambda, n_train=N, n_samples=1,
        deterministic_st=True,
    )

    print(f"\n  Training: {total_epochs} epochs, deterministic ST (fixed-tau)")
    print(f"  τ = {args.tau_fixed} (fixed, no annealing)")
    print(f"  lr={args.lr}, lambda={args.mdl_lambda}, batch_size={bs}")
    _print_metric_legend("basic")

    best = {
        'n_perfect': -1,
        'gen_n': -1,
        'complexity_bits': math.inf,
        'params': None,
        'opt_state': None,
        'epoch': 0,
    }
    epochs_since_best = 0
    n_restarts = 0
    restart_patience = getattr(args, 'restart_patience', 0)

    is_full_batch = bs >= N

    if is_full_batch:
        step_nojit = make_train_step(
            args.mdl_lambda, n_train=N, n_samples=1,
            deterministic_st=True, jit=False,
        )
        fused = make_fused_epoch_fn_fixed_tau(
            step_nojit, x_train, y_train, mask_train,
        )
        print(f"  [FUSED] Full-batch mode: epochs fused via lax.scan")

    t0 = time.time()

    if is_full_batch:
        epoch = start_epoch
        while epoch < total_epochs:
            # Scan forward to next log/eval boundary.
            run_end = epoch + 1
            for e in range(epoch, total_epochs):
                run_end = e + 1
                needs_log = ((e + 1) % args.log_every == 0) or e == 0
                needs_eval = (e + 1) % args.eval_every == 0
                if needs_log or needs_eval:
                    break

            n_steps = run_end - epoch
            state, rng, metrics = fused(state, rng, n_steps)

            last_epoch = run_end - 1
            needs_log = ((last_epoch + 1) % args.log_every == 0) or last_epoch == 0
            needs_eval = (last_epoch + 1) % args.eval_every == 0

            if needs_log or needs_eval:
                print(_fmt_epoch_header(last_epoch + 1, args.tau_fixed))

            if needs_log:
                complexity_argmax_bits = _compute_discrete_hyp_bits(
                    state.params, grid_codelengths,
                )
                print(_fmt_train_line(metrics, complexity_argmax_bits))

            if needs_eval:
                best, is_best = _eval_and_update_best(
                    state, last_epoch, args.hidden_size,
                    grid_values, grid_codelengths,
                    val_inputs, val_targets,
                    best, run_dir,
                )
                if is_best:
                    epochs_since_best = 0
                else:
                    epochs_since_best += n_steps
                    state, epochs_since_best, did_restart = _maybe_restart(
                        state, best, epochs_since_best, restart_patience,
                    )
                    if did_restart:
                        n_restarts += 1
                        print(f"  ↻ RESTART #{n_restarts} at epoch {last_epoch + 1}"
                              f" → best checkpoint (epoch {best['epoch']})")
            else:
                epochs_since_best += n_steps

            epoch = run_end

    else:
        # Mini-batch fallback (tau is fixed, no annealing).
        for epoch in range(start_epoch, total_epochs):
            state, rng, metrics = _run_epoch(
                state, x_train, y_train, mask_train, N, bs, rng, train_step,
            )

            if (epoch + 1) % args.log_every == 0 or epoch == 0:
                complexity_argmax_bits = _compute_discrete_hyp_bits(
                    state.params, grid_codelengths,
                )
                print(_fmt_epoch_header(epoch + 1, args.tau_fixed))
                print(_fmt_train_line(metrics, complexity_argmax_bits))

            if (epoch + 1) % args.eval_every == 0:
                best, is_best = _eval_and_update_best(
                    state, epoch, args.hidden_size,
                    grid_values, grid_codelengths,
                    val_inputs, val_targets,
                    best, run_dir,
                )
                if is_best:
                    epochs_since_best = 0
                else:
                    epochs_since_best += args.eval_every
                    state, epochs_since_best, did_restart = _maybe_restart(
                        state, best, epochs_since_best, restart_patience,
                    )
                    if did_restart:
                        n_restarts += 1
                        print(f"  ↻ RESTART #{n_restarts} at epoch {epoch + 1}"
                              f" → best checkpoint (epoch {best['epoch']})")

    elapsed = time.time() - t0
    print("-" * 70)
    print(f"Training complete in {elapsed:.1f}s"
          f"{f' ({n_restarts} restarts)' if n_restarts else ''}")

    if run_dir is not None:
        save_checkpoint(state.params, checkpoint_path(run_dir, "final.npz"))
        print(f"  [CKPT] Final checkpoint saved")

    return state, best['params'], best['n_perfect'], best['epoch']


def _run_epoch_shared(state, x_train, y_train, mask_train, N, bs, rng,
                      train_step, p_base):
    """Run one shared-weight training epoch."""
    rng, perm_rng = jrandom.split(rng)
    perm = jrandom.permutation(perm_rng, N)
    n_batches = max(N // bs, 1)

    epoch_obj = 0.0
    epoch_data_nll_bits = 0.0
    epoch_complexity_expected_bits = 0.0
    epoch_code_cross_entropy_bits = 0.0
    epoch_entropy_weights_bits = 0.0
    epoch_reg_complexity = 0.0
    epoch_reg_entropy_bonus = 0.0
    epoch_reg_net = 0.0
    epoch_kl_pi_phi = 0.0
    epoch_kl_phi_pbase = 0.0
    epoch_phi_entropy_bits = 0.0
    epoch_phi_min_prob = 0.0
    epoch_phi_max_prob = 0.0

    for b in range(n_batches):
        idx = perm[b * bs:(b + 1) * bs] if N >= bs else jnp.arange(N)
        xb, yb, mb = x_train[idx], y_train[idx], mask_train[idx]

        rng, batch_rng = jrandom.split(rng)
        state, loss, aux = train_step(state, xb, yb, mb, batch_rng, p_base)

        epoch_obj += float(aux["objective_total_bits"])
        epoch_data_nll_bits += float(aux["data_nll_bits"])
        epoch_complexity_expected_bits += float(aux["complexity_expected_bits"])
        epoch_code_cross_entropy_bits += float(aux["code_cross_entropy_bits"])
        epoch_entropy_weights_bits += float(aux["entropy_weights_bits"])
        epoch_reg_complexity += float(aux["reg_complexity_weighted_bits"])
        epoch_reg_entropy_bonus += float(aux["reg_entropy_bonus_bits"])
        epoch_reg_net += float(aux["reg_net_bits"])
        epoch_kl_pi_phi += float(aux["kl_pi_phi_bits"])
        epoch_kl_phi_pbase += float(aux["kl_phi_pbase_bits"])
        epoch_phi_entropy_bits += float(aux["phi_entropy_bits"])
        epoch_phi_min_prob += float(aux["phi_min_prob"])
        epoch_phi_max_prob += float(aux["phi_max_prob"])

    return state, rng, {
        "objective_total_bits": epoch_obj / n_batches,
        "data_nll_bits": epoch_data_nll_bits / n_batches,
        "complexity_expected_bits": epoch_complexity_expected_bits / n_batches,
        "code_cross_entropy_bits": epoch_code_cross_entropy_bits / n_batches,
        "entropy_weights_bits": epoch_entropy_weights_bits / n_batches,
        "reg_complexity_weighted_bits": epoch_reg_complexity / n_batches,
        "reg_entropy_bonus_bits": epoch_reg_entropy_bonus / n_batches,
        "reg_net_bits": epoch_reg_net / n_batches,
        "kl_pi_phi_bits": epoch_kl_pi_phi / n_batches,
        "kl_phi_pbase_bits": epoch_kl_phi_pbase / n_batches,
        "phi_entropy_bits": epoch_phi_entropy_bits / n_batches,
        "phi_min_prob": epoch_phi_min_prob / n_batches,
        "phi_max_prob": epoch_phi_max_prob / n_batches,
    }


def run_training_shared(args, model, grid_values, grid_codelengths,
                        x_train, y_train, mask_train,
                        val_inputs, val_targets, rng, grid,
                        run_dir=None, start_epoch=0, init_params=None):
    """Run training with the shared-weight MDL objective."""
    rng, init_rng = jrandom.split(rng)
    N = x_train.shape[0]
    max_seq_len = x_train.shape[1]
    bs = args.batch_size if args.batch_size > 0 else N

    state = create_shared_mdl_state(
        init_rng, model, grid_values, grid_codelengths,
        seq_len=max_seq_len,
        batch_size=min(bs, N),
        lr=args.lr,
        tau_init=args.tau_start,
    )
    if init_params is not None:
        state = state.replace(params=init_params)
        print(f"  Loaded params from checkpoint (resuming from epoch {start_epoch})")

    print(f"  Number of LSTM+output parameters: {state.params['logits'].shape[0]}")
    print(f"  Logit array shape: {state.params['logits'].shape}")
    print(f"  Phi logits shape: {state.params['phi_logits'].shape}")

    p_base = compute_p_base(grid_codelengths)

    total_epochs = args.epochs
    effective_n_samples = 1 if args.mode_forward else args.n_samples
    train_step = make_shared_train_step(
        args.lambda1, args.lambda2, args.epsilon, n_train=N,
        n_samples=effective_n_samples,
        deterministic_st=args.deterministic_st,
    )

    est_name = "deterministic ST" if args.deterministic_st else f"Gumbel ST ({effective_n_samples} samples)"
    print(f"\n  Training: {total_epochs} epochs, {est_name}")
    print(f"  τ: {args.tau_start} → {args.tau_end} (exponential annealing)")
    print(f"  lr={args.lr}, lambda1={args.lambda1}, lambda2={args.lambda2}, "
          f"eps={args.epsilon}, batch_size={bs}")
    _print_metric_legend("shared")

    best_val_n_perfect = -1
    best_val_gen_n = -1
    best_complexity_bits = math.inf
    best_params = None
    best_opt_state = None
    epochs_since_best = 0
    n_restarts = 0
    restart_patience = getattr(args, 'restart_patience', 0)

    t0 = time.time()
    for epoch in range(start_epoch, total_epochs):
        tau = anneal_tau(epoch, total_epochs, args.tau_start, args.tau_end)
        state = state.replace(tau=tau)

        state, rng, metrics = _run_epoch_shared(
            state, x_train, y_train, mask_train, N, bs, rng,
            train_step, p_base,
        )

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            lan_hyp_bits = _compute_discrete_hyp_bits(
                state.params, grid_codelengths,
            )
            shared_disc = _compute_shared_discrete_complexity_bits(
                state.params, grid_codelengths, p_base,
                args.lambda1, args.lambda2, args.epsilon,
            )
            print(_fmt_epoch_header(epoch + 1, tau))
            print(_fmt_train_line_shared(metrics, lan_hyp_bits, shared_disc))

        if (epoch + 1) % args.eval_every == 0:
            model_params = {"logits": state.params["logits"]}
            val_result = evaluate_deterministic_accuracy(
                state.apply_fn, model_params, grid_values,
                val_inputs, val_targets,
            )
            n_perfect = val_result["n_perfect"]
            gen_n = val_result["gen_n"]
            current_complexity_bits = _compute_discrete_hyp_bits(
                model_params, grid_codelengths,
            )
            n_val = len(val_inputs)
            is_best = _should_update_best(
                n_perfect,
                gen_n,
                current_complexity_bits,
                best_val_n_perfect,
                best_val_gen_n,
                best_complexity_bits,
            )
            best_tag = "  ★ NEW BEST" if is_best else ""
            lan_total = current_complexity_bits + integer_code_length(args.hidden_size)
            print(_fmt_val_line(
                n_perfect, n_val, gen_n, lan_total,
                best_tag=best_tag,
            ))
            if is_best:
                best_val_n_perfect = n_perfect
                best_val_gen_n = gen_n
                best_complexity_bits = current_complexity_bits
                best_params = jax.tree.map(lambda x: x.copy(), state.params)
                best_opt_state = jax.tree.map(lambda x: x.copy(), state.opt_state)
                best_epoch = epoch + 1
                epochs_since_best = 0
                if run_dir is not None:
                    save_checkpoint(best_params, checkpoint_path(run_dir, "best.npz"))
                    save_checkpoint_meta(
                        run_dir,
                        epoch + 1,
                        best_val_n_perfect,
                        best_checkpoint_epoch=epoch + 1,
                    )
                    print(f"              ↳ [CKPT] checkpoint saved")
            else:
                epochs_since_best += args.eval_every
                if (restart_patience > 0 and best_params is not None
                        and epochs_since_best >= restart_patience):
                    new_opt_state = _zero_adam_moments(best_opt_state)
                    state = state.replace(params=best_params, opt_state=new_opt_state)
                    epochs_since_best = 0
                    n_restarts += 1
                    print(f"  ↻ RESTART #{n_restarts} at epoch {epoch + 1}"
                          f" → best checkpoint (epoch {best_epoch})")

    elapsed = time.time() - t0
    print("-" * 70)
    print(f"Training complete in {elapsed:.1f}s"
          f"{f' ({n_restarts} restarts)' if n_restarts else ''}")

    if run_dir is not None:
        save_checkpoint(state.params, checkpoint_path(run_dir, "final.npz"))
        print(f"  [CKPT] Final checkpoint saved")

    return state, best_params, best_val_n_perfect


def _resolve_config_path(config_arg):
    """Resolve a config argument to an existing YAML path."""
    if config_arg is None:
        return None
    p = Path(config_arg)
    if p.exists():
        return p
    for ext in (".yaml", ".yml"):
        p_ext = Path(f"{config_arg}{ext}")
        if p_ext.exists():
            return p_ext
    return None


def _load_yaml_defaults(config_path):
    """Load flat key/value defaults from a YAML config file."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config must be a mapping of argument names to values: {config_path}"
        )
    return raw


def _build_arg_parser(defaults=None):
    """Build argument parser, optionally seeded by config defaults."""
    parser = argparse.ArgumentParser(description="Differentiable MDL experiment")
    parser.add_argument(
        "config", nargs="?", default=None,
        help="Optional YAML config path. CLI flags override YAML values.",
    )
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
    parser.add_argument("--data_seed", type=int, default=None,
                        help="Seed for data generation (defaults to --seed if unset)")
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
    parser.add_argument("--deterministic_st", action="store_true",
                        help="Use deterministic straight-through instead of Gumbel ST")
    parser.add_argument("--mode_forward", action="store_true",
                        help="Use mode of pi (not Gumbel argmax) in forward pass "
                             "(Lee et al. 2021 Semi-Relaxed Quantization)")
    parser.add_argument("--init_cl_scale", type=float, default=0.0,
                        help="Scale for codelength-informed logit initialization "
                             "(0 = legacy noise-only, >0 = bias toward simple rationals)")
    parser.add_argument("--training_mode", type=str, default="annealed",
                        choices=["annealed", "dst_fixed"],
                        help="Training mode: 'annealed' (tau annealing, existing) "
                             "or 'dst_fixed' (fixed-tau deterministic ST)")
    parser.add_argument("--tau_fixed", type=float, default=1.0,
                        help="Fixed backward-pass temperature for dst_fixed mode "
                             "(no annealing; replaces tau_start/tau_end)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    # Shared-weight mode parameters
    parser.add_argument("--lambda1", type=float, default=1.0,
                        help="Shared code-term weight (cross-entropy to phi, shared mode)")
    parser.add_argument("--lambda2", type=float, default=1.0,
                        help="Dictionary-prior KL weight (shared mode)")
    parser.add_argument("--epsilon", type=float, default=1e-6,
                        help="Min probability for adaptive prior (shared mode)")
    # Evaluation
    parser.add_argument("--test_max_n", type=int, default=1500,
                        help="Max n for test set (can be overridden in --eval mode)")
    parser.add_argument("--val_min_n", type=int, default=22,
                        help="Minimum n included in the structured validation set")
    parser.add_argument("--val_max_n", type=int, default=71,
                        help="Maximum n included in the structured validation set")
    parser.add_argument("--restart_patience", type=int, default=0,
                        help="Epochs without improvement before resetting to best "
                             "checkpoint (0 = disabled)")
    parser.add_argument("--eval_every", type=int, default=100,
                        help="Evaluate every N epochs")
    parser.add_argument("--log_every", type=int, default=50,
                        help="Log training metrics every N epochs")
    parser.add_argument("--deterministic", action="store_true",
                        help="Force deterministic GPU ops (slower but fully reproducible)")
    parser.add_argument("--analyze", action="store_true",
                        help="Run analytical network analysis (golden check, failure prediction)")
    parser.add_argument("--analyze_max_n", type=int, default=100_000,
                        help="Max n for analytical golden check (default: 100000)")
    # Run management
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to a results directory (used with --eval or --resume)")
    parser.add_argument("--ckpt_select", type=str, default="auto",
                        choices=["auto", "best", "final"],
                        help="Which checkpoint to load from --ckpt (default: auto)")
    parser.add_argument("--eval", action="store_true",
                        help="Load best checkpoint from --ckpt and run test evaluation only")
    parser.add_argument("--resume", action="store_true",
                        help="Load best checkpoint from --ckpt and resume training")
    parser.add_argument("--resume_epoch", type=int, default=None,
                        help="Override the starting epoch when resuming")
    parser.add_argument("--resume_in_place", action="store_true",
                        help="Resume inside --ckpt instead of copying into a fresh run dir")

    if defaults:
        valid_dests = {a.dest for a in parser._actions}
        unknown = sorted(k for k in defaults if k not in valid_dests)
        if unknown:
            raise ValueError(
                "Unknown config keys: "
                f"{', '.join(unknown)}"
            )
        parser.set_defaults(**defaults)
    return parser


def _resolve_resume_checkpoint(run_dir: Path, selection: str = "auto") -> tuple[Path, str]:
    """Find checkpoint file for eval/resume, supporting legacy filenames."""
    candidates_by_kind = {
        "best": [
            checkpoint_path(run_dir, "best.npz", create=False),
            run_dir / "checkpoint_best.npz",
        ],
        "final": [
            checkpoint_path(run_dir, "final.npz", create=False),
            run_dir / "checkpoint_final.npz",
        ],
    }
    if selection == "auto":
        search_order = [("best", candidates_by_kind["best"]),
                        ("final", candidates_by_kind["final"])]
    else:
        search_order = [(selection, candidates_by_kind[selection])]
    for kind, candidates in search_order:
        for c in candidates:
            if c.exists():
                return c, kind
    raise FileNotFoundError(f"No {selection} checkpoint found in {run_dir}")


def _read_resume_meta(run_dir: Path) -> dict:
    """Read checkpoint metadata, supporting legacy path."""
    candidates = [
        checkpoint_path(run_dir, "meta.json", create=False),
        run_dir / "checkpoint_meta.json",
    ]
    for meta_path in candidates:
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
    return {}


def _resolve_resume_start_epoch(run_dir: Path, checkpoint_kind: str,
                                default_final_epoch: int = 0) -> int:
    """Choose a sensible start epoch for the selected checkpoint kind."""
    meta = _read_resume_meta(run_dir)
    if checkpoint_kind == "best":
        if "best_checkpoint_epoch" in meta:
            return int(meta["best_checkpoint_epoch"])
        # Legacy runs do not record the epoch for the best checkpoint; safest is
        # to restart the schedule from the beginning unless the user overrides it.
        return 0
    if "last_epoch" in meta:
        return int(meta["last_epoch"])
    return int(default_final_epoch)


def _write_resume_info(run_dir: Path, source_run_dir: Path, source_ckpt_path: Path,
                       checkpoint_kind: str, start_epoch: int):
    """Record the provenance of a copied resume run."""
    info = {
        "source_run_dir": str(source_run_dir.resolve()),
        "source_checkpoint_path": str(source_ckpt_path.resolve()),
        "source_checkpoint_kind": checkpoint_kind,
        "resume_start_epoch": int(start_epoch),
    }
    with open(run_dir / "resume_info.json", "w") as f:
        json.dump(info, f, indent=2)


def _prepare_resume_run_dir(args, source_run_dir: Path, source_ckpt_path: Path,
                            checkpoint_kind: str, start_epoch: int) -> Path:
    """Create a fresh run directory seeded by a copied source checkpoint."""
    run_dir = make_run_dir(args, suffix=f"_resume_{checkpoint_kind}")
    copied_ckpt = run_dir / f"resume_source_{checkpoint_kind}.npz"
    shutil.copy2(source_ckpt_path, copied_ckpt)
    _write_resume_info(run_dir, source_run_dir, source_ckpt_path, checkpoint_kind,
                       start_epoch)
    return run_dir


def _print_resolved_parameters(args):
    """Print resolved effective parameters at startup."""
    params = vars(args)

    print("\nResolved parameters")
    print("-" * 60)
    training_mode = getattr(args, 'training_mode', 'annealed')
    print(f"  mode={args.mode}")
    if training_mode == 'dst_fixed':
        print(f"  training_mode=dst_fixed  (tau_fixed={args.tau_fixed})")
    if args.mode == "basic":
        print("  J = L_D + λ/N·E[Σℓ(θᵢ)] − τ/N·ΣH₂(πᵢ)  (Eq. 86)")
        print(f"  mdl_lambda={args.mdl_lambda}")
    else:
        print("  J = L_D + λ₁/N·CE₂(π,φ) + λ₂/N·KL(φ‖P) − τ/N·ΣH₂(πᵢ)  (Eq. 156)")
        print(f"  lambda1={args.lambda1}")
        print(f"  lambda2={args.lambda2}")
        print(f"  epsilon={args.epsilon}")

    for key in sorted(params):
        print(f"  {key}={params[key]}")
    print("-" * 60)


def main():
    # Parse config path first so YAML defaults can seed argparse.
    # Only treat argv[1] as a config candidate if it is actually positional.
    pre_config_arg = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        pre_config_arg = sys.argv[1]
    yaml_defaults = {}
    pre_cfg_path = _resolve_config_path(pre_config_arg)
    if pre_cfg_path is not None:
        yaml_defaults = _load_yaml_defaults(pre_cfg_path)

    try:
        parser = _build_arg_parser(defaults=yaml_defaults)
    except ValueError as e:
        raise SystemExit(str(e))
    args = parser.parse_args()

    cfg_path = _resolve_config_path(args.config)
    if args.config is not None and cfg_path is None:
        parser.error(f"Config file not found: {args.config}")
    args.config = str(cfg_path) if cfg_path else None
    args.config_name = cfg_path.stem if cfg_path else None

    # If data_seed not given, tie it to seed so --seed controls all randomness.
    if args.data_seed is None:
        args.data_seed = args.seed

    # --- Validate run management flags ---
    if args.eval and args.resume:
        parser.error("--eval and --resume are mutually exclusive")
    if (args.eval or args.resume) and args.ckpt is None:
        parser.error("--ckpt is required when using --eval or --resume")
    if (args.eval or args.resume) and not Path(args.ckpt).exists():
        parser.error(f"Checkpoint directory not found: {args.ckpt}")

    # --- Override args from saved config when eval/resuming ---
    if args.eval or args.resume:
        # Preserve explicitly provided CLI overrides when loading saved config.
        override_flags = {
            "test_max_n": "--test_max_n",
            "analyze": "--analyze",
            "analyze_max_n": "--analyze_max_n",
            "epochs": "--epochs",
            "lr": "--lr",
            "mdl_lambda": "--mdl_lambda",
            "lambda1": "--lambda1",
            "lambda2": "--lambda2",
            "epsilon": "--epsilon",
            "tau_start": "--tau_start",
            "tau_end": "--tau_end",
            "batch_size": "--batch_size",
            "n_samples": "--n_samples",
            "deterministic_st": "--deterministic_st",
            "mode_forward": "--mode_forward",
            "init_cl_scale": "--init_cl_scale",
            "training_mode": "--training_mode",
            "tau_fixed": "--tau_fixed",
            "deterministic": "--deterministic",
            "restart_patience": "--restart_patience",
            "eval_every": "--eval_every",
            "log_every": "--log_every",
        }
        run_mgmt_overrides = {
            "ckpt_select": args.ckpt_select,
            "resume_epoch": args.resume_epoch,
            "resume_in_place": args.resume_in_place,
        }
        parser_defaults = {
            a.dest: a.default for a in parser._actions
            if a.dest != "help"
        }
        cli_overrides = {
            dest: getattr(args, dest)
            for dest, flag in override_flags.items()
            if flag in sys.argv
        }

        saved_args = load_run_config(args.ckpt)
        saved_args.eval = args.eval
        saved_args.resume = args.resume
        saved_args.ckpt = args.ckpt
        for dest, default in parser_defaults.items():
            if not hasattr(saved_args, dest):
                setattr(saved_args, dest, default)
        if not hasattr(saved_args, "config"):
            saved_args.config = None
        if not hasattr(saved_args, "config_name"):
            saved_args.config_name = None
        for dest, value in cli_overrides.items():
            setattr(saved_args, dest, value)
        for dest, value in run_mgmt_overrides.items():
            setattr(saved_args, dest, value)
        args = saved_args

    # --- Set up run directory and logging ---
    if args.eval:
        run_dir = Path(args.ckpt)
        ckpt_path, _ = _resolve_resume_checkpoint(run_dir, args.ckpt_select)
        loaded_params = load_checkpoint(ckpt_path)
        start_epoch = 0
        log_mode = "a"
    elif args.resume:
        source_run_dir = Path(args.ckpt)
        ckpt_path, checkpoint_kind = _resolve_resume_checkpoint(
            source_run_dir, args.ckpt_select,
        )
        loaded_params = load_checkpoint(ckpt_path)
        if args.resume_epoch is not None:
            start_epoch = int(args.resume_epoch)
        else:
            start_epoch = _resolve_resume_start_epoch(
                source_run_dir, checkpoint_kind, default_final_epoch=args.epochs,
            )
        if args.resume_in_place:
            run_dir = source_run_dir
            log_mode = "a"
        else:
            run_dir = _prepare_resume_run_dir(
                args, source_run_dir, ckpt_path, checkpoint_kind, start_epoch,
            )
            log_mode = "w"
        print(f"Resuming from epoch {start_epoch}/{args.epochs}")
    else:
        run_dir = make_run_dir(args)
        loaded_params = None
        start_epoch = 0
        log_mode = "w"

    _tee = TeeLogger(run_dir / "train.log", mode=log_mode)
    _tee.__enter__()
    try:
        _main_inner(args, run_dir, loaded_params, start_epoch)
    finally:
        _tee.__exit__(None, None, None)


def _main_inner(args, run_dir, loaded_params, start_epoch):
    """Inner main logic (runs inside TeeLogger context)."""
    # Seed the global numpy RNG so any library path that uses it is reproducible.
    np.random.seed(args.seed)

    print("=" * 60)
    if args.deterministic:
        print("Deterministic mode: ON  (--xla_gpu_deterministic_ops=true)")
    else:
        print("Deterministic mode: OFF (pass --deterministic for full reproducibility)")
    print("Differentiable MDL for a^n b^n")
    print(f"Mode: {args.mode}")
    if getattr(args, "config", None):
        print(f"Config: {args.config}")
    if args.eval:
        print(f"[EVAL MODE] checkpoint: {args.ckpt}")
    elif args.resume:
        print(f"[RESUME from epoch {start_epoch}] checkpoint: {args.ckpt}")
        if run_dir != Path(args.ckpt):
            print(f"Resume run directory: {run_dir}")
    else:
        print(f"Run directory: {run_dir}")
    print("=" * 60)
    _print_resolved_parameters(args)

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

    # Structured validation set
    val_inputs, val_targets = make_validation_set(
        train_max_n, val_max_n=args.val_max_n, val_min_n=args.val_min_n,
    )
    if val_inputs:
        first_val_n = len(val_inputs[0]) // 2
        last_val_n = len(val_inputs[-1]) // 2
        print(
            f"  Structured val set: {len(val_inputs)} strings "
            f"(n={first_val_n}..{last_val_n})"
        )
    else:
        print("  Structured val set: 0 strings")

    # Test set (skip generation for very large n - float64 simulation handles it)
    if args.test_max_n <= 10_000:
        test_inputs, test_targets = make_test_set(max_n=args.test_max_n)
        print(f"  Test set: {len(test_inputs)} strings (n=1..{args.test_max_n})")
    else:
        test_inputs, test_targets = [], []
        print(f"  Test set: n=1..{args.test_max_n} (will use float64 simulation)")

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
        mode_forward=args.mode_forward,
        init_cl_scale=args.init_cl_scale,
    )

    rng = jrandom.PRNGKey(args.seed)

    # --- Training (or eval/resume) ---
    if args.eval:
        # Reconstruct a state with loaded params just to get apply_fn
        N = x_train.shape[0]
        bs = args.batch_size if args.batch_size > 0 else N
        rng, init_rng = jrandom.split(rng)
        if args.mode == "basic":
            tau_for_eval = (args.tau_fixed
                            if getattr(args, 'training_mode', 'annealed') == 'dst_fixed'
                            else args.tau_end)
            state = create_mdl_state(
                init_rng, model,
                seq_len=x_train.shape[1],
                batch_size=min(bs, N),
                lr=args.lr,
                tau_init=tau_for_eval,
            )
        else:
            state = create_shared_mdl_state(
                init_rng, model, grid_values, grid_codelengths,
                seq_len=x_train.shape[1],
                batch_size=min(bs, N),
                lr=args.lr,
                tau_init=args.tau_end,
            )
        state = state.replace(params=loaded_params)
        best_params = loaded_params
        best_val_n_perfect = None
        best_epoch = None
    elif args.mode == "basic" and getattr(args, 'training_mode', 'annealed') == 'dst_fixed':
        state, best_params, best_val_n_perfect, best_epoch = run_training_dst_fixed(
            args, model, grid_values, grid_codelengths,
            x_train, y_train, mask_train,
            val_inputs, val_targets, rng, grid,
            run_dir=run_dir, start_epoch=start_epoch,
            init_params=loaded_params,
        )
    elif args.mode == "basic":
        state, best_params, best_val_n_perfect, best_epoch = run_training_basic(
            args, model, grid_values, grid_codelengths,
            x_train, y_train, mask_train,
            val_inputs, val_targets, rng, grid,
            run_dir=run_dir, start_epoch=start_epoch,
            init_params=loaded_params,
        )
    else:
        state, best_params, best_val_n_perfect = run_training_shared(
            args, model, grid_values, grid_codelengths,
            x_train, y_train, mask_train,
            val_inputs, val_targets, rng, grid,
            run_dir=run_dir, start_epoch=start_epoch,
            init_params=loaded_params,
        )
        best_epoch = None

    if not args.eval and run_dir is not None:
        save_checkpoint_meta(run_dir, args.epochs, best_val_n_perfect or 0)

    # --- Final evaluation ---
    metrics = run_final_evaluation(
        args, state, best_params,
        grid, grid_values, grid_codelengths,
        test_inputs, test_targets,
        train_inputs, train_targets,
        golden_mdl, golden_result,
        best_epoch=best_epoch,
    )
    if run_dir is not None:
        save_results(run_dir, metrics)
        print(f"\nResults saved to: {run_dir}/")


def run_final_evaluation(args, state, best_params,
                         grid, grid_values, grid_codelengths,
                         test_inputs, test_targets,
                         train_inputs, train_targets,
                         golden_mdl, golden_result,
                         best_epoch=None):
    """Run final test evaluation and print the comparison table.

    For test_max_n > 10000, uses efficient float64 simulation instead of
    JAX batched evaluation (avoids memory issues with very long sequences).

    Returns a dict of metrics for results.json.
    """
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    if best_params is not None and best_epoch is not None:
        print(f"  (using best checkpoint from epoch {best_epoch})")
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

    # Compute discrete MDL score (Lan-style)
    total_hyp_bits = compute_discrete_mdl_score(eval_model_params, grid, grid_values)
    arch_bits = integer_code_length(args.hidden_size)
    total_mdl_bits = arch_bits + total_hyp_bits
    print(f"  Discrete |H| (Lan): {total_mdl_bits} bits ({arch_bits} arch + {total_hyp_bits} weights)")

    if args.mode == "shared":
        from src.mdl.shared_weights import compute_p_base
        p_base = compute_p_base(grid_codelengths)
        shared_disc = _compute_shared_discrete_complexity_bits(
            eval_params, grid_codelengths, p_base,
            args.lambda1, args.lambda2, args.epsilon,
        )
        print(
            f"  Shared-code complexity: {shared_disc['shared_complexity']:.1f} bits "
            f"(code_ce={shared_disc['code_ce']:.1f} + kl_dict={shared_disc['kl_dict']:.1f})"
        )

    # Test accuracy
    test_max_n = args.test_max_n
    use_f64_eval = test_max_n > 10_000

    if use_f64_eval:
        # For very large n, use binary-search to efficiently find the
        # generalization boundary instead of testing all N strings.
        print(f"\nEvaluating on test set (n=1..{test_max_n}) using float64 simulation...")
        extracted = extract_weights(eval_model_params, grid, grid_values)
        print(f"  Finding generalization boundary (binary search)...")
        first_failure = find_failure_n(extracted["named"], max_n=test_max_n)
        if first_failure is None:
            our_gen_n = test_max_n
            our_n_perfect = test_max_n
            all_correct = True
        else:
            our_gen_n = first_failure - 1
            our_n_perfect = our_gen_n  # conservative: at least 1..gen_n are correct
            all_correct = False
        mean_acc = our_n_perfect / test_max_n
    else:
        print(f"\nEvaluating on test set (n=1..{test_max_n})...")
        test_result = evaluate_deterministic_accuracy(
            state.apply_fn, eval_model_params, grid_values,
            test_inputs, test_targets, max_n=test_max_n,
        )
        our_n_perfect = test_result["n_perfect"]
        our_gen_n = test_result["gen_n"]
        first_failure = test_result["first_failure_n"]
        all_correct = test_result["all_correct"]
        mean_acc = float(test_result["mean_accuracy"])

    print(f"  Perfect strings: {our_n_perfect}/{test_max_n}")
    print(f"  Generalisation range: n=1..{our_gen_n}")
    if not all_correct:
        print(f"  First failure at n={first_failure}")

    # --- Summary comparison table ---
    n_params = len(discrete_weights)
    trivial_hyp = n_params * 5  # all-zero weights, 5 bits each
    trivial_mdl = arch_bits + trivial_hyp

    golden_gen_n = (
        test_max_n if golden_result["all_correct"]
        else (golden_result["first_failure_n"] - 1)
    )
    golden_n_perfect = golden_gen_n

    print("\n" + "=" * 70)
    print("COMPARISON TABLE (cf. Lan et al. 2024, Table 1)")
    print("=" * 70)
    print(f"{'Method':<30} {'|H| (bits)':>10} {'gen_n':>8} {'n_perfect':>16}")
    print("-" * 70)
    print(f"{'Lan et al. golden':<30} {golden_mdl['total_bits']:>10d} {golden_gen_n:>8d} "
          f"{golden_n_perfect:>10d}/{test_max_n}")
    print(f"{'Trivial (always-b)':<30} {trivial_mdl:>10d} {0:>8d} "
          f"{0:>10d}/{test_max_n}")
    mode_name = "Ours (basic MDL)" if args.mode == "basic" else "Ours (shared MDL)"
    print(f"{mode_name:<30} {total_mdl_bits:>10d} {our_gen_n:>8d} "
          f"{our_n_perfect:>10d}/{test_max_n}")
    print("=" * 70)

    # --- Paper-comparable metrics (Abudy et al. 2025) ---
    print("\n" + "=" * 70)
    print("PAPER-COMPARABLE METRICS (Abudy et al. 2025)")
    print("=" * 70)

    # Golden baselines
    print("  Evaluating golden baselines...")
    golden_opt_test = compute_optimal_dh_test(
        max_n=test_max_n, p=args.p, batch_size=64, verbose=True,
    )
    print(f"  Golden test |D:H|: {golden_opt_test['data_dh_bits']:.4f} bits")
    print(f"  Golden |H| (LSTM): {golden_opt_test['h_bits']} bits")

    golden_opt_train = compute_optimal_dh_train(
        train_inputs, train_targets, p=args.p, batch_size=64, verbose=True,
    )
    print(f"  Golden train |D:H|: {golden_opt_train['train_dh_data_bits']:.2f} bits")

    # Trained network |D:H| via discrete forward pass
    def our_discrete_fwd(x):
        logits_out, _ = state.apply_fn(
            {"params": eval_model_params}, x, tau=1.0, train=False,
        )
        return logits_out

    print("  Evaluating trained network...")
    our_test_result = compute_grammar_weighted_nll_bits(
        our_discrete_fwd, max_n=test_max_n, p=args.p, batch_size=64,
        verbose=True,
    )
    our_test_data_dh = our_test_result["data_dh_bits"]

    our_train_result = compute_train_dh(
        our_discrete_fwd, train_inputs, train_targets, batch_size=64,
        verbose=True,
    )
    our_train_data_dh = our_train_result["train_dh_data_bits"]

    our_h = compute_trained_h_bits(
        eval_model_params, grid_codelengths, args.hidden_size,
    )

    delta_test = compute_delta_pct(
        our_test_data_dh, golden_opt_test["data_dh_bits"],
    )
    delta_train = compute_delta_pct(
        our_train_data_dh, golden_opt_train["train_dh_data_bits"],
    )

    print(f"\n  Our test |D:H|:  {our_test_data_dh:.4f} bits  "
          f"(Δ_test = {delta_test:+.1f}%)")
    print(f"  Our train |D:H|: {our_train_data_dh:.2f} bits  "
          f"(Δ_train = {delta_train:+.1f}%)")
    print(f"  Our |H|:         {our_h['h_bits']} bits "
          f"({our_h['arch_bits']} arch + {our_h['weight_bits']} weights)")

    # Summary table
    table = format_abudy_comparison_table(
        our_test_data_dh=our_test_data_dh,
        our_train_data_dh=our_train_data_dh,
        our_h_bits=our_h["h_bits"],
        opt_test_data_dh=golden_opt_test["data_dh_bits"],
        opt_train_data_dh=golden_opt_train["train_dh_data_bits"],
        golden_h_bits=golden_opt_test["h_bits"],
    )
    print(f"\n{table}")

    # --- Analytical network analysis ---
    if getattr(args, "analyze", False):
        analysis_result = analyze_model(
            eval_model_params, grid, grid_values,
            max_test_n=args.analyze_max_n, p=args.p,
        )

    return {
        "mode": args.mode,
        "gen_n": int(our_gen_n),
        "n_perfect": int(our_n_perfect),
        "total_mdl_bits": int(total_mdl_bits),
        "arch_bits": int(arch_bits),
        "weight_bits": int(total_hyp_bits),
        "first_failure_n": first_failure,
        "mean_det_accuracy": float(mean_acc),
        # Paper-comparable metrics
        "test_data_dh_bits": float(our_test_data_dh),
        "train_data_dh_bits": float(our_train_data_dh),
        "delta_test_pct": float(delta_test),
        "delta_train_pct": float(delta_train),
        "golden_test_data_dh_bits": float(golden_opt_test["data_dh_bits"]),
        "golden_train_data_dh_bits": float(golden_opt_train["train_dh_data_bits"]),
        "golden_h_bits": int(golden_opt_test["h_bits"]),
    }


if __name__ == "__main__":
    main()
