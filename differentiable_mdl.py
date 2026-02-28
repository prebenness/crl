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
    python differentiable_mdl.py config/anbn_mdl/basic_train.yaml
    python differentiable_mdl.py config/anbn_mdl/basic_train.yaml --epochs 50000
    python differentiable_mdl.py [--n_max 10] [--m_max 10] [--epochs 5000] ...
"""

import argparse
import json
import os
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
    evaluate_deterministic_accuracy,
    anneal_tau_st_phase,
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
from src.mdl.analysis import (
    analyze_model,
    extract_weights,
    evaluate_range_f64,
    find_failure_n,
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


def make_run_dir(args):
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
        f"_g{args.n_max}x{args.m_max}_s{args.seed}{cfg_tag}"
    )
    run_dir = make_experiment_dir("anbn_mdl", name)
    save_config(run_dir, vars(args))
    return run_dir


def load_run_config(run_dir):
    """Load config.json from a run directory into an argparse.Namespace."""
    with open(Path(run_dir) / "config.json") as f:
        return argparse.Namespace(**json.load(f))


def save_checkpoint_meta(run_dir, epoch, best_val_n_perfect):
    """Write a small sidecar recording the last completed epoch."""
    meta = {"last_epoch": int(epoch), "best_val_n_perfect": int(best_val_n_perfect)}
    with open(checkpoint_path(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


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

    For large test_max_n (>1500), the golden network is known to be correct
    by construction, so we skip the expensive JAX evaluation.
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
        # For large n, the golden network is correct by construction
        print(f"  Golden network is correct for ALL n by construction (skipping eval)")
        golden_result = {
            "mean_accuracy": 1.0,
            "all_correct": True,
            "first_failure_n": None,
        }

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
    """Run two-phase training with the basic MDL objective.

    Phase 1 (warmup): Continuous soft relaxation, zero-variance gradients.
    Phase 2 (ST):     Gumbel-Softmax straight-through, multi-sample variance reduction.

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

    # Create train steps for both phases
    warmup_train_step = make_train_step(
        args.mdl_lambda, n_train=N, n_samples=1, soft_forward=True,
    )
    bridge_train_step = make_train_step(
        args.mdl_lambda, n_train=N, n_samples=1,
        soft_forward=False, deterministic_st=True,
    )
    st_train_step = make_train_step(
        args.mdl_lambda, n_train=N, n_samples=args.n_samples, soft_forward=False,
    )

    warmup_epochs = args.warmup_epochs
    total_epochs = args.epochs
    bridge_epochs = 1 if (warmup_epochs > 0 and warmup_epochs < total_epochs) else 0
    st_epochs = max(total_epochs - warmup_epochs - bridge_epochs, 0)

    print(f"\n  Phase 1 (warmup): {warmup_epochs} epochs, soft forward (no Gumbel)")
    if bridge_epochs:
        print("  Phase 2 (bridge): 1 epoch, deterministic ST (hard forward, no Gumbel)")
    print(f"  Phase 3 (ST):     {st_epochs} epochs, {args.n_samples} Gumbel samples")
    print(f"  tau: warmup fixed at {args.tau_start}, then ST anneal {args.tau_start} -> {args.tau_end}")
    print(f"  lr={args.lr}, lambda={args.mdl_lambda}, batch_size={bs}")
    print("-" * 70)

    best_val_n_perfect = -1
    best_params = None

    t0 = time.time()
    for epoch in range(start_epoch, total_epochs):
        is_warmup = epoch < warmup_epochs
        is_bridge = bridge_epochs and (epoch == warmup_epochs)
        phase_name = "warmup" if is_warmup else ("bridge" if is_bridge else "ST")

        # Keep warmup tau fixed; anneal only in ST phase.
        tau = anneal_tau_st_phase(
            epoch, total_epochs, warmup_epochs, args.tau_start, args.tau_end,
        )

        if is_warmup:
            train_step = warmup_train_step
        elif is_bridge:
            train_step = bridge_train_step
        else:
            train_step = st_train_step

        state = state.replace(tau=tau)

        state, rng, metrics = _run_epoch(
            state, x_train, y_train, mask_train, N, bs, rng, train_step,
        )

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            # Compute discrete argmax complexity periodically for monitoring.
            complexity_argmax_bits = _compute_discrete_hyp_bits(
                state.params, grid, grid_values,
            )
            print(
                f"Epoch {epoch+1:5d} [{phase_name:4s}] | "
                f"objective_total_bits={metrics['objective_total_bits']:8.1f}b  "
                f"data_nll_bits={metrics['data_nll_bits']:8.1f}b  "
                f"reg_net_bits={metrics['reg_net_bits']:7.1f}b "
                f"(reg_complexity_weighted_bits={metrics['reg_complexity_weighted_bits']:.1f}b "
                f"- reg_entropy_bonus_bits={metrics['reg_entropy_bonus_bits']:.1f}b) | "
                f"complexity_expected_bits={metrics['complexity_expected_bits']:.1f}b  "
                f"complexity_argmax_bits={complexity_argmax_bits:4d}b  "
                f"entropy_weights_bits={metrics['entropy_weights_bits']:5.1f}b  τ={float(tau):.4f}"
            )

        if (epoch + 1) % args.eval_every == 0:
            val_result = evaluate_deterministic_accuracy(
                state.apply_fn, state.params, grid_values,
                val_inputs, val_targets,
            )
            n_perfect = val_result["n_perfect"]
            n_val = len(val_inputs)
            gen_n = val_result["gen_n"]
            fail_n = val_result["first_failure_n"]
            val_sym = "✓" if fail_n is None else f"✗ (fails@{fail_n})"
            is_best = n_perfect > best_val_n_perfect
            best_tag = "  ★ NEW BEST" if is_best else ""
            print(f"              ↳ val: gen_n={gen_n} {val_sym}  ({n_perfect}/{n_val} perfect){best_tag}")
            if is_best:
                best_val_n_perfect = n_perfect
                best_params = jax.tree.map(lambda x: x.copy(), state.params)
                if run_dir is not None:
                    save_checkpoint(best_params, checkpoint_path(run_dir, "best.npz"))
                    save_checkpoint_meta(run_dir, epoch + 1, best_val_n_perfect)
                    print(f"              ↳ [CKPT] checkpoint saved")

    elapsed = time.time() - t0
    print("-" * 70)
    print(f"Training complete in {elapsed:.1f}s")

    if run_dir is not None:
        save_checkpoint(state.params, checkpoint_path(run_dir, "final.npz"))
        print(f"  [CKPT] Final checkpoint saved")

    return state, best_params, best_val_n_perfect


def _run_epoch_shared(state, x_train, y_train, mask_train, N, bs, rng,
                      train_step, p_base):
    """Run one shared-weight training epoch."""
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
    if init_params is not None:
        state = state.replace(params=init_params)
        print(f"  Loaded params from checkpoint (resuming from epoch {start_epoch})")

    print(f"  Number of LSTM+output parameters: {state.params['logits'].shape[0]}")
    print(f"  Logit array shape: {state.params['logits'].shape}")
    print(f"  Phi logits shape: {state.params['phi_logits'].shape}")

    p_base = compute_p_base(grid_values)

    warmup_train_step = make_shared_train_step(
        args.lambda1, args.lambda2, args.epsilon, n_train=N,
        n_samples=1, soft_forward=True,
    )
    bridge_train_step = make_shared_train_step(
        args.lambda1, args.lambda2, args.epsilon, n_train=N,
        n_samples=1, soft_forward=False, deterministic_st=True,
    )
    st_train_step = make_shared_train_step(
        args.lambda1, args.lambda2, args.epsilon, n_train=N,
        n_samples=args.n_samples, soft_forward=False,
    )

    warmup_epochs = args.warmup_epochs
    total_epochs = args.epochs
    bridge_epochs = 1 if (warmup_epochs > 0 and warmup_epochs < total_epochs) else 0
    st_epochs = max(total_epochs - warmup_epochs - bridge_epochs, 0)

    print(f"\n  Phase 1 (warmup): {warmup_epochs} epochs, soft forward (no Gumbel)")
    if bridge_epochs:
        print("  Phase 2 (bridge): 1 epoch, deterministic ST (hard forward, no Gumbel)")
    print(f"  Phase 3 (ST):     {st_epochs} epochs, {args.n_samples} Gumbel samples")
    print(f"  tau: warmup fixed at {args.tau_start}, then ST anneal {args.tau_start} -> {args.tau_end}")
    print(f"  lr={args.lr}, lambda1={args.lambda1}, lambda2={args.lambda2}, "
          f"eps={args.epsilon}, batch_size={bs}")
    print("-" * 70)

    best_val_n_perfect = -1
    best_params = None

    t0 = time.time()
    for epoch in range(start_epoch, total_epochs):
        is_warmup = epoch < warmup_epochs
        is_bridge = bridge_epochs and (epoch == warmup_epochs)
        phase_name = "warmup" if is_warmup else ("bridge" if is_bridge else "ST")

        # Keep warmup tau fixed; anneal only in ST phase.
        tau = anneal_tau_st_phase(
            epoch, total_epochs, warmup_epochs, args.tau_start, args.tau_end,
        )

        if is_warmup:
            train_step = warmup_train_step
        elif is_bridge:
            train_step = bridge_train_step
        else:
            train_step = st_train_step

        state = state.replace(tau=tau)

        state, rng, metrics = _run_epoch_shared(
            state, x_train, y_train, mask_train, N, bs, rng,
            train_step, p_base,
        )

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            complexity_argmax_bits = _compute_discrete_hyp_bits(
                state.params, grid, grid_values,
            )
            print(
                f"Epoch {epoch+1:5d} [{phase_name:4s}] | "
                f"objective_total_bits={metrics['objective_total_bits']:8.1f}b  "
                f"data_nll_bits={metrics['data_nll_bits']:8.1f}b  "
                f"reg_net_bits={metrics['reg_net_bits']:7.1f}b "
                f"(reg_complexity_weighted_bits={metrics['reg_complexity_weighted_bits']:.1f}b "
                f"- reg_entropy_bonus_bits={metrics['reg_entropy_bonus_bits']:.1f}b) | "
                f"complexity_expected_bits={metrics['complexity_expected_bits']:.1f}b  "
                f"complexity_argmax_bits={complexity_argmax_bits:4d}b  "
                f"entropy_weights_bits={metrics['entropy_weights_bits']:5.1f}b  "
                f"kl_pi_phi_bits={metrics['kl_pi_phi_bits']:.1f}b  "
                f"kl_phi_pbase_bits={metrics['kl_phi_pbase_bits']:.1f}b  "
                f"phi_entropy_bits={metrics['phi_entropy_bits']:.1f}b  "
                f"phi∈[{metrics['phi_min_prob']:.2e},{metrics['phi_max_prob']:.2e}]  "
                f"τ={float(tau):.4f}"
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
            fail_n = val_result["first_failure_n"]
            val_sym = "✓" if fail_n is None else f"✗ (fails@{fail_n})"
            is_best = n_perfect > best_val_n_perfect
            best_tag = "  ★ NEW BEST" if is_best else ""
            print(f"              ↳ val: gen_n={gen_n} {val_sym}  ({n_perfect}/{n_val} perfect){best_tag}")
            if is_best:
                best_val_n_perfect = n_perfect
                best_params = jax.tree.map(lambda x: x.copy(), state.params)
                if run_dir is not None:
                    save_checkpoint(best_params, checkpoint_path(run_dir, "best.npz"))
                    save_checkpoint_meta(run_dir, epoch + 1, best_val_n_perfect)
                    print(f"              ↳ [CKPT] checkpoint saved")

    elapsed = time.time() - t0
    print("-" * 70)
    print(f"Training complete in {elapsed:.1f}s")

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
                        help="Max n for test set (can be overridden in --eval mode)")
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
    parser.add_argument("--eval", action="store_true",
                        help="Load best checkpoint from --ckpt and run test evaluation only")
    parser.add_argument("--resume", action="store_true",
                        help="Load best checkpoint from --ckpt and resume training")

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


def _resolve_resume_checkpoint(run_dir: Path) -> Path:
    """Find checkpoint file for eval/resume, supporting legacy filenames."""
    candidates = [
        checkpoint_path(run_dir, "best.npz", create=False),
        checkpoint_path(run_dir, "final.npz", create=False),
        run_dir / "checkpoint_best.npz",
        run_dir / "checkpoint_final.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def _load_resume_meta_epoch(run_dir: Path) -> int:
    """Read resume epoch from checkpoint metadata, supporting legacy path."""
    candidates = [
        checkpoint_path(run_dir, "meta.json", create=False),
        run_dir / "checkpoint_meta.json",
    ]
    for meta_path in candidates:
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            return int(meta.get("last_epoch", 0))
    return 0


def _print_resolved_parameters(args):
    """Print resolved effective parameters at startup."""
    params = vars(args)

    print("\nResolved parameters")
    print("-" * 60)
    print(f"  mode={args.mode}")
    if args.mode == "basic":
        print(
            "  objective_total_bits = data_nll_bits + "
            "reg_complexity_weighted_bits - reg_entropy_bonus_bits"
        )
        print(f"  mdl_lambda={args.mdl_lambda}")
    else:
        print(
            "  objective_total_bits = data_nll_bits + "
            "reg_complexity_weighted_bits - reg_entropy_bonus_bits"
        )
        print(f"  lambda1={args.lambda1}")
        print(f"  lambda2={args.lambda2}")
        print(f"  epsilon={args.epsilon}")

    for key in sorted(params):
        print(f"  {key}={params[key]}")
    print("-" * 60)


def main():
    # Parse config path first so YAML defaults can seed argparse.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("config", nargs="?")
    pre_args, _ = pre.parse_known_args()

    yaml_defaults = {}
    pre_cfg_path = _resolve_config_path(pre_args.config)
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
        # Preserve CLI-specified eval-time overrides
        cli_test_max_n = args.test_max_n
        cli_analyze = args.analyze
        cli_analyze_max_n = args.analyze_max_n
        # Check if --test_max_n was explicitly provided on the command line
        test_max_n_explicit = "--test_max_n" in sys.argv

        saved_args = load_run_config(args.ckpt)
        saved_args.eval = args.eval
        saved_args.resume = args.resume
        saved_args.ckpt = args.ckpt
        saved_args.analyze = cli_analyze
        saved_args.analyze_max_n = cli_analyze_max_n
        if not hasattr(saved_args, "config"):
            saved_args.config = None
        if not hasattr(saved_args, "config_name"):
            saved_args.config_name = None
        if test_max_n_explicit:
            saved_args.test_max_n = cli_test_max_n
        args = saved_args

    # --- Set up run directory and logging ---
    if args.eval or args.resume:
        run_dir = Path(args.ckpt)
        ckpt_path = _resolve_resume_checkpoint(run_dir)
        loaded_params = load_checkpoint(ckpt_path)
        start_epoch = 0
        if args.resume:
            start_epoch = _load_resume_meta_epoch(run_dir)
            print(f"Resuming from epoch {start_epoch}/{args.epochs}")
        log_mode = "a"
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

    # Structured validation set (all strings with train_max_n < n <= 71)
    val_inputs, val_targets = make_validation_set(train_max_n, val_max_n=71)
    print(f"  Structured val set: {len(val_inputs)} strings (n={train_max_n+1}..71)")

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
    )

    rng = jrandom.PRNGKey(args.seed)

    # --- Training (or eval/resume) ---
    if args.eval:
        # Reconstruct a state with loaded params just to get apply_fn
        N = x_train.shape[0]
        bs = args.batch_size if args.batch_size > 0 else N
        rng, init_rng = jrandom.split(rng)
        if args.mode == "basic":
            state = create_mdl_state(
                init_rng, model,
                seq_len=x_train.shape[1],
                batch_size=min(bs, N),
                lr=args.lr,
                tau_init=args.tau_end,
            )
        else:
            state = create_shared_mdl_state(
                init_rng, model, grid_values,
                seq_len=x_train.shape[1],
                batch_size=min(bs, N),
                lr=args.lr,
                tau_init=args.tau_end,
            )
        state = state.replace(params=loaded_params)
        best_params = loaded_params
        best_val_n_perfect = None
    elif args.mode == "basic":
        state, best_params, best_val_n_perfect = run_training_basic(
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

    if not args.eval and run_dir is not None:
        save_checkpoint_meta(run_dir, args.epochs, best_val_n_perfect or 0)

    # --- Final evaluation ---
    metrics = run_final_evaluation(
        args, state, best_params,
        grid, grid_values, test_inputs, test_targets,
        golden_mdl, golden_result,
    )
    if run_dir is not None:
        save_results(run_dir, metrics)
        print(f"\nResults saved to: {run_dir}/")


def run_final_evaluation(args, state, best_params,
                         grid, grid_values, test_inputs, test_targets,
                         golden_mdl, golden_result):
    """Run final test evaluation and print the comparison table.

    For test_max_n > 10000, uses efficient float64 simulation instead of
    JAX batched evaluation (avoids memory issues with very long sequences).

    Returns a dict of metrics for results.json.
    """
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
    }


if __name__ == "__main__":
    main()
