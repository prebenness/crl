"""Continuous prime-exponent relaxation for MDL-style rational regularization.

Each neural network weight is parameterized as:
    s_i = tanh(u_i) * exp(clamp(z_i^T log(primes)))

where z_i in R^P are exponent coordinates and u_i in R is a sign parameter.
The MDL regularizer is a weighted L1 penalty on exponents:
    R_mdl = sum_i sum_r |z_{i,r}| * log(p_r)

This is fully differentiable and requires no Gumbel sampling.

Supports two tasks:
    - xor: 2D binary classification toy problem (PrimeExpMLP)
    - anbn: a^n b^n language recognition (PrimeExpLSTM)

Usage:
    python prime_rationals.py config/prime_rationals/xor.yaml
    python prime_rationals.py config/prime_rationals/anbn.yaml
"""

import math
import os
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

# XLA_FLAGS must be set before JAX initialises.
if "--deterministic" in sys.argv:
    _xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_deterministic_ops" not in _xla_flags:
        os.environ["XLA_FLAGS"] = (
            _xla_flags + " --xla_gpu_deterministic_ops=true"
        ).strip()

import jax
import jax.numpy as jnp
from jax import random as jrandom
import flax.linen as nn
import numpy as np
import optax
import yaml
from flax.training import train_state

from src.mdl.data import (
    make_anbn_dataset,
    make_test_set,
    make_validation_set,
    sequences_to_padded_arrays,
    SYMBOL_B,
    NUM_SYMBOLS,
)
from src.mdl.evaluation import (
    compute_per_string_nll_bits,
    compute_grammar_weighted_nll_bits,
    compute_train_dh,
    compute_anbn_grammar_weights,
)
from src.utils.checkpointing import (
    TeeLogger,
    make_experiment_dir,
    checkpoint_path,
    save_checkpoint,
    load_checkpoint,
    save_results,
    save_config,
)


# ===========================================================================
# Math utilities
# ===========================================================================

def first_primes(P: int) -> list[int]:
    """Return the first P prime numbers."""
    primes = []
    n = 2
    while len(primes) < P:
        if all(n % p != 0 for p in primes if p * p <= n):
            primes.append(n)
        n += 1
    return primes


def get_log_primes(P: int) -> jnp.ndarray:
    """Return log of first P primes as a JAX array."""
    return jnp.array([math.log(p) for p in first_primes(P)], dtype=jnp.float32)


def reconstruct_weight(z, u, log_primes, clamp_val=10.0):
    """Reconstruct weight from exponent coordinates and sign parameter.

    Args:
        z: (..., P) exponent coordinates.
        u: (...) sign parameters.
        log_primes: (P,) log of prime basis.
        clamp_val: clamp range for log-magnitude.

    Returns:
        weight: (...) reconstructed weights.
    """
    logmag = jnp.sum(z * log_primes, axis=-1)
    logmag = jnp.clip(logmag, -clamp_val, clamp_val)
    sign = jnp.tanh(u)
    return sign * jnp.exp(logmag)


def compute_mdl_penalty(z, log_primes):
    """Compute MDL penalty: sum of |z_{i,r}| * log(p_r).

    Args:
        z: (..., P) exponent coordinates.
        log_primes: (P,) log of prime basis.

    Returns:
        Scalar penalty.
    """
    return jnp.sum(jnp.abs(z) * log_primes)


def discretize(z, u):
    """Round exponents to nearest integer, harden signs.

    Args:
        z: (..., P) exponent coordinates.
        u: (...) sign parameters.

    Returns:
        (rounded_z, hard_sign) tuple.
    """
    rounded_z = jnp.round(z)
    hard_sign = jnp.sign(u)
    hard_sign = jnp.where(hard_sign == 0, jnp.ones_like(hard_sign), hard_sign)
    return rounded_z, hard_sign


def exponents_to_rational(e, primes):
    """Convert integer exponent vector to (numerator, denominator).

    Args:
        e: 1D array/list of integer exponents.
        primes: list of prime numbers.

    Returns:
        (num, den) as Python ints, automatically coprime.
    """
    num = 1
    den = 1
    for exp_val, p in zip(e, primes):
        exp_int = int(round(float(exp_val)))
        if exp_int > 0:
            num *= p ** exp_int
        elif exp_int < 0:
            den *= p ** (-exp_int)
    return num, den


def discretize_params(params, log_primes):
    """Discretize all prime-exponent parameters in a params tree.

    Rounds z_* parameters to nearest integer and hardens u_* signs.
    Returns a new params dict suitable for evaluation.
    """
    def process_leaf(path_str, val):
        if 'z_' in path_str:
            return jnp.round(val)
        elif 'u_' in path_str:
            sign = jnp.sign(val)
            return jnp.where(sign == 0, jnp.ones_like(sign), sign)
        return val

    def traverse(d, prefix=""):
        out = {}
        for k, v in d.items():
            path = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                out[k] = traverse(v, path)
            else:
                out[k] = process_leaf(path, v)
        return out

    return traverse(params)


def reconstruct_weight_discrete(z_rounded, sign_hard, log_primes, clamp_val=10.0):
    """Reconstruct weight from discretized exponents and hard sign.

    Same as reconstruct_weight but uses hard sign directly (no tanh).
    """
    logmag = jnp.sum(z_rounded * log_primes, axis=-1)
    logmag = jnp.clip(logmag, -clamp_val, clamp_val)
    return sign_hard * jnp.exp(logmag)


# ===========================================================================
# Flax modules
# ===========================================================================

def _normal_init(std):
    """Normal initializer with given std."""
    def init(rng, shape, dtype=jnp.float32):
        return std * jrandom.normal(rng, shape, dtype=dtype)
    return init


class PrimeExpLinear(nn.Module):
    """Dense layer with prime-exponent weight parameterization.

    Each weight is s = tanh(u) * exp(clamp(z^T log(primes))).
    """
    features: int
    P: int = 6
    init_std: float = 0.01
    clamp_logmag: float = 10.0
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]
        log_primes = get_log_primes(self.P)

        z_w = self.param('z_weight', _normal_init(self.init_std),
                         (self.features, in_features, self.P))
        u_w = self.param('u_weight', _normal_init(self.init_std),
                         (self.features, in_features))
        W = reconstruct_weight(z_w, u_w, log_primes, self.clamp_logmag)
        out = x @ W.T

        if self.use_bias:
            z_b = self.param('z_bias', _normal_init(self.init_std),
                             (self.features, self.P))
            u_b = self.param('u_bias', _normal_init(self.init_std),
                             (self.features,))
            b = reconstruct_weight(z_b, u_b, log_primes, self.clamp_logmag)
            out = out + b

        return out


class PrimeExpMLP(nn.Module):
    """Simple MLP with prime-exponent weights for toy experiments."""
    hidden_dim: int = 32
    output_dim: int = 1
    P: int = 6
    init_std: float = 0.01
    clamp_logmag: float = 10.0

    @nn.compact
    def __call__(self, x):
        x = PrimeExpLinear(self.hidden_dim, P=self.P,
                           init_std=self.init_std,
                           clamp_logmag=self.clamp_logmag)(x)
        x = nn.relu(x)
        x = PrimeExpLinear(self.output_dim, P=self.P,
                           init_std=self.init_std,
                           clamp_logmag=self.clamp_logmag)(x)
        return x


class PrimeExpLSTM(nn.Module):
    """LSTM with prime-exponent weight parameterization for ANBN.

    Mirrors GumbelSoftmaxLSTM weight layout (108 parameters for H=I=O=3)
    but uses continuous prime-exponent coordinates instead of a categorical
    grid. No tau, rng, or sampling — forward pass is deterministic.
    """
    hidden_size: int = 3
    input_size: int = 3
    output_size: int = 3
    P: int = 6
    init_std: float = 0.01
    clamp_logmag: float = 10.0

    @nn.compact
    def __call__(self, x):
        B, T = x.shape
        H = self.hidden_size
        I = self.input_size
        P = self.P

        # Parameter counts — same layout as GumbelSoftmaxLSTM
        n_lstm_w = 4 * I * H + 4 * H * H   # 72
        n_lstm_b = 4 * H + 4 * H            # 24
        n_out_w = H * self.output_size       # 9
        n_out_b = self.output_size           # 3
        n_total = n_lstm_w + n_lstm_b + n_out_w + n_out_b  # 108

        log_primes = get_log_primes(P)

        z = self.param('z_exponents', _normal_init(self.init_std),
                       (n_total, P))
        u = self.param('u_signs', _normal_init(self.init_std),
                       (n_total,))

        all_weights = reconstruct_weight(z, u, log_primes, self.clamp_logmag)

        # --- Unpack weights (identical layout to GumbelSoftmaxLSTM) ---
        offset = 0

        def take(size):
            nonlocal offset
            w = all_weights[offset:offset + size]
            offset += size
            return w

        W_ii = take(I * H).reshape(I, H)
        W_if = take(I * H).reshape(I, H)
        W_ig = take(I * H).reshape(I, H)
        W_io = take(I * H).reshape(I, H)

        W_hi = take(H * H).reshape(H, H)
        W_hf = take(H * H).reshape(H, H)
        W_hg = take(H * H).reshape(H, H)
        W_ho = take(H * H).reshape(H, H)

        b_ii = take(H)
        b_if = take(H)
        b_ig = take(H)
        b_io = take(H)
        b_hi = take(H)
        b_hf = take(H)
        b_hg = take(H)
        b_ho = take(H)

        W_out = take(H * self.output_size).reshape(H, self.output_size)
        b_out = take(self.output_size)

        assert offset == n_total

        # --- One-hot encode input ---
        x_onehot = jax.nn.one_hot(x, I)  # (B, T, I)

        # --- LSTM scan ---
        def lstm_step(carry, x_t):
            h, c = carry
            i_t = jax.nn.sigmoid(x_t @ W_ii + b_ii + h @ W_hi + b_hi)
            f_t = jax.nn.sigmoid(x_t @ W_if + b_if + h @ W_hf + b_hf)
            g_t = jnp.tanh(x_t @ W_ig + b_ig + h @ W_hg + b_hg)
            o_t = jax.nn.sigmoid(x_t @ W_io + b_io + h @ W_ho + b_ho)
            c = f_t * c + i_t * g_t
            h = o_t * jnp.tanh(c)
            return (h, c), h

        h0 = jnp.zeros((B, H))
        c0 = jnp.zeros((B, H))
        x_seq = jnp.transpose(x_onehot, (1, 0, 2))  # (T, B, I)
        (h_final, c_final), h_seq = jax.lax.scan(lstm_step, (h0, c0), x_seq)
        h_seq = jnp.transpose(h_seq, (1, 0, 2))  # (B, T, H)

        # --- Output layer ---
        logits = h_seq @ W_out + b_out  # (B, T, output_size)

        mdl_pen = compute_mdl_penalty(z, log_primes)

        aux = {
            "mdl_penalty": mdl_pen,
            "z_exponents": z,
            "u_signs": u,
            "n_params": n_total,
        }
        return logits, aux


class StandardLSTM(nn.Module):
    """Standard LSTM with real-valued weights for baseline comparison.

    Same architecture and weight layout as PrimeExpLSTM (108 parameters for
    H=I=O=3) but with unconstrained real-valued weights. Serves as a baseline
    for no-reg, L1, and L2 experiments.
    """
    hidden_size: int = 3
    input_size: int = 3
    output_size: int = 3
    init_std: float = 0.1

    @nn.compact
    def __call__(self, x):
        B, T = x.shape
        H = self.hidden_size
        I = self.input_size

        n_lstm_w = 4 * I * H + 4 * H * H  # 72
        n_lstm_b = 4 * H + 4 * H           # 24
        n_out_w = H * self.output_size      # 9
        n_out_b = self.output_size          # 3
        n_total = n_lstm_w + n_lstm_b + n_out_w + n_out_b  # 108

        weights = self.param('weights', _normal_init(self.init_std), (n_total,))

        # --- Unpack weights (identical layout to PrimeExpLSTM) ---
        offset = 0

        def take(size):
            nonlocal offset
            w = weights[offset:offset + size]
            offset += size
            return w

        W_ii = take(I * H).reshape(I, H)
        W_if = take(I * H).reshape(I, H)
        W_ig = take(I * H).reshape(I, H)
        W_io = take(I * H).reshape(I, H)

        W_hi = take(H * H).reshape(H, H)
        W_hf = take(H * H).reshape(H, H)
        W_hg = take(H * H).reshape(H, H)
        W_ho = take(H * H).reshape(H, H)

        b_ii = take(H)
        b_if = take(H)
        b_ig = take(H)
        b_io = take(H)
        b_hi = take(H)
        b_hf = take(H)
        b_hg = take(H)
        b_ho = take(H)

        W_out = take(H * self.output_size).reshape(H, self.output_size)
        b_out = take(self.output_size)

        assert offset == n_total

        # --- One-hot encode input ---
        x_onehot = jax.nn.one_hot(x, I)  # (B, T, I)

        # --- LSTM scan ---
        def lstm_step(carry, x_t):
            h, c = carry
            i_t = jax.nn.sigmoid(x_t @ W_ii + b_ii + h @ W_hi + b_hi)
            f_t = jax.nn.sigmoid(x_t @ W_if + b_if + h @ W_hf + b_hf)
            g_t = jnp.tanh(x_t @ W_ig + b_ig + h @ W_hg + b_hg)
            o_t = jax.nn.sigmoid(x_t @ W_io + b_io + h @ W_ho + b_ho)
            c = f_t * c + i_t * g_t
            h = o_t * jnp.tanh(c)
            return (h, c), h

        h0 = jnp.zeros((B, H))
        c0 = jnp.zeros((B, H))
        x_seq = jnp.transpose(x_onehot, (1, 0, 2))  # (T, B, I)
        (h_final, c_final), h_seq = jax.lax.scan(lstm_step, (h0, c0), x_seq)
        h_seq = jnp.transpose(h_seq, (1, 0, 2))  # (B, T, H)

        # --- Output layer ---
        logits = h_seq @ W_out + b_out  # (B, T, output_size)

        aux = {
            "l1_penalty": jnp.sum(jnp.abs(weights)),
            "l2_penalty": jnp.sum(weights ** 2),
            "n_params": n_total,
        }
        return logits, aux


# ===========================================================================
# Loss functions
# ===========================================================================

def collect_mdl_penalty_from_params(params, P):
    """Traverse params tree and sum MDL penalty over all z_* tensors."""
    log_primes = get_log_primes(P)
    total = 0.0

    def traverse(d):
        nonlocal total
        for k, v in d.items():
            if isinstance(v, dict):
                traverse(v)
            elif 'z_' in k:
                total += compute_mdl_penalty(v, log_primes)

    traverse(params)
    return total


def cross_entropy_bits(logits, targets, mask):
    """Cross-entropy loss in bits, averaged over valid positions.

    Args:
        logits: (B, T, V) output logits.
        targets: (B, T) int target indices.
        mask: (B, T) float mask (1 where valid).

    Returns:
        Scalar loss in bits.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    nll_bits = nll / jnp.log(2.0)
    n_valid = jnp.sum(mask)
    return jnp.sum(nll_bits * mask) / jnp.maximum(n_valid, 1.0)


def binary_cross_entropy_with_logits(logits, targets):
    """Binary cross-entropy from logits, averaged over samples."""
    return jnp.mean(
        jnp.maximum(logits, 0) - logits * targets
        + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )


# ===========================================================================
# Training
# ===========================================================================

def create_xor_train_state(model, rng, lr):
    """Initialize train state for XOR experiment."""
    dummy_x = jnp.zeros((1, 2), dtype=jnp.float32)
    params = model.init(rng, dummy_x)["params"]
    tx = optax.adam(lr)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
    )


def create_anbn_train_state(model, rng, seq_len, batch_size, lr):
    """Initialize train state for ANBN experiment."""
    dummy_x = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_x)["params"]
    tx = optax.adam(lr)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
    )


def make_xor_train_step(lambda_mdl, P):
    """Create JIT-compiled XOR training step."""
    log_primes = get_log_primes(P)

    @jax.jit
    def train_step(state, x, y):
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, x)
            task_loss = binary_cross_entropy_with_logits(logits, y)
            mdl_reg = collect_mdl_penalty_from_params(params, P)
            loss = task_loss + lambda_mdl * mdl_reg
            return loss, {"task_loss": task_loss, "mdl_reg": mdl_reg}

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, aux

    return train_step


def make_anbn_train_step(lambda_mdl, n_train, P):
    """Create JIT-compiled ANBN training step."""
    log_primes = get_log_primes(P)

    @jax.jit
    def train_step(state, x, y, mask):
        def loss_fn(params):
            logits, aux = state.apply_fn({"params": params}, x)
            data_nll = cross_entropy_bits(logits, y, mask)
            mdl_reg = aux["mdl_penalty"]
            loss = data_nll + (lambda_mdl / n_train) * mdl_reg
            return loss, {
                "data_nll": data_nll,
                "mdl_reg": mdl_reg,
            }

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, aux

    return train_step


def make_anbn_baseline_train_step(lambda_mdl, n_train, reg):
    """Create JIT-compiled ANBN baseline training step.

    Args:
        reg: "none", "l1", or "l2".
    """
    @jax.jit
    def train_step(state, x, y, mask):
        def loss_fn(params):
            logits, aux = state.apply_fn({"params": params}, x)
            data_nll = cross_entropy_bits(logits, y, mask)
            if reg == "l1":
                penalty = aux["l1_penalty"]
            elif reg == "l2":
                penalty = aux["l2_penalty"]
            else:
                penalty = 0.0
            loss = data_nll + (lambda_mdl / n_train) * penalty
            return loss, {
                "data_nll": data_nll,
                "reg_penalty": penalty if reg != "none" else 0.0,
            }

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, aux

    return train_step


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_xor(model, params, x_test, y_test):
    """Evaluate XOR model: accuracy and loss."""
    logits = model.apply({"params": params}, x_test)
    preds = (logits.squeeze(-1) > 0).astype(jnp.float32)
    accuracy = jnp.mean(preds == y_test)
    loss = binary_cross_entropy_with_logits(logits, y_test[:, None])
    return {"accuracy": float(accuracy), "loss": float(loss)}


def evaluate_anbn_accuracy(model, params, test_inputs, test_targets,
                           batch_size=64):
    """Evaluate deterministic accuracy on a^n b^n test set.

    Same logic as training.evaluate_deterministic_accuracy but without
    tau/rng arguments (prime-exp forward is deterministic).

    Returns per-string accuracies and summary statistics.
    """
    N = len(test_inputs)
    accs = np.zeros(N, dtype=np.float32)

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_inputs = test_inputs[batch_start:batch_end]
        batch_targets = test_targets[batch_start:batch_end]
        B = len(batch_inputs)

        max_len = max(len(s) for s in batch_inputs)
        x_pad = np.zeros((B, max_len), dtype=np.int32)
        y_pad = np.zeros((B, max_len), dtype=np.int32)
        det_mask = np.zeros((B, max_len), dtype=np.float32)

        for i, (inp, tgt) in enumerate(zip(batch_inputs, batch_targets)):
            L = len(inp)
            x_pad[i, :L] = inp
            y_pad[i, :L] = tgt
            for t in range(L):
                if inp[t] == SYMBOL_B:
                    det_mask[i, t] = 1.0

        x_jnp = jnp.array(x_pad)
        y_jnp = jnp.array(y_pad)
        det_mask_jnp = jnp.array(det_mask)

        logits, _ = model.apply({"params": params}, x_jnp)
        preds = jnp.argmax(logits, axis=-1)

        correct = (preds == y_jnp).astype(jnp.float32)
        n_det = jnp.sum(det_mask_jnp, axis=-1)
        n_correct = jnp.sum(correct * det_mask_jnp, axis=-1)
        batch_accs = jnp.where(n_det > 0, n_correct / n_det, 1.0)
        accs[batch_start:batch_end] = np.array(batch_accs)

    perfect = accs >= 1.0 - 1e-6
    n_perfect = int(np.sum(perfect))

    # gen_n: largest n such that all strings 1..n are perfect
    gen_n = 0
    for i in range(N):
        if perfect[i]:
            gen_n = i + 1
        else:
            break

    first_failure = None
    for i in range(N):
        if not perfect[i]:
            first_failure = i + 1  # n is 1-indexed
            break

    return {
        "mean_accuracy": float(np.mean(accs)),
        "n_perfect": n_perfect,
        "gen_n": gen_n,
        "first_failure_n": first_failure,
        "per_string_acc": accs,
    }


def make_forward_fn(model, params):
    """Create forward_fn compatible with evaluation.compute_per_string_nll_bits."""
    def forward_fn(x_batch):
        logits, _ = model.apply({"params": params}, x_batch)
        return logits
    return forward_fn


def compute_h_bits(params, P):
    """Compute hypothesis codelength |H| for discretized prime-exp model.

    After rounding exponents to integers, the codelength is the MDL penalty
    evaluated at the rounded values: sum |e_{i,r}| * log(p_r).
    """
    log_primes = get_log_primes(P)
    disc_params = discretize_params(params, log_primes)
    return float(collect_mdl_penalty_from_params(disc_params, P))


# ===========================================================================
# Discretization & analysis
# ===========================================================================

def print_rational_weights_lstm(params, P):
    """Print LSTM weights as rational fractions after discretization."""
    primes = first_primes(P)
    log_primes = get_log_primes(P)

    z = np.array(params["z_exponents"])
    u = np.array(params["u_signs"])

    z_rounded = np.round(z).astype(int)
    signs = np.sign(u)
    signs[signs == 0] = 1

    H = 3  # hidden size for ANBN
    I = 3  # input size
    O = 3  # output size

    labels = []
    for gate in ["i", "f", "g", "o"]:
        for row in range(I):
            for col in range(H):
                labels.append(f"W_{gate}i[{row},{col}]")
    for gate in ["i", "f", "g", "o"]:
        for row in range(H):
            for col in range(H):
                labels.append(f"W_{gate}h[{row},{col}]")
    for gate in ["i", "f", "g", "o"]:
        for col in range(H):
            labels.append(f"b_{gate}i[{col}]")
    for gate in ["i", "f", "g", "o"]:
        for col in range(H):
            labels.append(f"b_{gate}h[{col}]")
    for row in range(H):
        for col in range(O):
            labels.append(f"W_out[{row},{col}]")
    for col in range(O):
        labels.append(f"b_out[{col}]")

    n_total = len(labels)
    n_head = 5
    n_tail = 5

    def format_weight(i):
        num, den = exponents_to_rational(z_rounded[i], primes)
        sign_str = "-" if signs[i] < 0 else "+"
        exp_str = ", ".join(f"{int(z_rounded[i, r])}" for r in range(len(primes)))
        frac_str = f"{num}/{den}" if den > 1 else f"{num}"
        w_val = float(reconstruct_weight_discrete(
            jnp.array(z_rounded[i], dtype=jnp.float32),
            jnp.array(signs[i], dtype=jnp.float32),
            log_primes,
        ))
        return (f"  {labels[i]:20s} = {sign_str}{frac_str:>10s}  "
                f"(exp=[{exp_str}], val={w_val:+.6f})")

    print("\n=== Discretized rational weights ===")
    if n_total <= n_head + n_tail + 2:
        for i in range(n_total):
            print(format_weight(i))
    else:
        for i in range(n_head):
            print(format_weight(i))
        print(f"  {'...':20s}   ({n_total - n_head - n_tail} more)")
        for i in range(n_total - n_tail, n_total):
            print(format_weight(i))


def print_rational_weights_mlp(params, P):
    """Print MLP weights as rational fractions after discretization."""
    primes = first_primes(P)
    log_primes = get_log_primes(P)

    print("\n=== Discretized rational weights ===")
    for layer_name, layer_params in params.items():
        print(f"\n  Layer: {layer_name}")
        for param_name, val in layer_params.items():
            if 'z_' in param_name:
                # Find corresponding u parameter
                u_name = param_name.replace('z_', 'u_')
                u_val = np.array(layer_params[u_name])
                z_val = np.array(val)

                z_rounded = np.round(z_val).astype(int)
                signs = np.sign(u_val)
                signs[signs == 0] = 1

                flat_z = z_rounded.reshape(-1, z_rounded.shape[-1])
                flat_signs = signs.flatten()

                n_total = len(flat_z)
                n_head = 5
                n_tail = 5
                if n_total <= n_head + n_tail + 2:
                    for i in range(n_total):
                        num, den = exponents_to_rational(flat_z[i], primes)
                        sign_str = "-" if flat_signs[i] < 0 else "+"
                        frac = f"{num}/{den}" if den > 1 else str(num)
                        print(f"    [{i}] = {sign_str}{frac}")
                else:
                    for i in range(n_head):
                        num, den = exponents_to_rational(flat_z[i], primes)
                        sign_str = "-" if flat_signs[i] < 0 else "+"
                        frac = f"{num}/{den}" if den > 1 else str(num)
                        print(f"    [{i}] = {sign_str}{frac}")
                    print(f"    ... ({n_total - n_head - n_tail} more)")
                    for i in range(n_total - n_tail, n_total):
                        num, den = exponents_to_rational(flat_z[i], primes)
                        sign_str = "-" if flat_signs[i] < 0 else "+"
                        frac = f"{num}/{den}" if den > 1 else str(num)
                        print(f"    [{i}] = {sign_str}{frac}")


# ===========================================================================
# XOR experiment
# ===========================================================================

def _zero_adam_moments(opt_state):
    """Zero mu and nu in an Adam optimizer state, preserving count."""
    adam_state, scale_state = opt_state
    zeroed = adam_state._replace(
        mu=jax.tree.map(jnp.zeros_like, adam_state.mu),
        nu=jax.tree.map(jnp.zeros_like, adam_state.nu),
    )
    return (zeroed, scale_state)


def _maybe_restart(state, best_params, best_opt_state, epochs_since_best,
                   restart_patience):
    """Check stall and restart from best checkpoint if patience exceeded.

    Returns (state, epochs_since_best, did_restart).
    """
    if restart_patience <= 0 or best_params is None:
        return state, epochs_since_best, False
    if epochs_since_best < restart_patience:
        return state, epochs_since_best, False
    new_opt_state = _zero_adam_moments(best_opt_state)
    state = state.replace(params=best_params, opt_state=new_opt_state)
    return state, 0, True


def _should_update_best(n_perfect, gen_n, complexity_bits,
                        best_val_n_perfect, best_val_gen_n,
                        best_complexity_bits):
    """Prefer better validation coverage, then lower complexity.

    Same logic as differentiable_mdl.py.
    """
    if n_perfect != best_val_n_perfect:
        return n_perfect > best_val_n_perfect
    if gen_n != best_val_gen_n:
        return gen_n > best_val_gen_n
    return complexity_bits < best_complexity_bits


def make_xor_data(n_train, noise_std, seed=0):
    """Generate noisy XOR dataset.

    Points in 4 quadrants of [-1,1]^2, label = XOR of signs.
    """
    rng = np.random.RandomState(seed)
    centers = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    labels = np.array([0.0, 1.0, 1.0, 0.0])

    indices = rng.randint(0, 4, size=n_train)
    x = centers[indices] + rng.randn(n_train, 2) * noise_std
    y = labels[indices]

    return x.astype(np.float32), y.astype(np.float32)


def make_xor_test_grid(resolution=50):
    """Create a dense test grid for XOR evaluation."""
    xs = np.linspace(-1.5, 1.5, resolution)
    ys = np.linspace(-1.5, 1.5, resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    labels = ((grid[:, 0] > 0) ^ (grid[:, 1] > 0)).astype(np.float32)
    return grid, labels


def run_xor_experiment(config):
    """Run XOR toy experiment with prime-exponent MLP."""
    print("=" * 60)
    print("Prime-Exponent XOR Experiment")
    print("=" * 60)

    run_dir = make_experiment_dir("prime_rationals_xor", f"P{config.P}_lam{config.lambda_mdl}")

    print(f"Results dir: {run_dir}")
    save_config(run_dir, vars(config))

    rng = jrandom.PRNGKey(config.seed)

    # Data
    x_train, y_train = make_xor_data(config.xor_n_train, config.xor_noise_std,
                                     seed=config.seed)
    x_test, y_test = make_xor_test_grid()
    x_train_jnp = jnp.array(x_train)
    y_train_jnp = jnp.array(y_train)
    x_test_jnp = jnp.array(x_test)
    y_test_jnp = jnp.array(y_test)

    print(f"Train: {len(x_train)} samples, Test grid: {len(x_test)} points")
    print(f"P={config.P}, lambda={config.lambda_mdl}, lr={config.lr}")

    # Model
    model = PrimeExpMLP(
        hidden_dim=config.hidden_dim,
        output_dim=1,
        P=config.P,
        init_std=config.init_std,
        clamp_logmag=config.clamp_logmag,
    )
    state = create_xor_train_state(model, rng, config.lr)
    n_params = sum(p.size for p in jax.tree.leaves(state.params))
    print(f"Model parameters: {n_params}")

    train_step = make_xor_train_step(config.lambda_mdl, config.P)

    # Training loop
    t0 = time.monotonic()
    best_acc = 0.0
    best_xor_params = None
    best_xor_opt_state = None
    best_xor_epoch = 0
    epochs_since_best = 0
    n_restarts = 0

    for epoch in range(1, config.epochs + 1):
        state, loss, aux = train_step(state, x_train_jnp,
                                      y_train_jnp[:, None])

        if epoch % config.log_every == 0 or epoch == 1:
            elapsed = time.monotonic() - t0
            train_metrics = evaluate_xor(model, state.params,
                                         x_train_jnp, y_train_jnp)
            print(f"  epoch {epoch:5d} | loss={float(loss):.4f} | "
                  f"bce={float(aux['task_loss']):.4f} | "
                  f"mdl={float(aux['mdl_reg']):.4f} | "
                  f"train_acc={train_metrics['accuracy']:.4f} | "
                  f"time={elapsed:.1f}s")

        if epoch % config.eval_every == 0 or epoch == config.epochs:
            metrics = evaluate_xor(model, state.params, x_test_jnp,
                                   y_test_jnp)
            is_best = metrics['accuracy'] > best_acc
            best_tag = "  * NEW BEST" if is_best else ""
            print(f"  [eval] test_acc={metrics['accuracy']:.4f}{best_tag}")
            if is_best:
                best_acc = metrics['accuracy']
                best_xor_params = jax.tree.map(lambda x: x.copy(), state.params)
                best_xor_opt_state = state.opt_state
                best_xor_epoch = epoch
                epochs_since_best = 0
                save_checkpoint({"params": state.params},
                                str(checkpoint_path(run_dir, "best.npz")))
                print(f"              -> [CKPT] best checkpoint saved (epoch {epoch})")
            else:
                epochs_since_best += config.eval_every
                state, epochs_since_best, did_restart = _maybe_restart(
                    state, best_xor_params, best_xor_opt_state,
                    epochs_since_best, config.restart_patience,
                )
                if did_restart:
                    n_restarts += 1
                    print(f"  ↻ RESTART #{n_restarts} at epoch {epoch}"
                          f" → best checkpoint (epoch {best_xor_epoch})")

    elapsed = time.monotonic() - t0
    print(f"Training complete in {elapsed:.1f}s"
          f"{f' ({n_restarts} restarts)' if n_restarts else ''}")

    # Save final checkpoint
    save_checkpoint({"params": state.params},
                    str(checkpoint_path(run_dir, "final.npz")))
    print(f"  [CKPT] Final checkpoint saved")

    # Final evaluation
    print("\n--- Final evaluation (continuous) ---")
    final_metrics = evaluate_xor(model, state.params, x_test_jnp, y_test_jnp)
    print(f"  Test accuracy: {final_metrics['accuracy']:.4f}")

    # Discretized evaluation
    print("\n--- Discretized evaluation ---")
    disc_params = discretize_params(state.params, get_log_primes(config.P))
    disc_metrics = evaluate_xor(model, disc_params, x_test_jnp, y_test_jnp)
    print(f"  Test accuracy (discrete): {disc_metrics['accuracy']:.4f}")

    # Print rational weights
    print_rational_weights_mlp(state.params, config.P)

    # Exponent statistics
    all_z = []
    for leaf in jax.tree.leaves(state.params):
        arr = np.array(leaf)
        if arr.ndim >= 1 and arr.shape[-1] == config.P:
            all_z.append(arr.flatten())
    if all_z:
        all_z = np.concatenate(all_z)
        print(f"\n  Exponent stats: mean={np.mean(np.abs(all_z)):.4f}, "
              f"max={np.max(np.abs(all_z)):.4f}, "
              f"near-zero (<0.1): {np.mean(np.abs(all_z) < 0.1)*100:.1f}%")

    # Save results
    results = {
        "continuous_accuracy": final_metrics["accuracy"],
        "discrete_accuracy": disc_metrics["accuracy"],
        "best_accuracy": best_acc,
    }
    save_results(run_dir, results)

    print(f"\nResults saved to {run_dir}")
    return state, results


# ===========================================================================
# ANBN experiment
# ===========================================================================

def run_anbn_experiment(config):
    """Run ANBN experiment with prime-exponent LSTM."""
    print("=" * 60)
    print("Prime-Exponent ANBN Experiment")
    print("=" * 60)

    run_dir = make_experiment_dir(
        "prime_rationals_anbn",
        f"P{config.P}_lam{config.lambda_mdl}_lr{config.lr}",
    )

    print(f"Results dir: {run_dir}")
    save_config(run_dir, vars(config))

    rng = jrandom.PRNGKey(config.seed)

    # --- Data ---
    train_inputs, train_targets = make_anbn_dataset(
        num_strings=config.num_train, p=config.p, seed=config.data_seed,
    )
    x_train, y_train, mask_train = sequences_to_padded_arrays(
        train_inputs, train_targets,
    )
    n_train = len(train_inputs)

    test_inputs, test_targets = make_test_set(max_n=config.test_max_n)

    # Determine max n in training data for validation range
    train_max_n = max(
        sum(1 for tok in inp if tok == 1)  # count 'a' tokens
        for inp in train_inputs
    )
    val_inputs, val_targets = make_validation_set(
        train_max_n, val_max_n=config.val_max_n, val_min_n=config.val_min_n,
    )

    print(f"Train: {n_train} strings (max_n={train_max_n})")
    print(f"Validation: {len(val_inputs)} strings")
    print(f"Test: {len(test_inputs)} strings (max_n={config.test_max_n})")
    print(f"P={config.P}, lambda={config.lambda_mdl}, lr={config.lr}")

    # --- Model ---
    model = PrimeExpLSTM(
        hidden_size=config.hidden_size,
        input_size=NUM_SYMBOLS,
        output_size=NUM_SYMBOLS,
        P=config.P,
        init_std=config.init_std,
        clamp_logmag=config.clamp_logmag,
    )

    state = create_anbn_train_state(
        model, rng, x_train.shape[1], x_train.shape[0], config.lr,
    )
    n_model_params = sum(p.size for p in jax.tree.leaves(state.params))
    print(f"Model parameters: {n_model_params} "
          f"(108 weights x P={config.P} exponents + 108 signs)")

    train_step = make_anbn_train_step(config.lambda_mdl, n_train, config.P)

    # --- Training loop ---
    t0 = time.monotonic()
    best_val_n_perfect = -1
    best_val_gen_n = -1
    best_complexity = math.inf
    best_params = state.params
    best_opt_state = state.opt_state
    best_epoch = 0
    epochs_since_best = 0
    n_restarts = 0

    for epoch in range(1, config.epochs + 1):
        state, loss, aux = train_step(state, x_train, y_train, mask_train)

        if epoch % config.log_every == 0 or epoch == 1:
            elapsed = time.monotonic() - t0
            print(f"  epoch {epoch:5d} | loss={float(loss):.6f} | "
                  f"nll={float(aux['data_nll']):.4f} bits | "
                  f"mdl={float(aux['mdl_reg']):.2f} | "
                  f"time={elapsed:.1f}s")

        if epoch % config.eval_every == 0 or epoch == config.epochs:
            val_result = evaluate_anbn_accuracy(
                model, state.params, val_inputs, val_targets,
            )
            current_complexity = float(aux['mdl_reg'])
            n_perfect = val_result['n_perfect']
            gen_n = val_result['gen_n']
            n_val = len(val_inputs)

            # Same best-model logic as differentiable_mdl.py:
            # prefer n_perfect > gen_n > lower complexity
            is_best = _should_update_best(
                n_perfect, gen_n, current_complexity,
                best_val_n_perfect, best_val_gen_n, best_complexity,
            )
            best_tag = "  * NEW BEST" if is_best else ""

            print(f"  [val] acc={val_result['mean_accuracy']:.4f} | "
                  f"n_perfect={n_perfect}/{n_val} | "
                  f"gen_n={gen_n} | "
                  f"mdl={current_complexity:.2f}{best_tag}")

            if is_best:
                best_val_n_perfect = n_perfect
                best_val_gen_n = gen_n
                best_complexity = current_complexity
                best_params = jax.tree.map(lambda x: x.copy(), state.params)
                best_opt_state = state.opt_state
                best_epoch = epoch
                epochs_since_best = 0
                save_checkpoint(
                    {"params": state.params},
                    str(checkpoint_path(run_dir, "best.npz")),
                )
                print(f"              -> [CKPT] best checkpoint saved (epoch {epoch})")
            else:
                epochs_since_best += config.eval_every
                state, epochs_since_best, did_restart = _maybe_restart(
                    state, best_params, best_opt_state,
                    epochs_since_best, config.restart_patience,
                )
                if did_restart:
                    n_restarts += 1
                    print(f"  ↻ RESTART #{n_restarts} at epoch {epoch}"
                          f" → best checkpoint (epoch {best_epoch})")

    elapsed = time.monotonic() - t0
    print(f"Training complete in {elapsed:.1f}s"
          f"{f' ({n_restarts} restarts)' if n_restarts else ''}")

    # Save final checkpoint
    save_checkpoint({"params": state.params},
                    str(checkpoint_path(run_dir, "final.npz")))
    print(f"  [CKPT] Final checkpoint saved")

    # --- Final evaluation ---
    print("\n" + "=" * 60)
    print("Final Evaluation (continuous weights)")
    print("=" * 60)

    # Accuracy on test set
    test_result = evaluate_anbn_accuracy(
        model, best_params, test_inputs, test_targets,
    )
    print(f"  Test accuracy: {test_result['mean_accuracy']:.4f}")
    print(f"  Perfect strings: {test_result['n_perfect']}/{len(test_inputs)}")
    print(f"  gen_n: {test_result['gen_n']}")
    if test_result['first_failure_n'] is not None:
        print(f"  First failure at n={test_result['first_failure_n']}")

    # Grammar-weighted |D:H|
    forward_fn = make_forward_fn(model, best_params)
    test_dh = compute_grammar_weighted_nll_bits(
        forward_fn, config.test_max_n, p=config.p, verbose=True,
    )
    print(f"  Test |D:H| = {test_dh['data_dh_bits']:.4f} bits")

    # Train |D:H|
    train_dh = compute_train_dh(forward_fn, train_inputs, train_targets)
    print(f"  Train |D:H| = {train_dh['train_dh_data_bits']:.4f} bits")

    # |H| (hypothesis codelength)
    h_bits = compute_h_bits(best_params, config.P)
    print(f"  |H| = {h_bits:.4f} bits")
    print(f"  Total MDL = {test_dh['data_dh_bits'] + h_bits:.4f} bits")

    # --- Discretized evaluation ---
    print("\n" + "=" * 60)
    print("Discretized Evaluation (rounded exponents)")
    print("=" * 60)

    disc_params = discretize_params(best_params, get_log_primes(config.P))
    disc_test = evaluate_anbn_accuracy(
        model, disc_params, test_inputs, test_targets,
    )
    print(f"  Test accuracy: {disc_test['mean_accuracy']:.4f}")
    print(f"  Perfect strings: {disc_test['n_perfect']}/{len(test_inputs)}")
    print(f"  gen_n: {disc_test['gen_n']}")

    disc_forward_fn = make_forward_fn(model, disc_params)
    disc_test_dh = compute_grammar_weighted_nll_bits(
        disc_forward_fn, config.test_max_n, p=config.p,
    )
    print(f"  Test |D:H| (discrete) = {disc_test_dh['data_dh_bits']:.4f} bits")

    disc_h_bits = compute_h_bits(disc_params, config.P)
    print(f"  |H| (discrete) = {disc_h_bits:.4f} bits")
    print(f"  Total MDL (discrete) = "
          f"{disc_test_dh['data_dh_bits'] + disc_h_bits:.4f} bits")

    # Print rational weights
    print_rational_weights_lstm(best_params, config.P)

    # Exponent statistics
    z = np.array(best_params["z_exponents"])
    print(f"\n  Exponent stats: mean|z|={np.mean(np.abs(z)):.4f}, "
          f"max|z|={np.max(np.abs(z)):.4f}, "
          f"near-zero (<0.1): {np.mean(np.abs(z) < 0.1)*100:.1f}%")

    # --- Save results ---
    results = {
        "best_epoch": best_epoch,
        "best_val_n_perfect": best_val_n_perfect,
        "test_accuracy": test_result["mean_accuracy"],
        "test_n_perfect": test_result["n_perfect"],
        "test_gen_n": test_result["gen_n"],
        "test_first_failure_n": test_result["first_failure_n"],
        "test_dh_bits": test_dh["data_dh_bits"],
        "train_dh_bits": train_dh["train_dh_data_bits"],
        "h_bits": h_bits,
        "disc_test_accuracy": disc_test["mean_accuracy"],
        "disc_test_n_perfect": disc_test["n_perfect"],
        "disc_test_gen_n": disc_test["gen_n"],
        "disc_test_dh_bits": disc_test_dh["data_dh_bits"],
        "disc_h_bits": disc_h_bits,
    }
    save_results(run_dir, results)

    print(f"\nResults saved to {run_dir}")
    return state, results


# ===========================================================================
# ANBN baseline experiment (no reg / L1 / L2)
# ===========================================================================

def run_anbn_baseline_experiment(config):
    """Run ANBN experiment with standard LSTM (no reg / L1 / L2)."""
    reg = config.reg
    print("=" * 60)
    print(f"Standard LSTM ANBN Baseline (reg={reg})")
    print("=" * 60)

    run_dir = make_experiment_dir(
        f"anbn_baseline_{reg}",
        f"lam{config.lambda_mdl}_lr{config.lr}",
    )
    print(f"Results dir: {run_dir}")
    save_config(run_dir, vars(config))

    rng = jrandom.PRNGKey(config.seed)

    # --- Data ---
    train_inputs, train_targets = make_anbn_dataset(
        num_strings=config.num_train, p=config.p, seed=config.data_seed,
    )
    x_train, y_train, mask_train = sequences_to_padded_arrays(
        train_inputs, train_targets,
    )
    n_train = len(train_inputs)

    test_inputs, test_targets = make_test_set(max_n=config.test_max_n)

    train_max_n = max(
        sum(1 for tok in inp if tok == 1)
        for inp in train_inputs
    )
    val_inputs, val_targets = make_validation_set(
        train_max_n, val_max_n=config.val_max_n, val_min_n=config.val_min_n,
    )

    print(f"Train: {n_train} strings (max_n={train_max_n})")
    print(f"Validation: {len(val_inputs)} strings")
    print(f"Test: {len(test_inputs)} strings (max_n={config.test_max_n})")
    lam_str = f"lambda={config.lambda_mdl}" if reg != "none" else "lambda=N/A"
    print(f"reg={reg}, {lam_str}, lr={config.lr}")

    # --- Model ---
    model = StandardLSTM(
        hidden_size=config.hidden_size,
        input_size=NUM_SYMBOLS,
        output_size=NUM_SYMBOLS,
        init_std=config.init_std,
    )
    state = create_anbn_train_state(
        model, rng, x_train.shape[1], x_train.shape[0], config.lr,
    )
    n_model_params = sum(p.size for p in jax.tree.leaves(state.params))
    print(f"Model parameters: {n_model_params}")

    train_step = make_anbn_baseline_train_step(
        config.lambda_mdl, n_train, reg,
    )

    # --- Training loop ---
    t0 = time.monotonic()
    best_val_n_perfect = -1
    best_val_gen_n = -1
    best_complexity = math.inf
    best_params = state.params
    best_opt_state = state.opt_state
    best_epoch = 0
    epochs_since_best = 0
    n_restarts = 0

    for epoch in range(1, config.epochs + 1):
        state, loss, aux = train_step(state, x_train, y_train, mask_train)

        if epoch % config.log_every == 0 or epoch == 1:
            elapsed = time.monotonic() - t0
            reg_str = f"reg={float(aux['reg_penalty']):.4f} | " if reg != "none" else ""
            print(f"  epoch {epoch:5d} | loss={float(loss):.6f} | "
                  f"nll={float(aux['data_nll']):.4f} bits | "
                  f"{reg_str}"
                  f"time={elapsed:.1f}s")

        if epoch % config.eval_every == 0 or epoch == config.epochs:
            val_result = evaluate_anbn_accuracy(
                model, state.params, val_inputs, val_targets,
            )
            current_complexity = float(aux['reg_penalty']) if reg != "none" else 0.0
            n_perfect = val_result['n_perfect']
            gen_n = val_result['gen_n']
            n_val = len(val_inputs)

            is_best = _should_update_best(
                n_perfect, gen_n, current_complexity,
                best_val_n_perfect, best_val_gen_n, best_complexity,
            )
            best_tag = "  * NEW BEST" if is_best else ""

            print(f"  [val] acc={val_result['mean_accuracy']:.4f} | "
                  f"n_perfect={n_perfect}/{n_val} | "
                  f"gen_n={gen_n}{best_tag}")

            if is_best:
                best_val_n_perfect = n_perfect
                best_val_gen_n = gen_n
                best_complexity = current_complexity
                best_params = jax.tree.map(lambda x: x.copy(), state.params)
                best_opt_state = state.opt_state
                best_epoch = epoch
                epochs_since_best = 0
                save_checkpoint(
                    {"params": state.params},
                    str(checkpoint_path(run_dir, "best.npz")),
                )
                print(f"              -> [CKPT] best checkpoint saved (epoch {epoch})")
            else:
                epochs_since_best += config.eval_every
                state, epochs_since_best, did_restart = _maybe_restart(
                    state, best_params, best_opt_state,
                    epochs_since_best, config.restart_patience,
                )
                if did_restart:
                    n_restarts += 1
                    print(f"  ↻ RESTART #{n_restarts} at epoch {epoch}"
                          f" → best checkpoint (epoch {best_epoch})")

    elapsed = time.monotonic() - t0
    print(f"Training complete in {elapsed:.1f}s"
          f"{f' ({n_restarts} restarts)' if n_restarts else ''}")

    # Save final checkpoint
    save_checkpoint({"params": state.params},
                    str(checkpoint_path(run_dir, "final.npz")))
    print(f"  [CKPT] Final checkpoint saved")

    # --- Final evaluation ---
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    test_result = evaluate_anbn_accuracy(
        model, best_params, test_inputs, test_targets,
    )
    print(f"  Test accuracy: {test_result['mean_accuracy']:.4f}")
    print(f"  Perfect strings: {test_result['n_perfect']}/{len(test_inputs)}")
    print(f"  gen_n: {test_result['gen_n']}")
    if test_result['first_failure_n'] is not None:
        print(f"  First failure at n={test_result['first_failure_n']}")

    forward_fn = make_forward_fn(model, best_params)
    test_dh = compute_grammar_weighted_nll_bits(
        forward_fn, config.test_max_n, p=config.p, verbose=True,
    )
    print(f"  Test |D:H| = {test_dh['data_dh_bits']:.4f} bits")

    train_dh = compute_train_dh(forward_fn, train_inputs, train_targets)
    print(f"  Train |D:H| = {train_dh['train_dh_data_bits']:.4f} bits")

    # Weight statistics
    w = np.array(best_params["weights"])
    print(f"\n  Weight stats: mean|w|={np.mean(np.abs(w)):.4f}, "
          f"max|w|={np.max(np.abs(w)):.4f}, "
          f"near-zero (<0.01): {np.mean(np.abs(w) < 0.01)*100:.1f}%")

    # --- Save results ---
    results = {
        "reg": reg,
        "best_epoch": best_epoch,
        "best_val_n_perfect": int(best_val_n_perfect),
        "test_accuracy": test_result["mean_accuracy"],
        "test_n_perfect": test_result["n_perfect"],
        "test_gen_n": test_result["gen_n"],
        "test_first_failure_n": test_result["first_failure_n"],
        "test_dh_bits": test_dh["data_dh_bits"],
        "train_dh_bits": train_dh["train_dh_data_bits"],
    }
    save_results(run_dir, results)

    print(f"\nResults saved to {run_dir}")
    return state, results


# ===========================================================================
# Config & CLI
# ===========================================================================

@dataclass
class PrimeRationalConfig:
    """Configuration for prime-exponent relaxation experiments."""
    task: str = "xor"
    P: int = 6
    hidden_dim: int = 32
    lambda_mdl: float = 1e-3
    lr: float = 1e-3
    epochs: int = 5000
    seed: int = 42
    init_std: float = 0.01
    clamp_logmag: float = 10.0
    eval_every: int = 100
    log_every: int = 100

    # XOR-specific
    xor_n_train: int = 200
    xor_noise_std: float = 0.1

    # ANBN-specific
    num_train: int = 1000
    p: float = 0.3
    data_seed: int = 0
    test_max_n: int = 1500
    hidden_size: int = 3
    val_min_n: int = 22
    val_max_n: int = 71

    # Training schedule
    restart_patience: int = 0

    # Baseline-specific
    reg: str = "none"


def _build_arg_parser(defaults=None):
    """Build argument parser. YAML defaults seed argparse so CLI overrides them."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Prime-exponent relaxation for MDL-style rational regularization",
    )
    parser.add_argument(
        "config", nargs="?", default=None,
        help="Optional YAML config path. CLI flags override YAML values.",
    )

    # Task
    parser.add_argument("--task", type=str, default="xor",
                        choices=["xor", "anbn", "anbn_baseline"],
                        help="Experiment task (default: xor)")
    # Prime basis
    parser.add_argument("--P", type=int, default=6,
                        help="Number of primes in basis (default: 6)")
    # Architecture
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="MLP hidden layer width for XOR (default: 32)")
    parser.add_argument("--hidden_size", type=int, default=3,
                        help="LSTM hidden size for ANBN (3 matches Lan et al.)")
    # Training
    parser.add_argument("--epochs", type=int, default=5000,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate")
    parser.add_argument("--lambda_mdl", type=float, default=1e-3,
                        help="MDL regularization weight")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--init_std", type=float, default=0.01,
                        help="Std for exponent/sign parameter initialization")
    parser.add_argument("--clamp_logmag", type=float, default=10.0,
                        help="Clamp range for log-magnitude before exp()")
    # Data — ANBN
    parser.add_argument("--num_train", type=int, default=1000,
                        help="Number of training strings (ANBN)")
    parser.add_argument("--p", type=float, default=0.3,
                        help="PCFG termination probability (ANBN)")
    parser.add_argument("--data_seed", type=int, default=None,
                        help="Seed for data generation (defaults to --seed)")
    # Data — XOR
    parser.add_argument("--xor_n_train", type=int, default=200,
                        help="Number of training samples (XOR)")
    parser.add_argument("--xor_noise_std", type=float, default=0.1,
                        help="Noise std for XOR data generation")
    # Evaluation
    parser.add_argument("--test_max_n", type=int, default=1500,
                        help="Max n for ANBN test set")
    parser.add_argument("--val_min_n", type=int, default=22,
                        help="Minimum n in ANBN validation set")
    parser.add_argument("--val_max_n", type=int, default=71,
                        help="Maximum n in ANBN validation set")
    parser.add_argument("--eval_every", type=int, default=100,
                        help="Evaluate every N epochs")
    parser.add_argument("--log_every", type=int, default=50,
                        help="Log training metrics every N epochs")
    parser.add_argument("--restart_patience", type=int, default=0,
                        help="Epochs without improvement before resetting to best "
                             "checkpoint (0 = disabled)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Force deterministic GPU ops")
    # Baseline-specific
    parser.add_argument("--reg", type=str, default="none",
                        choices=["none", "l1", "l2"],
                        help="Regularization type for anbn_baseline task")

    if defaults:
        valid_dests = {a.dest for a in parser._actions}
        unknown = sorted(k for k in defaults if k not in valid_dests)
        if unknown:
            print(f"Warning: ignoring unknown config keys: {', '.join(unknown)}")
            defaults = {k: v for k, v in defaults.items() if k in valid_dests}
        parser.set_defaults(**defaults)
    return parser


def main():
    # Parse config path first so YAML defaults can seed argparse.
    pre_config_arg = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        pre_config_arg = sys.argv[1]

    yaml_defaults = {}
    if pre_config_arg is not None:
        config_path = Path(pre_config_arg)
        if config_path.exists() and config_path.suffix in (".yaml", ".yml"):
            with open(config_path) as f:
                yaml_defaults = yaml.safe_load(f) or {}

    parser = _build_arg_parser(defaults=yaml_defaults)
    args = parser.parse_args()

    # If data_seed not given, tie it to seed.
    if args.data_seed is None:
        args.data_seed = args.seed

    # Build config from parsed args
    config = PrimeRationalConfig()
    for f in fields(PrimeRationalConfig):
        if hasattr(args, f.name):
            setattr(config, f.name, getattr(args, f.name))

    print(f"Config: {config}")
    print(f"Primes ({config.P}): {first_primes(config.P)}")

    if config.task == "xor":
        run_xor_experiment(config)
    elif config.task == "anbn":
        run_anbn_experiment(config)
    elif config.task == "anbn_baseline":
        run_anbn_baseline_experiment(config)
    else:
        print(f"Unknown task: {config.task}")
        sys.exit(1)


if __name__ == "__main__":
    main()
