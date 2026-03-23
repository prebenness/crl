"""LSTM with categorical weight parameterization for differentiable MDL.

Each weight is parameterized as a categorical distribution over a finite
grid of rational numbers S. During training, Gumbel-Softmax with
straight-through is used to sample discrete weights while allowing
gradient flow through the soft samples.

Architecture matches Lan et al. (2024): LSTM cell + single linear output
layer + softmax, with hidden_size=3 and input/output size=3 (#, a, b).
"""

import jax
import jax.numpy as jnp
from jax import random as jrandom
import flax.linen as nn
from typing import Any


def codelength_informed_init(grid_codelengths, scale=1.0):
    """Initialize logits proportional to -scale * l(s_m).

    Makes the initial categorical approximate P_base, concentrating mass
    on simple (low-codelength) rationals like 0, ±1, ±1/2.  When scale=0
    this falls back to pure noise (similar to the old normal(0.1) init).

    Reference: informed by Louizos et al. (2019) "Relaxed Quantization
    for Discretized Neural Networks" (arXiv:1810.01875) which notes that
    initialization strongly affects which grid region the optimizer explores.
    """
    cl = jnp.asarray(grid_codelengths)

    def init_fn(rng, shape, dtype=jnp.float32):
        n_total, M = shape
        base_logits = -scale * cl  # (M,)
        noise = jrandom.normal(rng, shape=(n_total, M)) * 0.1
        return jnp.broadcast_to(base_logits[None, :], (n_total, M)) + noise

    return init_fn


class GumbelSoftmaxLSTM(nn.Module):
    """LSTM where every weight/bias is a categorical over a rational grid.

    The model stores logits (alpha) for each parameter, and at forward time
    uses Gumbel-Softmax ST to produce discrete-valued weights from the grid.

    Attributes:
        hidden_size: LSTM hidden dimension (3 for Lan et al.)
        input_size: input dimension (3 for {#, a, b})
        output_size: output dimension (3 for {#, a, b})
        grid_values: float32 array (M,) of rational grid values
        grid_codelengths: float32 array (M,) of per-weight codelengths
        mode_forward: if True, use mode of π (not Gumbel argmax) in the
            forward pass during stochastic ST.  Reference: Lee et al. (2021)
            "Semi-Relaxed Quantization with DropBits" (arXiv:1911.12990).
        init_cl_scale: scale for codelength-informed initialization.
            0 = noise-only (legacy), >0 = bias toward simple rationals.
    """
    hidden_size: int
    input_size: int
    output_size: int
    grid_values: Any  # (M,) array
    grid_codelengths: Any  # (M,) array
    mode_forward: bool = False
    init_cl_scale: float = 0.0

    @nn.compact
    def __call__(self, x, tau, train=True, rng=None,
                 deterministic_st=False):
        """Forward pass through the categorical LSTM.

        Forward modes:
            train=True, deterministic_st=True: deterministic straight-through
                Hard argmax in forward, softmax(logits/tau) gradients in backward.
            train=True, deterministic_st=False: Gumbel-Softmax straight-through
                Discrete samples in forward, soft gradients in backward.
            train=False: deterministic argmax (evaluation)

        Args:
            x: int32 (batch, seq_len) input token indices
            tau: Gumbel-Softmax temperature
            train: whether in training mode
            rng: PRNG key for Gumbel noise (needed when train=True, deterministic_st=False)
            deterministic_st: if True, use deterministic straight-through

        Returns:
            logits: float32 (batch, seq_len, output_size) output logits
            aux: dict with 'expected_codelength', 'all_probs', etc.
        """
        B, T = x.shape
        H = self.hidden_size
        I = self.input_size
        M = len(self.grid_values)

        # --- Define all logit parameters ---
        n_lstm_w = 4 * I * H + 4 * H * H
        n_lstm_b = 4 * H + 4 * H
        n_out_w = H * self.output_size
        n_out_b = self.output_size
        n_total = n_lstm_w + n_lstm_b + n_out_w + n_out_b

        # Single logit array for all parameters.
        # When init_cl_scale > 0, logits are biased toward simple rationals
        # (codelength-informed init).  When init_cl_scale == 0, falls back to
        # small random noise that breaks the symmetry trapping the all-zero
        # LSTM at a saddle point (h=0 always, gradient of data term = 0).
        if self.init_cl_scale > 0:
            init_fn = codelength_informed_init(
                self.grid_codelengths, scale=self.init_cl_scale,
            )
        else:
            init_fn = nn.initializers.normal(stddev=0.1)
        all_logits = self.param("logits", init_fn, (n_total, M))

        grid = jnp.asarray(self.grid_values)

        if train and deterministic_st:
            # Deterministic straight-through: hard argmax forward, soft grads.
            y_soft = jax.nn.softmax(all_logits / tau, axis=-1)
            idx = jnp.argmax(y_soft, axis=-1)
            y_hard = jax.nn.one_hot(idx, M)
            y_st = y_hard - jax.lax.stop_gradient(y_soft) + y_soft
            all_weights = jnp.sum(y_st * grid[None, :], axis=-1)
        elif train and rng is not None:
            # Gumbel-Softmax straight-through
            keys = jrandom.split(rng, n_total)

            def sample_one(logit_row, key):
                gumbel_noise = jrandom.gumbel(key, shape=(M,))
                perturbed = (logit_row + gumbel_noise) / tau
                y_soft = jax.nn.softmax(perturbed, axis=-1)
                # Mode forward: use argmax of *unperturbed* logits (mode of π)
                # instead of argmax of Gumbel-perturbed logits.  This avoids
                # catastrophic forward-pass samples where Gumbel noise selects
                # a grid point far from the distribution mode.
                # Ref: Lee et al. (2021) "Semi-Relaxed Quantization" §3.1
                idx = jax.lax.cond(
                    mode_fwd,
                    lambda: jnp.argmax(logit_row, axis=-1),
                    lambda: jnp.argmax(y_soft, axis=-1),
                )
                y_hard = jax.nn.one_hot(idx, M)
                y_st = y_hard - jax.lax.stop_gradient(y_soft) + y_soft
                w = jnp.dot(y_st, grid)
                return w

            mode_fwd = jnp.bool_(self.mode_forward)
            all_weights = jax.vmap(sample_one)(all_logits, keys)
        else:
            # Deterministic: pick argmax
            idx = jnp.argmax(all_logits, axis=-1)
            all_weights = grid[idx]

        # Compute probabilities for expected codelength (always, no Gumbel)
        all_probs = jax.nn.softmax(all_logits, axis=-1)

        # Expected codelength: sum_i sum_m pi_{i,m} * l(s_m)
        cl = jnp.asarray(self.grid_codelengths)
        expected_codelength = jnp.sum(all_probs * cl[None, :])

        # --- Unpack weights ---
        offset = 0

        def take(size):
            nonlocal offset
            w = all_weights[offset:offset + size]
            offset += size
            return w

        # LSTM input weights: W_ii, W_if, W_ig, W_io each (I, H)
        W_ii = take(I * H).reshape(I, H)
        W_if = take(I * H).reshape(I, H)
        W_ig = take(I * H).reshape(I, H)
        W_io = take(I * H).reshape(I, H)

        # LSTM hidden weights: W_hi, W_hf, W_hg, W_ho each (H, H)
        W_hi = take(H * H).reshape(H, H)
        W_hf = take(H * H).reshape(H, H)
        W_hg = take(H * H).reshape(H, H)
        W_ho = take(H * H).reshape(H, H)

        # LSTM biases
        b_ii = take(H)
        b_if = take(H)
        b_ig = take(H)
        b_io = take(H)
        b_hi = take(H)
        b_hf = take(H)
        b_hg = take(H)
        b_ho = take(H)

        # Output layer
        W_out = take(H * self.output_size).reshape(H, self.output_size)
        b_out = take(self.output_size)

        assert offset == n_total

        # --- One-hot encode input ---
        x_onehot = jax.nn.one_hot(x, I)  # (B, T, I)

        # --- Run LSTM ---
        def lstm_step(carry, x_t):
            h, c = carry  # (B, H) each
            # x_t: (B, I)
            i_t = jax.nn.sigmoid(x_t @ W_ii + b_ii + h @ W_hi + b_hi)
            f_t = jax.nn.sigmoid(x_t @ W_if + b_if + h @ W_hf + b_hf)
            g_t = jnp.tanh(x_t @ W_ig + b_ig + h @ W_hg + b_hg)
            o_t = jax.nn.sigmoid(x_t @ W_io + b_io + h @ W_ho + b_ho)
            c = f_t * c + i_t * g_t
            h = o_t * jnp.tanh(c)
            return (h, c), h

        h0 = jnp.zeros((B, H))
        c0 = jnp.zeros((B, H))

        # Transpose to (T, B, I) for scan
        x_seq = jnp.transpose(x_onehot, (1, 0, 2))
        (h_final, c_final), h_seq = jax.lax.scan(lstm_step, (h0, c0), x_seq)
        # h_seq: (T, B, H) -> (B, T, H)
        h_seq = jnp.transpose(h_seq, (1, 0, 2))

        # --- Output layer ---
        logits = h_seq @ W_out + b_out  # (B, T, output_size)

        aux = {
            "expected_codelength": expected_codelength,
            "all_probs": all_probs,
            "all_logits": all_logits,
            "n_params": n_total,
        }
        return logits, aux


def decode_weights(params, grid_values):
    """Extract the discrete weights (argmax) from trained logits.

    Returns a dict mapping parameter roles to their rational values.
    """
    logits = params["params"]["logits"]  # (n_total, M)
    grid = jnp.asarray(grid_values)
    idx = jnp.argmax(logits, axis=-1)
    return grid[idx]
