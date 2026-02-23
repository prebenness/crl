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
    """
    hidden_size: int
    input_size: int
    output_size: int
    grid_values: Any  # (M,) array
    grid_codelengths: Any  # (M,) array

    def _sample_weights(self, logits, tau, rng):
        """Gumbel-Softmax ST: hard one-hot in forward, soft in backward.

        Args:
            logits: (..., M) unnormalized log-probabilities
            tau: temperature (= 1/beta)
            rng: PRNG key

        Returns:
            weights: (...) discrete weight values from grid
            probs: (..., M) softmax probabilities (for computing expected codelength)
        """
        M = logits.shape[-1]

        # Gumbel noise
        gumbel_noise = jrandom.gumbel(rng, shape=logits.shape)
        perturbed = (logits + gumbel_noise) / tau

        # Soft probabilities
        y_soft = jax.nn.softmax(perturbed, axis=-1)

        # Hard one-hot (straight-through)
        idx = jnp.argmax(y_soft, axis=-1)
        y_hard = jax.nn.one_hot(idx, M)

        # ST trick: hard in forward, soft gradients in backward
        y_st = y_hard - jax.lax.stop_gradient(y_soft) + y_soft

        # Map to grid values
        grid = jnp.asarray(self.grid_values)
        weights = jnp.sum(y_st * grid[None, :], axis=-1) if y_st.ndim == 2 else jnp.dot(y_st, grid)

        # Probabilities for codelength computation (no Gumbel, just softmax of logits)
        probs = jax.nn.softmax(logits, axis=-1)

        return weights, probs

    @nn.compact
    def __call__(self, x, tau, train=True, rng=None):
        """Forward pass through the categorical LSTM.

        Args:
            x: int32 (batch, seq_len) input token indices
            tau: Gumbel-Softmax temperature
            train: whether to use stochastic sampling
            rng: PRNG key for Gumbel noise

        Returns:
            logits: float32 (batch, seq_len, output_size) output logits
            aux: dict with 'expected_codelength' and 'probs_dict'
        """
        B, T = x.shape
        H = self.hidden_size
        I = self.input_size
        M = len(self.grid_values)

        # --- Define all logit parameters ---
        # LSTM has 4 gates (i, f, g, o), each with input and hidden weights + biases
        # W_ii, W_if, W_ig, W_io: (input_size, hidden_size) each
        # W_hi, W_hf, W_hg, W_ho: (hidden_size, hidden_size) each
        # b_ii, b_if, b_ig, b_io: (hidden_size,) each
        # b_hi, b_hf, b_hg, b_ho: (hidden_size,) each
        # Output layer: W_out (hidden_size, output_size), b_out (output_size,)

        # Total parameters:
        # Input weights: 4 * I * H
        # Hidden weights: 4 * H * H
        # Input biases: 4 * H
        # Hidden biases: 4 * H
        # Output weights: H * O
        # Output bias: O
        n_lstm_w = 4 * I * H + 4 * H * H
        n_lstm_b = 4 * H + 4 * H
        n_out_w = H * self.output_size
        n_out_b = self.output_size
        n_total = n_lstm_w + n_lstm_b + n_out_w + n_out_b

        # Single logit array for all parameters
        all_logits = self.param(
            "logits",
            nn.initializers.zeros_init(),
            (n_total, M),
        )

        if train and rng is not None:
            # Split RNG for each parameter
            keys = jrandom.split(rng, n_total)
            # Vectorized Gumbel-Softmax over all parameters
            def sample_one(logit_row, key):
                gumbel_noise = jrandom.gumbel(key, shape=(M,))
                perturbed = (logit_row + gumbel_noise) / tau
                y_soft = jax.nn.softmax(perturbed, axis=-1)
                idx = jnp.argmax(y_soft, axis=-1)
                y_hard = jax.nn.one_hot(idx, M)
                y_st = y_hard - jax.lax.stop_gradient(y_soft) + y_soft
                grid = jnp.asarray(self.grid_values)
                w = jnp.dot(y_st, grid)
                return w

            all_weights = jax.vmap(sample_one)(all_logits, keys)
        else:
            # Deterministic: pick argmax
            grid = jnp.asarray(self.grid_values)
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
