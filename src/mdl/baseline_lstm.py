"""Standard LSTM baseline for comparison with differentiable MDL.

Implements the same architecture as GumbelSoftmaxLSTM (LSTM cell + linear
output layer) but with standard continuous float weights trained via
backpropagation with cross-entropy loss and optional L1/L2 regularization.

This reproduces the baseline experiments from Lan et al. (2024) Section 4.2:
    - Architecture: LSTM hidden_size=3, input/output size=3
    - Training: Adam, CE loss + optional L1/L2
    - Optional dropout and early stopping
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from fractions import Fraction

from src.mdl.coding import rational_codelength, integer_code_length


class BaselineLSTM(nn.Module):
    """Standard LSTM with continuous weights.

    Same architecture as GumbelSoftmaxLSTM but with regular float parameters
    instead of categorical distributions over a rational grid.

    Attributes:
        hidden_size: LSTM hidden dimension (3 for Lan et al.)
        input_size: input dimension (3 for {#, a, b})
        output_size: output dimension (3 for {#, a, b})
        dropout_rate: dropout probability (0 = no dropout)
    """
    hidden_size: int
    input_size: int
    output_size: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train=True, rng=None, tau=None, **kwargs):
        """Forward pass through the standard LSTM.

        Args:
            x: int32 (batch, seq_len) input token indices
            train: whether in training mode (affects dropout)
            rng: PRNG key for dropout
            tau: ignored (accepted for compatibility with MDL evaluation)

        Returns:
            logits: float32 (batch, seq_len, output_size)
            aux: dict (empty for baseline, keeps interface consistent)
        """
        B, _ = x.shape
        H = self.hidden_size
        I = self.input_size

        # LSTM input weights (I -> 4H)
        W_ii = self.param("W_ii", nn.initializers.glorot_normal(), (I, H))
        W_if = self.param("W_if", nn.initializers.glorot_normal(), (I, H))
        W_ig = self.param("W_ig", nn.initializers.glorot_normal(), (I, H))
        W_io = self.param("W_io", nn.initializers.glorot_normal(), (I, H))

        # LSTM hidden weights (H -> 4H)
        W_hi = self.param("W_hi", nn.initializers.glorot_normal(), (H, H))
        W_hf = self.param("W_hf", nn.initializers.glorot_normal(), (H, H))
        W_hg = self.param("W_hg", nn.initializers.glorot_normal(), (H, H))
        W_ho = self.param("W_ho", nn.initializers.glorot_normal(), (H, H))

        # LSTM biases
        b_ii = self.param("b_ii", nn.initializers.zeros, (H,))
        b_if = self.param("b_if", nn.initializers.zeros, (H,))
        b_ig = self.param("b_ig", nn.initializers.zeros, (H,))
        b_io = self.param("b_io", nn.initializers.zeros, (H,))
        b_hi = self.param("b_hi", nn.initializers.zeros, (H,))
        b_hf = self.param("b_hf", nn.initializers.zeros, (H,))
        b_hg = self.param("b_hg", nn.initializers.zeros, (H,))
        b_ho = self.param("b_ho", nn.initializers.zeros, (H,))

        # Output layer
        W_out = self.param("W_out", nn.initializers.glorot_normal(),
                           (H, self.output_size))
        b_out = self.param("b_out", nn.initializers.zeros, (self.output_size,))

        # One-hot encode input
        x_onehot = jax.nn.one_hot(x, I)  # (B, T, I)

        # LSTM scan
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

        x_seq = jnp.transpose(x_onehot, (1, 0, 2))
        _, h_seq = jax.lax.scan(lstm_step, (h0, c0), x_seq)
        h_seq = jnp.transpose(h_seq, (1, 0, 2))  # (B, T, H)

        # Optional dropout on hidden states
        if self.dropout_rate > 0:
            deterministic = not train or rng is None
            h_seq = nn.Dropout(rate=self.dropout_rate)(
                h_seq, deterministic=deterministic, rng=rng)

        # Output layer
        logits = h_seq @ W_out + b_out  # (B, T, output_size)

        return logits, {}


def flatten_params(params):
    """Flatten a params dict to a 1D array of all weight values."""
    leaves = jax.tree.leaves(params)
    return jnp.concatenate([jnp.ravel(l) for l in leaves])


def compute_baseline_mdl_score(params, hidden_size, max_denominator=1000):
    """Compute MDL hypothesis score for a baseline network.

    Converts continuous weights to closest rationals (Lan et al. method)
    and computes the encoding length.

    Args:
        params: Flax params dict
        hidden_size: LSTM hidden size (for architecture encoding)
        max_denominator: maximum denominator for rationalization

    Returns:
        dict with total_bits, arch_bits, weight_bits, n_nonzero
    """
    all_weights = flatten_params(params)
    arch_bits = integer_code_length(hidden_size)

    weight_bits = 0
    n_nonzero = 0
    for w in all_weights:
        w_float = float(w)
        frac = Fraction(w_float).limit_denominator(max_denominator)
        weight_bits += rational_codelength(frac)
        if frac != 0:
            n_nonzero += 1

    return {
        "total_bits": arch_bits + weight_bits,
        "arch_bits": arch_bits,
        "weight_bits": weight_bits,
        "n_nonzero": n_nonzero,
        "n_params": len(all_weights),
    }


def make_baseline_loss_fn(reg_type=None, reg_lambda=0.0):
    """Create CE + optional regularization loss function.

    Args:
        reg_type: None, "l1", or "l2"
        reg_lambda: regularization coefficient

    Returns:
        loss function (params, apply_fn, x, y, mask, rng) -> (loss, aux)
    """
    def loss_fn(params, apply_fn, x, y, mask, rng):
        logits, _ = apply_fn({"params": params}, x, train=True, rng=rng)

        # Cross-entropy
        ce_nats = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        ce_bits = ce_nats / jnp.log(2.0)
        data_nll_bits = jnp.sum(ce_bits * mask)

        objective_total_bits = data_nll_bits

        # Regularization
        reg_regularizer = jnp.array(0.0)
        if reg_type == "l1" and reg_lambda > 0:
            all_weights = flatten_params(params)
            reg_regularizer = reg_lambda * jnp.sum(jnp.abs(all_weights))
            objective_total_bits = objective_total_bits + reg_regularizer
        elif reg_type == "l2" and reg_lambda > 0:
            all_weights = flatten_params(params)
            reg_regularizer = reg_lambda * jnp.sum(all_weights ** 2)
            objective_total_bits = objective_total_bits + reg_regularizer

        aux = {
            "objective_total_bits": objective_total_bits,
            "data_nll_bits": data_nll_bits,
            "reg_regularizer": reg_regularizer,
        }
        return objective_total_bits, aux

    return loss_fn


def create_baseline_state(rng, model, seq_len, batch_size, lr):
    """Initialize training state for baseline LSTM."""
    dummy_x = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_x, train=False)["params"]
    tx = optax.adam(lr)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


def make_baseline_train_step(reg_type=None, reg_lambda=0.0):
    """Create a JIT-compiled baseline training step."""
    loss_fn = make_baseline_loss_fn(reg_type=reg_type, reg_lambda=reg_lambda)

    @jax.jit
    def train_step(state, x, y, mask, rng):
        def _loss(params):
            return loss_fn(params, state.apply_fn, x, y, mask, rng)

        (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
            state.params
        )
        state = state.apply_gradients(grads=grads)
        return state, loss, aux

    return train_step
