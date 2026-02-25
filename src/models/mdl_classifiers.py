"""MLP classifiers with categorical weight parameterization for MDL.

Each weight is parameterized as a categorical distribution over a finite
grid of rational numbers S. During training, Gumbel-Softmax with
straight-through is used to sample discrete weights while allowing
gradient flow. This is the MLP analogue of src/mdl/lstm.py.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random as jrandom
import flax.linen as nn
from typing import Any


def kaiming_categorical_init(grid_values, layer_dims, peak_logit=10.0):
    """Flax initializer that peaks categorical logits at Kaiming He targets.

    The standard normal(stddev=0.1) init gives near-uniform softmax over the
    symmetric grid, so E[w] ≈ 0 for every parameter — a dead ReLU network.
    This initializer instead:
      1. Draws a target weight from N(0, sqrt(2)) for each weight param
         (biases target 0). Since the forward pass divides by sqrt(fan_in),
         the effective init is sqrt(2/fan_in) = Kaiming He for ReLU.
      2. Finds the nearest grid value to each target.
      3. Sets the logit at that grid index to peak_logit, rest to 0.

    Uses pure JAX ops so the function is traceable under JIT.

    Args:
        grid_values: (M,) array of rational grid values
        layer_dims: [input_dim, h1, bottleneck, h3, num_classes]
        peak_logit: logit value at the target grid index (rest get 0)
    """
    grid_jnp = jnp.asarray(grid_values)

    def init_fn(rng, shape, dtype=jnp.float32):
        n_total, M = shape

        # Build per-parameter target weights using JAX random
        targets = jnp.zeros(n_total)
        offset = 0
        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]

            # Weights: N(0, sqrt(2)) — forward pass divides by sqrt(fan_in)
            n_w = fan_in * fan_out
            rng, rng_layer = jrandom.split(rng)
            w_targets = jrandom.normal(rng_layer, shape=(n_w,)) * jnp.sqrt(2.0)
            targets = targets.at[offset:offset + n_w].set(w_targets)
            offset += n_w

            # Biases: target 0 (already zero)
            offset += fan_out

        # Find nearest grid value for each parameter
        nearest_idx = jnp.argmin(
            jnp.abs(targets[:, None] - grid_jnp[None, :]), axis=-1,
        )

        # Set logit at nearest grid index to peak_logit, rest stay at 0
        logits = jnp.zeros((n_total, M), dtype=dtype)
        logits = logits.at[jnp.arange(n_total), nearest_idx].set(peak_logit)
        return logits

    return init_fn


class GumbelSoftmaxMLP(nn.Module):
    """MLP where every weight/bias is a categorical over a rational grid.

    Architecture: flatten -> Dense(h1) -> relu -> Dense(bottleneck) -> relu
                  -> Dense(h3) -> relu -> Dense(num_classes)

    The bottleneck layer activations serve as the representation z for HSIC
    in pair mode (analogous to mu in VIB models).

    Attributes:
        num_classes: number of output classes
        grid_values: float32 array (M,) of rational grid values
        grid_codelengths: float32 array (M,) of per-weight codelengths
        h1: first hidden layer width
        bottleneck: bottleneck layer width (representation for HSIC)
        h3: third hidden layer width
    """
    num_classes: int
    grid_values: Any   # (M,) array
    grid_codelengths: Any  # (M,) array
    h1: int = 100
    bottleneck: int = 50
    h3: int = 100

    @nn.compact
    def __call__(self, x, tau, train=True, rng=None, soft_forward=False):
        """Forward pass through the categorical MLP.

        Three forward modes (matching GumbelSoftmaxLSTM):
            train=True, soft_forward=True:  continuous relaxation (warmup)
            train=True, soft_forward=False: Gumbel-Softmax straight-through
            train=False: deterministic argmax (evaluation)

        Args:
            x: float32 (batch, ...) input (will be flattened)
            tau: Gumbel-Softmax temperature
            train: whether in training mode
            rng: PRNG key for Gumbel noise (needed when train=True, soft_forward=False)
            soft_forward: if True, use continuous relaxation instead of ST

        Returns:
            logits: float32 (batch, num_classes) output logits
            aux: dict with 'expected_codelength', 'all_probs', 'z', 'mu'
        """
        B = x.shape[0]
        input_dim = 1
        for d in x.shape[1:]:
            input_dim *= d
        M = len(self.grid_values)

        # Total parameter count across all layers
        layer_dims = [input_dim, self.h1, self.bottleneck, self.h3, self.num_classes]
        n_total = sum(
            layer_dims[i] * layer_dims[i + 1] + layer_dims[i + 1]
            for i in range(len(layer_dims) - 1)
        )

        all_logits = self.param(
            "logits",
            kaiming_categorical_init(self.grid_values, layer_dims),
            (n_total, M),
        )

        grid = jnp.asarray(self.grid_values)

        # === Weight materialization ===
        if train and soft_forward:
            # Continuous relaxation: weight = E[grid | softmax(logits/tau)]
            y_soft = jax.nn.softmax(all_logits / tau, axis=-1)
            all_weights = jnp.sum(y_soft * grid[None, :], axis=-1)
        elif train and rng is not None:
            # Vectorized Gumbel-Softmax straight-through
            gumbel_noise = jrandom.gumbel(rng, shape=(n_total, M))
            perturbed = (all_logits + gumbel_noise) / tau
            y_soft = jax.nn.softmax(perturbed, axis=-1)
            idx = jnp.argmax(y_soft, axis=-1)
            y_hard = jax.nn.one_hot(idx, M)
            y_st = y_hard - lax.stop_gradient(y_soft) + y_soft
            all_weights = jnp.sum(y_st * grid[None, :], axis=-1)
        else:
            # Deterministic: pick argmax
            idx = jnp.argmax(all_logits, axis=-1)
            all_weights = grid[idx]

        # Expected codelength (always computed from softmax, no Gumbel)
        all_probs = jax.nn.softmax(all_logits, axis=-1)
        cl = jnp.asarray(self.grid_codelengths)
        expected_codelength = jnp.sum(all_probs * cl[None, :])

        # === Unpack weights and run forward pass ===
        offset = 0

        def take(size):
            nonlocal offset
            w = lax.dynamic_slice_in_dim(all_weights, offset, size)
            offset += size
            return w

        h = x.reshape((B, -1))  # flatten

        # Each layer applies fan-in scaling: W / sqrt(fan_in).
        # Without this, random grid values (up to ±5) with large fan-in
        # produce extreme activations — unlike LSTMs where sigmoid/tanh
        # gates naturally bound values. The scaling is part of the
        # architecture; codelengths still describe the unscaled rationals.

        # Layer 1: input -> h1
        W1 = take(input_dim * self.h1).reshape(input_dim, self.h1)
        b1 = take(self.h1)
        h = h @ W1 / jnp.sqrt(jnp.float32(input_dim)) + b1
        h = nn.relu(h)

        # Layer 2: h1 -> bottleneck
        W2 = take(self.h1 * self.bottleneck).reshape(self.h1, self.bottleneck)
        b2 = take(self.bottleneck)
        z = h @ W2 / jnp.sqrt(jnp.float32(self.h1)) + b2
        h = nn.relu(z)

        # Layer 3: bottleneck -> h3
        W3 = take(self.bottleneck * self.h3).reshape(self.bottleneck, self.h3)
        b3 = take(self.h3)
        h = h @ W3 / jnp.sqrt(jnp.float32(self.bottleneck)) + b3
        h = nn.relu(h)

        # Layer 4: h3 -> num_classes
        W4 = take(self.h3 * self.num_classes).reshape(self.h3, self.num_classes)
        b4 = take(self.num_classes)
        logits = h @ W4 / jnp.sqrt(jnp.float32(self.h3)) + b4

        aux = {
            "expected_codelength": expected_codelength,
            "all_probs": all_probs,
            "z": z,
            "mu": z,  # alias for compatibility with HSIC pair mode
        }
        return logits, aux
