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
            nn.initializers.normal(stddev=0.1),
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

        # Layer 1: input -> h1
        W1 = take(input_dim * self.h1).reshape(input_dim, self.h1)
        b1 = take(self.h1)
        h = h @ W1 + b1
        h = nn.relu(h)

        # Layer 2: h1 -> bottleneck
        W2 = take(self.h1 * self.bottleneck).reshape(self.h1, self.bottleneck)
        b2 = take(self.bottleneck)
        z = h @ W2 + b2  # bottleneck activations (for HSIC)
        h = nn.relu(z)

        # Layer 3: bottleneck -> h3
        W3 = take(self.bottleneck * self.h3).reshape(self.bottleneck, self.h3)
        b3 = take(self.h3)
        h = h @ W3 + b3
        h = nn.relu(h)

        # Layer 4: h3 -> num_classes
        W4 = take(self.h3 * self.num_classes).reshape(self.h3, self.num_classes)
        b4 = take(self.num_classes)
        logits = h @ W4 + b4

        aux = {
            "expected_codelength": expected_codelength,
            "all_probs": all_probs,
            "z": z,
            "mu": z,  # alias for compatibility with HSIC pair mode
        }
        return logits, aux
