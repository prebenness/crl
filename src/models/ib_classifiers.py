# ============================================================
# 4) Deterministic Bottleneck Classifier
# ============================================================


import numpy as np

import jax.numpy as jnp
from jax import random as jrandom
from jax.scipy.special import logsumexp


import flax.linen as nn


class IBClassifier(nn.Module):
    bottleneck_width: int
    num_classes: int
    lamb: float      # interpreted as noise std


    @nn.compact
    def __call__(self, x, train=True):
        # --- Encoder ---
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), (2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), (2, 2))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        # --- The Bottleneck (Z) ---
        z = nn.Dense(self.bottleneck_width)(x)
        # Tanh is recommended for IB to bound the embedding space [-1, 1]
        z = nn.tanh(z)

        # --- Add noise to channel ---
        # Additive Gaussian noise in the bottleneck
        if train:
            noise = jrandom.normal(self.make_rng("noise"), z.shape) * self.lamb
            z = z + noise

        # --- Classifier ---
        logits = nn.Dense(self.num_classes)(z)
        

        # Aux dict: keeps z for loss/logging
        aux = {
            "z": z,
        }

        return logits, aux


class SimpleDecoder(nn.Module):
    # Tiny MLP decoder: z -> x_recon (NHWC)
    output_shape: tuple  # (H, W, C)
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, z):
        h = nn.Dense(self.hidden_dim)(z)
        h = nn.relu(h)
        out_dim = int(np.prod(self.output_shape))
        h = nn.Dense(out_dim)(h)
        h = nn.Dense(out_dim)(h)
        # return logits. Use optax.sigmoid_binary_cross_entropy for a stable BCE.
        x_recon_logits = h.reshape((z.shape[0],) + self.output_shape)
        return x_recon_logits


class VIBClassifier(nn.Module):
    bottleneck_width: int
    num_classes: int

    # Optional numeric-stability knobs (defaults are sane)
    min_logvar: float = -10.0
    max_logvar: float =  10.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        x_in = x
        # --- Encoder (same as IBClassifier for apples-to-apples) ---
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), (2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), (2, 2))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        # --- Variational bottleneck: q(z|x) = N(mu, diag(var)) ---
        mu = nn.Dense(self.bottleneck_width, name="z_mu")(x)
        logvar = nn.Dense(self.bottleneck_width, name="z_logvar")(x)
        logvar = jnp.clip(logvar, self.min_logvar, self.max_logvar)

        std = jnp.exp(0.5 * logvar)

        if train:
            eps = jrandom.normal(self.make_rng("noise"), mu.shape)
            z = mu + std * eps
        else:
            # Deterministic at eval for lower-variance metrics
            z = mu

        # --- Classifier head ---
        logits = nn.Dense(self.num_classes)(z)

        # --- Decoder head (reconstruction) ---
        x_recon_logits = SimpleDecoder(output_shape=x_in.shape[1:])(z)

        # --- KL(q(z|x) || N(0, I)) ---
        # Per-example KL: 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)
        kl_per_example = 0.5 * jnp.sum(
            (mu * mu) + jnp.exp(logvar) - 1.0 - logvar,
            axis=-1
        )
        kl = jnp.mean(kl_per_example)

        # Aux dict: keeps z plus VIB stats for loss/logging
        aux = {
            "z": z,
            "x_recon_logits": x_recon_logits,
            "mu": mu,
            "logvar": logvar,
            "kl": kl,
            "kl_per_example": kl_per_example,
        }

        return logits, aux