import numpy as np

import jax.numpy as jnp
from jax import random as jrandom

import flax.linen as nn


def _compute_kl_and_aux(z, mu, logvar, x_recon_logits):
    """Shared VIB auxiliary computation: KL divergence and output dict."""
    kl_per_example = 0.5 * jnp.sum(
        (mu * mu) + jnp.exp(logvar) - 1.0 - logvar,
        axis=-1,
    )
    kl = jnp.mean(kl_per_example)
    return {
        "z": z,
        "x_recon_logits": x_recon_logits,
        "mu": mu,
        "logvar": logvar,
        "kl": kl,
        "kl_per_example": kl_per_example,
    }


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

        # MLP
        h = nn.Dense(128)(z)
        h = nn.relu(h)

        # --- Classifier head ---
        logits = nn.Dense(self.num_classes)(h)

        x_recon_logits = SimpleDecoder(output_shape=x_in.shape[1:])(z)

        return logits, _compute_kl_and_aux(z, mu, logvar, x_recon_logits)


class ULAMLPVarClassifier(nn.Module):
    num_classes: int

    # Optional numeric-stability knobs (defaults are sane)
    min_logvar: float = -10.0
    max_logvar: float =  10.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        x_in = x

        # Expect x in NHWC from your pipeline: (B, 28, 28, 3)
        x = x.reshape((x.shape[0], -1))  # flatten

        # uLA: 3 hidden layers, 100 neurons each
        x = nn.Dense(100)(x); x = nn.relu(x)
        
        mu = nn.Dense(50)(x); mu = nn.relu(mu)
        logvar = nn.Dense(50)(x); logvar = nn.relu(logvar) 
        logvar = jnp.clip(logvar, self.min_logvar, self.max_logvar)

        # --- Variational "bottleneck": q(z|x) = N(mu, diag(var)) ---
        std = jnp.exp(0.5 * logvar)
        if train:
            eps = jrandom.normal(self.make_rng("noise"), mu.shape)
            z = mu + std * eps
        else:
            # Deterministic at eval for lower-variance metrics
            z = mu

        h = nn.Dense(100)(z); h = nn.relu(h)

        logits = nn.Dense(self.num_classes)(h)

        x_recon_logits = SimpleDecoder(output_shape=x_in.shape[1:])(z)

        return logits, _compute_kl_and_aux(z, mu, logvar, x_recon_logits)