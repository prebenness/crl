import numpy as np

import jax.numpy as jnp
from jax import random as jrandom


import flax.linen as nn


class StdClassifier(nn.Module):
    rep_dim: int
    num_classes: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        # --- Encoder (mirror VIB/IB for comparability) ---
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), (2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), (2, 2))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)

        # Penultimate rep (this is z2)
        h = nn.Dense(self.rep_dim)(x)
        h = nn.relu(h)

        logits = nn.Dense(self.num_classes)(h)

        aux = {"h": h}
        return logits, aux
