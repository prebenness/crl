import jax.numpy as jnp
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
        z = nn.relu(x)

        h = nn.Dense(self.rep_dim)(z)
        h = nn.relu(h)

        logits = nn.Dense(self.num_classes)(h)

        aux = {"z": z}
        return logits, aux


class OracleMLP(nn.Module):
    """Deterministic MLP for oracle color classification.

    Architecture matches ULAMLPVarClassifier's encoder path but without
    variational components (no KL, no reconstruction, no sampling).
    Bottleneck is 50-d for direct comparability with VIB mu.
    """
    num_classes: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        z = nn.Dense(50)(x)
        logits = nn.Dense(self.num_classes)(nn.relu(nn.Dense(100)(z)))
        return logits, {"z": z}


class ULAMLPClassifier(nn.Module):
    """
    cMNIST classifier matching uLA / CCDB architecture:
      - 3 hidden layers
      - 100 units each
      - ReLU after each hidden
      - logits -> num_classes

    Same signature as StdClassifier: (rep_dim, num_classes).
    Hidden widths are fixed at 100 for strict fidelity to the uLA protocol.
    If rep_dim != 100 a warning is emitted; the value is ignored.
    """
    rep_dim: int
    num_classes: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        if self.rep_dim != 100:
            import warnings
            warnings.warn(
                f"ULAMLPClassifier ignores rep_dim={self.rep_dim}; "
                f"hidden width is fixed at 100 for uLA protocol fidelity."
            )
        # Expect x in NHWC from your pipeline: (B, 28, 28, 3)
        x = x.reshape((x.shape[0], -1))  # flatten

        # uLA: 3 hidden layers, 100 neurons each (fixed for paper compatibility)
        x = nn.Dense(100)(x); x = nn.relu(x)
        x = nn.Dense(100)(x); z = nn.relu(x)
        h = nn.Dense(100)(z); h = nn.relu(h)

        # z is penpenultimate rep (100-d), h is penultimate (100-d)
        logits = nn.Dense(self.num_classes)(h)
        return logits, {"z": z, "h": h}


class CBAOMMlp(nn.Module):
    """Colour-conditioned classifier for CBA-OM (backdoor adjustment).

    Takes (x, s) where s is the oracle-provided colour label.
    Late fusion: encoder processes x alone, then [encoder(x); embed(s)]
    is fed to a single linear classification head.
    At test time, marginalise over all colours externally.
    """
    num_classes: int
    num_colors: int = 10
    embed_dim: int = 16

    @nn.compact
    def __call__(self, x, s, train: bool = True):
        # Early fusion: concatenate colour embedding with flattened image
        # (proposal recommends late fusion for larger encoders, but with
        # 3x100 the linear head lacks capacity for deconfounding —
        # early fusion lets colour modulate all 3 layers)
        x = x.reshape((x.shape[0], -1))
        s_emb = nn.Embed(num_embeddings=self.num_colors,
                         features=self.embed_dim)(s)
        x = jnp.concatenate([x, s_emb], axis=-1)

        x = nn.Dense(100)(x); x = nn.relu(x)
        x = nn.Dense(100)(x); z = nn.relu(x)
        h = nn.Dense(100)(z); h = nn.relu(h)

        logits = nn.Dense(self.num_classes)(h)
        return logits, {"z": z}

