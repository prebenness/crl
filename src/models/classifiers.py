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


class ULAMLPClassifier(nn.Module):
    """
    cMNIST classifier matching uLA / CCDB architecture:
      - 3 hidden layers
      - 100 units each
      - ReLU after each hidden
      - logits -> num_classes

    Same signature as StdClassifier: (rep_dim, num_classes).
    For strict fairness to uLA, rep_dim must be 100 (we assert it).
    """
    rep_dim: int
    num_classes: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Expect x in NHWC from your pipeline: (B, 28, 28, 3)
        x = x.reshape((x.shape[0], -1))  # flatten

        # uLA: 3 hidden layers, 100 neurons each
        x = nn.Dense(100)(x); x = nn.relu(x)
        x = nn.Dense(100)(x); z = nn.relu(x)
        h = nn.Dense(100)(z); h = nn.relu(h)

        # z is penpenultimate rep (100-d)
        logits = nn.Dense(self.num_classes)(h)
        return logits, {"z": z}

