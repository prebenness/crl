# =========================
# Model (Flax)
# =========================
from flax import linen as nn


class SimpleMLP(nn.Module):
    hidden_sizes: tuple[int, ...] = (512, 256)
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, train: bool = True):
        r = self.get_reps(x, train)
        x = nn.Dense(self.num_classes)(r)  # logits
        return x

    @nn.compact
    def get_reps(self, x, train: bool = True):
        # x: [B, C * W * H]
        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.gelu(x)
        return x