"""IPSN (Interventional Path Surgery Network) Flax module.

Disentangled encoder E(x) = (c, b) with:
  - c: content code (digit features, colour-invariant via gradient reversal)
  - b: colour code (predicts oracle colour)
  - g(c): digit classifier
  - h_s(b): colour classifier
  - a_s(grad_rev(c)): colour adversary
  - D(c, s): decoder / recolourer

Reference: Ganin et al. (2016), "Domain-Adversarial Training of Neural
Networks", JMLR 17(1), adapted as path-specific causal intervention.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
import flax.linen as nn


class IPSNMlp(nn.Module):
    """IPSN for cMNIST: shared encoder trunk + disentangled heads.

    The encoder→g(c) inference path is 3 hidden ReLU layers of width 100,
    matching the ULA protocol.  Auxiliary heads (h_s, a_s, D) are small.
    """
    num_classes: int
    num_colors: int = 10
    c_dim: int = 50
    b_dim: int = 16
    embed_dim: int = 16
    decoder_hidden: int = 64
    grad_rev_scale: float = 1.0

    def setup(self):
        # Shared encoder trunk (layers 1-2 of 3x100)
        self.trunk1 = nn.Dense(100)
        self.trunk2 = nn.Dense(100)
        # Projection heads
        self.c_proj = nn.Dense(self.c_dim)
        self.b_proj = nn.Dense(self.b_dim)
        # Digit classifier g(c) — layer 3 of 3x100
        self.g_hidden = nn.Dense(100)
        self.g_out = nn.Dense(self.num_classes)
        # Colour classifier h_s(b) — single linear
        self.h_s = nn.Dense(self.num_colors)
        # Colour adversary a_s(c) — single linear (gradient reversal applied externally)
        self.a_s = nn.Dense(self.num_colors)
        # Decoder D(c, s)
        self.dec_embed = nn.Embed(
            num_embeddings=self.num_colors, features=self.embed_dim,
        )
        self.dec_hidden_layer = nn.Dense(self.decoder_hidden)
        self.dec_out = nn.Dense(28 * 28 * 3)

    def encode(self, x):
        """x -> (c, b)."""
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(self.trunk1(x))
        x = nn.relu(self.trunk2(x))
        c = self.c_proj(x)
        b = self.b_proj(x)
        return c, b

    def encode_content(self, x):
        """x -> c (content code only, for cycle consistency)."""
        c, _ = self.encode(x)
        return c

    def classify_digit(self, c):
        """c -> digit logits."""
        h = nn.relu(self.g_hidden(c))
        return self.g_out(h)

    def decode(self, c, s):
        """(c, s) -> reconstruction logits (B, 2352)."""
        s_emb = self.dec_embed(s)
        h = jnp.concatenate([c, s_emb], axis=-1)
        h = nn.relu(self.dec_hidden_layer(h))
        return self.dec_out(h)

    def __call__(self, x, s=None, train=True):
        c, b = self.encode(x)
        digit_logits = self.classify_digit(c)

        if s is None:
            # Eval: content path only
            return digit_logits, {"z": c}

        # Training: all heads
        color_logits = self.h_s(b)

        # Gradient reversal: identity forward, negated backward
        c_rev = (
            (1.0 + self.grad_rev_scale) * lax.stop_gradient(c)
            - self.grad_rev_scale * c
        )
        adversary_logits = self.a_s(c_rev)

        x_recon_logits = self.decode(c, s)

        return digit_logits, {
            "z": c,
            "c": c,
            "b": b,
            "digit_logits": digit_logits,
            "color_logits": color_logits,
            "adversary_logits": adversary_logits,
            "x_recon_logits": x_recon_logits,
        }
