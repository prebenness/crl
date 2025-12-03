import jax
import jax.numpy as jnp

from src.utils.cfg import CFG
from src.utils.utils import jax_mean

# =========================
# Loss & metrics (pure fns)
# =========================


def cross_entropy_loss(logits, labels):
    onehot = jax.nn.one_hot(labels, num_classes=CFG.num_classes)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(onehot * log_probs, axis=-1))


def accuracy(logits, labels):
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean((preds == labels).astype(jnp.float32))


def sum_of_squared_params(params) -> float:
    """Compute L2 penalty = sum of squared params (as a scalar float)."""
    sq_sums = [ jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params) ]
    return float(jnp.sum(jnp.stack(sq_sums)))


def mutual_information_ry(y_counts, ces):
        y_counts = y_counts.astype(jnp.float64)     # 64 bit precision for stability
        p = y_counts / y_counts.sum()               # Approx P_Y(y)
        p = p[p > 0]                                # Fine since we take plogp = 0 for p = 0
        H_y = float(-(p * jnp.log(p)).sum())
        CE  = jax_mean(ces)
        I = H_y - CE

        return I, H_y
