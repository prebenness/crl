import jax
import jax.numpy as jnp

from src.utils.cfg import CFG

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
