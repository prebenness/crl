import jax
import jax.numpy as jnp
from jax import value_and_grad
from flax.training import train_state
import optax

from src.pure_fns.losses_metrics import cross_entropy_loss, accuracy


# =========================
# JIT-compiled steps
# =========================
@jax.jit
def train_step(state, batch_x, batch_y):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch_x, train=True)
        loss = cross_entropy_loss(logits, batch_y)
        return loss, logits

    (loss, logits), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params=state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state)
    acc = accuracy(logits, batch_y)
    return new_state, loss, acc


def make_eval_step(apply_fn):
    """
    Factory to avoid passing Python callables into jitted fns.
    Captures apply_fn in a closure; the jitted function only takes arrays.
    """
    @jax.jit
    def eval_step(params, batch_x, batch_y):
        logits = apply_fn({"params": params}, batch_x, train=False)
        return accuracy(logits, batch_y)
    return eval_step

# =========================
# Train state (Flax+Optax)
# =========================
def create_train_state(rng, model, learning_rate, weight_decay, batch_size, input_dim):
    dummy = jnp.ones((batch_size, input_dim), jnp.float32)
    variables = model.init(rng, dummy)
    params = variables["params"]
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    return train_state.TrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=tx.init(params),
    )