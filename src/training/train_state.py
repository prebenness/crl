"""Custom Flax train states and state constructors."""

import jax.numpy as jnp
from flax.training import train_state
import optax


class ControlTrainState(train_state.TrainState):
    """TrainState extended with ControlVAE controller state."""
    beta: jnp.ndarray
    int_err: jnp.ndarray


def create_state_inner(rng, model, input_shape, cfg):
    """Initialize inner (VIB) model state with ControlVAE fields."""
    params = model.init(
        rng,
        jnp.ones(input_shape, jnp.float32),
        train=True,
    )["params"]
    tx = optax.adamw(cfg.training.lr, cfg.training.weight_decay_inner)
    opt_state = tx.init(params)

    return ControlTrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        beta=jnp.array(cfg.controller.beta_min, dtype=jnp.float32),
        int_err=jnp.array(0.0, dtype=jnp.float32),
    )


def create_state_outer(rng, model, input_shape, cfg):
    """Initialize outer (standard classifier) model state."""
    params = model.init(
        rng,
        jnp.ones(input_shape, jnp.float32),
        train=True,
    )["params"]
    tx = optax.adamw(cfg.training.lr, cfg.training.weight_decay_outer)
    opt_state = tx.init(params)

    return train_state.TrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
    )
