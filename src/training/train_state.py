"""Custom Flax train states and state constructors."""

import numpy as np
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


class MDLTrainState(train_state.TrainState):
    """TrainState with Gumbel temperature for MDL models."""
    tau: jnp.ndarray


def create_state_mdl(rng, model, input_shape, cfg):
    """Initialize MDL model state with temperature field."""
    params = model.init(
        rng,
        jnp.ones(input_shape, jnp.float32),
        tau=cfg.mdl.tau_start,
        train=False,
    )["params"]
    tx = optax.adamw(cfg.training.lr, cfg.training.weight_decay_inner)
    opt_state = tx.init(params)

    return MDLTrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        tau=jnp.array(cfg.mdl.tau_start, dtype=jnp.float32),
    )


def create_state_mdl_shared(rng, model, input_shape, cfg):
    """Initialize shared-MDL model state with model logits + phi logits.

    This mirrors the ANBN shared-weight setup: the underlying MDL MLP keeps
    its categorical weight logits, and a learned shared prior ``phi`` is
    represented by a separate trainable ``phi_logits`` vector in the same
    optimizer state.
    """
    model_params = model.init(
        rng,
        jnp.ones(input_shape, jnp.float32),
        tau=cfg.mdl.tau_start,
        train=False,
    )["params"]
    grid_size = int(len(model.grid_values))
    # Initialize φ logits ∝ log P_base = -l(s_m) * ln(2), so the shared
    # prior starts near P_base rather than uniform.  The optimizer only needs
    # to learn deviations from the coding-scheme-induced prior.
    cl = np.asarray(model.grid_codelengths, dtype=np.float32)
    phi_init = jnp.asarray(-cl * np.log(2.0), dtype=jnp.float32)
    params = {
        "logits": model_params["logits"],
        "phi_logits": phi_init,
    }
    tx = optax.adamw(cfg.training.lr, cfg.training.weight_decay_inner)
    opt_state = tx.init(params)

    return MDLTrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        tau=jnp.array(cfg.mdl.tau_start, dtype=jnp.float32),
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
