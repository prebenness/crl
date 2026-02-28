"""Regression test: mdl inner update parity between single and paired training."""

import jax
import jax.numpy as jnp
from jax import random as jrandom

from src.config import load_config
from src.mdl.coding import grid_values_and_codelengths
from src.models.classifiers import ULAMLPClassifier
from src.models.mdl_classifiers import GumbelSoftmaxMLP
from src.training.steps import make_train_step_mdl, make_train_step_mdl_pair
from src.training.train_state import create_state_mdl, create_state_outer


def test_mdl_inner_step_matches_pair_when_hsic_is_zero():
    """Inner MDL update should match exactly when pairing adds no gradient."""
    cfg = load_config("config/colored_mnist/mdl_pair_sweep.yaml")
    cfg.training.batch_size = 16
    cfg.mdl.n_samples = 3

    grid_vals, grid_codes = grid_values_and_codelengths(
        cfg.mdl.n_max, cfg.mdl.m_max,
    )
    inner_model = GumbelSoftmaxMLP(
        num_classes=cfg.model.num_classes,
        grid_values=grid_vals,
        grid_codelengths=grid_codes,
    )
    outer_model = ULAMLPClassifier(
        rep_dim=cfg.model.outer_rep_dim,
        num_classes=cfg.model.num_classes,
    )

    # Start both inner states from exactly the same initialization.
    init_key = jrandom.PRNGKey(11)
    inner_single = create_state_mdl(
        init_key, inner_model, (cfg.training.batch_size, 28, 28, 3), cfg,
    )
    inner_pair = create_state_mdl(
        init_key, inner_model, (cfg.training.batch_size, 28, 28, 3), cfg,
    )
    outer_state = create_state_outer(
        jrandom.PRNGKey(12), outer_model, (cfg.training.batch_size, 28, 28, 3), cfg,
    )

    x = jrandom.uniform(
        jrandom.PRNGKey(21), (cfg.training.batch_size, 28, 28, 3),
    )
    y = jrandom.randint(
        jrandom.PRNGKey(22), (cfg.training.batch_size,), 0, cfg.model.num_classes,
    )
    step_rng = jrandom.PRNGKey(23)
    mdl_lambda = 1e-1
    n_train = 60_000

    single_step = make_train_step_mdl(cfg, soft_forward=False)
    pair_step = make_train_step_mdl_pair(cfg, soft_forward=False)

    inner_single, m_single = single_step(
        inner_single, (x, y), step_rng, mdl_lambda, n_train,
    )
    inner_pair, _, m_pair = pair_step(
        inner_pair, outer_state, (x, y), step_rng,
        mdl_lambda, n_train, 0.0, cfg.model.num_classes,
    )

    flat_single, _ = jax.flatten_util.ravel_pytree(inner_single.params)
    flat_pair, _ = jax.flatten_util.ravel_pytree(inner_pair.params)

    assert jnp.allclose(flat_single, flat_pair, atol=0.0, rtol=0.0)
    assert float(m_single["objective_total_nats"]) == float(m_pair["objective_total_nats"])
    assert float(m_single["data_nll_nats"]) == float(m_pair["data_nll_nats"])
