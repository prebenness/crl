"""Epoch-level training/eval loops (lax.scan wrappers)."""

import jax
import jax.numpy as jnp
from jax import random as jrandom
import jax.lax as lax


def make_train_epoch(train_step_fn):
    """Return a JIT-compiled train_epoch using the given train_step."""
    def train_epoch(state, xb, yb, rng, lamb, alpha):
        n_batches = xb.shape[0]
        rngs = jrandom.split(rng, n_batches)

        def body(carry, inputs):
            st = carry
            x, y, r = inputs
            st, metrics = train_step_fn(st, (x, y), r, lamb, alpha)
            return st, metrics

        state, metrics_history = lax.scan(
            body, state, (xb, yb, rngs),
        )
        avg_metrics = {k: jnp.mean(v) for k, v in metrics_history.items()}
        return state, avg_metrics

    return jax.jit(train_epoch, donate_argnums=(0,))


def make_train_epoch_cba_om(train_step_fn):
    """Return a JIT-compiled CBA-OM train_epoch (batches include colour labels)."""
    def train_epoch(state, xb, yb, sb, rng, lamb, alpha):
        n_batches = xb.shape[0]
        rngs = jrandom.split(rng, n_batches)

        def body(carry, inputs):
            st = carry
            x, y, s, r = inputs
            st, metrics = train_step_fn(st, (x, y, s), r, lamb, alpha)
            return st, metrics

        state, metrics_history = lax.scan(
            body, state, (xb, yb, sb, rngs),
        )
        avg_metrics = {k: jnp.mean(v) for k, v in metrics_history.items()}
        return state, avg_metrics

    return jax.jit(train_epoch, donate_argnums=(0,))


def make_train_epoch_pair(train_step_pair_fn):
    """Return a JIT-compiled train_epoch_pair using the given train_step_pair."""
    def train_epoch_pair(inner_state, outer_state, xb, yb, rng, lamb, alpha,
                         hsic_w, num_classes):
        n_batches = xb.shape[0]
        rngs = jrandom.split(rng, n_batches)

        def body(carry, inputs):
            st1, st2 = carry
            x, y, r = inputs
            st1, st2, metrics = train_step_pair_fn(
                st1, st2, (x, y), r, lamb, alpha, hsic_w, num_classes,
            )
            return (st1, st2), metrics

        (inner_state, outer_state), metrics_history = lax.scan(
            body, (inner_state, outer_state), (xb, yb, rngs),
        )
        avg_metrics = {k: jnp.mean(v) for k, v in metrics_history.items()}
        return inner_state, outer_state, avg_metrics

    return jax.jit(
        train_epoch_pair, donate_argnums=(0, 1),
        static_argnames=("num_classes",),
    )


def make_train_epoch_mdl(train_step_fn):
    """Return a JIT-compiled MDL train_epoch."""
    def train_epoch(state, xb, yb, rng, mdl_lambda, n_train):
        n_batches = xb.shape[0]
        rngs = jrandom.split(rng, n_batches)

        def body(carry, inputs):
            st = carry
            x, y, r = inputs
            st, metrics = train_step_fn(st, (x, y), r, mdl_lambda, n_train)
            return st, metrics

        state, metrics_history = lax.scan(
            body, state, (xb, yb, rngs),
        )
        avg_metrics = {k: jnp.mean(v) for k, v in metrics_history.items()}
        return state, avg_metrics

    return jax.jit(train_epoch, donate_argnums=(0,))


def make_train_epoch_mdl_pair(train_step_pair_fn):
    """Return a JIT-compiled MDL pair train_epoch."""
    def train_epoch_pair(inner_state, outer_state, xb, yb, rng,
                         mdl_lambda, n_train, hsic_w, num_classes):
        n_batches = xb.shape[0]
        rngs = jrandom.split(rng, n_batches)

        def body(carry, inputs):
            st1, st2 = carry
            x, y, r = inputs
            st1, st2, metrics = train_step_pair_fn(
                st1, st2, (x, y), r, mdl_lambda, n_train, hsic_w, num_classes,
            )
            return (st1, st2), metrics

        (inner_state, outer_state), metrics_history = lax.scan(
            body, (inner_state, outer_state), (xb, yb, rngs),
        )
        avg_metrics = {k: jnp.mean(v) for k, v in metrics_history.items()}
        return inner_state, outer_state, avg_metrics

    return jax.jit(
        train_epoch_pair, donate_argnums=(0, 1),
        static_argnames=("num_classes",),
    )


def make_train_epoch_pair_mmd(train_step_pair_fn):
    """Return a JIT-compiled MMD pair train_epoch (VIB/oracle inner)."""
    def train_epoch_pair(inner_state, outer_state, xb, yb, sb, wb, rng,
                         lamb, alpha, mmd_w, num_classes):
        n_batches = xb.shape[0]
        rngs = jrandom.split(rng, n_batches)

        def body(carry, inputs):
            st1, st2 = carry
            x, y, s, w, r = inputs
            st1, st2, metrics = train_step_pair_fn(
                st1, st2, (x, y, s, w), r, lamb, alpha, mmd_w, num_classes,
            )
            return (st1, st2), metrics

        (inner_state, outer_state), metrics_history = lax.scan(
            body, (inner_state, outer_state), (xb, yb, sb, wb, rngs),
        )
        avg_metrics = {k: jnp.mean(v) for k, v in metrics_history.items()}
        return inner_state, outer_state, avg_metrics

    return jax.jit(
        train_epoch_pair, donate_argnums=(0, 1),
        static_argnames=("num_classes",),
    )


def make_train_epoch_mdl_pair_mmd(train_step_pair_fn):
    """Return a JIT-compiled MMD pair train_epoch (MDL/shared-MDL inner)."""
    def train_epoch_pair(inner_state, outer_state, xb, yb, sb, wb, rng,
                         mdl_lambda, n_train, mmd_w, num_classes):
        n_batches = xb.shape[0]
        rngs = jrandom.split(rng, n_batches)

        def body(carry, inputs):
            st1, st2 = carry
            x, y, s, w, r = inputs
            st1, st2, metrics = train_step_pair_fn(
                st1, st2, (x, y, s, w), r, mdl_lambda, n_train, mmd_w,
                num_classes,
            )
            return (st1, st2), metrics

        (inner_state, outer_state), metrics_history = lax.scan(
            body, (inner_state, outer_state), (xb, yb, sb, wb, rngs),
        )
        avg_metrics = {k: jnp.mean(v) for k, v in metrics_history.items()}
        return inner_state, outer_state, avg_metrics

    return jax.jit(
        train_epoch_pair, donate_argnums=(0, 1),
        static_argnames=("num_classes",),
    )


def make_eval_epoch(eval_step_fn):
    """Return a JIT-compiled eval_epoch using the given eval_step.

    Args:
        eval_step_fn: per-batch eval function returning (loss, acc).

    The returned ``eval_epoch(state, xb, yb, rng, counts)`` expects a
    ``counts`` array (from ``make_eval_batches``) giving the number of
    valid samples per batch.  Loss and accuracy are weighted by counts so
    that zero-padding in the last batch does not bias the result.
    """
    def eval_epoch(state, xb, yb, rng, counts):
        n_batches = xb.shape[0]
        rngs = jrandom.split(rng, n_batches)

        def body(carry, batch):
            x, y, r = batch
            loss, acc = eval_step_fn(state, (x, y), r)
            return carry, (loss, acc)

        _, (losses, accs) = lax.scan(body, None, (xb, yb, rngs))

        # Weighted mean: each batch's mean loss/acc is weighted by
        # the number of valid samples it contains.
        weights = counts.astype(jnp.float32)
        total = weights.sum()
        return (losses * weights).sum() / total, (accs * weights).sum() / total

    return jax.jit(eval_epoch)
