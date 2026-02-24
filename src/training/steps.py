"""JIT-compiled training and evaluation step functions.

Each make_* factory closes over config values and returns a JIT-compiled
function with the same call signature as the original code.
"""

import jax
import jax.numpy as jnp
from jax import random as jrandom
import jax.lax as lax
from jax.scipy.special import logsumexp
import optax

from src.loss_fns.reg_loss_fns import class_cond_hsic_rbf


def _vib_loss(apply_fn, params, x, y, rng, train_mc_samples, alpha,
              beta, ctrl_ki, beta_min, beta_max, lamb):
    """Shared VIB inner-model loss: MC sampling, CE, recon, KL, ControlVAE.

    Returns (total_loss, (logits, ce, recon, kl, beta_used, mu)).
    """
    lamb = jnp.asarray(lamb, jnp.float32)
    alpha = jnp.asarray(alpha, jnp.float32)
    _beta_min = jnp.array(beta_min, dtype=jnp.float32)
    _beta_max = jnp.array(beta_max, dtype=jnp.float32)

    keys = jrandom.split(rng, train_mc_samples)

    def one_sample(k):
        logits, aux = apply_fn(
            {"params": params}, x, train=True, rngs={"noise": k},
        )
        return logits, aux.get("kl", 0.0), aux["x_recon_logits"], aux["mu"]

    logits_K, kl_K, x_recon_logits_K, mu_K = jax.vmap(one_sample)(keys)

    def ce_from_logits(lgts):
        return optax.softmax_cross_entropy_with_integer_labels(lgts, y).mean()
    ce_loss = jnp.mean(jax.vmap(ce_from_logits)(logits_K))

    def recon_from_logits(xl):
        bce = optax.sigmoid_binary_cross_entropy(xl, x)
        return bce.mean()
    recon_loss = jnp.mean(jax.vmap(recon_from_logits)(x_recon_logits_K))

    kl = (kl_K[0]).astype(jnp.float32)
    mu = mu_K[0]
    logits = jnp.mean(logits_K, axis=0)

    # ControlVAE dual-ascent controller
    kl_sg = lax.stop_gradient(kl)
    beta_candidate = beta + (ctrl_ki * (kl_sg - lamb))
    beta_used = jnp.clip(beta_candidate, _beta_min, _beta_max)

    lam_sg = lax.stop_gradient(beta_used)
    task_loss = (ce_loss * alpha) + (recon_loss * (1.0 - alpha))
    total_loss = task_loss * (1.0 - lam_sg) + lam_sg * kl

    return total_loss, (logits, ce_loss, recon_loss, kl, beta_used, mu)


def make_train_step(cfg):
    """Return a JIT-compiled single-model train_step."""
    beta_min = cfg.controller.beta_min
    beta_max = cfg.controller.beta_max
    ctrl_ki = cfg.controller.ctrl_ki
    train_mc_samples = cfg.mc_samples.train

    @jax.jit
    def train_step(state, batch, rng, lamb, alpha):
        x, y = batch

        def loss_fn(params):
            total_loss, (logits, ce, recon, kl, beta_used, _mu) = _vib_loss(
                state.apply_fn, params, x, y, rng,
                train_mc_samples, alpha,
                state.beta, ctrl_ki, beta_min, beta_max, lamb,
            )
            return total_loss, (logits, ce, recon, kl, beta_used)

        (loss, (logits, ce_loss, recon_loss, kl, beta_used)), grads = (
            jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        )
        state = state.apply_gradients(grads=grads)
        state = state.replace(beta=beta_used)

        acc = (jnp.argmax(logits, -1) == y).mean()
        metrics = {
            "loss": loss, "ce": ce_loss, "recon": recon_loss,
            "kl": kl, "beta": beta_used, "acc": acc,
        }
        return state, metrics

    return train_step


def make_train_step_pair(cfg):
    """Return a JIT-compiled dual-model train_step_pair."""
    beta_min = cfg.controller.beta_min
    beta_max = cfg.controller.beta_max
    ctrl_ki = cfg.controller.ctrl_ki
    train_mc_samples = cfg.mc_samples.train

    def train_step_pair(inner_state, outer_state, batch, rng, lamb, alpha,
                        hsic_w, num_classes):
        x, y = batch
        hsic_w = jnp.asarray(hsic_w, jnp.float32)

        rng1, rng2 = jrandom.split(rng, 2)

        # --- Inner model (VIB) ---
        def loss_fn1(params):
            return _vib_loss(
                inner_state.apply_fn, params, x, y, rng1,
                train_mc_samples, alpha,
                inner_state.beta, ctrl_ki, beta_min, beta_max, lamb,
            )

        (loss1, (logits1, ce1, recon1, kl1, beta1, mu1)), grads1 = (
            jax.value_and_grad(loss_fn1, has_aux=True)(inner_state.params)
        )
        inner_state = inner_state.apply_gradients(grads=grads1)
        inner_state = inner_state.replace(beta=beta1)

        mu1_sg = lax.stop_gradient(mu1)

        # --- Outer model (standard classifier + HSIC) ---
        def loss_fn2(params):
            logits2, aux2 = outer_state.apply_fn(
                {"params": params}, x, train=True,
            )
            z2 = aux2["z"]

            ce2 = optax.softmax_cross_entropy_with_integer_labels(
                logits2, y
            ).mean()
            hsic = class_cond_hsic_rbf(
                mu1_sg, z2, y, num_classes=num_classes,
            )

            total = ce2 * (1 - hsic_w) + hsic * hsic_w
            return total, (logits2, ce2, hsic)

        (loss2, (logits2, ce2, hsic_loss)), grads2 = (
            jax.value_and_grad(loss_fn2, has_aux=True)(outer_state.params)
        )
        outer_state = outer_state.apply_gradients(grads=grads2)

        acc1 = (jnp.argmax(logits1, -1) == y).mean()
        acc2 = (jnp.argmax(logits2, -1) == y).mean()

        metrics = {
            "loss1": loss1, "acc1": acc1, "ce1": ce1,
            "recon1": recon1, "kl1": kl1, "beta1": beta1,
            "loss2": loss2, "acc2": acc2, "ce2": ce2,
            "hsic": hsic_loss,
        }
        return inner_state, outer_state, metrics

    return jax.jit(train_step_pair, static_argnames=("num_classes",))


def _mdl_loss(apply_fn, params, x, y, rng, tau, mdl_lambda,
              n_train, n_samples=1, soft_forward=False):
    """MDL inner-model loss: CE + lambda * hyp_codelength - entropy_bonus.

    Returns (total_loss, (logits, ce, hyp_cl, entropy, z)).
    """
    from src.mdl.training import _compute_hyp_and_entropy

    B = x.shape[0]
    batch_scale = B / jnp.maximum(n_train, 1)

    if soft_forward or n_samples <= 1:
        logits, aux = apply_fn(
            {"params": params}, x, tau=tau, train=True, rng=rng,
            soft_forward=soft_forward,
        )
    else:
        keys = jrandom.split(rng, n_samples)

        def one_sample(k):
            return apply_fn(
                {"params": params}, x, tau=tau, train=True, rng=k,
            )

        all_logits_K, all_aux_K = jax.vmap(one_sample)(keys)
        logits = jnp.mean(all_logits_K, axis=0)
        # probs/codelength are identical across Gumbel samples
        aux = jax.tree.map(lambda a: a[0], all_aux_K)

    ce = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    hyp_cl, entropy, entropy_bonus = _compute_hyp_and_entropy(aux, tau)

    total_loss = ce + mdl_lambda * batch_scale * hyp_cl - batch_scale * entropy_bonus

    z = aux.get("z", None)

    return total_loss, (logits, ce, hyp_cl, entropy, z)


def make_train_step_mdl(cfg, soft_forward=False):
    """Return a JIT-compiled single-model MDL train_step.

    Call twice: once with soft_forward=True (warmup), once with False (training).
    """
    n_samples = cfg.mdl.n_samples

    @jax.jit
    def train_step(state, batch, rng, mdl_lambda, n_train):
        x, y = batch

        def loss_fn(params):
            return _mdl_loss(
                state.apply_fn, params, x, y, rng, state.tau,
                mdl_lambda, n_train, n_samples, soft_forward,
            )

        (loss, (logits, ce, hyp_cl, entropy, _z)), grads = (
            jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        )
        state = state.apply_gradients(grads=grads)

        acc = (jnp.argmax(logits, -1) == y).mean()
        metrics = {
            "loss": loss, "ce": ce, "hyp_cl": hyp_cl,
            "entropy": entropy, "acc": acc,
        }
        return state, metrics

    return train_step


def make_train_step_mdl_pair(cfg, soft_forward=False):
    """Return a JIT-compiled dual-model train step: MDL inner + standard outer with HSIC."""
    n_samples = cfg.mdl.n_samples

    def train_step_pair(inner_state, outer_state, batch, rng, mdl_lambda,
                        n_train, hsic_w, num_classes):
        x, y = batch
        hsic_w = jnp.asarray(hsic_w, jnp.float32)

        rng1, rng2 = jrandom.split(rng, 2)

        # --- Inner model (MDL) ---
        def loss_fn1(params):
            return _mdl_loss(
                inner_state.apply_fn, params, x, y, rng1,
                inner_state.tau, mdl_lambda, n_train, n_samples, soft_forward,
            )

        (loss1, (logits1, ce1, hyp_cl, entropy, z1)), grads1 = (
            jax.value_and_grad(loss_fn1, has_aux=True)(inner_state.params)
        )
        inner_state = inner_state.apply_gradients(grads=grads1)

        z1_sg = lax.stop_gradient(z1)

        # --- Outer model (standard classifier + HSIC) ---
        def loss_fn2(params):
            logits2, aux2 = outer_state.apply_fn(
                {"params": params}, x, train=True,
            )
            z2 = aux2["z"]

            ce2 = optax.softmax_cross_entropy_with_integer_labels(
                logits2, y
            ).mean()
            hsic = class_cond_hsic_rbf(
                z1_sg, z2, y, num_classes=num_classes,
            )

            total = ce2 * (1 - hsic_w) + hsic * hsic_w
            return total, (logits2, ce2, hsic)

        (loss2, (logits2, ce2, hsic_loss)), grads2 = (
            jax.value_and_grad(loss_fn2, has_aux=True)(outer_state.params)
        )
        outer_state = outer_state.apply_gradients(grads=grads2)

        acc1 = (jnp.argmax(logits1, -1) == y).mean()
        acc2 = (jnp.argmax(logits2, -1) == y).mean()

        metrics = {
            "loss1": loss1, "acc1": acc1, "ce1": ce1,
            "hyp_cl": hyp_cl, "entropy": entropy,
            "loss2": loss2, "acc2": acc2, "ce2": ce2,
            "hsic": hsic_loss,
        }
        return inner_state, outer_state, metrics

    return jax.jit(train_step_pair, static_argnames=("num_classes",))


def make_eval_step_mdl(cfg):
    """Return a JIT-compiled eval step for MDL models (deterministic argmax)."""
    @jax.jit
    def eval_step(state, batch, rng):
        x, y = batch
        logits, _ = state.apply_fn(
            {"params": state.params}, x, tau=1.0, train=False,
        )
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        acc = (jnp.argmax(logits, -1) == y).mean()
        return nll, acc

    return eval_step


def make_eval_step(cfg):
    """Return a JIT-compiled eval_step with MC averaging."""
    eval_mc_samples = cfg.mc_samples.eval

    @jax.jit
    def eval_step(state, batch, rng):
        x, y = batch
        keys = jrandom.split(rng, eval_mc_samples)

        def one_sample(k):
            logits, _ = state.apply_fn(
                {"params": state.params}, x, train=True, rngs={"noise": k},
            )
            return logits

        logits_K = jax.vmap(one_sample)(keys)

        log_probs_K = jax.nn.log_softmax(logits_K, axis=-1)
        log_probs = (logsumexp(log_probs_K, axis=0)
                     - jnp.log(eval_mc_samples))

        nll = -jnp.take_along_axis(
            log_probs, y[:, None], axis=-1
        ).mean()

        probs = jnp.exp(log_probs)
        acc = (jnp.argmax(probs, axis=-1) == y).mean()
        return nll, acc

    return eval_step
