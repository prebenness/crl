"""Training loop for the differentiable MDL experiment.

Implements the relaxed objective J_beta from the proposal:
    J_beta(alpha) = E[L_MDL(theta)] - (1/beta) * sum_i H(pi_i)

where:
    L_MDL(theta) = L_D(theta) + lambda * sum_i l(theta_i)

In practice:
    - L_D is estimated via Gumbel-Softmax ST (biased but practical)
    - The coding term sum_i l(theta_i) is computed exactly in expectation
      (since it's linear in pi): E[sum l(theta_i)] = sum_i sum_m pi_{i,m} l(s_m)
    - The entropy bonus is computed analytically
"""

import jax
import jax.numpy as jnp
from jax import random as jrandom
import optax
from flax.training import train_state

from src.mdl.data import NUM_SYMBOLS


class MDLTrainState(train_state.TrainState):
    """TrainState extended with temperature (tau = 1/beta)."""
    tau: jnp.ndarray


def create_mdl_state(rng, model, seq_len, batch_size, lr, tau_init):
    """Initialize model state for MDL training."""
    dummy_x = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    params = model.init(
        rng,
        dummy_x,
        tau=tau_init,
        train=False,
    )["params"]
    tx = optax.adam(lr)
    return MDLTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        tau=jnp.array(tau_init, dtype=jnp.float32),
    )


def _compute_hyp_and_entropy(model_aux, tau):
    """Compute hypothesis codelength and entropy terms from model aux outputs.

    These terms depend only on the categorical logits (not Gumbel noise),
    so they are computed once regardless of n_samples.
    """
    expected_hyp_codelength = model_aux["expected_codelength"]

    all_probs = model_aux["all_probs"]  # (n_params, M)
    log_probs = jnp.log2(all_probs + 1e-10)
    entropy_per_param = -jnp.sum(all_probs * log_probs, axis=-1)
    total_entropy = jnp.sum(entropy_per_param)

    # beta = 1/tau, so 1/beta = tau
    entropy_bonus = tau * total_entropy

    return expected_hyp_codelength, total_entropy, entropy_bonus


def _compute_data_codelength(logits, y, mask):
    """Compute data codelength (cross-entropy in bits) for one forward pass."""
    ce_nats = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    ce_bits = ce_nats / jnp.log(2.0)
    data_codelength = jnp.sum(ce_bits * mask)
    ce_per_token = jnp.sum(ce_nats * mask) / jnp.sum(mask)
    return data_codelength, ce_per_token


def make_loss_fn(mdl_lambda: float, n_train: int = 1, n_samples: int = 1,
                 soft_forward: bool = False):
    """Create the MDL loss function.

    The loss combines:
    1. Data term: cross-entropy (= negative log-likelihood = codelength of data)
    2. Hypothesis term: expected codelength of weights under categorical dist
    3. Entropy bonus: -(1/beta) * H(pi) to encourage exploration

    When n_samples > 1, the data term is averaged over multiple independent
    Gumbel-Softmax samples to reduce gradient variance.

    When soft_forward=True, uses continuous relaxation (no Gumbel noise)
    for zero-variance gradients during warmup phase.

    Args:
        mdl_lambda: trade-off parameter for hypothesis codelength
        n_train: total number of training sequences (for batch scaling)
        n_samples: number of Gumbel samples to average data term over
        soft_forward: if True, use continuous relaxation (no Gumbel)
    """
    mdl_lambda = float(mdl_lambda)
    n_train = float(max(n_train, 1))

    def loss_fn(params, apply_fn, x, y, mask, tau, rng):
        B = x.shape[0]
        batch_scale = B / n_train

        if soft_forward:
            # Single forward pass with continuous relaxation
            logits, model_aux = apply_fn(
                {"params": params}, x, tau=tau, train=True, rng=rng,
                soft_forward=True,
            )
            data_codelength, ce_per_token = _compute_data_codelength(
                logits, y, mask,
            )
        elif n_samples > 1:
            # Multi-sample: average data_cl over K Gumbel-Softmax passes
            keys = jrandom.split(rng, n_samples)

            def single_sample(key):
                logits_k, aux_k = apply_fn(
                    {"params": params}, x, tau=tau, train=True, rng=key,
                )
                data_cl, ce = _compute_data_codelength(logits_k, y, mask)
                return data_cl, ce, aux_k

            data_cls, ce_per_tokens, all_aux = jax.vmap(single_sample)(keys)
            data_codelength = jnp.mean(data_cls)
            ce_per_token = jnp.mean(ce_per_tokens)

            # model_aux is identical across samples (doesn't depend on
            # Gumbel noise), so just take the first.
            model_aux = jax.tree.map(lambda x: x[0], all_aux)
        else:
            # Single Gumbel-Softmax sample
            logits, model_aux = apply_fn(
                {"params": params}, x, tau=tau, train=True, rng=rng,
            )
            data_codelength, ce_per_token = _compute_data_codelength(
                logits, y, mask,
            )

        # Hypothesis and entropy (exact, independent of Gumbel noise)
        expected_hyp_codelength, total_entropy, entropy_bonus = \
            _compute_hyp_and_entropy(model_aux, tau)

        # Total relaxed MDL objective with batch scaling
        mdl_loss = data_codelength + mdl_lambda * batch_scale * expected_hyp_codelength
        total_loss = mdl_loss - batch_scale * entropy_bonus

        aux = {
            "data_codelength": data_codelength,
            "hyp_codelength": expected_hyp_codelength,
            "entropy": total_entropy,
            "entropy_bonus": entropy_bonus,
            "mdl_total": mdl_loss,
            "ce_per_token": ce_per_token,
        }
        return total_loss, aux

    return loss_fn


def make_train_step(mdl_lambda: float, n_train: int = 1, n_samples: int = 1,
                    soft_forward: bool = False):
    """Create a JIT-compiled training step.

    Args:
        mdl_lambda: MDL trade-off parameter
        n_train: total training sequences (for batch scaling)
        n_samples: Gumbel samples for variance reduction (ST phase)
        soft_forward: use continuous relaxation (warmup phase)
    """
    loss_fn = make_loss_fn(
        mdl_lambda, n_train=n_train, n_samples=n_samples,
        soft_forward=soft_forward,
    )

    @jax.jit
    def train_step(state, x, y, mask, rng):
        def _loss(params):
            return loss_fn(
                params, state.apply_fn, x, y, mask, state.tau, rng,
            )

        (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
            state.params
        )
        state = state.apply_gradients(grads=grads)
        return state, loss, aux

    return train_step


def deterministic_accuracy_single(
    apply_fn, params, grid_values, inp, tgt,
):
    """Compute deterministic accuracy on a single a^n b^n string.

    Deterministic accuracy (Lan et al.): ratio of correct predictions
    at positions where the next token is fully determined. Per Lan et al.,
    this is "the phase that starts once the first 'b' appears, including
    the end-of-sequence symbol."

    The deterministic positions are those where the INPUT is 'b': at these
    positions the network has already seen a 'b' and all future symbols
    are determined (more b's, then #).

    Args:
        apply_fn: model.apply
        params: trained parameters
        grid_values: rational grid values
        inp: (seq_len,) input token sequence
        tgt: (seq_len,) target token sequence

    Returns:
        accuracy: float, deterministic accuracy for this string
    """
    from src.mdl.data import SYMBOL_B

    x = jnp.array(inp)[None, :]  # (1, T)
    logits, _ = apply_fn(
        {"params": params}, x, tau=1.0, train=False,
    )
    preds = jnp.argmax(logits[0], axis=-1)  # (T,)

    inp_arr = jnp.array(inp)
    tgt_arr = jnp.array(tgt)
    n = len(inp)
    correct = (preds[:n] == tgt_arr[:n]).astype(jnp.float32)

    # Deterministic positions: where the input is 'b'
    # At these positions the network knows the rest of the string is b...b#
    det_mask = (inp_arr == SYMBOL_B).astype(jnp.float32)
    n_det = jnp.sum(det_mask)

    # Avoid division by zero for n=0 strings (no b's in input)
    acc = jnp.where(n_det > 0, jnp.sum(correct * det_mask) / n_det, 1.0)
    return acc


def evaluate_deterministic_accuracy(
    apply_fn, params, grid_values, test_inputs, test_targets,
    max_n: int = 1500,
    batch_size: int = 64,
):
    """Evaluate deterministic accuracy on a^n b^n test set.

    Uses batched evaluation to avoid per-string JAX recompilation.
    Strings are grouped into batches of similar length and padded.

    Returns per-string accuracies and overall accuracy.
    """
    import numpy as np
    from src.mdl.data import SYMBOL_B

    N = len(test_inputs)
    accs = np.zeros(N, dtype=np.float32)

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_inputs = test_inputs[batch_start:batch_end]
        batch_targets = test_targets[batch_start:batch_end]
        B = len(batch_inputs)

        max_len = max(len(s) for s in batch_inputs)
        x_pad = np.zeros((B, max_len), dtype=np.int32)
        y_pad = np.zeros((B, max_len), dtype=np.int32)
        det_mask = np.zeros((B, max_len), dtype=np.float32)

        for i, (inp, tgt) in enumerate(zip(batch_inputs, batch_targets)):
            L = len(inp)
            x_pad[i, :L] = inp
            y_pad[i, :L] = tgt
            for t in range(L):
                if inp[t] == SYMBOL_B:
                    det_mask[i, t] = 1.0

        x_jnp = jnp.array(x_pad)
        y_jnp = jnp.array(y_pad)
        det_mask_jnp = jnp.array(det_mask)

        logits, _ = apply_fn(
            {"params": params}, x_jnp, tau=1.0, train=False,
        )
        preds = jnp.argmax(logits, axis=-1)

        correct = (preds == y_jnp).astype(jnp.float32)
        n_det = jnp.sum(det_mask_jnp, axis=-1)
        n_correct = jnp.sum(correct * det_mask_jnp, axis=-1)
        batch_accs = jnp.where(n_det > 0, n_correct / n_det, 1.0)

        accs[batch_start:batch_end] = np.array(batch_accs)

    accs_arr = jnp.array(accs)
    all_correct = bool(jnp.all(accs_arr > 1.0 - 1e-6))
    mean_acc = jnp.mean(accs_arr)

    if not all_correct:
        failures = jnp.where(
            accs_arr < 1.0 - 1e-6, jnp.arange(len(accs_arr)), len(accs_arr),
        )
        first_fail = int(jnp.min(failures)) + 1  # +1 because test starts at n=1
    else:
        first_fail = None

    n_perfect = int(jnp.sum(accs_arr > 1.0 - 1e-6))

    # gen_n: largest n such that all strings 1..n have 100% accuracy
    if all_correct:
        gen_n = len(test_inputs)
    elif first_fail is not None:
        gen_n = first_fail - 1
    else:
        gen_n = 0

    return {
        "mean_accuracy": float(mean_acc),
        "all_correct": all_correct,
        "first_failure_n": first_fail,
        "per_string_acc": accs_arr,
        "n_perfect": n_perfect,
        "gen_n": gen_n,
    }


def anneal_tau(epoch, total_epochs, tau_start, tau_end):
    """Exponential temperature annealing: tau_start -> tau_end over training."""
    progress = epoch / max(total_epochs - 1, 1)
    log_tau = jnp.log(tau_start) + progress * (jnp.log(tau_end) - jnp.log(tau_start))
    return jnp.exp(log_tau)
