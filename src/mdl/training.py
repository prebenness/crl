"""Training loop for the differentiable MDL experiment.

Implements the relaxed objective J_beta from the proposal:
    J_beta(alpha) = E[L_MDL(theta)] + lambda * E[sum l(theta_i)]
                    - tau * sum_i H(pi_i)

where tau = 1/beta.  The entropy bonus (subtracted) encourages
exploration by penalizing peaked weight distributions.

In practice:
    - L_D is estimated via Gumbel-Softmax ST (biased but practical)
    - The coding term sum_i l(theta_i) is computed exactly in expectation
      (since it's linear in pi): E[sum l(theta_i)] = sum_i sum_m pi_{i,m} l(s_m)
    - The entropy bonus (subtracted) is computed analytically
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


def _compute_complexity_and_entropy_bits(model_aux, tau):
    """Compute complexity and entropy terms in bits from model aux outputs.

    These terms depend only on the categorical logits (not Gumbel noise),
    so they are computed once regardless of n_samples.
    """
    complexity_expected_bits = model_aux["expected_codelength"]

    all_probs = model_aux["all_probs"]  # (n_params, M)
    log_probs = jnp.log2(all_probs + 1e-10)
    entropy_per_param_bits = -jnp.sum(all_probs * log_probs, axis=-1)
    entropy_weights_bits = jnp.sum(entropy_per_param_bits)

    # Entropy bonus (subtracted): tau * H, where tau = 1/beta
    entropy_bonus_bits = tau * entropy_weights_bits

    return complexity_expected_bits, entropy_weights_bits, entropy_bonus_bits


def _compute_data_nll_bits(logits, y, mask):
    """Compute data NLL (cross-entropy) in bits, averaged over valid positions."""
    ce_nats = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    ce_bits = ce_nats / jnp.log(2.0)
    return jnp.sum(ce_bits * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def make_loss_fn(mdl_lambda: float, n_train: int = 1, n_samples: int = 1,
                 soft_forward: bool = False, deterministic_st: bool = False):
    """Create the MDL loss function.

    The loss combines:
    1. Data term: cross-entropy averaged over valid positions (bits)
    2. Hypothesis term: expected codelength of weights, scaled by 1/N
    3. Entropy bonus (subtracted): tau * H(pi), scaled by 1/N

    Uses averaged data NLL + 1/N scaling on regularization terms, matching
    the cMNIST convention.

    When n_samples > 1, the data term is averaged over multiple independent
    Gumbel-Softmax samples to reduce gradient variance.

    When soft_forward=True, uses continuous relaxation (no Gumbel noise)
    for zero-variance gradients during warmup phase.
    When deterministic_st=True, uses hard argmax weights in forward with
    softmax(logits/tau) gradients (no Gumbel noise) for a stable bridge phase.

    Args:
        mdl_lambda: trade-off parameter for hypothesis codelength
        n_train: total number of training sequences (for 1/N reg scaling)
        n_samples: number of Gumbel samples to average data term over
        soft_forward: if True, use continuous relaxation (no Gumbel)
        deterministic_st: if True, use deterministic straight-through
    """
    mdl_lambda = float(mdl_lambda)
    n_train = float(max(n_train, 1))

    def loss_fn(params, apply_fn, x, y, mask, tau, rng):
        hyp_scale = 1.0 / n_train

        if soft_forward:
            # Single forward pass with continuous relaxation
            logits, model_aux = apply_fn(
                {"params": params}, x, tau=tau, train=True, rng=rng,
                soft_forward=True,
            )
            data_nll_bits = _compute_data_nll_bits(logits, y, mask)
        elif deterministic_st:
            # Single deterministic ST pass (hard argmax + soft gradients).
            logits, model_aux = apply_fn(
                {"params": params}, x, tau=tau, train=True,
                deterministic_st=True,
            )
            data_nll_bits = _compute_data_nll_bits(logits, y, mask)
        elif n_samples > 1:
            # Multi-sample: average data_cl over K Gumbel-Softmax passes
            keys = jrandom.split(rng, n_samples)

            def single_sample(key):
                logits_k, _ = apply_fn(
                    {"params": params}, x, tau=tau, train=True, rng=key,
                )
                data_nll_k = _compute_data_nll_bits(logits_k, y, mask)
                return data_nll_k

            # Keep one full aux tree, but avoid stacking K copies of large
            # tensors like all_probs across the Monte Carlo samples.
            logits_0, model_aux = apply_fn(
                {"params": params}, x, tau=tau, train=True, rng=keys[0],
            )
            data_nll_0 = _compute_data_nll_bits(logits_0, y, mask)
            rest_data_nll = jax.vmap(single_sample)(keys[1:])
            data_nll_bits = (data_nll_0 + jnp.sum(rest_data_nll)) / n_samples
        else:
            # Single Gumbel-Softmax sample
            logits, model_aux = apply_fn(
                {"params": params}, x, tau=tau, train=True, rng=rng,
            )
            data_nll_bits = _compute_data_nll_bits(logits, y, mask)

        # Hypothesis and entropy (exact, independent of Gumbel noise)
        complexity_expected_bits, entropy_weights_bits, entropy_bonus_bits = \
            _compute_complexity_and_entropy_bits(model_aux, tau)

        # Total relaxed MDL objective: averaged data NLL + 1/N scaling on reg terms.
        reg_complexity_weighted = mdl_lambda * hyp_scale * complexity_expected_bits
        reg_entropy_bonus = hyp_scale * entropy_bonus_bits
        reg_net = reg_complexity_weighted - reg_entropy_bonus
        objective_total_bits = data_nll_bits + reg_net

        aux = {
            # Unified naming (bits): objective decomposition
            "objective_total_bits": objective_total_bits,
            "data_nll_bits": data_nll_bits,
            "complexity_expected_bits": complexity_expected_bits,
            "entropy_weights_bits": entropy_weights_bits,
            "reg_complexity_weighted_bits": reg_complexity_weighted,
            "reg_entropy_bonus_bits": reg_entropy_bonus,
            "reg_net_bits": reg_net,
        }
        return objective_total_bits, aux

    return loss_fn


def make_train_step(mdl_lambda: float, n_train: int = 1, n_samples: int = 1,
                    soft_forward: bool = False,
                    deterministic_st: bool = False):
    """Create a JIT-compiled training step.

    Args:
        mdl_lambda: MDL trade-off parameter
        n_train: total training sequences (for batch scaling)
        n_samples: Gumbel samples for variance reduction (ST phase)
        soft_forward: use continuous relaxation (warmup phase)
        deterministic_st: deterministic straight-through bridge phase
    """
    loss_fn = make_loss_fn(
        mdl_lambda, n_train=n_train, n_samples=n_samples,
        soft_forward=soft_forward, deterministic_st=deterministic_st,
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


def anneal_tau_st_phase(epoch, total_epochs, warmup_epochs, tau_start, tau_end):
    """Anneal tau only after warmup; keep tau_start during warmup."""
    if epoch < warmup_epochs:
        return jnp.asarray(tau_start)
    st_epoch = epoch - warmup_epochs
    st_total = max(total_epochs - warmup_epochs, 1)
    return anneal_tau(st_epoch, st_total, tau_start, tau_end)
