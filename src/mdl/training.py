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


def make_loss_fn(mdl_lambda: float):
    """Create the MDL loss function.

    The loss combines:
    1. Data term: cross-entropy (= negative log-likelihood = codelength of data)
    2. Hypothesis term: expected codelength of weights under categorical dist
    3. Entropy bonus: -(1/beta) * H(pi) to encourage exploration

    Args:
        mdl_lambda: trade-off parameter for hypothesis codelength
    """
    mdl_lambda = float(mdl_lambda)

    def loss_fn(params, apply_fn, x, y, mask, tau, rng):
        """Compute the relaxed MDL objective.

        Args:
            params: model parameters (the logits)
            apply_fn: model.apply function
            x: (B, T) input tokens
            y: (B, T) target tokens
            mask: (B, T) valid position mask
            tau: Gumbel-Softmax temperature
            rng: PRNG key

        Returns:
            loss: scalar
            aux: dict with component losses
        """
        logits, model_aux = apply_fn(
            {"params": params}, x, tau=tau, train=True, rng=rng,
        )

        # --- Data codelength: cross-entropy in bits ---
        # logits: (B, T, num_symbols), y: (B, T)
        ce_nats = optax.softmax_cross_entropy_with_integer_labels(
            logits, y
        )  # (B, T)
        # Mask out padding and convert to bits
        ce_bits = ce_nats / jnp.log(2.0)
        # Sum over valid positions (total codelength for the dataset)
        data_codelength = jnp.sum(ce_bits * mask)

        # --- Hypothesis codelength: computed exactly in expectation ---
        # E[sum_i l(theta_i)] = sum_i sum_m pi_{i,m} * l(s_m)
        expected_hyp_codelength = model_aux["expected_codelength"]

        # --- Entropy bonus ---
        all_probs = model_aux["all_probs"]  # (n_params, M)
        # H(pi_i) = -sum_m pi_{i,m} log2(pi_{i,m})
        log_probs = jnp.log2(all_probs + 1e-10)
        entropy_per_param = -jnp.sum(all_probs * log_probs, axis=-1)
        total_entropy = jnp.sum(entropy_per_param)

        # beta = 1/tau, so 1/beta = tau
        entropy_bonus = tau * total_entropy

        # --- Total relaxed MDL objective ---
        # J_beta = E[L_D] + lambda * E[sum l(theta_i)] - (1/beta) * H
        mdl_loss = data_codelength + mdl_lambda * expected_hyp_codelength
        total_loss = mdl_loss - entropy_bonus

        aux = {
            "data_codelength": data_codelength,
            "hyp_codelength": expected_hyp_codelength,
            "entropy": total_entropy,
            "entropy_bonus": entropy_bonus,
            "mdl_total": mdl_loss,
            "ce_per_token": jnp.sum(ce_nats * mask) / jnp.sum(mask),
        }
        return total_loss, aux

    return loss_fn


def make_train_step(mdl_lambda: float):
    """Create a JIT-compiled training step."""
    loss_fn = make_loss_fn(mdl_lambda)

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

    return {
        "mean_accuracy": float(mean_acc),
        "all_correct": all_correct,
        "first_failure_n": first_fail,
        "per_string_acc": accs_arr,
    }
