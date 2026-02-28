"""Shared-weight MDL extension with an adaptive prior (Section 8).

Extends the basic categorical MDL approach by introducing a learned shared
prior phi over the rational grid S. Instead of penalizing each weight's
codelength independently, we use a composite objective that encourages
weight-sharing through KL divergence terms.

Composite objective (Section 8.1):

    J(alpha, phi; beta) = E[L_D(theta)]
                        + lambda1 * sum_i DKL(pi_i || phi)
                        + lambda2 * DKL(phi || P_base)
                        - (1/beta) * sum_i H(pi_i)

where:
    pi_i = softmax(alpha_i)        per-weight categorical distribution
    phi in Delta^{M-1}_epsilon     learned shared adaptive prior (epsilon-bounded)
    P_base(s_m) ~ 1/(|s_m| + 1)   fixed hyper-prior favoring simple rationals
    lambda1                        weight-sharing KL weight
    lambda2                        dictionary cost weight

The adaptive prior phi is parameterized via unconstrained logits and mapped
onto the epsilon-bounded simplex:

    phi = softmax(phi_logits) * (1 - M * epsilon) + epsilon

This ensures phi_m >= epsilon for all m, preventing KL divergence from
blowing up when some grid values are unused.
"""

import jax
import jax.numpy as jnp
from jax import random as jrandom
import optax
from flax.training import train_state

from src.mdl.data import NUM_SYMBOLS


# ---------------------------------------------------------------------------
# Hyper-prior and simplex utilities
# ---------------------------------------------------------------------------

def compute_p_base(grid_values):
    """Fixed hyper-prior P_base(s_m) proportional to 1 / (|s_m| + 1).

    Assigns higher probability to simpler rationals (those closer to 0
    or with small absolute value like 0, +/-1).

    Args:
        grid_values: float32 array of shape (M,) with rational grid values.

    Returns:
        p_base: float32 array of shape (M,), normalized probability vector.
    """
    grid_values = jnp.asarray(grid_values)
    unnormalized = 1.0 / (jnp.abs(grid_values) + 1.0)
    p_base = unnormalized / jnp.sum(unnormalized)
    return p_base


def epsilon_bound_simplex(phi_logits, epsilon):
    """Map unconstrained logits to the epsilon-bounded probability simplex.

    phi = softmax(phi_logits) * (1 - M * epsilon) + epsilon

    This guarantees phi_m >= epsilon for all m, which is essential to keep
    KL(pi_i || phi) finite even when pi_i concentrates on a grid value that
    phi would otherwise assign zero probability.

    Args:
        phi_logits: float32 array of shape (M,), unconstrained logits.
        epsilon: float, minimum probability for each grid element.

    Returns:
        phi: float32 array of shape (M,), epsilon-bounded probability vector.
    """
    M = phi_logits.shape[-1]
    soft = jax.nn.softmax(phi_logits, axis=-1)
    phi = soft * (1.0 - M * epsilon) + epsilon
    return phi


# ---------------------------------------------------------------------------
# KL divergence (in bits, using log2)
# ---------------------------------------------------------------------------

def _kl_divergence(p, q):
    """DKL(p || q) in bits.

    DKL(p || q) = sum_m p_m * log2(p_m / q_m)

    Both p and q must be strictly positive where p > 0 to avoid NaN.
    A small additive constant is used for numerical stability.

    Args:
        p: float32 (..., M) probability distributions.
        q: float32 (..., M) probability distributions.

    Returns:
        kl: float32 (...) KL divergence per distribution.
    """
    eps = 1e-10
    return jnp.sum(p * jnp.log2((p + eps) / (q + eps)), axis=-1)


# ---------------------------------------------------------------------------
# Train state
# ---------------------------------------------------------------------------

class SharedMDLTrainState(train_state.TrainState):
    """TrainState extended with temperature and shared prior metadata.

    The phi_logits are stored inside ``params`` under the key
    ``"phi_logits"`` so that they are optimized jointly with the model
    logits by the same optimizer.  This avoids the need for a separate
    optimizer or manual gradient handling.

    Additional fields:
        tau: Gumbel-Softmax temperature (= 1/beta).
    """
    tau: jnp.ndarray


def create_shared_mdl_state(
    rng,
    model,
    grid_values,
    seq_len,
    batch_size,
    lr,
    tau_init,
):
    """Initialize training state with both model logits and phi_logits.

    The params dict has the structure::

        {
            "logits": (n_params, M),   # per-weight categorical logits (alpha)
            "phi_logits": (M,),        # shared prior logits (unconstrained)
        }

    phi_logits is initialized to zeros, which corresponds to a uniform
    adaptive prior at the start of training.

    Args:
        rng: PRNG key.
        model: GumbelSoftmaxLSTM instance.
        grid_values: float32 array (M,) of rational grid values.
        seq_len: sequence length for dummy initialization.
        batch_size: batch size for dummy initialization.
        lr: learning rate for Adam optimizer.
        tau_init: initial Gumbel-Softmax temperature.

    Returns:
        SharedMDLTrainState with joint params.
    """
    dummy_x = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    model_params = model.init(
        rng,
        dummy_x,
        tau=tau_init,
        train=False,
    )["params"]

    M = len(grid_values)
    phi_logits = jnp.zeros((M,), dtype=jnp.float32)

    # Joint params dict: model logits + phi_logits side by side.
    params = {
        "logits": model_params["logits"],
        "phi_logits": phi_logits,
    }

    tx = optax.adam(lr)
    return SharedMDLTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        tau=jnp.array(tau_init, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def _shared_compute_data_nll_bits(logits, y, mask):
    """Compute data NLL (cross-entropy) in bits for one forward pass."""
    ce_nats = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    ce_bits = ce_nats / jnp.log(2.0)
    return jnp.sum(ce_bits * mask)


def make_shared_loss_fn(lambda1=1.0, lambda2=1.0, epsilon=1e-6, n_train=1,
                        n_samples=1, soft_forward=False,
                        deterministic_st=False):
    """Create the shared-weight MDL loss function (Section 8.1).

    The composite objective is:

        J = E[L_D(theta)]
          + lambda1 * sum_i DKL(pi_i || phi)
          + lambda2 * DKL(phi || P_base)
          - (1/beta) * sum_i H(pi_i)

    where beta = 1/tau and phi is epsilon-bounded.

    Args:
        lambda1: weight for the per-weight KL term (weight sharing).
        lambda2: weight for the dictionary cost KL term.
        epsilon: minimum probability for each grid element in phi.
        n_train: total number of training sequences (for batch scaling).
        n_samples: number of Gumbel samples for variance reduction.
        soft_forward: if True, use continuous relaxation (no Gumbel).
    """
    lambda1 = float(lambda1)
    lambda2 = float(lambda2)
    epsilon = float(epsilon)
    n_train = float(max(n_train, 1))

    def loss_fn(params, apply_fn, x, y, mask, tau, rng, p_base):
        model_params = {"logits": params["logits"]}

        if soft_forward:
            logits, model_aux = apply_fn(
                {"params": model_params}, x, tau=tau, train=True, rng=rng,
                soft_forward=True,
            )
            data_nll_bits = _shared_compute_data_nll_bits(logits, y, mask)
        elif deterministic_st:
            logits, model_aux = apply_fn(
                {"params": model_params}, x, tau=tau, train=True,
                deterministic_st=True,
            )
            data_nll_bits = _shared_compute_data_nll_bits(logits, y, mask)
        elif n_samples > 1:
            keys = jrandom.split(rng, n_samples)

            def single_sample(key):
                logits_k, aux_k = apply_fn(
                    {"params": model_params}, x, tau=tau, train=True, rng=key,
                )
                data_nll_k = _shared_compute_data_nll_bits(logits_k, y, mask)
                return data_nll_k, aux_k

            data_nll_samples, all_aux = jax.vmap(single_sample)(keys)
            data_nll_bits = jnp.mean(data_nll_samples)

            # model_aux is identical across samples (doesn't depend on
            # Gumbel noise), so just take the first.
            model_aux = jax.tree.map(lambda x: x[0], all_aux)
        else:
            logits, model_aux = apply_fn(
                {"params": model_params}, x, tau=tau, train=True, rng=rng,
            )
            data_nll_bits = _shared_compute_data_nll_bits(logits, y, mask)

        # Per-weight distributions
        all_probs = model_aux["all_probs"]  # (n_params, M)

        # Shared adaptive prior (epsilon-bounded)
        phi = epsilon_bound_simplex(params["phi_logits"], epsilon)  # (M,)

        # KL(pi_i || phi) for each weight, summed
        kl_per_weight = _kl_divergence(all_probs, phi[None, :])
        kl_weight_sharing = jnp.sum(kl_per_weight)

        # KL(phi || P_base)
        p_base = jnp.asarray(p_base)
        kl_dictionary = _kl_divergence(phi, p_base)

        # Entropy bonus
        log_probs = jnp.log2(all_probs + 1e-10)
        entropy_per_param = -jnp.sum(all_probs * log_probs, axis=-1)
        entropy_weights_bits = jnp.sum(entropy_per_param)
        entropy_bonus_bits = tau * entropy_weights_bits

        # Batch scaling
        B = x.shape[0]
        batch_scale = B / n_train

        # Composite objective (bits) with explicit decomposition.
        complexity_expected_bits = (
            lambda1 * kl_weight_sharing + lambda2 * kl_dictionary
        )
        reg_complexity_weighted = batch_scale * complexity_expected_bits
        reg_entropy_bonus = batch_scale * entropy_bonus_bits
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
            "kl_pi_phi_bits": kl_weight_sharing,
            "kl_phi_pbase_bits": kl_dictionary,
            "phi_min_prob": jnp.min(phi),
            "phi_max_prob": jnp.max(phi),
            "phi_entropy_bits": -jnp.sum(phi * jnp.log2(phi + 1e-10)),
        }
        return objective_total_bits, aux

    return loss_fn


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def make_shared_train_step(lambda1=1.0, lambda2=1.0, epsilon=1e-6, n_train=1,
                           n_samples=1, soft_forward=False,
                           deterministic_st=False):
    """Create a JIT-compiled training step for the shared-weight objective.

    Args:
        lambda1: weight for the per-weight KL term.
        lambda2: weight for the dictionary cost KL term.
        epsilon: minimum probability for phi.
        n_train: total number of training sequences (for batch scaling).
        n_samples: Gumbel samples for variance reduction.
        soft_forward: use continuous relaxation (warmup phase).
        deterministic_st: deterministic straight-through bridge phase.
    """
    loss_fn = make_shared_loss_fn(
        lambda1, lambda2, epsilon, n_train=n_train,
        n_samples=n_samples, soft_forward=soft_forward,
        deterministic_st=deterministic_st,
    )

    @jax.jit
    def train_step(state, x, y, mask, rng, p_base):
        def _loss(params):
            return loss_fn(
                params, state.apply_fn, x, y, mask, state.tau, rng, p_base,
            )

        (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
            state.params
        )
        state = state.apply_gradients(grads=grads)
        return state, loss, aux

    return train_step
