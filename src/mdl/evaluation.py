"""Paper-comparable evaluation metrics for differentiable MDL.

Implements the metrics from Abudy et al. (2025, arXiv:2505.13398v2) and
Lan et al. (2024, arXiv:2402.10013v2) for direct comparison in tables.

Key conventions (matching Abudy et al. 2025):
    |D:H| = data term only (NLL in bits), NOT including |H|.
    |H|   = hypothesis codelength (Lan-style encoding, bits).
    Δ%    = (|D:H|_ours - |D:H|_golden) / |D:H|_golden × 100.

    Train |D:H| = raw sum of per-string NLL over the training corpus.
    Test  |D:H| = grammar-weighted expected NLL over the exhaustive test set.

    Smoothing: additive 1e-10 to output probs before log (no re-norm),
    matching Abudy et al. (2025, Section 4).
"""

import jax
import jax.numpy as jnp
import numpy as np

from src.mdl.data import make_test_set, make_anbn_fixed_n


# ---------------------------------------------------------------------------
# Grammar weights
# ---------------------------------------------------------------------------

def compute_anbn_grammar_weights(
    max_n: int, p: float = 0.3, min_n: int = 0,
) -> np.ndarray:
    """Raw PCFG weights P(n) = p * (1-p)^n for n=min_n..max_n.

    Returns unnormalized probabilities from the geometric PCFG.
    For full PCFG (min_n=0, large max_n), these sum to ~1.

    The default min_n=0 includes the empty string, matching the
    information-theoretic |D:H| = E_{s~PCFG}[-log P_H(s)] over
    the full grammar distribution.

    Args:
        max_n: largest n in the test set.
        p: PCFG termination probability.
        min_n: smallest n (default 0, includes empty string).

    Returns:
        (max_n - min_n + 1,) float64 array of PCFG weights.
    """
    ns = np.arange(min_n, max_n + 1)
    return p * (1 - p) ** ns


# ---------------------------------------------------------------------------
# Per-string NLL (core computation)
# ---------------------------------------------------------------------------

def compute_per_string_nll_bits(
    forward_fn,
    inputs,
    targets,
    batch_size: int = 64,
    smoothing: float = 1e-10,
) -> np.ndarray:
    """Compute total NLL in bits per string (sum over positions).

    Uses the Abudy et al. (2025) smoothing convention: additive epsilon
    to output probabilities, no re-normalization.

    Args:
        forward_fn: callable (x_batch: int32 (B,T)) -> logits (B,T,V).
        inputs: list of variable-length int lists (input sequences).
        targets: list of variable-length int lists (target sequences).
        batch_size: strings per batch for padded evaluation.
        smoothing: additive constant for probability smoothing.

    Returns:
        (N,) float64 array of per-string total NLL in bits.
    """
    N = len(inputs)
    nll_per_string = np.zeros(N, dtype=np.float64)

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_inputs = inputs[batch_start:batch_end]
        batch_targets = targets[batch_start:batch_end]
        B = len(batch_inputs)

        max_len = max(len(s) for s in batch_inputs)
        x_pad = np.zeros((B, max_len), dtype=np.int32)
        y_pad = np.zeros((B, max_len), dtype=np.int32)
        mask = np.zeros((B, max_len), dtype=np.float32)

        for i, (inp, tgt) in enumerate(zip(batch_inputs, batch_targets)):
            L = len(inp)
            x_pad[i, :L] = inp
            y_pad[i, :L] = tgt
            mask[i, :L] = 1.0

        x_jnp = jnp.array(x_pad)
        y_jnp = jnp.array(y_pad)
        mask_jnp = jnp.array(mask)

        logits = forward_fn(x_jnp)  # (B, T, V)

        # Smoothed NLL in bits per position
        probs = jax.nn.softmax(logits, axis=-1)
        probs_smoothed = probs + smoothing
        log_probs_bits = jnp.log2(probs_smoothed)
        nll_bits = -jnp.take_along_axis(
            log_probs_bits, y_jnp[..., None], axis=-1,
        ).squeeze(-1)  # (B, T)

        # Sum over positions per string (not average)
        per_string = jnp.sum(nll_bits * mask_jnp, axis=-1)  # (B,)
        nll_per_string[batch_start:batch_end] = np.array(per_string)

    return nll_per_string


# ---------------------------------------------------------------------------
# Grammar-weighted test NLL  (|D:H| data term, test)
# ---------------------------------------------------------------------------

def compute_grammar_weighted_nll_bits(
    forward_fn,
    max_n: int,
    p: float = 0.3,
    batch_size: int = 64,
) -> dict:
    """Grammar-weighted |D:H| data term on the exhaustive test set.

    |D:H|_test = Σ_{n=0}^{max_n} P(n) × NLL_total(n)

    where NLL_total(n) is the total NLL in bits for string aⁿbⁿ,
    P(n) = p*(1-p)^n is the raw PCFG probability (unnormalized),
    and n=0 is the empty string "# #".

    Including n=0 with raw PCFG weights matches the information-
    theoretic convention |D:H| = E_{s~PCFG}[-log P_H(s)].
    For the ideal golden predictor this gives ~2.94 bits, matching
    Abudy et al. (2025, arXiv:2505.13398v2, line 803).

    Args:
        forward_fn: callable (x_batch) -> logits.
        max_n: largest n in the test set.
        p: PCFG termination probability.
        batch_size: strings per batch.

    Returns:
        dict with data_dh_bits, nll_per_string, grammar_weights, max_n.
    """
    # Build test set including n=0 (empty string)
    test_inputs = []
    test_targets = []
    for n in range(0, max_n + 1):
        inp, tgt = make_anbn_fixed_n(n)
        test_inputs.append(inp)
        test_targets.append(tgt)

    weights = compute_anbn_grammar_weights(max_n, p=p, min_n=0)

    nll_per_string = compute_per_string_nll_bits(
        forward_fn, test_inputs, test_targets, batch_size=batch_size,
    )

    data_dh = float(np.sum(weights * nll_per_string))

    return {
        "data_dh_bits": data_dh,
        "nll_per_string": nll_per_string,
        "grammar_weights": weights,
        "max_n": max_n,
    }


# ---------------------------------------------------------------------------
# Train |D:H| (raw NLL sum over training corpus)
# ---------------------------------------------------------------------------

def compute_train_dh(
    forward_fn,
    train_inputs,
    train_targets,
    batch_size: int = 64,
) -> dict:
    """Compute train |D:H| as raw NLL sum over training strings.

    Abudy et al. (2025) convention: train |D:H| data term is the
    sum (not grammar-weighted average) of per-string total NLL.

    Args:
        forward_fn: callable (x_batch) -> logits.
        train_inputs: list of input sequences from the training set.
        train_targets: list of target sequences from the training set.
        batch_size: strings per batch.

    Returns:
        dict with train_dh_data_bits, nll_per_string, n_strings.
    """
    nll_per_string = compute_per_string_nll_bits(
        forward_fn, train_inputs, train_targets, batch_size=batch_size,
    )
    total_nll = float(np.sum(nll_per_string))

    return {
        "train_dh_data_bits": total_nll,
        "nll_per_string": nll_per_string,
        "n_strings": len(train_inputs),
    }


# ---------------------------------------------------------------------------
# Optimal |D:H| (golden network baseline)
# ---------------------------------------------------------------------------

def compute_optimal_dh_test(
    max_n: int,
    p: float = 0.3,
    batch_size: int = 64,
) -> dict:
    """Golden network's test |D:H| and |H|.

    Returns the analytical optimum against which Δ_test% is computed.

    Args:
        max_n: largest n in the test set.
        p: PCFG termination probability.
        batch_size: strings per batch.

    Returns:
        dict with data_dh_bits, h_bits, mdl_score.
    """
    from src.mdl.golden import (
        build_golden_network_params, golden_forward, golden_mdl_score,
    )

    params = build_golden_network_params(p=p)

    def golden_fwd(x):
        return golden_forward(params, x)

    data_result = compute_grammar_weighted_nll_bits(
        golden_fwd, max_n=max_n, p=p, batch_size=batch_size,
    )

    mdl_score = golden_mdl_score(p=p)

    return {
        "data_dh_bits": data_result["data_dh_bits"],
        "h_bits": mdl_score["total_bits"],
        "mdl_score": mdl_score,
    }


def compute_optimal_dh_train(
    train_inputs,
    train_targets,
    p: float = 0.3,
    batch_size: int = 64,
) -> dict:
    """Golden network's train |D:H| data term.

    Returns the baseline against which Δ_train% is computed.

    Args:
        train_inputs: list of input sequences from the training set.
        train_targets: list of target sequences from the training set.
        p: PCFG termination probability.
        batch_size: strings per batch.

    Returns:
        dict with train_dh_data_bits.
    """
    from src.mdl.golden import build_golden_network_params, golden_forward

    params = build_golden_network_params(p=p)

    def golden_fwd(x):
        return golden_forward(params, x)

    return compute_train_dh(
        golden_fwd, train_inputs, train_targets, batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Trained network composite |D:H|
# ---------------------------------------------------------------------------

def compute_trained_h_bits(params, grid_codelengths, hidden_size: int) -> dict:
    """Compute |H| for a trained discretised network.

    Discretises weights via argmax over the rational grid, then sums
    per-weight Lan-style codelengths.

    Args:
        params: params dict with "logits" key, shape (n_params, M).
        grid_codelengths: float array (M,) of per-grid-point codelengths.
        hidden_size: LSTM hidden size (for architecture encoding).

    Returns:
        dict with h_bits, arch_bits, weight_bits.
    """
    from src.mdl.coding import integer_code_length

    logits = params["logits"]
    idx = jnp.argmax(logits, axis=-1)
    cl = jnp.asarray(grid_codelengths)
    weight_bits = float(jnp.sum(cl[idx]))
    arch_bits = integer_code_length(hidden_size)

    return {
        "h_bits": arch_bits + int(weight_bits),
        "arch_bits": arch_bits,
        "weight_bits": int(weight_bits),
    }


def evaluate_trained_network_dh(
    apply_fn,
    params,
    grid_codelengths,
    hidden_size: int,
    test_max_n: int,
    p: float = 0.3,
    batch_size: int = 64,
) -> dict:
    """Composite evaluation of a trained network for paper comparison.

    Returns test |D:H| (data term), |H|, and the total, using the
    discretised (argmax) network for both.

    Args:
        apply_fn: model.apply function.
        params: params dict (basic mode: has "logits"; shared mode:
            caller should extract model-only params first).
        grid_codelengths: float array (M,) of per-grid-point codelengths.
        hidden_size: LSTM hidden size.
        test_max_n: largest n in the test set.
        p: PCFG termination probability.
        batch_size: strings per batch.

    Returns:
        dict with data_dh_bits, h_bits, arch_bits, weight_bits.
    """
    # |H|
    h_result = compute_trained_h_bits(params, grid_codelengths, hidden_size)

    # |D:H| data term via discrete forward pass
    def discrete_fwd(x):
        logits, _ = apply_fn(
            {"params": params}, x, tau=1.0, train=False,
        )
        return logits

    data_result = compute_grammar_weighted_nll_bits(
        discrete_fwd, max_n=test_max_n, p=p, batch_size=batch_size,
    )

    return {
        "data_dh_bits": data_result["data_dh_bits"],
        **h_result,
    }


# ---------------------------------------------------------------------------
# Δ%
# ---------------------------------------------------------------------------

def compute_delta_pct(score: float, optimal: float) -> float:
    """Δ% = (score - optimal) / optimal × 100.

    Abudy et al. (2025) convention: operates on the data term |D:H|,
    not the total |D:H| + |H|.
    """
    if optimal == 0:
        return float("inf") if score > 0 else 0.0
    return (score - optimal) / optimal * 100.0


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def format_abudy_comparison_table(
    our_test_data_dh: float,
    our_train_data_dh: float,
    our_h_bits: int,
    opt_test_data_dh: float,
    opt_train_data_dh: float,
    golden_h_bits: int,
) -> str:
    """Format results as a comparison table matching Abudy et al. (2025).

    Columns: |D:H|_train, |D:H|_test, Δ_train%, Δ_test%, |H|.
    """
    delta_train = compute_delta_pct(our_train_data_dh, opt_train_data_dh)
    delta_test = compute_delta_pct(our_test_data_dh, opt_test_data_dh)

    hdr = (
        f"{'Method':>25} {'|D:H| train':>12} {'|D:H| test':>12} "
        f"{'Δ_train%':>10} {'Δ_test%':>10} {'|H|':>8}"
    )
    sep = "-" * len(hdr)

    golden_row = (
        f"{'Golden (optimal)':>25} {opt_train_data_dh:>12.2f} "
        f"{opt_test_data_dh:>12.2f} {'---':>10} {'---':>10} "
        f"{golden_h_bits:>8d}"
    )
    ours_row = (
        f"{'Ours (diff. MDL)':>25} {our_train_data_dh:>12.2f} "
        f"{our_test_data_dh:>12.2f} {delta_train:>9.1f}% "
        f"{delta_test:>9.1f}% {our_h_bits:>8d}"
    )

    lines = [
        "=" * len(hdr),
        "PAPER-COMPARABLE RESULTS (cf. Abudy et al. 2025, Tables 1-2)",
        "=" * len(hdr),
        hdr,
        sep,
        golden_row,
        ours_row,
        "=" * len(hdr),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Golden network under different regularisers
# ---------------------------------------------------------------------------

def evaluate_golden_under_regularisers(
    max_n: int,
    p: float = 0.3,
    batch_size: int = 64,
) -> dict:
    """Evaluate the golden network under CE, L1, L2, and MDL.

    Reports raw norms so the comparison holds for any λ. The core
    argument from Lan et al. (2024, arXiv:2402.10013v2, Table 4):
    only MDL has the golden network at or near the optimum, because
    L1/L2 heavily penalise the saturated gates (LARGE=127).

    Note: our golden is an LSTM (Lan et al. 2024), which differs
    architecturally from the free-form RNN goldens in Abudy et al.
    (2025). L1/L2 norms are not directly comparable across architectures.

    Args:
        max_n: largest n in the test set.
        p: PCFG termination probability.
        batch_size: strings per batch.

    Returns:
        dict with ce_test_bits, l1_norm, l2_norm_squared, mdl_bits,
        n_params, mdl_score, note.
    """
    from src.mdl.golden import (
        build_golden_network_params, golden_forward, golden_mdl_score,
    )

    params = build_golden_network_params(p=p)

    # CE: grammar-weighted test NLL
    def golden_fwd(x):
        return golden_forward(params, x)

    data_result = compute_grammar_weighted_nll_bits(
        golden_fwd, max_n=max_n, p=p, batch_size=batch_size,
    )
    ce_bits = data_result["data_dh_bits"]

    # Parameter norms
    all_weights = jnp.concatenate([v.ravel() for v in params.values()])
    l1_norm = float(jnp.sum(jnp.abs(all_weights)))
    l2_norm_sq = float(jnp.sum(all_weights ** 2))

    # MDL
    mdl_score = golden_mdl_score(p=p)

    return {
        "ce_test_bits": ce_bits,
        "l1_norm": l1_norm,
        "l2_norm_squared": l2_norm_sq,
        "mdl_bits": mdl_score["total_bits"],
        "n_params": int(len(all_weights)),
        "mdl_score": mdl_score,
        "note": (
            "L1/L2 norms are for our LSTM golden (Lan et al. 2024), "
            "not the free-form RNN used by Abudy et al. (2025). "
            "Norms are not directly comparable across architectures."
        ),
    }


def format_golden_regulariser_table(result: dict) -> str:
    """Format golden-under-regularisers as a readable table.

    Shows that MDL keeps the golden network near-optimal while
    L1/L2 impose large penalties due to saturated gates.
    """
    ce = result["ce_test_bits"]
    l1 = result["l1_norm"]
    l2 = result["l2_norm_squared"]
    mdl = result["mdl_bits"]

    lines = [
        "=" * 65,
        "GOLDEN NETWORK UNDER DIFFERENT REGULARISERS",
        f"(LSTM golden, {result['n_params']} params, Lan et al. 2024)",
        "=" * 65,
        f"  CE (test |D:H|):     {ce:.4f} bits",
        f"  L1 norm:             {l1:.2f}",
        f"  L2 norm squared:     {l2:.2f}",
        f"  MDL |H|:             {mdl} bits",
        "",
        "  For any λ > 0, total objective = CE + λ × reg:",
        f"    CE only:           {ce:.4f}",
        f"    CE + λ·L1:         {ce:.4f} + λ·{l1:.2f}",
        f"    CE + λ·L2:         {ce:.4f} + λ·{l2:.2f}",
        f"    CE + λ·MDL:        {ce:.4f} + λ·{mdl}",
        "",
        f"  Note: {result['note']}",
        "=" * 65,
    ]
    return "\n".join(lines)
