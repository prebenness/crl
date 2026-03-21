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

from src.mdl.data import (
    make_test_set, make_anbn_fixed_n,
    SYMBOL_HASH, SYMBOL_A, SYMBOL_B, NUM_SYMBOLS,
)


# ---------------------------------------------------------------------------
# Grammar weights
# ---------------------------------------------------------------------------

def compute_anbn_grammar_weights(
    max_n: int, p: float = 0.3, min_n: int = 1,
) -> np.ndarray:
    """Raw PCFG weights P(n) = p * (1-p)^n for n=min_n..max_n.

    Returns unnormalized probabilities from the geometric PCFG.
    Default min_n=1 matches Abudy et al. (2025) test convention
    (test sets start at n=1, excluding the empty string).

    Args:
        max_n: largest n in the test set.
        p: PCFG termination probability.
        min_n: smallest n (default 1, Abudy et al. convention).

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
    verbose: bool = False,
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
        verbose: if True, print batch progress.

    Returns:
        (N,) float64 array of per-string total NLL in bits.
    """
    import time as _time

    N = len(inputs)
    nll_per_string = np.zeros(N, dtype=np.float64)
    n_batches = (N + batch_size - 1) // batch_size
    t0 = _time.monotonic() if verbose else None

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

        if verbose:
            batch_idx = batch_start // batch_size + 1
            report_interval = max(1, n_batches // 5)
            if batch_idx % report_interval == 0 or batch_idx == n_batches:
                elapsed = _time.monotonic() - t0
                eta = elapsed / batch_idx * (n_batches - batch_idx)
                print(
                    f"    batch {batch_idx}/{n_batches} "
                    f"(max_len={max_len}, "
                    f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s)",
                    flush=True,
                )

    return nll_per_string


# ---------------------------------------------------------------------------
# Grammar-weighted test NLL  (|D:H| data term, test)
# ---------------------------------------------------------------------------

def compute_grammar_weighted_nll_bits(
    forward_fn,
    max_n: int,
    p: float = 0.3,
    batch_size: int = 64,
    verbose: bool = False,
) -> dict:
    """Grammar-weighted |D:H| data term on the exhaustive test set.

    |D:H|_test = Σ_{n=1}^{max_n} P(n) × NLL_total(n)

    where NLL_total(n) is the total NLL in bits for string aⁿbⁿ,
    P(n) = p*(1-p)^n is the raw PCFG probability (unnormalized).

    Test set starts at n=1 following Abudy et al. (2025,
    arXiv:2505.13398v2, Appendix A.1).

    Args:
        forward_fn: callable (x_batch) -> logits.
        max_n: largest n in the test set.
        p: PCFG termination probability.
        batch_size: strings per batch.
        verbose: if True, print batch progress.

    Returns:
        dict with data_dh_bits, nll_per_string, grammar_weights, max_n.
    """
    if verbose:
        print(f"  Computing grammar-weighted |D:H|_test (n=1..{max_n})...")
    test_inputs = []
    test_targets = []
    for n in range(1, max_n + 1):
        inp, tgt = make_anbn_fixed_n(n)
        test_inputs.append(inp)
        test_targets.append(tgt)

    weights = compute_anbn_grammar_weights(max_n, p=p)

    nll_per_string = compute_per_string_nll_bits(
        forward_fn, test_inputs, test_targets, batch_size=batch_size,
        verbose=verbose,
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
    verbose: bool = False,
) -> dict:
    """Compute train |D:H| as raw NLL sum over training strings.

    Abudy et al. (2025) convention: train |D:H| data term is the
    sum (not grammar-weighted average) of per-string total NLL.

    Args:
        forward_fn: callable (x_batch) -> logits.
        train_inputs: list of input sequences from the training set.
        train_targets: list of target sequences from the training set.
        batch_size: strings per batch.
        verbose: if True, print batch progress.

    Returns:
        dict with train_dh_data_bits, nll_per_string, n_strings.
    """
    nll_per_string = compute_per_string_nll_bits(
        forward_fn, train_inputs, train_targets, batch_size=batch_size,
        verbose=verbose,
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
    verbose: bool = False,
) -> dict:
    """Golden network's test |D:H| and |H|.

    Returns the analytical optimum against which Δ_test% is computed.

    Args:
        max_n: largest n in the test set.
        p: PCFG termination probability.
        batch_size: strings per batch.
        verbose: if True, print batch progress.

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
        verbose=verbose,
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
    verbose: bool = False,
) -> dict:
    """Golden network's train |D:H| data term.

    Returns the baseline against which Δ_train% is computed.

    Args:
        train_inputs: list of input sequences from the training set.
        train_targets: list of target sequences from the training set.
        p: PCFG termination probability.
        batch_size: strings per batch.
        verbose: if True, print batch progress.

    Returns:
        dict with train_dh_data_bits.
    """
    from src.mdl.golden import build_golden_network_params, golden_forward

    params = build_golden_network_params(p=p)

    def golden_fwd(x):
        return golden_forward(params, x)

    return compute_train_dh(
        golden_fwd, train_inputs, train_targets, batch_size=batch_size,
        verbose=verbose,
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


# ---------------------------------------------------------------------------
# Recognition accuracy (FLaRe / Deletang comparison)
# ---------------------------------------------------------------------------

def compute_full_string_accuracy(
    forward_fn,
    inputs,
    targets,
    batch_size: int = 64,
) -> np.ndarray:
    """Per-string boolean: True iff argmax correct at every position.

    Unlike evaluate_deterministic_accuracy (which checks only b-phase
    positions), this checks ALL positions — equivalent to the recognition
    accuracy notion used in FLaRe (ICLR 2025) and Deletang et al.
    (2023, "Neural Networks and the Chomsky Hierarchy", ICLR 2023).

    Args:
        forward_fn: callable (x_batch: int32 (B,T)) -> logits (B,T,V).
        inputs: list of variable-length int lists.
        targets: list of variable-length int lists.
        batch_size: strings per batch.

    Returns:
        (N,) bool array. True = all positions correct ("accepted").
    """
    N = len(inputs)
    accepted = np.zeros(N, dtype=bool)

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

        logits = forward_fn(jnp.array(x_pad))  # (B, T, V)
        preds = jnp.argmax(logits, axis=-1)  # (B, T)
        correct = (preds == jnp.array(y_pad)).astype(jnp.float32)
        # A position is "ok" if correct OR padding
        ok = correct * jnp.array(mask) + (1.0 - jnp.array(mask))
        per_string = jnp.all(ok > 0.5, axis=-1)  # (B,)
        accepted[batch_start:batch_end] = np.array(per_string)

    return accepted


def generate_negative_anbn(
    num_examples: int,
    max_n: int,
    p: float = 0.3,
    seed: int = 0,
) -> tuple[list[list[int]], list[list[int]]]:
    """Generate invalid strings over {#, a, b} for recognition testing.

    Produces strings that are NOT valid a^n b^n, formatted as
    language-modeling pairs (input=s[:-1], target=s[1:]).

    Four negative types in equal proportion:
      1. Wrong counts: a^m b^n with m != n
      2. Interleaved: a's and b's mixed (e.g., abab)
      3. Single symbol: only a's or only b's (e.g., #aaa#)
      4. Random: random tokens from {a, b}

    Args:
        num_examples: total negative strings to generate.
        max_n: controls length range (lengths drawn from geometric(p)).
        p: geometric parameter for length distribution.
        seed: random seed.

    Returns:
        (negative_inputs, negative_targets): lists of int lists.
    """
    rng = np.random.RandomState(seed)
    neg_inputs = []
    neg_targets = []

    per_type = num_examples // 4
    remainder = num_examples - 4 * per_type

    def _sample_length():
        # Geometric distribution on {1, 2, ...} to avoid empty strings
        return int(rng.geometric(p)) if rng.random() > 0.5 else rng.randint(1, max(max_n, 2))

    def _add(string):
        """Add a #-delimited string as input/target pair."""
        full = [SYMBOL_HASH] + string + [SYMBOL_HASH]
        neg_inputs.append(full[:-1])
        neg_targets.append(full[1:])

    # Type 1: wrong counts (a^m b^n, m != n)
    for _ in range(per_type + remainder):
        m = _sample_length()
        n = _sample_length()
        if m == n:
            n = m + 1  # ensure m != n
        _add([SYMBOL_A] * m + [SYMBOL_B] * n)

    # Type 2: interleaved a's and b's
    for _ in range(per_type):
        length = _sample_length() * 2
        s = []
        for j in range(length):
            s.append(SYMBOL_A if rng.random() < 0.5 else SYMBOL_B)
        # Ensure it's not accidentally a valid a^nb^n
        if len(s) >= 2 and all(c == SYMBOL_A for c in s[:len(s)//2]) \
                and all(c == SYMBOL_B for c in s[len(s)//2:]) \
                and s.count(SYMBOL_A) == s.count(SYMBOL_B):
            s[0] = SYMBOL_B  # break validity
        _add(s)

    # Type 3: single symbol (only a's or only b's)
    for _ in range(per_type):
        length = _sample_length()
        sym = SYMBOL_A if rng.random() < 0.5 else SYMBOL_B
        _add([sym] * length)

    # Type 4: random tokens from {a, b}
    for _ in range(per_type):
        length = _sample_length()
        s = [rng.choice([SYMBOL_A, SYMBOL_B]) for _ in range(length)]
        # Break if accidentally valid
        na = s.count(SYMBOL_A)
        nb = s.count(SYMBOL_B)
        if na == nb and na > 0:
            is_valid = all(c == SYMBOL_A for c in s[:na]) and \
                       all(c == SYMBOL_B for c in s[na:])
            if is_valid:
                s[0] = SYMBOL_B
        _add(s)

    return neg_inputs, neg_targets


def compute_recognition_accuracy(
    forward_fn,
    positive_inputs,
    positive_targets,
    negative_inputs,
    negative_targets,
    batch_size: int = 64,
) -> dict:
    """Binary accept/reject recognition accuracy.

    A string is "accepted" iff the model's argmax prediction is correct
    at every position; otherwise "rejected". For comparison with
    FLaRe (ICLR 2025) and Deletang et al. (2023, ICLR 2023).

    Args:
        forward_fn: callable (x_batch) -> logits.
        positive_inputs, positive_targets: valid strings.
        negative_inputs, negative_targets: invalid strings.
        batch_size: strings per batch.

    Returns:
        dict with accuracy, tp_rate, tn_rate, n_positive, n_negative.
    """
    pos_accepted = compute_full_string_accuracy(
        forward_fn, positive_inputs, positive_targets, batch_size,
    )
    neg_accepted = compute_full_string_accuracy(
        forward_fn, negative_inputs, negative_targets, batch_size,
    )

    tp = int(np.sum(pos_accepted))
    tn = int(np.sum(~neg_accepted))
    n_pos = len(positive_inputs)
    n_neg = len(negative_inputs)

    return {
        "accuracy": (tp + tn) / max(n_pos + n_neg, 1),
        "tp_rate": tp / max(n_pos, 1),
        "tn_rate": tn / max(n_neg, 1),
        "n_positive": n_pos,
        "n_negative": n_neg,
    }
