"""Golden a^n b^n LSTM from Lan et al. (2024), Appendix B.

A manually-constructed LSTM with hidden_size=3 that perfectly recognizes
the a^n b^n language. The network uses saturated gates and a counting
mechanism in the cell state to track the difference #a - #b.

This module provides:
    - build_golden_network_params(): construct all LSTM weights as JAX arrays
    - golden_forward(): run the golden LSTM on batched token sequences
    - golden_mdl_score(): compute hypothesis codelength |H| under MDL coding
    - evaluate_golden_network(): test deterministic accuracy on a^n b^n strings

References:
    Lan et al. (2024), Appendix B
"""

import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np

from src.mdl.coding import integer_code_length, rational_codelength
from src.mdl.data import (
    SYMBOL_HASH,
    SYMBOL_A,
    SYMBOL_B,
    NUM_SYMBOLS,
    make_anbn_fixed_n,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LARGE = 2**7 - 1  # 127
HIDDEN_SIZE = 3
INPUT_SIZE = NUM_SYMBOLS   # 3: {#, a, b}
OUTPUT_SIZE = NUM_SYMBOLS  # 3: {#, a, b}


# ---------------------------------------------------------------------------
# Parameter construction
# ---------------------------------------------------------------------------

def build_golden_network_params(p: float = 0.3) -> dict:
    """Build all weight matrices and biases for the golden LSTM.

    The LSTM equations (matching the codebase convention in lstm.py) are:

        it = sigmoid(x_t @ W_ii + b_ii + h @ W_hi + b_hi)
        ft = sigmoid(x_t @ W_if + b_if + h @ W_hf + b_hf)
        gt = tanh  (x_t @ W_ig + b_ig + h @ W_hg + b_hg)
        ot = sigmoid(x_t @ W_io + b_io + h @ W_ho + b_ho)
        ct = ft * ct-1 + it * gt
        ht = ot * tanh(ct)

    where x_t has shape (B, I) and W_ii has shape (I, H), so x_t @ W_ii
    gives (B, H).  The paper specifies weights in math convention
    (W_paper @ x), so we transpose them for the code convention (x @ W_code).

    Returns:
        dict with keys:
            W_ii, W_if, W_ig, W_io: (I, H) input-to-hidden weights
            W_hi, W_hf, W_hg, W_ho: (H, H) hidden-to-hidden weights
            b_ii, b_if, b_ig, b_io: (H,) input biases
            b_hi, b_hf, b_hg, b_ho: (H,) hidden biases
            W_out: (H, O) output weights
            b_out: (O,) output bias
    """
    I, H, O = INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
    L = float(LARGE)

    # -----------------------------------------------------------------------
    # Cell input gate (g): gt = tanh(Wig @ xt)
    # -----------------------------------------------------------------------
    # Paper: Wig = LARGE * [[1, 0, 0],
    #                        [1, 0, 0],
    #                        [0, 1, -1]]   shape (H, I)
    # Code:  W_ig = Wig.T  shape (I, H)
    Wig_paper = L * jnp.array([[1.0, 0.0,  0.0],
                                [1.0, 0.0,  0.0],
                                [0.0, 1.0, -1.0]])
    W_ig = Wig_paper.T  # (I, H)

    # All other g-gate parameters are zero
    b_ig = jnp.zeros(H)
    W_hg = jnp.zeros((H, H))
    b_hg = jnp.zeros(H)

    # -----------------------------------------------------------------------
    # Input gate (i): it = sigmoid(b_ii) ≈ 1  (always open)
    # -----------------------------------------------------------------------
    W_ii = jnp.zeros((I, H))
    b_ii = L * jnp.ones(H)
    W_hi = jnp.zeros((H, H))
    b_hi = jnp.zeros(H)

    # -----------------------------------------------------------------------
    # Forget gate (f): ft = sigmoid(b_if) ≈ 1  (always remember)
    # -----------------------------------------------------------------------
    W_if = jnp.zeros((I, H))
    b_if = L * jnp.ones(H)
    W_hf = jnp.zeros((H, H))
    b_hf = jnp.zeros(H)

    # -----------------------------------------------------------------------
    # Output gate (o): ot = sigmoid(Wio @ xt + bio) = one-hot mask
    # -----------------------------------------------------------------------
    # Paper: Wio = LARGE * [[2, 0, 0],
    #                        [0, 2, 0],
    #                        [0, 0, 2]]   shape (H, I)
    # Code:  W_io = Wio.T  (symmetric here, so same)
    Wio_paper = L * jnp.array([[2.0, 0.0, 0.0],
                                [0.0, 2.0, 0.0],
                                [0.0, 0.0, 2.0]])
    W_io = Wio_paper.T  # (I, H)

    b_io = L * jnp.array([-1.0, -1.0, -1.0])
    W_ho = jnp.zeros((H, H))
    b_ho = jnp.zeros(H)

    # -----------------------------------------------------------------------
    # Output layer: logits = ht @ W_out + b_out
    # -----------------------------------------------------------------------
    epsilon = 1.0 / (2**14 - 1)

    # Target probability table (4 states x 3 symbols):
    #   row 0 = start (#):          [p,   1-p, 0  ]
    #   row 1 = a phase:            [0,   1-p, p  ]
    #   row 2 = b phase (not last): [0,   0,   1  ]
    #   row 3 = last b (balanced):  [1,   0,   0  ]
    targets = jnp.array([
        [p,   1 - p, 0.0],
        [0.0, 1 - p, p  ],
        [0.0, 0.0,   1.0],
        [1.0, 0.0,   0.0],
    ])

    Wlog = jnp.log(targets + epsilon)  # (4, 3)

    Wout_prime = Wlog[:3, :]   # (3, 3) -- rows 0..2
    bout = Wlog[3, :]          # (3,)   -- row 3

    # Wout'' = (Wout' - bout).T   shape (3, 3) = (O, H) in math notation
    Wout_dprime = (Wout_prime - bout[None, :]).T  # (O, H) -- but see below

    # Actually Wout_dprime is (O, H) = (3, 3) in the paper's convention
    # (it maps from hidden to output: logits = Wout @ ht).
    # In code we need W_out of shape (H, O) so that ht @ W_out gives (B, O).
    # Wout_dprime.T would be (H, O).
    # But wait: the paper says Wout'' = (Wout' - bout).T
    #   Wout' is (3, 3) -- 3 hidden states x 3 output symbols
    #   (Wout' - bout) is (3, 3) -- each row minus bout
    #   .T is (3, 3) = (output_dim, hidden_dim)
    # So logits = Wout'' @ ht  in math notation
    # In code: logits = ht @ Wout''.T = ht @ (Wout' - bout)
    # So W_out (code, shape H x O) = (Wout' - bout)  which is (3, 3)

    tanh_1 = float(jnp.tanh(1.0))
    # Paper: Wout = Wout'' / tanh(1)
    # In code convention: W_out = (Wout' - bout) / tanh(1)
    W_out = (Wout_prime - bout[None, :]) / tanh_1  # (H=3, O=3)

    b_out = bout  # (O=3,)

    return {
        "W_ii": W_ii, "W_if": W_if, "W_ig": W_ig, "W_io": W_io,
        "W_hi": W_hi, "W_hf": W_hf, "W_hg": W_hg, "W_ho": W_ho,
        "b_ii": b_ii, "b_if": b_if, "b_ig": b_ig, "b_io": b_io,
        "b_hi": b_hi, "b_hf": b_hf, "b_hg": b_hg, "b_ho": b_ho,
        "W_out": W_out, "b_out": b_out,
    }


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def golden_forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Run the golden LSTM on a batch of token sequences.

    Args:
        params: dict from build_golden_network_params()
        x: int32 array of shape (batch, seq_len) with token indices

    Returns:
        logits: float32 array of shape (batch, seq_len, 3)
    """
    B, T = x.shape
    H = HIDDEN_SIZE
    I = INPUT_SIZE

    # Unpack parameters
    W_ii, W_if, W_ig, W_io = params["W_ii"], params["W_if"], params["W_ig"], params["W_io"]
    W_hi, W_hf, W_hg, W_ho = params["W_hi"], params["W_hf"], params["W_hg"], params["W_ho"]
    b_ii, b_if, b_ig, b_io = params["b_ii"], params["b_if"], params["b_ig"], params["b_io"]
    b_hi, b_hf, b_hg, b_ho = params["b_hi"], params["b_hf"], params["b_hg"], params["b_ho"]
    W_out, b_out = params["W_out"], params["b_out"]

    # One-hot encode input: (B, T) -> (B, T, I)
    x_onehot = jax.nn.one_hot(x, I)

    # LSTM scan
    def lstm_step(carry, x_t):
        """Single LSTM time step.

        Args:
            carry: (h, c) each of shape (B, H)
            x_t: (B, I) one-hot input at this time step

        Returns:
            (h_new, c_new), h_new
        """
        h, c = carry
        i_t = jax.nn.sigmoid(x_t @ W_ii + b_ii + h @ W_hi + b_hi)
        f_t = jax.nn.sigmoid(x_t @ W_if + b_if + h @ W_hf + b_hf)
        g_t = jnp.tanh(x_t @ W_ig + b_ig + h @ W_hg + b_hg)
        o_t = jax.nn.sigmoid(x_t @ W_io + b_io + h @ W_ho + b_ho)
        c_new = f_t * c + i_t * g_t
        h_new = o_t * jnp.tanh(c_new)
        return (h_new, c_new), h_new

    h0 = jnp.zeros((B, H))
    c0 = jnp.zeros((B, H))

    # Transpose to (T, B, I) for jax.lax.scan
    x_seq = jnp.transpose(x_onehot, (1, 0, 2))
    _, h_seq = jax.lax.scan(lstm_step, (h0, c0), x_seq)
    # h_seq: (T, B, H) -> (B, T, H)
    h_seq = jnp.transpose(h_seq, (1, 0, 2))

    # Output layer: logits = h @ W_out + b_out
    logits = h_seq @ W_out + b_out  # (B, T, O)
    return logits


# ---------------------------------------------------------------------------
# MDL hypothesis codelength
# ---------------------------------------------------------------------------

def _collect_nonzero_rational_weights(p: float = 0.3) -> list[Fraction]:
    """Collect all non-zero weights from the golden network as exact Fractions.

    We reconstruct the exact rational values rather than converting from
    float, since the golden network weights are defined by exact formulas.

    Returns:
        List of Fraction values for every non-zero weight/bias entry.
    """
    L = LARGE  # 127, integer

    nonzero = []

    # --- LSTM weights ---
    # W_ig (code convention, shape I=3 x H=3):
    #   Wig_paper.T = LARGE * [[1, 1, 0], [0, 0, 1], [0, 0, -1]]
    # Non-zero entries: LARGE at positions (0,0), (0,1), (1,2), and -LARGE at (2,2)
    nonzero.extend([Fraction(L), Fraction(L), Fraction(L), Fraction(-L)])

    # b_ii: LARGE * [1, 1, 1] -- 3 entries
    nonzero.extend([Fraction(L)] * 3)

    # b_if: LARGE * [1, 1, 1] -- 3 entries
    nonzero.extend([Fraction(L)] * 3)

    # W_io (code convention, shape I=3 x H=3):
    #   LARGE * [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    # Non-zero entries: 2*LARGE on diagonal -- 3 entries
    nonzero.extend([Fraction(2 * L)] * 3)

    # b_io: LARGE * [-1, -1, -1] -- 3 entries
    nonzero.extend([Fraction(-L)] * 3)

    # --- Output layer ---
    # These are computed from the target probability table.
    # We compute them as exact Fractions.
    epsilon = Fraction(1, 2**14 - 1)
    p_frac = Fraction(p).limit_denominator(1000)  # p = 0.3 = 3/10 exactly

    # Target probability table rows
    targets = [
        [p_frac, 1 - p_frac, Fraction(0)],     # start (#)
        [Fraction(0), 1 - p_frac, p_frac],      # a phase
        [Fraction(0), Fraction(0), Fraction(1)], # b phase (not last)
        [Fraction(1), Fraction(0), Fraction(0)], # last b (balanced)
    ]

    # Wlog = ln(targets + epsilon)
    # But ln of a rational is irrational -- we need the actual float values
    # converted back to the closest rational for MDL coding.
    #
    # The MDL coding scheme encodes each weight as a rational number.
    # For the golden network, the output layer weights are defined by:
    #   Wlog[i,j] = ln(targets[i,j] + epsilon)
    #   W_out[h,o] = (Wlog[h,o] - Wlog[3,o]) / tanh(1)    for h in {0,1,2}
    #   b_out[o] = Wlog[3,o]
    #
    # These are transcendental numbers. We approximate them as rationals
    # using high-precision Fraction conversion from the float values.

    tanh_1 = math.tanh(1.0)
    eps_float = float(epsilon)

    # Compute Wlog as floats, then convert to Fractions
    Wlog = []
    for row in targets:
        Wlog_row = []
        for val in row:
            Wlog_row.append(math.log(float(val) + eps_float))
        Wlog.append(Wlog_row)

    bout_floats = Wlog[3]
    bout_fracs = [Fraction(v).limit_denominator(10**12) for v in bout_floats]

    # W_out (code convention H=3 x O=3):
    # W_out[h, o] = (Wlog[h, o] - Wlog[3, o]) / tanh(1)
    Wout_fracs = []
    for h in range(3):
        for o in range(3):
            val = (Wlog[h][o] - Wlog[3][o]) / tanh_1
            Wout_fracs.append(Fraction(val).limit_denominator(10**12))

    # Collect non-zero output weights
    for f in Wout_fracs:
        if f != 0:
            nonzero.append(f)

    # Collect non-zero output biases
    for f in bout_fracs:
        if f != 0:
            nonzero.append(f)

    return nonzero


def golden_mdl_score(p: float = 0.3) -> dict:
    """Compute the hypothesis codelength |H| of the golden network.

    Uses the coding scheme from coding.py:
        |H| = |E(hidden_size)| + sum_{w != 0} l(w)

    where l(w) = 1 + |E(numerator)| + |E(denominator)| for each non-zero
    rational weight w = +/- n/m, and |E(hidden_size)| encodes the
    architecture prefix (hidden dimension).

    Args:
        p: PCFG termination probability (default 0.3)

    Returns:
        dict with:
            total_bits: total hypothesis codelength in bits
            arch_bits: architecture encoding bits
            weight_bits: total weight encoding bits
            n_nonzero: number of non-zero weights
            weights: list of (Fraction, bits) for each non-zero weight
    """
    # Architecture prefix: encode hidden_size
    arch_bits = integer_code_length(HIDDEN_SIZE)

    # Collect all non-zero weights as Fractions
    nonzero_weights = _collect_nonzero_rational_weights(p)

    # Compute per-weight codelengths
    weight_details = []
    total_weight_bits = 0
    for w in nonzero_weights:
        bits = rational_codelength(w)
        weight_details.append((w, bits))
        total_weight_bits += bits

    total_bits = arch_bits + total_weight_bits

    return {
        "total_bits": total_bits,
        "arch_bits": arch_bits,
        "weight_bits": total_weight_bits,
        "n_nonzero": len(nonzero_weights),
        "weights": weight_details,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_golden_network(
    max_n: int = 1500,
    p: float = 0.3,
    batch_size: int = 128,
) -> dict:
    """Evaluate the golden LSTM on all a^n b^n strings for n=1..max_n.

    Deterministic accuracy (Lan et al.'s metric): fraction of correct
    predictions at positions where the next token is fully determined,
    i.e., from the first b input onward (including the final #).

    Strings are grouped into batches of similar length and padded to
    avoid per-string JAX recompilation.

    Args:
        max_n: maximum n to test (default 1500, matching Lan et al.)
        p: PCFG termination probability
        batch_size: number of strings per batch for padded evaluation

    Returns:
        dict with:
            mean_accuracy: average deterministic accuracy over all strings
            all_correct: bool, whether every string is perfectly predicted
            first_failure_n: n of first imperfect string (None if all correct)
            per_string_acc: array of per-string accuracies
    """
    params = build_golden_network_params(p=p)

    # Build all test strings
    all_inputs = []
    all_targets = []
    for n in range(1, max_n + 1):
        inp, tgt = make_anbn_fixed_n(n)
        all_inputs.append(inp)
        all_targets.append(tgt)

    accs = np.zeros(max_n)

    # Process in batches, grouping strings by length to minimize padding
    # Since strings are already sorted by n (and hence by length), we
    # just chunk them sequentially.
    for batch_start in range(0, max_n, batch_size):
        batch_end = min(batch_start + batch_size, max_n)
        batch_inputs = all_inputs[batch_start:batch_end]
        batch_targets = all_targets[batch_start:batch_end]
        B = len(batch_inputs)

        # Pad to max length in this batch
        max_len = max(len(s) for s in batch_inputs)
        x_pad = np.zeros((B, max_len), dtype=np.int32)
        y_pad = np.zeros((B, max_len), dtype=np.int32)
        inp_pad = np.zeros((B, max_len), dtype=np.int32)
        det_mask = np.zeros((B, max_len), dtype=np.float32)

        for i, (inp, tgt) in enumerate(zip(batch_inputs, batch_targets)):
            L = len(inp)
            x_pad[i, :L] = inp
            y_pad[i, :L] = tgt
            inp_pad[i, :L] = inp
            # Deterministic positions: where the INPUT is b
            for t in range(L):
                if inp[t] == SYMBOL_B:
                    det_mask[i, t] = 1.0

        x_jnp = jnp.array(x_pad)
        y_jnp = jnp.array(y_pad)
        det_mask_jnp = jnp.array(det_mask)

        # Forward pass on batch
        logits = golden_forward(params, x_jnp)  # (B, max_len, 3)
        preds = jnp.argmax(logits, axis=-1)      # (B, max_len)

        # Per-string deterministic accuracy
        correct = (preds == y_jnp).astype(jnp.float32)
        n_det = jnp.sum(det_mask_jnp, axis=-1)                 # (B,)
        n_correct = jnp.sum(correct * det_mask_jnp, axis=-1)   # (B,)
        batch_accs = jnp.where(n_det > 0, n_correct / n_det, 1.0)

        accs[batch_start:batch_end] = np.array(batch_accs)

    accs_arr = jnp.array(accs)
    # Use tolerance for all_correct check to handle float32 rounding
    # (e.g., 33/33 can round to 0.99999994 in float32)
    all_correct = bool(jnp.all(accs_arr > 1.0 - 1e-6))

    # Find first failure (using same tolerance)
    if not all_correct:
        failures = jnp.where(
            accs_arr < 1.0 - 1e-6, jnp.arange(len(accs_arr)), len(accs_arr)
        )
        first_fail = int(jnp.min(failures)) + 1  # +1 because n starts at 1
    else:
        first_fail = None

    return {
        "mean_accuracy": float(jnp.mean(accs_arr)),
        "all_correct": all_correct,
        "first_failure_n": first_fail,
        "per_string_acc": accs_arr,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building golden network...")
    params = build_golden_network_params()

    # Print parameter shapes
    for k, v in params.items():
        print(f"  {k}: shape={v.shape}, nonzero={int(jnp.sum(v != 0))}")

    # Test on a small string: a^3 b^3
    print("\nTest on a^3 b^3:")
    inp, tgt = make_anbn_fixed_n(3)
    print(f"  input:  {inp}")
    print(f"  target: {tgt}")

    x = jnp.array(inp, dtype=jnp.int32)[None, :]
    logits = golden_forward(params, x)
    probs = jax.nn.softmax(logits[0], axis=-1)
    preds = jnp.argmax(logits[0], axis=-1)

    symbol_names = {0: "#", 1: "a", 2: "b"}
    for t in range(len(inp)):
        tgt_sym = symbol_names[tgt[t]]
        pred_sym = symbol_names[int(preds[t])]
        p_vec = probs[t]
        match = "OK" if int(preds[t]) == tgt[t] else "WRONG"
        print(
            f"  t={t}: input={symbol_names[inp[t]]}, "
            f"target={tgt_sym}, pred={pred_sym}, "
            f"probs=[{p_vec[0]:.4f}, {p_vec[1]:.4f}, {p_vec[2]:.4f}] "
            f"{match}"
        )

    # MDL score
    print("\nMDL hypothesis codelength:")
    mdl = golden_mdl_score()
    print(f"  Architecture bits: {mdl['arch_bits']}")
    print(f"  Weight bits: {mdl['weight_bits']} ({mdl['n_nonzero']} non-zero weights)")
    print(f"  Total |H|: {mdl['total_bits']} bits")

    # Quick accuracy check on small n
    print("\nEvaluating on n=1..20...")
    result = evaluate_golden_network(max_n=20)
    print(f"  Mean accuracy: {result['mean_accuracy']:.4f}")
    print(f"  All correct: {result['all_correct']}")
    if not result["all_correct"]:
        print(f"  First failure at n={result['first_failure_n']}")
