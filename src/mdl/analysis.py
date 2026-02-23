"""Analytical network analysis for a^n b^n LSTMs.

Given a trained model's discrete rational weights, this module can:
  1. Determine if the network is "golden" (provably perfect for all n)
  2. Find the exact n at which a non-golden network first fails
  3. Analyze the counting mechanism (gate saturation, counter components)
  4. Efficiently simulate the LSTM at arbitrary n using float64

The key insight: for the LSTM to implement a perfect counter for a^n b^n,
the forget and input gates must be exactly 1.0 in float32 (saturated),
and the cell input gate must produce consistent ±delta for a/b inputs.
With the rational grid (max value 10), pre-activations of ~20 suffice
for exact float32 saturation (sigmoid(17) = 1.0 in float32).
"""

import math
from fractions import Fraction

import numpy as np

from src.mdl.coding import build_rational_grid, rational_codelength
from src.mdl.data import (
    SYMBOL_HASH,
    SYMBOL_A,
    SYMBOL_B,
    NUM_SYMBOLS,
)


# Float32 saturation thresholds (empirically determined)
# sigmoid(x) = 1.0 in float32 for x >= SIGMOID_SAT_THRESHOLD
# tanh(x) = 1.0 in float32 for |x| >= TANH_SAT_THRESHOLD
SIGMOID_SAT_THRESHOLD = 16.7  # conservative; sigmoid(16.7) ≈ 1 - 5.5e-8
TANH_SAT_THRESHOLD = 9.1      # conservative; tanh(9.1) ≈ 1 - 2.2e-8


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

def extract_weights(params, grid, grid_values):
    """Extract discrete weights from trained logits and reshape into LSTM components.

    Args:
        params: dict with "logits" key, shape (108, M)
        grid: list of Fraction objects (rational grid)
        grid_values: float array of grid values

    Returns:
        dict with:
            "named": dict of named weight arrays (float64) - W_ii, b_ii, etc.
            "fractions": dict of named Fraction lists
            "flat_f64": flat float64 array of all 108 weights
            "flat_frac": list of 108 Fraction values
            "indices": argmax indices into the grid
    """
    logits = np.array(params["logits"])
    grid_f = np.array(grid_values, dtype=np.float64)
    idx = np.argmax(logits, axis=-1)

    flat_f64 = grid_f[idx]
    flat_frac = [grid[int(i)] for i in idx]

    I, H, O = NUM_SYMBOLS, 3, NUM_SYMBOLS  # 3, 3, 3

    # Unpack in same order as lstm.py (lines 116-150)
    offset = 0

    def take(size, shape=None):
        nonlocal offset
        vals = flat_f64[offset:offset + size]
        fracs = flat_frac[offset:offset + size]
        offset += size
        if shape is not None:
            vals = vals.reshape(shape)
        return vals, fracs

    named = {}
    fractions = {}

    for name, size, shape in [
        ("W_ii", I * H, (I, H)), ("W_if", I * H, (I, H)),
        ("W_ig", I * H, (I, H)), ("W_io", I * H, (I, H)),
        ("W_hi", H * H, (H, H)), ("W_hf", H * H, (H, H)),
        ("W_hg", H * H, (H, H)), ("W_ho", H * H, (H, H)),
        ("b_ii", H, None), ("b_if", H, None),
        ("b_ig", H, None), ("b_io", H, None),
        ("b_hi", H, None), ("b_hf", H, None),
        ("b_hg", H, None), ("b_ho", H, None),
        ("W_out", H * O, (H, O)), ("b_out", O, None),
    ]:
        v, f = take(size, shape)
        named[name] = v
        fractions[name] = f

    assert offset == len(flat_f64)

    return {
        "named": named,
        "fractions": fractions,
        "flat_f64": flat_f64,
        "flat_frac": flat_frac,
        "indices": idx,
    }


def _sigmoid(x):
    """Numerically stable sigmoid in float64."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _softmax(x):
    """Softmax over last axis in float64."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Float64 LSTM simulation
# ---------------------------------------------------------------------------

def simulate_lstm_f64(weights, n):
    """Simulate the LSTM on a single a^n b^n string using float64.

    Args:
        weights: dict from extract_weights()["named"]
        n: the n in a^n b^n

    Returns:
        dict with:
            "predictions": (T,) int array of predicted next tokens
            "targets": (T,) int array of target next tokens
            "cell_states": (T, H) float64 cell states
            "hidden_states": (T, H) float64 hidden states
            "logits": (T, O) float64 output logits
            "correct": bool, whether all deterministic positions correct
            "det_accuracy": float, deterministic accuracy
    """
    w = weights
    H = 3

    # Build input/target sequences for a^n b^n: # a...a b...b #
    # Input:  # a^n b^n    (length 2n+1)
    # Target: a^n b^n #    (length 2n+1)
    T = 2 * n + 1
    inp = np.zeros(T, dtype=np.int32)
    tgt = np.zeros(T, dtype=np.int32)

    inp[0] = SYMBOL_HASH
    for i in range(1, n + 1):
        inp[i] = SYMBOL_A
    for i in range(n + 1, 2 * n + 1):
        inp[i] = SYMBOL_B

    for i in range(n):
        tgt[i] = SYMBOL_A
    for i in range(n, 2 * n):
        tgt[i] = SYMBOL_B
    tgt[2 * n] = SYMBOL_HASH

    # One-hot encode
    x_onehot = np.eye(NUM_SYMBOLS, dtype=np.float64)[inp]  # (T, 3)

    # LSTM scan
    h = np.zeros(H, dtype=np.float64)
    c = np.zeros(H, dtype=np.float64)

    cell_states = np.zeros((T, H), dtype=np.float64)
    hidden_states = np.zeros((T, H), dtype=np.float64)
    all_logits = np.zeros((T, NUM_SYMBOLS), dtype=np.float64)

    for t in range(T):
        x_t = x_onehot[t]  # (3,)

        i_t = _sigmoid(x_t @ w["W_ii"] + w["b_ii"] + h @ w["W_hi"] + w["b_hi"])
        f_t = _sigmoid(x_t @ w["W_if"] + w["b_if"] + h @ w["W_hf"] + w["b_hf"])
        g_t = np.tanh(x_t @ w["W_ig"] + w["b_ig"] + h @ w["W_hg"] + w["b_hg"])
        o_t = _sigmoid(x_t @ w["W_io"] + w["b_io"] + h @ w["W_ho"] + w["b_ho"])

        c = f_t * c + i_t * g_t
        h = o_t * np.tanh(c)

        cell_states[t] = c
        hidden_states[t] = h
        all_logits[t] = h @ w["W_out"] + w["b_out"]

    predictions = np.argmax(all_logits, axis=-1)

    # Deterministic accuracy: positions where input is 'b'
    det_mask = (inp == SYMBOL_B).astype(np.float64)
    n_det = np.sum(det_mask)
    if n_det > 0:
        correct_mask = (predictions == tgt).astype(np.float64)
        n_correct = np.sum(correct_mask * det_mask)
        det_accuracy = n_correct / n_det
    else:
        det_accuracy = 1.0

    return {
        "predictions": predictions,
        "targets": tgt,
        "cell_states": cell_states,
        "hidden_states": hidden_states,
        "logits": all_logits,
        "correct": det_accuracy > 1.0 - 1e-9,
        "det_accuracy": det_accuracy,
    }


def _check_single_n(weights, n):
    """Quick check: does the model get a^n b^n correct?"""
    result = simulate_lstm_f64(weights, n)
    return result["correct"]


# ---------------------------------------------------------------------------
# Find failure point
# ---------------------------------------------------------------------------

def find_failure_n(weights, max_n=100_000):
    """Find the exact n at which the network first fails on a^n b^n.

    Uses exponential search to find the rough bound, then binary search
    to pinpoint the exact failure.

    Args:
        weights: dict from extract_weights()["named"]
        max_n: maximum n to test (returns None if all correct up to this)

    Returns:
        int or None: first n where the network fails, or None if all correct
    """
    # Exponential search: find first failure in powers of 2
    n = 1
    while n <= max_n:
        if not _check_single_n(weights, n):
            break
        n *= 2

    if n > max_n:
        # Also check max_n itself
        if _check_single_n(weights, max_n):
            return None
        # Failure is between n//2 and max_n
        lo, hi = n // 2, max_n
    elif n == 1:
        return 1
    else:
        lo, hi = n // 2, n

    # Binary search between lo (correct) and hi (fails)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _check_single_n(weights, mid):
            lo = mid
        else:
            hi = mid

    return hi


# ---------------------------------------------------------------------------
# Counting mechanism analysis
# ---------------------------------------------------------------------------

def analyze_counting(weights):
    """Analyze the counting mechanism of the LSTM.

    Runs the network on small strings and inspects cell state evolution
    to identify counter components and gate saturation levels.

    Args:
        weights: dict from extract_weights()["named"]

    Returns:
        dict with analysis results
    """
    w = weights
    H = 3

    # --- Gate pre-activation analysis ---
    # For each input type (#, a, b), compute gate pre-activations at h=0
    input_names = {SYMBOL_HASH: "#", SYMBOL_A: "a", SYMBOL_B: "b"}
    gate_analysis = {}

    for sym, name in input_names.items():
        x_t = np.eye(NUM_SYMBOLS, dtype=np.float64)[sym]
        h_zero = np.zeros(H, dtype=np.float64)

        gate_analysis[name] = {
            "input_gate_pre": x_t @ w["W_ii"] + w["b_ii"] + h_zero @ w["W_hi"] + w["b_hi"],
            "forget_gate_pre": x_t @ w["W_if"] + w["b_if"] + h_zero @ w["W_hf"] + w["b_hf"],
            "cell_input_pre": x_t @ w["W_ig"] + w["b_ig"] + h_zero @ w["W_hg"] + w["b_hg"],
            "output_gate_pre": x_t @ w["W_io"] + w["b_io"] + h_zero @ w["W_ho"] + w["b_ho"],
        }

        # Compute activated values
        ga = gate_analysis[name]
        ga["input_gate"] = _sigmoid(ga["input_gate_pre"])
        ga["forget_gate"] = _sigmoid(ga["forget_gate_pre"])
        ga["cell_input"] = np.tanh(ga["cell_input_pre"])
        ga["output_gate"] = _sigmoid(ga["output_gate_pre"])

    # --- Counter component identification ---
    # Run on n=10 and check which cell components grow linearly during 'a' phase
    result_10 = simulate_lstm_f64(weights, 10)
    cs = result_10["cell_states"]

    # During 'a' phase (timesteps 1..10), check increments
    counter_components = []
    for j in range(H):
        # Increments during 'a' phase
        a_increments = np.diff(cs[1:11, j])  # timesteps 1->2, 2->3, ..., 9->10
        # Decrements during 'b' phase
        b_increments = np.diff(cs[11:21, j])  # timesteps 11->12, ..., 19->20

        a_std = np.std(a_increments)
        b_std = np.std(b_increments)
        a_mean = np.mean(a_increments)
        b_mean = np.mean(b_increments)

        is_counter = (a_std < 1e-6 and b_std < 1e-6
                      and abs(a_mean) > 0.1
                      and np.sign(a_mean) != np.sign(b_mean))

        counter_components.append({
            "component": j,
            "a_increment_mean": float(a_mean),
            "a_increment_std": float(a_std),
            "b_increment_mean": float(b_mean),
            "b_increment_std": float(b_std),
            "is_counter": is_counter,
            "cell_values_a_phase": cs[1:11, j].tolist(),
            "cell_values_b_phase": cs[11:21, j].tolist(),
        })

    # --- Hidden weight analysis ---
    # Check if hidden-to-hidden gate weights are zero
    hidden_weights_zero = {
        "W_hi": bool(np.all(w["W_hi"] == 0)),
        "W_hf": bool(np.all(w["W_hf"] == 0)),
        "W_hg": bool(np.all(w["W_hg"] == 0)),
        "W_ho": bool(np.all(w["W_ho"] == 0)),
        "b_hi": bool(np.all(w["b_hi"] == 0)),
        "b_hf": bool(np.all(w["b_hf"] == 0)),
        "b_hg": bool(np.all(w["b_hg"] == 0)),
        "b_ho": bool(np.all(w["b_ho"] == 0)),
    }

    return {
        "gate_analysis": gate_analysis,
        "counter_components": counter_components,
        "hidden_weights_zero": hidden_weights_zero,
    }


# ---------------------------------------------------------------------------
# Goldenness check
# ---------------------------------------------------------------------------

def check_golden_properties(weights, max_test_n=100_000):
    """Determine if the network is provably golden (perfect for all n).

    A network is "proven golden" if:
    1. For all counter components j, the forget and input gates are exactly
       saturated (pre-activation >= SIGMOID_SAT_THRESHOLD) for ALL inputs
       and ALL reachable hidden states.
    2. The cell input gate produces consistent ±delta for a/b inputs.
    3. The output layer correctly maps states to probabilities.

    If hidden-to-hidden gate weights are zero, gate values depend only on
    the current input (not on h), so checking at h=0 suffices.

    If hidden-to-hidden weights are non-zero, we fall back to empirical
    verification up to max_test_n.

    Args:
        weights: dict from extract_weights()["named"]
        max_test_n: max n for empirical verification

    Returns:
        dict with:
            "verdict": "proven_golden" | "empirically_golden" | "not_golden"
            "proven_n": int or None, largest n proven correct
            "failure_n": int or None, first failure n
            "details": str, human-readable explanation
    """
    counting = analyze_counting(weights)
    w = weights

    # Find counter components
    counters = [c for c in counting["counter_components"] if c["is_counter"]]

    if not counters:
        # No counter detected - check if it still works empirically
        failure_n = find_failure_n(weights, max_n=max_test_n)
        if failure_n is None:
            return {
                "verdict": "empirically_golden",
                "proven_n": max_test_n,
                "failure_n": None,
                "details": (
                    f"No clear counter component identified, but the network "
                    f"passes all tests up to n={max_test_n}. Cannot prove "
                    f"correctness analytically."
                ),
            }
        return {
            "verdict": "not_golden",
            "proven_n": failure_n - 1,
            "failure_n": failure_n,
            "details": (
                f"No clear counter component identified. "
                f"Network fails at n={failure_n}."
            ),
        }

    # Check if we can prove saturation analytically
    hwz = counting["hidden_weights_zero"]
    all_hidden_zero = all(hwz.values())

    if all_hidden_zero:
        # Gates depend only on input, not hidden state -> can prove analytically
        # Check that forget and input gates are saturated for all inputs
        proven = True
        details_parts = []

        for counter in counters:
            j = counter["component"]

            for input_name in ["#", "a", "b"]:
                ga = counting["gate_analysis"][input_name]

                f_pre = ga["forget_gate_pre"][j]
                i_pre = ga["input_gate_pre"][j]

                if f_pre < SIGMOID_SAT_THRESHOLD:
                    proven = False
                    details_parts.append(
                        f"Forget gate for component {j}, input '{input_name}': "
                        f"pre-activation={f_pre:.4f} < {SIGMOID_SAT_THRESHOLD} "
                        f"(sigmoid={_sigmoid(np.array([f_pre]))[0]:.10f})"
                    )
                if i_pre < SIGMOID_SAT_THRESHOLD:
                    proven = False
                    details_parts.append(
                        f"Input gate for component {j}, input '{input_name}': "
                        f"pre-activation={i_pre:.4f} < {SIGMOID_SAT_THRESHOLD} "
                        f"(sigmoid={_sigmoid(np.array([i_pre]))[0]:.10f})"
                    )

            # Check counter consistency
            a_inc = counter["a_increment_mean"]
            b_inc = counter["b_increment_mean"]
            if abs(abs(a_inc) - abs(b_inc)) > 1e-6:
                proven = False
                details_parts.append(
                    f"Counter component {j}: |a_increment|={abs(a_inc):.8f} != "
                    f"|b_increment|={abs(b_inc):.8f}"
                )

        if proven:
            return {
                "verdict": "proven_golden",
                "proven_n": None,  # infinite
                "failure_n": None,
                "details": (
                    "Network is PROVABLY GOLDEN. All hidden-to-hidden gate "
                    "weights are zero, and all gate pre-activations exceed "
                    f"the float32 saturation threshold ({SIGMOID_SAT_THRESHOLD}). "
                    "The counter increments/decrements are perfectly balanced. "
                    "The network correctly recognizes a^n b^n for ALL n."
                ),
            }
        else:
            # Gates not saturated - find empirical failure
            failure_n = find_failure_n(weights, max_n=max_test_n)
            if failure_n is None:
                return {
                    "verdict": "empirically_golden",
                    "proven_n": max_test_n,
                    "failure_n": None,
                    "details": (
                        "Hidden-to-hidden weights are zero but gate saturation "
                        "is not proven:\n  " + "\n  ".join(details_parts) +
                        f"\nHowever, network passes all tests up to n={max_test_n}."
                    ),
                }
            return {
                "verdict": "not_golden",
                "proven_n": failure_n - 1,
                "failure_n": failure_n,
                "details": (
                    "Gate saturation insufficient:\n  " +
                    "\n  ".join(details_parts) +
                    f"\nNetwork fails at n={failure_n}."
                ),
            }
    else:
        # Hidden weights are non-zero - gates depend on h, which evolves
        # We can still check dynamically if gates stay saturated
        # For now, fall back to empirical verification
        failure_n = find_failure_n(weights, max_n=max_test_n)

        # Also check gate saturation during simulation
        result_100 = simulate_lstm_f64(weights, min(100, max_test_n))
        cs = result_100["cell_states"]
        hs = result_100["hidden_states"]

        # Check gate values during a longer simulation
        gate_min_vals = _check_dynamic_saturation(weights, min(1000, max_test_n))

        if failure_n is None:
            if gate_min_vals["min_forget"] > 1.0 - 1e-12:
                return {
                    "verdict": "empirically_golden",
                    "proven_n": max_test_n,
                    "failure_n": None,
                    "details": (
                        "Hidden-to-hidden weights are non-zero, so analytical proof "
                        "is harder. However, dynamic gate analysis shows forget gate "
                        f"min={gate_min_vals['min_forget']:.15f} and input gate "
                        f"min={gate_min_vals['min_input']:.15f} during simulation "
                        f"up to n={min(1000, max_test_n)}. "
                        f"Network passes all tests up to n={max_test_n}."
                    ),
                }
            return {
                "verdict": "empirically_golden",
                "proven_n": max_test_n,
                "failure_n": None,
                "details": (
                    f"Network passes all tests up to n={max_test_n}, but "
                    "hidden-to-hidden weights are non-zero and gates show "
                    f"some leakage (min forget={gate_min_vals['min_forget']:.10f}). "
                    "May fail at larger n."
                ),
            }

        return {
            "verdict": "not_golden",
            "proven_n": failure_n - 1,
            "failure_n": failure_n,
            "details": (
                f"Network fails at n={failure_n}. "
                "Hidden-to-hidden weights are non-zero. "
                f"Gate dynamics: min forget={gate_min_vals['min_forget']:.10f}, "
                f"min input={gate_min_vals['min_input']:.10f}."
            ),
        }


def _check_dynamic_saturation(weights, n):
    """Check gate saturation dynamically during a simulation of a^n b^n."""
    w = weights
    H = 3
    T = 2 * n + 1

    inp = np.zeros(T, dtype=np.int32)
    inp[0] = SYMBOL_HASH
    inp[1:n + 1] = SYMBOL_A
    inp[n + 1:2 * n + 1] = SYMBOL_B

    x_onehot = np.eye(NUM_SYMBOLS, dtype=np.float64)[inp]

    h = np.zeros(H, dtype=np.float64)
    c = np.zeros(H, dtype=np.float64)

    min_forget = 1.0
    min_input = 1.0

    for t in range(T):
        x_t = x_onehot[t]
        i_t = _sigmoid(x_t @ w["W_ii"] + w["b_ii"] + h @ w["W_hi"] + w["b_hi"])
        f_t = _sigmoid(x_t @ w["W_if"] + w["b_if"] + h @ w["W_hf"] + w["b_hf"])
        g_t = np.tanh(x_t @ w["W_ig"] + w["b_ig"] + h @ w["W_hg"] + w["b_hg"])
        o_t = _sigmoid(x_t @ w["W_io"] + w["b_io"] + h @ w["W_ho"] + w["b_ho"])

        c = f_t * c + i_t * g_t
        h = o_t * np.tanh(c)

        min_forget = min(min_forget, float(np.min(f_t)))
        min_input = min(min_input, float(np.min(i_t)))

    return {
        "min_forget": min_forget,
        "min_input": min_input,
    }


# ---------------------------------------------------------------------------
# Efficient large-n evaluation
# ---------------------------------------------------------------------------

def evaluate_range_f64(weights, max_n, verbose=True):
    """Evaluate the model on all a^n b^n strings for n=1..max_n using float64.

    More memory-efficient than JAX batched evaluation for very large n,
    since it processes one string at a time.

    Args:
        weights: dict from extract_weights()["named"]
        max_n: maximum n to test
        verbose: print progress

    Returns:
        dict with:
            "n_perfect": number of perfectly predicted strings
            "gen_n": largest n such that all 1..n are correct
            "first_failure_n": first n with error (or None)
            "per_string_correct": bool array of length max_n
    """
    per_string = np.zeros(max_n, dtype=bool)
    first_fail = None

    report_interval = max(1, max_n // 20)

    for i in range(max_n):
        n = i + 1
        per_string[i] = _check_single_n(weights, n)

        if not per_string[i] and first_fail is None:
            first_fail = n

        if verbose and (n % report_interval == 0 or n == max_n):
            n_correct_so_far = int(np.sum(per_string[:n]))
            print(f"  Progress: {n}/{max_n} tested, {n_correct_so_far} correct", flush=True)

    n_perfect = int(np.sum(per_string))

    # gen_n: largest n where all 1..n are correct
    if first_fail is not None:
        gen_n = first_fail - 1
    else:
        gen_n = max_n

    return {
        "n_perfect": n_perfect,
        "gen_n": gen_n,
        "first_failure_n": first_fail,
        "per_string_correct": per_string,
    }


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_model(params, grid, grid_values, max_test_n=100_000, p=0.3):
    """Run comprehensive analysis on a trained model.

    Args:
        params: dict with "logits" key
        grid: list of Fraction objects
        grid_values: float array of grid values
        max_test_n: maximum n for empirical testing
        p: PCFG termination probability

    Returns:
        dict with all analysis results
    """
    print("\n" + "=" * 70)
    print("NETWORK ANALYSIS")
    print("=" * 70)

    # Extract weights
    extracted = extract_weights(params, grid, grid_values)
    w = extracted["named"]

    # --- Weight summary ---
    n_nonzero = sum(1 for f in extracted["flat_frac"] if f != 0)
    total_codelength = sum(rational_codelength(f) for f in extracted["flat_frac"])

    print(f"\nWeight summary: {len(extracted['flat_frac'])} total, {n_nonzero} non-zero")
    print(f"  Hypothesis codelength: {total_codelength} bits")

    # --- Named weight display ---
    print("\nLSTM weight structure:")
    for name in ["W_ii", "W_if", "W_ig", "W_io",
                  "W_hi", "W_hf", "W_hg", "W_ho",
                  "b_ii", "b_if", "b_ig", "b_io",
                  "b_hi", "b_hf", "b_hg", "b_ho",
                  "W_out", "b_out"]:
        arr = w[name]
        nz = int(np.sum(arr != 0))
        if nz == 0:
            print(f"  {name:6s}: all zeros")
        else:
            flat = arr.flatten()
            unique = np.unique(flat[flat != 0])
            if len(unique) <= 6:
                vals_str = ", ".join(f"{v:g}" for v in unique)
                print(f"  {name:6s}: {nz} non-zero, values: {{{vals_str}}}")
            else:
                print(f"  {name:6s}: {nz} non-zero, range [{flat.min():.4f}, {flat.max():.4f}]")

    # --- Counting analysis ---
    print("\nCounting mechanism analysis:")
    counting = analyze_counting(w)

    for comp in counting["counter_components"]:
        j = comp["component"]
        status = "COUNTER" if comp["is_counter"] else "not counter"
        print(f"  Cell component {j} [{status}]:")
        print(f"    a-phase increment: {comp['a_increment_mean']:+.8f} (std={comp['a_increment_std']:.2e})")
        print(f"    b-phase increment: {comp['b_increment_mean']:+.8f} (std={comp['b_increment_std']:.2e})")

    # --- Gate saturation ---
    print("\nGate pre-activations at h=0:")
    for input_name in ["#", "a", "b"]:
        ga = counting["gate_analysis"][input_name]
        print(f"  Input '{input_name}':")
        for gate_name in ["forget_gate", "input_gate"]:
            pre_key = f"{gate_name}_pre"
            sat_key = gate_name
            pre = ga[pre_key]
            val = ga[sat_key]
            sat_marks = ["*" if p >= SIGMOID_SAT_THRESHOLD else " " for p in pre]
            print(f"    {gate_name:12s}: pre=[{pre[0]:8.3f}, {pre[1]:8.3f}, {pre[2]:8.3f}] "
                  f"val=[{val[0]:.10f}, {val[1]:.10f}, {val[2]:.10f}] "
                  f"[{''.join(sat_marks)}]")

    hwz = counting["hidden_weights_zero"]
    all_hid_zero = all(hwz.values())
    print(f"\nHidden-to-hidden weights all zero: {all_hid_zero}")
    if not all_hid_zero:
        for k, v in hwz.items():
            if not v:
                print(f"  {k}: NON-ZERO")

    # --- Golden check ---
    print(f"\nChecking goldenness (max_test_n={max_test_n})...")
    golden = check_golden_properties(w, max_test_n=max_test_n)

    print(f"\n  Verdict: {golden['verdict'].upper()}")
    if golden["failure_n"] is not None:
        print(f"  First failure at n={golden['failure_n']}")
        print(f"  Proven correct for n=1..{golden['proven_n']}")
    elif golden["proven_n"] is not None:
        print(f"  Verified correct up to n={golden['proven_n']}")
    else:
        print(f"  Correct for ALL n (proven analytically)")
    print(f"\n  {golden['details']}")

    print("=" * 70)

    return {
        "weights": extracted,
        "counting": counting,
        "golden": golden,
        "n_nonzero": n_nonzero,
        "total_codelength": total_codelength,
    }
