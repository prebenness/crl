"""Tests for paper-comparable evaluation metrics (src/mdl/evaluation.py).

Covers:
    - Grammar weights: geometric distribution with n≥1 (Abudy convention)
    - Per-string NLL: correctness on known logits
    - Grammar-weighted test NLL: golden network sanity check
    - Optimal |D:H|: golden test baseline matches expected value
    - Δ%: arithmetic correctness
    - Golden under regularisers: norms and MDL match expectations
"""

import pytest
import math

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Grammar weights
# ---------------------------------------------------------------------------

class TestGrammarWeights:
    """Verify grammar weight computation for aⁿbⁿ."""

    def test_sums_to_one_minus_p(self):
        """Raw PCFG weights for n=1..max_n sum to ~(1-p) for large max_n."""
        from src.mdl.evaluation import compute_anbn_grammar_weights
        w = compute_anbn_grammar_weights(1000, p=0.3)
        assert abs(w.sum() - 0.7) < 1e-10

    def test_geometric_ratios(self):
        """Adjacent weights should have ratio (1-p)."""
        from src.mdl.evaluation import compute_anbn_grammar_weights
        p = 0.3
        w = compute_anbn_grammar_weights(100, p=p)
        for i in range(10):
            ratio = w[i + 1] / w[i]
            assert abs(ratio - (1 - p)) < 1e-12

    def test_n1_is_largest(self):
        """With default min_n=1, first weight (n=1) is largest."""
        from src.mdl.evaluation import compute_anbn_grammar_weights
        w = compute_anbn_grammar_weights(100, p=0.3)
        assert w[0] > w[1] > w[2]

    def test_shape_default(self):
        from src.mdl.evaluation import compute_anbn_grammar_weights
        w = compute_anbn_grammar_weights(50, p=0.3)
        assert w.shape == (50,)  # n=1..50

    def test_n1_weight_equals_p_times_one_minus_p(self):
        """P(1) = p * (1-p)^1 = p*(1-p)."""
        from src.mdl.evaluation import compute_anbn_grammar_weights
        w = compute_anbn_grammar_weights(10, p=0.3)
        assert abs(w[0] - 0.3 * 0.7) < 1e-12


# ---------------------------------------------------------------------------
# Per-string NLL
# ---------------------------------------------------------------------------

class TestPerStringNLL:
    """Verify per-string NLL computation on known logits."""

    def test_perfect_prediction_near_zero(self):
        """If model predicts the correct token with prob ~1, NLL ≈ 0."""
        from src.mdl.evaluation import compute_per_string_nll_bits

        def perfect_forward(x):
            B, T = x.shape
            # Return logits that put all mass on target
            # For a simple test: target is always token 1
            logits = jnp.full((B, T, 3), -100.0)
            logits = logits.at[:, :, 1].set(100.0)
            return logits

        inputs = [[0, 1, 1]]   # any tokens
        targets = [[1, 1, 1]]  # all token 1
        nll = compute_per_string_nll_bits(perfect_forward, inputs, targets)
        # Should be very close to zero (not exactly due to smoothing)
        assert nll[0] < 0.01

    def test_uniform_prediction(self):
        """Uniform prediction over 3 tokens: NLL = log2(3) per position."""
        from src.mdl.evaluation import compute_per_string_nll_bits

        def uniform_forward(x):
            B, T = x.shape
            return jnp.zeros((B, T, 3))  # uniform logits

        inputs = [[0, 1, 2]]
        targets = [[1, 2, 0]]
        nll = compute_per_string_nll_bits(
            uniform_forward, inputs, targets, smoothing=0.0,
        )
        # 3 positions × log2(3) ≈ 3 × 1.585 = 4.755
        expected = 3 * math.log2(3)
        assert abs(nll[0] - expected) < 0.01

    def test_sums_not_averages(self):
        """Verify NLL is summed over positions, not averaged."""
        from src.mdl.evaluation import compute_per_string_nll_bits

        def uniform_forward(x):
            B, T = x.shape
            return jnp.zeros((B, T, 3))

        # String of length 1
        nll_1 = compute_per_string_nll_bits(
            uniform_forward, [[0]], [[1]], smoothing=0.0,
        )
        # String of length 3
        nll_3 = compute_per_string_nll_bits(
            uniform_forward, [[0, 1, 2]], [[1, 2, 0]], smoothing=0.0,
        )
        # 3x length should give ~3x NLL
        assert abs(nll_3[0] / nll_1[0] - 3.0) < 0.01

    def test_multiple_strings(self):
        """Batched computation returns correct per-string values."""
        from src.mdl.evaluation import compute_per_string_nll_bits

        def uniform_forward(x):
            B, T = x.shape
            return jnp.zeros((B, T, 3))

        inputs = [[0], [0, 1], [0, 1, 2]]
        targets = [[1], [1, 2], [1, 2, 0]]
        nll = compute_per_string_nll_bits(
            uniform_forward, inputs, targets, smoothing=0.0,
        )
        per_pos = math.log2(3)
        assert abs(nll[0] - 1 * per_pos) < 0.01
        assert abs(nll[1] - 2 * per_pos) < 0.01
        assert abs(nll[2] - 3 * per_pos) < 0.01


# ---------------------------------------------------------------------------
# Grammar-weighted test NLL (golden network sanity)
# ---------------------------------------------------------------------------

class TestGrammarWeightedNLL:
    """Verify grammar-weighted NLL on the golden network (n=1 convention)."""

    def test_golden_test_dh(self):
        """Golden network test |D:H| with n≥1 (Abudy convention).

        Reference: Abudy et al. (2025, arXiv:2505.13398v2), line 803.
        For ideal predictor with p=0.3, n≥1:
        E[NLL_total(n) | n≥1] = E[n|n≥1] × (-log2(1-p)) + (-log2(p))
        where E[n|n≥1] = 1/p = 10/3 for geometric(0.3) conditioned on n≥1.
        So E[NLL|n≥1] ≈ (10/3)*0.5146 + 1.737 ≈ 3.452.
        But this is the conditional expectation; the grammar-weighted sum
        uses unnormalized weights summing to (1-p)=0.7, giving ~2.42.
        """
        from src.mdl.evaluation import compute_grammar_weighted_nll_bits
        from src.mdl.golden import build_golden_network_params, golden_forward

        params = build_golden_network_params(p=0.3)

        def golden_fwd(x):
            return golden_forward(params, x)

        result = compute_grammar_weighted_nll_bits(
            golden_fwd, max_n=200, p=0.3, batch_size=64,
        )

        # Analytical: Σ_{n=1}^{∞} p(1-p)^n × [n×(-log2(1-p)) + (-log2(p))]
        # = (1-p) × (-log2(p)) + p(1-p)/(1-(1-p))^2 × (-log2(1-p))  ... but
        # simplest: just check against the analytical value computed below.
        p = 0.3
        max_n_check = 5000
        ns = np.arange(1, max_n_check + 1)
        weights = p * (1 - p) ** ns
        nll_per_n = ns * (-math.log2(1 - p)) + (-math.log2(p))
        analytical = float(np.sum(weights * nll_per_n))

        assert abs(result["data_dh_bits"] - analytical) < 0.1, (
            f"Golden test |D:H| = {result['data_dh_bits']:.4f}, "
            f"expected ≈ {analytical:.4f}"
        )

    def test_analytical_expected_value(self):
        """Cross-check analytical formula for ideal predictor with n≥1.

        E[NLL_total(n)] = n × (-log2(1-p)) + (-log2(p))
        Σ_{n=1}^{∞} P(n) × E[NLL_total(n)] where P(n) = p(1-p)^n.
        """
        from src.mdl.evaluation import compute_anbn_grammar_weights

        p = 0.3
        max_n = 5000
        weights = compute_anbn_grammar_weights(max_n, p=p)

        ns = np.arange(1, max_n + 1)
        nll_per_n = ns * (-math.log2(1 - p)) + (-math.log2(p))
        expected_nll = np.sum(weights * nll_per_n)

        # Analytical closed-form:
        # Σ P(n)*n = p(1-p)/(1-(1-p))^2 = (1-p)/p
        # = E[n] for geometric on {0,1,...} = (1-p)/p, but here
        # Σ_{n=1} p(1-p)^n × n = (1-p)/p × p = (1-p) ... wait,
        # Σ_{n=1} n*p*(1-p)^n = p*(1-p)/(1-(1-p))^2 = (1-p)/p
        # So weighted NLL = (1-p)/p * (-log2(1-p)) + (1-p)*(-log2(p))
        closed = (1-p)/p * (-math.log2(1-p)) + (1-p) * (-math.log2(p))
        assert abs(expected_nll - closed) < 0.001
        assert abs(closed - 2.4165) < 0.01


# ---------------------------------------------------------------------------
# Optimal |D:H|
# ---------------------------------------------------------------------------

class TestOptimalDH:
    """Verify golden network optimal |D:H| computation."""

    def test_golden_h_bits_is_1137(self):
        """Golden LSTM |H| should be 1137 bits.

        This is our LSTM golden (Lan et al. 2024, arXiv:2402.10013v2),
        NOT the 139-bit free-form RNN golden from Abudy et al. (2025).
        The difference is due to LARGE=127 saturated gate weights.
        """
        from src.mdl.evaluation import compute_optimal_dh_test

        result = compute_optimal_dh_test(max_n=10, p=0.3)
        assert result["h_bits"] == 1137

    def test_returns_all_fields(self):
        from src.mdl.evaluation import compute_optimal_dh_test

        result = compute_optimal_dh_test(max_n=10, p=0.3)
        assert "data_dh_bits" in result
        assert "h_bits" in result
        assert "mdl_score" in result


# ---------------------------------------------------------------------------
# Δ%
# ---------------------------------------------------------------------------

class TestDeltaPct:
    """Verify Δ% computation."""

    def test_zero_gap(self):
        from src.mdl.evaluation import compute_delta_pct
        assert compute_delta_pct(2.94, 2.94) == 0.0

    def test_positive_gap(self):
        from src.mdl.evaluation import compute_delta_pct
        # 10% worse
        assert abs(compute_delta_pct(3.234, 2.94) - 10.0) < 0.01

    def test_negative_gap(self):
        """Score below optimal gives negative Δ% (unlikely but valid)."""
        from src.mdl.evaluation import compute_delta_pct
        assert compute_delta_pct(2.0, 4.0) == -50.0

    def test_zero_optimal(self):
        from src.mdl.evaluation import compute_delta_pct
        assert compute_delta_pct(1.0, 0.0) == float("inf")
        assert compute_delta_pct(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Golden under regularisers
# ---------------------------------------------------------------------------

class TestGoldenRegularisers:
    """Verify golden network norms and MDL."""

    def test_mdl_is_1137(self):
        """LSTM golden |H| = 1137 bits (not 139; see TestOptimalDH)."""
        from src.mdl.evaluation import evaluate_golden_under_regularisers
        result = evaluate_golden_under_regularisers(max_n=10, p=0.3)
        assert result["mdl_bits"] == 1137

    def test_l1_dominated_by_large_weights(self):
        """L1 norm should be large due to LARGE=127 saturated gates."""
        from src.mdl.evaluation import evaluate_golden_under_regularisers
        result = evaluate_golden_under_regularisers(max_n=10, p=0.3)
        # With LARGE=127 and many gate weights, L1 should be >> 100
        assert result["l1_norm"] > 100

    def test_108_params(self):
        from src.mdl.evaluation import evaluate_golden_under_regularisers
        result = evaluate_golden_under_regularisers(max_n=10, p=0.3)
        assert result["n_params"] == 108

    def test_ce_is_finite(self):
        from src.mdl.evaluation import evaluate_golden_under_regularisers
        result = evaluate_golden_under_regularisers(max_n=10, p=0.3)
        assert np.isfinite(result["ce_test_bits"])
        assert result["ce_test_bits"] > 0


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    """Verify table formatting functions produce output."""

    def test_comparison_table_produces_string(self):
        from src.mdl.evaluation import format_abudy_comparison_table
        table = format_abudy_comparison_table(
            our_test_data_dh=3.5,
            our_train_data_dh=1600.0,
            our_h_bits=200,
            opt_test_data_dh=2.94,
            opt_train_data_dh=1531.77,
            golden_h_bits=139,
        )
        assert "Golden" in table
        assert "Ours" in table
        assert "Δ" in table

    def test_regulariser_table_produces_string(self):
        from src.mdl.evaluation import (
            evaluate_golden_under_regularisers,
            format_golden_regulariser_table,
        )
        result = evaluate_golden_under_regularisers(max_n=10, p=0.3)
        table = format_golden_regulariser_table(result)
        assert "L1" in table
        assert "L2" in table
        assert "MDL" in table


# ---------------------------------------------------------------------------
# Recognition accuracy
# ---------------------------------------------------------------------------

class TestRecognitionAccuracy:
    """Verify recognition accuracy (binary accept/reject)."""

    def test_golden_fails_at_transition(self):
        """Golden network gets the a→b transition wrong (P(a)=0.7 > P(b)=0.3).

        This is correct behavior: the PCFG is memoryless, so the golden
        LM always predicts argmax='a' at the last a-position where the
        actual target is 'b'. Recognition accuracy is 0% on valid strings.
        """
        from src.mdl.evaluation import compute_full_string_accuracy
        from src.mdl.golden import build_golden_network_params, golden_forward
        from src.mdl.data import make_anbn_fixed_n

        params = build_golden_network_params(p=0.3)
        def golden_fwd(x):
            return golden_forward(params, x)

        inputs, targets = [], []
        for n in range(1, 21):
            inp, tgt = make_anbn_fixed_n(n)
            inputs.append(inp)
            targets.append(tgt)

        accepted = compute_full_string_accuracy(golden_fwd, inputs, targets)
        # Golden rejects all valid strings because of the a→b transition
        assert accepted.sum() == 0

    def test_golden_rejects_negatives(self):
        """Golden network should reject all invalid strings."""
        from src.mdl.evaluation import (
            compute_full_string_accuracy, generate_negative_anbn,
        )
        from src.mdl.golden import build_golden_network_params, golden_forward

        params = build_golden_network_params(p=0.3)
        def golden_fwd(x):
            return golden_forward(params, x)

        neg_inputs, neg_targets = generate_negative_anbn(
            num_examples=100, max_n=20, seed=42,
        )
        accepted = compute_full_string_accuracy(
            golden_fwd, neg_inputs, neg_targets,
        )
        # Golden should reject all (or nearly all) invalid strings
        assert accepted.sum() == 0, (
            f"Golden accepted {accepted.sum()}/100 invalid strings"
        )

    def test_negative_examples_are_invalid(self):
        """No generated negative should be a valid a^n b^n string."""
        from src.mdl.evaluation import generate_negative_anbn
        from src.mdl.data import SYMBOL_HASH, SYMBOL_A, SYMBOL_B

        neg_inputs, neg_targets = generate_negative_anbn(
            num_examples=200, max_n=30, seed=0,
        )
        for i, (inp, tgt) in enumerate(zip(neg_inputs, neg_targets)):
            # Reconstruct full string: input + last target token
            full = inp + [tgt[-1]]
            # Valid a^n b^n: [#, a, ..., a, b, ..., b, #] with equal a's and b's
            body = full[1:-1]  # strip delimiters
            if len(body) == 0:
                continue  # empty body is n=0 (valid but trivial)
            na = sum(1 for c in body if c == SYMBOL_A)
            nb = sum(1 for c in body if c == SYMBOL_B)
            if na != nb or na == 0:
                continue  # not equal counts or no symbols
            # Check structure: all a's then all b's
            is_valid = (
                all(c == SYMBOL_A for c in body[:na])
                and all(c == SYMBOL_B for c in body[na:])
            )
            assert not is_valid, (
                f"Negative example {i} is a valid a^{na} b^{nb} string"
            )

    def test_full_string_accuracy_perfect_fwd(self):
        """A forward function that always predicts token 1 accepts matching strings."""
        from src.mdl.evaluation import compute_full_string_accuracy

        # All targets are token 1 — forward always predicts token 1
        inputs = [[0, 0, 0], [0, 0]]
        targets = [[1, 1, 1], [1, 1]]

        def always_1_fwd(x):
            B, T = x.shape
            logits = jnp.full((B, T, 3), -100.0)
            logits = logits.at[:, :, 1].set(100.0)
            return logits

        accepted = compute_full_string_accuracy(always_1_fwd, inputs, targets)
        assert accepted.all()

        # Now with a target that doesn't match: token 2 instead of 1
        targets_bad = [[1, 2, 1], [1, 1]]
        accepted_bad = compute_full_string_accuracy(
            always_1_fwd, inputs, targets_bad,
        )
        assert not accepted_bad[0]  # first string has a mismatch
        assert accepted_bad[1]      # second string still all 1s
