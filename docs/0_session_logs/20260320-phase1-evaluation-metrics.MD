# Phase 1: Paper-Critical Metrics Infrastructure

**Date:** 2026-03-20
**Scope:** Implement all 5 Phase 1 items from the project roadmap (Section 3.C metrics).

## What Was Done

Created `src/mdl/evaluation.py` — a new module implementing paper-comparable evaluation metrics for direct comparison with Abudy et al. (2025, arXiv:2505.13398v2) and Lan et al. (2024, arXiv:2402.10013v2).

### Functions implemented

1. **Grammar weights**: `compute_anbn_grammar_weights(max_n, p, min_n=0)` — raw PCFG weights P(n) = p*(1-p)^n. Default includes n=0 (empty string).

2. **Per-string NLL**: `compute_per_string_nll_bits(forward_fn, inputs, targets, ...)` — total NLL in bits per string (sum over positions, not average). Uses Abudy et al. smoothing (additive 1e-10).

3. **Grammar-weighted test NLL**: `compute_grammar_weighted_nll_bits(forward_fn, max_n, p)` — |D:H|_test = Σ P(n) × NLL_total(n) including n=0. This is the data term in Abudy et al.'s tables.

4. **Optimal baselines**: `compute_optimal_dh_test(max_n, p)` and `compute_optimal_dh_train(train_inputs, train_targets, p)` — golden network's |D:H| data term and |H|.

5. **Trained network composite**: `evaluate_trained_network_dh(apply_fn, params, grid_codelengths, hidden_size, test_max_n, p)` — grammar-weighted test NLL + Lan-style |H| for argmax-discretised trained networks. Also `compute_trained_h_bits(params, grid_codelengths, hidden_size)`.

6. **Δ%**: `compute_delta_pct(score, optimal)` — (score - optimal) / optimal × 100. Operates on data term only (Abudy et al. convention).

7. **Train |D:H|**: `compute_train_dh(forward_fn, train_inputs, train_targets)` — raw NLL sum over training corpus.

8. **Golden under regularisers**: `evaluate_golden_under_regularisers(max_n, p)` — CE, L1 norm, L2 norm squared, MDL |H|.

9. **Formatting**: `format_abudy_comparison_table(...)` and `format_golden_regulariser_table(...)`.

### Tests

Created `tests/test_evaluation.py` with 24 tests covering all functions. Key validations:
- Golden test |D:H| ≈ 2.94 bits (matching Abudy et al. reference)
- Analytical cross-check: E[NLL] = E[n]×(-log2(1-p)) + (-log2(p)) ≈ 2.938
- Our LSTM golden |H| = 1137 bits (confirmed)
- 108 parameters, L1 norm > 100 (saturated gates)
- All 60 tests pass (36 existing + 24 new)

## Key Conventions Established

1. **n=0 included**: Grammar-weighted |D:H|_test uses the full PCFG expectation including the empty string (n=0). This gives ~2.94 bits for the ideal predictor. Excluding n=0 (conditioning on n≥1) gives ~3.45 — the difference is because E[n] = (1-p)/p ≈ 2.33 (full) vs E[n|n≥1] = 1/p ≈ 3.33.

2. **Raw PCFG weights**: Grammar weights are unnormalized P(n) = p*(1-p)^n. For large max_n they sum to ~1.

3. **|H| difference**: Our LSTM golden has |H| = 1137 bits (due to LARGE=127 saturated gates and transcendental output weights). Abudy et al.'s free-form RNN golden has |H| = 139 bits. Different architectures, not directly comparable for L1/L2.

4. **Abudy et al. notation**: |D:H| = data term only (not including |H|). Δ% operates on data term. |H| reported separately.

## Remaining Work

- Wire `evaluate_trained_network_dh` into `run_final_evaluation` in `differentiable_mdl.py`
- Add `compute_optimal_dh_test` call to `evaluate_golden_baseline` for automatic golden reference
- Print Abudy-style comparison table at end of training runs
