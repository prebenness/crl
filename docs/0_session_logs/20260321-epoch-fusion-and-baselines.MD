# Epoch Fusion + Baseline Metrics Integration

**Date:** 2026-03-21
**Scope:** Item 1.2 (lax.scan epoch fusion), items 6-7 (baseline metrics, results table).

## What Was Done

### 1. `lax.scan` epoch fusion for full-batch training (item 1.2)

**Files changed:**
- `src/mdl/training.py` — added `anneal_tau_traceable`, `make_fused_epoch_fn`; added `jit` param to `make_train_step`
- `differentiable_mdl.py` — added `_eval_and_update_best` helper; restructured `run_training_basic` with fused/non-fused branching

**How it works:**
For full-batch training (bs >= N), contiguous same-phase epoch runs are fused into single `lax.scan` calls. The fused path eliminates Python-loop dispatch overhead by keeping all computation within XLA.

Key components:
- `anneal_tau_traceable`: JAX-traceable version of `anneal_tau_st_phase` using `jnp.where` instead of Python `if`, safe for use inside `lax.scan`
- `make_fused_epoch_fn`: Creates a JIT'd function that runs N full-batch epochs via `lax.scan`. Closes over training data and tau schedule. `n_steps` is static (triggers recompilation per distinct value); `start_epoch` is dynamic.
- `make_train_step(jit=False)`: Returns un-JIT'd train step for use inside the scan body (the outer scan provides the JIT)

Chunk boundaries determined inline via scan-forward loop: starts at current epoch, advances until phase change, log boundary, or eval boundary. This avoids pre-computing chunks while correctly handling all edge cases (GST→DST transition via tau threshold, epoch 0 logging, resume).

**Performance:** ~2x speedup on 10-epoch debug run (4.9s fused vs 9.3s mini-batch). Larger speedups expected on longer runs.

**Refactoring:** Extracted `_eval_and_update_best` helper to avoid duplicating the 60-line validation eval block. Both fused and non-fused paths use this helper. `best_*` variables consolidated into a `best` dict.

### 2. Baseline metrics integration (items 6-7)

**File changed:** `scripts/experiments/baseline_lstm_experiments.py`

Added grammar-weighted |D:H| computation after training:
- `compute_grammar_weighted_nll_bits` for test |D:H|
- `compute_train_dh` for train |D:H|
- `compute_optimal_dh_test` / `compute_optimal_dh_train` for golden baselines
- `compute_delta_pct` for Δ%
- `format_abudy_comparison_table` for paper-style output

Results saved to `results.json` with full metric breakdown: `test_data_dh_bits`, `train_data_dh_bits`, `delta_test_pct`, `delta_train_pct`, golden reference values.

### Debug experiment results (10 epochs, seed=0, test_max_n=200)

All methods undertrained at 10 epochs — numbers are for infrastructure validation only:

| Method | |D:H| train | |D:H| test | Δ_train% | Δ_test% | |H| |
|--------|-------------|------------|----------|---------|-----|
| Golden | 2798.61 | 2.94 | --- | --- | 1137 |
| CE only | 8252.65 | 8.64 | +194.9% | +194.2% | 3713 |
| CE+L1 (λ=1) | 8252.72 | 8.64 | +194.9% | +194.2% | 3769 |
| CE+L2 (λ=1) | 8252.67 | 8.64 | +194.9% | +194.2% | 3771 |
| Diff. MDL | 9645.65 | 10.12 | +244.7% | +244.4% | 1481 |

MDL is worse at 10 epochs because the warmup phase hasn't finished. Production runs need 1000+ epochs.

## Tests

All 121 tests pass: 36 MDL + 48 math + 24 evaluation + 13 baselines.

## Remaining Work

- Long production runs for baselines (1000 epochs, multi-seed, matching Abudy et al. 2025 Table 10)
- Long MDL production runs with best schedule
- Combined multi-method comparison table from production results
