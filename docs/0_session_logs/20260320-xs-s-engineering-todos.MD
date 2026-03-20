# XS + S Engineering TODOs — 2026-03-20

## What was done

Implemented all 8 XS and S engineering TODOs from the consolidated code-cleanup list.

### XS items

**1. Entropy variable naming** — Updated docstrings and comments in `training.py`, `shared_weights.py`, `steps.py`, and `differentiable_mdl.py` to consistently use "entropy bonus (subtracted): tau * H(pi)" instead of the proposal's "-(1/beta) * H(pi)" notation.

**2. Short-circuit `n_samples` for `mode_forward`** — When `mode_forward=True`, the forward pass is deterministic, so multi-sampling wastes compute. Added `n_samples = 1 if cfg.mdl.mode_forward else cfg.mdl.n_samples` at all 4 cMNIST step factories in `steps.py`, and `effective_n_samples` guards at both ANBN train-step creation sites in `differentiable_mdl.py`.

**3. Vectorize `_compute_discrete_hyp_bits`** — Replaced the Python `for` loop + per-element `rational_codelength()` call with a single `jnp.sum(grid_codelengths[idx])` lookup. Updated all 4 call sites to pass `grid_codelengths` instead of `grid, grid_values`.

### S items

**4. Full fixed test sets during evaluation** — Added `make_eval_batches(x, y, batch_size)` in `datasets.py` that pads the last batch and returns a per-batch `counts` array. Updated `make_eval_epoch` in `epochs.py` to weight loss/accuracy by counts. Updated all 4 cMNIST runners to create eval batches once before the training loop.

**5. Train tail-batch handling** — Documented in `make_epoch_batches` docstring: random permutation means different samples are dropped each epoch, so all samples are seen over multiple epochs. Acceptable for training; no code change needed.

**6. Clarify `p_test` for `cmnist_ula`** — Added a `warnings.warn()` in `build_dataset` when `name == "cmnist_ula"` and `p_test != 0.1`, since `CMNISTuLA` always uses unbiased (uniform) test colors per the uLA protocol.

**7. Enforce `rep_dim` for ULA** — `ULAMLPClassifier` now emits a warning if `rep_dim != 100`. Hidden widths remain fixed at 100 for strict fidelity to the papers being replicated.

**8. Unify batch scaling convention** — Changed ANBN `_compute_data_nll_bits` and `_shared_compute_data_nll_bits` to return averaged NLL (divided by valid mask count) instead of summed. Changed `batch_scale = B / n_train` → `hyp_scale = 1.0 / n_train` in both `make_loss_fn` (basic) and `make_shared_loss_fn` (shared). Updated `mdl-metrics-and-logging.MD` formulas. **Important: existing ANBN hyperparameters (lambda, lr) will need recalibration.**

## Verification

All 82 tests pass (`test_mdl.py` + `test_math.py`).

## Files changed

- `differentiable_mdl.py` — docstring, `_compute_discrete_hyp_bits` vectorized, `effective_n_samples` guard
- `src/mdl/training.py` — docstrings, `_compute_data_nll_bits` averaged, `hyp_scale = 1/N`
- `src/mdl/shared_weights.py` — docstrings, `_shared_compute_data_nll_bits` averaged, `hyp_scale = 1/N`
- `src/training/steps.py` — docstrings, `n_samples` short-circuit in 4 factories
- `src/training/epochs.py` — `make_eval_epoch` accepts and uses `counts`
- `src/training/runners.py` — all 4 runners use `make_eval_batches` for test data
- `src/datasets/datasets.py` — new `make_eval_batches`, `p_test` warning, `make_epoch_batches` docstring
- `src/models/classifiers.py` — `ULAMLPClassifier` `rep_dim` warning
- `docs/1_closed_topics/mdl-metrics-and-logging.MD` — updated formulas to unified `1/N` convention
- `docs/2_active_discussions/code-cleanup-todos.MD` — marked 8 items done, 7 remain
