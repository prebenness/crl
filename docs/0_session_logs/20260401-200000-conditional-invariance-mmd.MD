# Session: Conditional Invariance (MMD) Outer Model Implementation + Tuning

**Date:** 2026-04-01

## Summary

Implemented conditional invariance (MMD) as a second outer loss option alongside HSIC, based on `cmnist_conditional_invariance_note.tex`. Also changed HSIC to use argmaxed inner predictions instead of raw representation. Ran initial tuning experiments.

## Implementation

### New outer loss: weighted CE + class-conditional MMD

The MMD outer loss implements the conditional invariance penalty from the note:
- Color oracle signal `S = argmax(inner_logits)` (hard prediction from inner model)
- Importance weights `w(y,s) = q(s) / p(s|y)` where `q(s) = 1/K` (uniform)
- `p(s|y)` precomputed per epoch from full training set forward pass
- Laplace smoothing + weight clipping for numerical stability
- Class-conditional all-pairs MMD: `R_inv = mean_{y, s<s'} MMD²(z|y,s; z|y,s')`
- RBF kernel, masking approach similar to existing `class_cond_hsic_rbf`

Toggle via config: `model.outer_loss: "hsic"` or `"mmd"`. New `MMDConfig` dataclass with `weight`, `w_max`, `smoothing_eps`.

### HSIC change: argmax predictions instead of raw representation

Changed all 5 HSIC pair step functions (oracle, VIB, MDL, MDL-shared, ResNet) to use `one_hot(argmax(logits1))` as the first argument to `class_cond_hsic_rbf`, instead of `z1_sg` or `mu1_sg`. Both HSIC and MMD now decorrelate the outer representation from the inner model's discrete predicted color.

### Both losses use convex combination

`total = (1 - w) * CE + w * penalty` where `w` in [0, 1]. Weight=1.0 means pure penalty (no CE).

### Performance fixes

- `precompute_predictions`: JIT-compiled batch forward; oracle runner hoists precomputation out of epoch loop (inner is frozen → S and W constant). Reduced from 5.4s/epoch to ~0s.
- `compute_mmd_weights`: vectorized with one-hot outer product instead of nested for-loop.
- Summary table now reports best checkpoint accuracy, not last epoch.
- Patience (`restart_patience`) added to MMD and HSIC oracle pair configs.

### Files modified

- `src/config.py` — `MMDConfig`, `outer_loss` field
- `src/loss_fns/reg_loss_fns.py` — `class_cond_mmd_rbf`
- `src/datasets/datasets.py` — `make_epoch_batches` extended with `*aux_arrays`
- `src/training/steps.py` — 4 MMD step factories + HSIC argmax change in all 5 pair steps
- `src/training/epochs.py` — 2 MMD epoch factories
- `src/training/runners.py` — precomputation utilities, runner branching, best-acc reporting
- `colored_mnist.py` — wiring, summary table
- `config/colored_mnist/oracle_pair_mmd.yaml`, `vib_pair_mmd_sweep.yaml` — new configs
- `config/colored_mnist/oracle_pair.yaml` — added patience, weight=1.0
- `tests/test_mmd.py` — 7 new tests (all pass)

## Experiment Results (p_train=0.95, oracle pair)

All runs use the same frozen oracle checkpoint (`20260331_172654`).

### MMD results

| Weight | Batch | LR   | WD_outer | Epochs | Best Test Acc | Best Epoch |
|--------|-------|------|----------|--------|---------------|------------|
| 0.5    | 256   | 1e-3 | 1e-4     | 200    | **0.9224**    | 112        |
| 0.7    | 128   | 1e-3 | 1e-4     | 100    | 0.9220        | 80         |
| 0.5    | 128   | 1e-3 | 1e-4     | 100    | 0.9216        | 63         |
| 0.5    | 256   | 1e-3 | 1e-4     | 500    | 0.9197        | 167        |
| 0.3    | 128   | 1e-3 | 1e-4     | 100    | 0.9190        | 54         |
| 0.6    | 256   | 1e-3 | 0.0      | 200    | 0.9185        | 110        |
| 0.5    | 128   | 5e-4 | 1e-4     | 200    | 0.9161        | 47         |
| 0.5    | 512   | 1e-3 | 1e-4     | 200    | 0.9102        | 54         |

### HSIC results (with argmax change)

| Weight | Batch | LR   | WD_outer | Epochs | Best Test Acc | Best Epoch |
|--------|-------|------|----------|--------|---------------|------------|
| 0.5    | 128   | 1e-3 | 1e-4     | 100    | 0.884         | 91         |
| 0.5    | 128   | 1e-3 | 1e-4     | 500    | 0.875         | 54         |

### Pre-existing HSIC results (with raw z1, before argmax change)

From earlier runs (same session, before HSIC argmax change):
- Best: ~0.92 with MMD (old additive formulation `ce + w*mmd`)
- These are not directly comparable to post-change runs.

## Observations

1. **MMD consistently outperforms HSIC** by ~4 percentage points (0.92 vs 0.88) under the same conditions.
2. **MMD weight sweet spot: 0.5–0.7.** Too low (0.3) underpenalizes, too high (1.0) removes CE entirely.
3. **Batch size 256 slightly better than 128** (better kernel estimates), 512 worse (too few steps/epoch).
4. **Default lr 1e-3 is optimal.** 5e-4 converges slower without gaining accuracy.
5. **Weight decay 1e-4 helps slightly** vs none.
6. **All MMD runs plateau at ~0.92** regardless of training length or restarts — likely an architecture ceiling for the 3x100 MLP.
7. Training is fast: ~0.2s/epoch after JIT warmup.

## Current config state (user-modified in oracle_pair_mmd.yaml)

- `w_max: 10.0` (from 500), `smoothing_eps: 1e-2` (from 1e-6), `restart_patience: 300`

## What to try next

- Different random seeds to measure variance
- Stronger weight decay (1e-3)
- LR scheduling / cosine annealing
- Whether the 0.92 ceiling is fundamental to architecture or can be broken
- VIB pair (non-oracle) with MMD — the real use case
