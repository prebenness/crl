# FG-CCDB Replication Script

**Date**: 2026-04-03

## Objective

Replicate the FG-CCDB method from Zhao, Zhang & Li, 2025 ("Fine-Grained Bias Exploration and Mitigation for Group-Robust Classification", arXiv:2505.06831v1) on cMNIST. Target: reproduce their Table 2 numbers.

**Paper targets** (test accuracy, 5 seeds):

| bc_ratio | FG-CCDB |
|----------|---------|
| 0.5% | 89.02 +/- 0.45 |
| 1% | 94.93 +/- 0.17 |
| 2% | 96.18 +/- 0.19 |
| 5% | 98.21 +/- 0.02 |

## What was done

### 1. Implementation (fg_ccdb_replication.py)

Standalone script at repo root implementing the full 3-stage FG-CCDB pipeline:

1. **BEO (Bias Exploration via Overfitting)**:
   - gamma=10% random subsample of training data
   - Train fresh 3x100 MLP with ERM for 20 epochs
   - 3 rounds of bias enhancement: per-class top-50% confidence selection, fresh model each round
   - Final model's argmax predictions = pseudo-bias labels

2. **FG-CCDB weights** (closed-form, Eqs. 3-5 in paper):
   - N_mat[bias, class] count matrix
   - G = N_mat / N, P[:,j] = G[:,j] / sum(G[:,j]), q = sum_j G[:,j]
   - W[i,j] = q[i] / P[i,j], w[i,j] = W[i,j] / N_mat[i,j]
   - Float64 to avoid numerical issues; zero-division safety for empty groups

3. **Debiased training**:
   - Weighted resampling (probability proportional to per-sample weight)
   - 5000 gradient steps, batch_size=256
   - Model selection: best worst-class accuracy on biased validation set

Hyperparameters match paper's Appendix table: 3x100 MLP, Adam lr=1e-2, wd=1e-4, batch 256, no scheduler, no augmentation.

### 2. First attempt: PyTorch

Initial implementation was pure PyTorch using DataLoader + WeightedRandomSampler. Problem: per-sample `CMNISTuLA.__getitem__` colorization through DataLoader was extremely slow — 85 seconds for a single run. Full 4x5 sweep would take ~28 minutes.

### 3. Rewrite: JAX/Flax

Rewrote using JAX/Flax, reusing:
- `CMNISTuLA` dataset + `dataset_to_jax_arrays()` for one-shot materialisation
- `make_epoch_batches()` for fast host-side shuffling
- `ULAMLPClassifier` model directly (no reimplementation needed)
- JIT-compiled `train_step` and `predict_batch`

Runtime dropped to ~55s per run. Stage 3 uses `np.random.choice(p=probs)` with normalised FG-CCDB weights instead of PyTorch's WeightedRandomSampler — mathematically equivalent.

### 4. Bug: optax.adamw positional args

Critical bug found: `optax.adamw(lr, wd)` was setting `b1=1e-4` (second positional arg is `b1`, not `weight_decay`). This killed Adam momentum and made training essentially broken.

**Fix**: `optax.adamw(learning_rate=lr, weight_decay=wd)`

Before fix: BEO bias accuracy = 0.19 (random). After fix: 0.94 (correct).

## Current Results (single seed=0, bc_ratio=5%)

- **BEO bias prediction accuracy**: 0.9402 (paper reports 100% on cMNIST)
- **N_mat diagonal fraction**: 0.9597
- **FG-CCDB weight range**: [0.000019, 593.5]
- **Stage 3 best val worst-class acc**: 0.4146
- **Test accuracy**: 67.75% (target: 98.21%)

## Open Issues

### Stage 3 accuracy gap (67% vs 98%)

The debiased training produces far lower accuracy than the paper reports. Diagnosis:

1. **Extreme weight dynamic range**: minority weights are ~30 million x larger than majority weights after normalisation. Each batch of 256 is almost entirely minority samples (~2700 unique points out of 54K). Model has 256K parameters trained on effectively 2700 samples → severe overfitting.

2. **BEO accuracy gap**: our BEO achieves 94% bias prediction vs. paper's reported 100%. The 6% error means some majority samples get minority weights and vice versa, corrupting the weight matrix. It's unclear whether this alone explains the accuracy gap.

3. **Training instability**: val worst-class accuracy oscillates wildly (0.08-0.41) across 5000 iterations. The learning dynamics under such extreme reweighting are unstable.

### Possible causes to investigate

- `optax.adamw` vs PyTorch `Adam(weight_decay=...)`: decoupled weight decay vs L2 reg. Small difference at wd=1e-4 but could matter.
- BEO enhancement: paper achieves 100% bias accuracy with 3 rounds. Our enhancement rounds don't improve much beyond the initial 94%. May need to investigate the confidence selection mechanism.
- The paper's description of "5000 iterations" may interact differently with PyTorch's WeightedRandomSampler epoch semantics vs. our per-step random sampling.

## Files

- `fg_ccdb_replication.py` — JAX/Flax replication script (current version)
- `results/fg_ccdb_replication/` — saved result JSONs
