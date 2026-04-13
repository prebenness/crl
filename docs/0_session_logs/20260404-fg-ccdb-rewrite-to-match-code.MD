# FG-CCDB Replication Rewrite: Matching Actual Code

**Date**: 2026-04-04

## Objective

Complete rewrite of `fg_ccdb_replication.py` to faithfully replicate what Zhao, Zhang & Li, 2025 ("Fine-Grained Bias Exploration and Mitigation for Group-Robust Classification", arXiv:2505.06831v1) **actually implemented** in their code repo, not what their paper describes.

## Key Code-vs-Paper Discrepancies Found

Detailed reading of their repository (`FG_CCDB/show_tsne.py`, `train_bias_classifier_20250509.py`, `dfr_bias_emerging_reweight_final.py`) revealed three major discrepancies:

1. **Architecture**: Paper claims "3-layer MLP". Code uses `MNIST_CNN_fc` — a 4-layer CNN (Conv→BN→ReLU ×4, strides [1,2,2,2], channels [64,128,256,256], GlobalAvgPool, Dense(256,10)).
2. **Bias labels**: Paper describes using BEO predictions. Code has `if 1: pseudo_g = gt_g` — always uses oracle ground truth alignment labels (binary 0=aligned, 1=misaligned).
3. **BEO training**: Paper says subsample gamma=10%. Code trains initial model on full training data.

Additional differences from paper:
- BEO lr=1e-4 (code) vs 1e-2 (paper)
- Stage 3 lr=1e-3 (code) vs 1e-2 (paper)
- Stage 3 2000 iters (code) vs 5000 (paper)
- BEO enhancement uses CE + contrastive loss (lambda_ctr=0.001), not just CE
- Confidence selection uses true-class logit, not max softmax
- Weight computation uses `group_weight_specific_v4` with class balancing factor (lamd = max_count/per_class_count)
- Double correction: both weighted sampling AND per-sample loss weighting in Stage 3
- Cosine LR schedule for BEO, constant LR for Stage 3
- Data augmentation in Stage 3: RandomResizedCrop(28, scale=(0.8,1.1)) + RandomRotation(30) + Normalize(0.5,0.5,0.5)

## What was Rewritten

Complete rewrite from the previous session's version:

1. **MNISTCNN**: New Flax module matching their 4-layer CNN exactly
2. **BNTrainState**: Custom TrainState subclass carrying batch_stats for BatchNorm
3. **Contrastive loss**: Intra-class pairwise distance loss matching `ctr_loss` from their code
4. **Batch affine augmentation**: `augment_batch()` using `torch.nn.functional.affine_grid` + `grid_sample` — approximates their PyTorch transforms but operates on entire batch at once (no per-sample Python loop)
5. **make_fns() factory**: Builds JIT-compiled train_step_ce, train_step_ctr, train_step_weighted, predict_batch — with separate BN and non-BN branches
6. **stage1_beo**: 1 initial training on full data (CE only) + 3 enhancement rounds (CE + contrastive), cosine LR schedule
7. **stage2_weights_v4**: Matches `group_weight_specific_v4` with class balancing, clamp 1e-4, normalize to [0,1]
8. **stage3_debiased**: Weighted sampling + per-sample weighted CE, augmentation on training batches, normalize-only on val
9. **CLI flags**: `--ula_mlp` (switch to uLA 3x100 MLP), `--use_beo` (use BEO predictions instead of oracle)

## Smoke Test Results (seed=0, bc_ratio=5%, target 98.21%)

- **CNN + Oracle**: best val worst-class ~92% (test eval timed out due to bash timeout)
- **MLP + Oracle** (no augmentation version): 90.18% test
- **BEO color accuracy**: ~99.8%

## Performance Fix

Initial augmentation implementation used a per-sample Python loop (`torch.stack([_STAGE3_AUG(t[i]) for i in range(B)])`) — 256 images per iteration, 2000 iterations. Stage 3 took 5+ minutes per 100 iterations.

Replaced with batch affine transform: build B×2×3 affine matrices (encoding scale, rotation, translation), then single `affine_grid` + `grid_sample` call. Orders of magnitude faster.

## Remaining Gap (~6pp from paper target)

- Training instability: val accuracy oscillates wildly (65–97%) with CNN + double-weighting
- Possible causes:
  - BatchNorm behavior differences JAX vs PyTorch
  - Weight init: Flax uses LeCun normal, PyTorch uses Kaiming
  - Affine augmentation is an approximation of their exact RandomResizedCrop
  - lr=1e-3 may need tuning for JAX optimizer differences

## Next Steps

1. Run full sweep across all bc_ratios (0.5%, 1%, 2%, 5%) and 5 seeds
2. Investigate training instability if gap persists
3. Compare CNN vs MLP results across all bc_ratios

## Files

- `fg_ccdb_replication.py` — complete rewrite (622 lines)
