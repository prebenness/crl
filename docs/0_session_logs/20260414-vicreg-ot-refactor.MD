# VICReg-OT Refactor (2026-04-14)

## Summary

Refactored `jepa_ot.py` from EMA-stabilized JEPA to VICReg-OT: replaced the EMA target encoder with VICReg regularization (Bardes, Ponce & LeCun 2022, arXiv:2105.04906) for collapse prevention, enabling fully end-to-end training through a single shared encoder.

## Motivation

JEPA-OT on cMNIST plateaued at 0.8787 after extensive tuning. Hypothesis: the EMA target encoder absorbs spurious correlations as "slow features" under 95% bias — it sees overwhelmingly color-aligned samples and drifts toward color-biased representations, providing poor OT anchors. VICReg replaces EMA with explicit variance and covariance regularization, allowing the encoder to receive gradients from all losses without collapsing.

Design spec: `agent-context/causal_vicreg_ot.tex`.

## Changes

### `jepa_ot.py`

**New loss functions:**
- `vicreg_variance_loss(z, gamma, eps)` — per-dimension hinge on std dev, prevents mode collapse.
- `vicreg_covariance_loss(z)` — off-diagonal covariance penalty, decorrelates dimensions.

**Config (`JEPAOTConfig`):**
- Removed: `ema_tau`, `lambda_ot`
- Added: `lambda_inv` (Sinkhorn weight), `lambda_var` (variance hinge), `lambda_cov` (covariance), `vicreg_gamma` (variance target)

**train_step rewrite:**
- Single shared encoder for both X_a and X_c (concatenate → encode → split). Correct BN statistics for ResNet-18.
- No stop-gradient on encoder from OT path — fully end-to-end.
- VICReg applied to both z_hat_trans and z_c.
- Mixed CE: 50% hallucinated (random color) + 50% raw anchor.
- Loss: L = L_task + L_geom, where L_geom = λ_inv * L_inv + λ_var * L_var + λ_cov * L_cov.

**Removed:**
- EMA target encoder (target_params, target_batch_stats, EMA update)
- All stop-gradient calls

**Expanded metrics:** loss_ce_u, loss_ce_a, loss_task, loss_inv, loss_var, loss_cov, loss_geom, loss_total, accuracy.

### Config files
- `config/colored_mnist/jepa_ot.yaml` — lambda_inv=1.0, lambda_var=25.0, lambda_cov=1.0, vicreg_gamma=1.0
- `config/ccifar10/jepa_ot.yaml` — lambda_inv=0.1, lambda_var=25.0, lambda_cov=1.0, vicreg_gamma=1.0

### Other
- `sweep_jepa_ot_2.sh` — updated to VICReg-OT parameter names
- `docs/2_active_discussions/jepa-ot-tuning-plan.MD` — updated to reflect VICReg-OT

## Preserved components

All Sinkhorn functions, class-conditional Sinkhorn, L2 normalization inside Sinkhorn, FiLM + residual predictor, BiasConflictingPool, all model classes, make_eval_fn (unchanged).

## Verification

| Test | Result |
|------|--------|
| cMNIST 2-epoch smoke | 0.54 test_acc, all losses finite, VICReg terms trending down, ~12s/epoch |
| cCIFAR-10 2-epoch smoke | 0.16 test_acc, ResNet-18 BN works with concat forward pass, ~40s/epoch |

## VICReg hyperparameter defaults

From Bardes et al. 2022: λ_var=25, λ_cov=1, γ=1 (dimension-normalized, scale-independent of z_dim). lambda_inv carried over from prior OT tuning (1.0 for cMNIST, 0.1 for cCIFAR-10).

## Next steps

1. Full 100-epoch cMNIST run — compare against 0.8787 EMA baseline
2. Tune lambda_inv, lambda_var, lambda_cov
3. Full 50-epoch cCIFAR-10 run
