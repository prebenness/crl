# VICReg-OT Tuning Plan

## Method

VICReg-OT: Counterfactual Optimal Transport with VICReg Collapse Prevention.
Entrypoint: `jepa_ot.py`.

Configs:
- cMNIST: `config/colored_mnist/jepa_ot.yaml` (MLP encoder, z_dim=100)
- cCIFAR-10: `config/ccifar10/jepa_ot.yaml` (ResNet-18 encoder, z_dim=512)

References:
- Bardes, Ponce & LeCun 2022, "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning", ICLR (arXiv:2105.04906)
- Feydy et al. 2019, "Interpolating between Optimal Transport and MMD using Sinkhorn Divergences", AISTATS (arXiv:1810.08278)
- Perez et al. 2018, "FiLM: Visual Reasoning with a General Conditioning Layer" (arXiv:1709.07871)

## Architecture

Configurable encoder (`jepa_ot.encoder`):
- `"mlp"` — 2-layer MLP (flatten → Dense(100) → ReLU → Dense(z_dim) → ReLU). For cMNIST.
- `"resnet18"` — ResNet-18 with cifar_mode stem (3×3, no maxpool). Returns 512-d GAP features. Has BatchNorm. For cCIFAR-10.

Single shared encoder for both anchor and bias-conflicting samples (concatenated forward pass for correct BN statistics). No EMA target encoder — VICReg regularization prevents collapse.

Predictor: FiLM + residual MLP.
- FiLM conditioning: `gamma(s) * z + beta(s)` — dimension-wise gating of spurious information.
- Residual: `z_hat = z + MLP([film_z, s])` — MLP learns the geometric delta, not identity.
- `predictor_hidden_dim` config controls MLP width (FiLM hidden = half).

Training:
- Augmentation: automatic per dataset (pad+crop+hflip+normalize for cCIFAR-10, none for cMNIST).
- End-to-end: all parameters (encoder, predictor, classifier) receive gradients from all losses.
- Geometry: class-conditional Sinkhorn (L2-normalized, debiased) + VICReg (variance hinge + covariance decorrelation) on both z_hat_trans and z_c.
- Task: 50/50 mix of CE on hallucinated debiased representations and CE on raw anchor representations.

## Diagnosed Problems and Fixes (from JEPA-OT phase)

1. **Epsilon scaling catastrophe** — L2-normalize point clouds inside Sinkhorn, eps=1.0. (kept)
2. **Weak predictor** — FiLM + residual MLP (was single linear layer). (kept)
3. **Cross-class OT contamination** — class-conditional Sinkhorn with fixed-size per-class buffers. (kept)
4. ~~**Encoder/predictor gradient race** — stop-gradient encoder from OT path.~~ Replaced by VICReg: variance and covariance regularization prevent collapse without requiring stop-gradient.

## VICReg-OT architectural change (2026-04-14)

Replaced EMA target encoder with VICReg regularization. Motivation: under extreme confounding (p_train=0.95), the EMA target absorbs spurious correlations as "slow features" — the target drifts toward color-biased representations, providing poor OT anchors.

Key changes:
- Single shared encoder (no EMA target, no target_params, no target_batch_stats)
- VICReg variance hinge + covariance decorrelation on z_hat_trans and z_c
- End-to-end gradient flow (no stop-gradient on encoder from OT path)
- Mixed CE: 50% hallucinated + 50% raw anchor
- Loss: L = L_task + L_geom, where L_geom = λ_inv * L_inv + λ_var * L_var + λ_cov * L_cov

## EMA-JEPA Results (prior to VICReg transition)

Best cMNIST: **0.8787** (lambda_ot=0.1, ema_tau=0.996, eps=1.0).
Plateau: all configurations converged to ~0.87–0.88 regardless of hyperparameters.

## VICReg-OT Results

### cMNIST

TBD — initial smoke test (2 epochs): 0.54 test_acc, all losses finite, VICReg terms trending down.

### cCIFAR-10

TBD.

### Comparison targets (cCIFAR-10, p_corr=0.95)

| Method | Test acc | Architecture | Source |
|--------|----------|-------------|--------|
| CCDB | TBD | ResNet-18 | Zhao et al. 2025, arXiv:2504.17314 |
| uLA | TBD | ResNet-18 | - |
| **VICReg-OT** | **TBD** | ResNet-18 | This work |

### Next steps
1. Run full 100-epoch cMNIST baseline with default VICReg-OT config
2. Compare against EMA-JEPA 0.8787 baseline
3. Tune lambda_inv, lambda_var, lambda_cov
4. Run cCIFAR-10 50-epoch baseline
5. Compare against CCDB/uLA numbers on cCIFAR-10
