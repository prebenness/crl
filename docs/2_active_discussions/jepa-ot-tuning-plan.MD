# JEPA-OT Tuning Plan

## Method

JEPA-OT: Action-Conditioned JEPA with Latent Counterfactuals via Sinkhorn Alignment.
Entrypoint: `jepa_ot.py`.

Configs:
- cMNIST: `config/colored_mnist/jepa_ot.yaml` (MLP encoder, z_dim=100)
- cCIFAR-10: `config/ccifar10/jepa_ot.yaml` (ResNet-18 encoder, z_dim=512)

References:
- Assran et al. 2023, "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", CVPR (arXiv:2301.08243)
- Feydy et al. 2019, "Interpolating between Optimal Transport and MMD using Sinkhorn Divergences", AISTATS (arXiv:1810.08278)
- Tarvainen & Valpola 2017, "Mean teachers are better role models", NeurIPS (arXiv:1703.01780)
- Perez et al. 2018, "FiLM: Visual Reasoning with a General Conditioning Layer" (arXiv:1709.07871)

## Architecture

Configurable encoder (`jepa_ot.encoder`):
- `"mlp"` — 2-layer MLP (flatten → Dense(100) → ReLU → Dense(z_dim) → ReLU). For cMNIST.
- `"resnet18"` — ResNet-18 with cifar_mode stem (3×3, no maxpool). Returns 512-d GAP features. Has BatchNorm (batch_stats tracked via EMA alongside params). For cCIFAR-10.

Predictor: FiLM + residual MLP.
- FiLM conditioning: `gamma(s) * z + beta(s)` — dimension-wise gating of spurious information.
- Residual: `z_hat = z + MLP([film_z, s])` — MLP learns the geometric delta, not identity.
- `predictor_hidden_dim` config controls MLP width (FiLM hidden = half).

Training:
- Augmentation: automatic per dataset (pad+crop+hflip+normalize for cCIFAR-10, none for cMNIST).
- Stop-gradient: encoder learns from CE only; predictor learns from OT only.
- Class-conditional Sinkhorn with L2 normalization and debiased divergence.
- EMA target encoder (both params and batch_stats).

## Diagnosed Problems and Fixes (all implemented)

1. **Epsilon scaling catastrophe** — L2-normalize point clouds, eps=1.0.
2. **Weak predictor** — FiLM + residual MLP (was single linear layer).
3. **Cross-class OT contamination** — class-conditional Sinkhorn with fixed-size per-class buffers.
4. **Encoder/predictor gradient race** — stop-gradient encoder from OT path.

## cMNIST Results

Sweep over lambda_ot and ema_tau with all 4 fixes active.

Best: **0.8787** (lambda_ot=0.1, ema_tau=0.996, eps=1.0).
Plateau: all configurations converge to ~0.87–0.88 regardless of hyperparameters.
This ceiling appears structural — the method works but doesn't match CCDB's 96.37% on cMNIST.

Paused cMNIST tuning due to uncertainty about whether CCDB's cMNIST numbers are valid (possible architecture discrepancy: Zhao, Li, Zhang 2025, arXiv:2504.17314 claims 3-layer MLP; FG-CCDB code uses CNN + oracle labels).

## cCIFAR-10 (current focus)

Moved to cCIFAR-10 for a cleaner comparison — both CCDB and uLA use ResNet-18 on this benchmark.

### Setup
- ResNet-18 cifar_mode (11.17M encoder params, 399K predictor, 5K classifier)
- Standard CIFAR augmentation (pad 4 + random crop 32×32 + hflip + CIFAR-10 normalize)
- p_train=0.95 (5% bias-conflicting)
- ~30s/epoch after JIT warmup

### Comparison targets (cCIFAR-10, p_corr=0.95)

| Method | Test acc | Architecture | Source |
|--------|----------|-------------|--------|
| CCDB | TBD | ResNet-18 | Zhao et al. 2025, arXiv:2504.17314 |
| uLA | TBD | ResNet-18 | - |
| **JEPA-OT** | **TBD** | ResNet-18 | This work |

### Next steps
1. Run full 50-epoch baseline with default config
2. Tune lambda_ot (0.05, 0.1, 0.3, 0.5)
3. Compare against CCDB/uLA numbers on cCIFAR-10
