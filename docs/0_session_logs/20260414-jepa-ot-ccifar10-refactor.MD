# JEPA-OT cCIFAR-10 Refactor (2026-04-14)

## Summary

Refactored `jepa_ot.py` to support cCIFAR-10 with ResNet-18 alongside the existing cMNIST/MLP path. All changes are backward-compatible — cMNIST runs produce identical results.

## Motivation

cMNIST JEPA-OT plateaued at ~0.88 after extensive tuning (sweeps over lambda_ot, ema_tau, batch size). Shifted focus to cCIFAR-10 for a cleaner comparison against CCDB (Zhao et al. 2025, arXiv:2504.17314) and uLA, which both use ResNet-18 on this benchmark.

## Changes

### `jepa_ot.py`
- `JEPAOTConfig`: added `encoder` ("mlp"/"resnet18") and `predictor_hidden_dim` fields.
- `ResNetEncoder`: new Flax module wrapping ResNet-18, returns 512-d GAP features. Has BatchNorm.
- Unified batch_stats approach: MLP uses `batch_stats={}` (no-op), ResNet tracks real BN stats. Same code path — no branching in the train step.
- `train_step`: augmentation hookup (resolved at JIT trace time), `mutable=["batch_stats"]` for encoder forward pass, EMA updates both params and batch_stats.
- `make_eval_fn`: accepts `batch_stats` and `eval_transform_fn`.
- `InterventionalPredictor`: FiLM hidden_dim now scales as `predictor_hidden_dim // 2` (preserves cMNIST 64/128 ratio, scales to 128/256 for cCIFAR).

### `src/datasets/augmentations.py`
- Added `ccifar10_train_augment`: pad 4px + random crop 32x32 + hflip + CIFAR-10 normalize.
- Added `ccifar10_eval_transform`: CIFAR-10 normalize only.
- Fixed `random_hflip` bug: `rngs[:, 0]` extracted raw ints, not valid PRNG keys.

### `config/ccifar10/jepa_ot.yaml`
- New config: ResNet-18 encoder, z_dim=512, predictor_hidden_dim=256, lambda_ot=0.1, 50 epochs, p_train=0.95.

## Verification

- cMNIST 2-epoch test: identical output (MLP, z_dim=100, ~8s/epoch).
- cCIFAR-10 2-epoch test: ResNet-18 (11.17M params), ~30s/epoch steady state, JIT warmup ~60s first epoch.
- Standalone tests confirmed unified batch_stats approach works for both MLP (empty dict) and ResNet (real BN stats).

## Parameter counts (cCIFAR-10)

| Component | Params |
|-----------|--------|
| ResNet-18 encoder | 11,173,962 |
| Predictor (FiLM + residual MLP) | 398,976 |
| Classifier head | 5,130 |
| **Total trainable** | **11,578,068** |
| Target (EMA) copy | 11,173,962 |
| Batch stats (BN running averages) | 9,600 |
