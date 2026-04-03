# 2026-03-31 Five Benchmark Dataset Implementation

## Summary

Implemented the full infrastructure for five spurious correlation benchmarks: cMNIST (updated), cCIFAR10, Waterbirds, CelebA, and CivilComments. This follows the specification in `additional-crl-datasets.tex`.

## Changes

### Phase 0: cMNIST Fixes
- **Color palette fix** (`src/datasets/datasets.py`): Replaced the old palette (which had linearly dependent orange/brown colors causing oracle accuracy ceiling at ~96%) with the DFA hue-wheel palette (Lee et al., 2021). All 10 colors are pairwise linearly independent.
- **Validation split** (`src/datasets/datasets.py`): Added `split="train"/"val"/"test"` parameter to `CMNISTuLA`. Val is a stratified ~10% split from the biased training set (preserves bias structure). Train+val = 60000 MNIST training images.
- **Beta config** (`src/config.py`): Added `beta` field to `DatasetConfig` (bias-conflicting ratio; p_corr = 1 - beta).

### Phase 1: Infrastructure
- **Flax ResNet-18/50** (`src/models/resnet.py`): BasicBlock and BottleneckBlock implementations with BatchNorm. ResNet-18 has `cifar_mode` (3x3 stem, no maxpool for 32x32 inputs). ResNet-50 for 224x224 ImageNet-scale inputs. Both follow convention: `__call__(x, train) -> (logits, {"z": gap_output})`.
- **BatchNormTrainState** (`src/training/train_state.py`): Extends `TrainState` with `batch_stats` field. `create_state_resnet()` initializes BatchNorm models correctly.
- **ResNet training steps** (`src/training/steps.py`): `make_train_step_resnet(cfg, augment_fn)` and `make_eval_step_resnet(cfg, transform_fn)` handle `mutable=['batch_stats']` during training and stored batch_stats during eval. Optional augmentation functions applied per-batch.
- **JAX augmentations** (`src/datasets/augmentations.py`): JIT-compatible image transforms: normalize (ImageNet/CIFAR-10), random crop, center crop, random hflip, resize. Composed pipelines for Waterbirds and CelebA.
- **Group-balanced evaluation** (`src/training/group_eval.py`): `eval_group_accuracies()` computes per-group and worst-group accuracy.
- **Streaming data loader** (`src/datasets/streaming.py`): For datasets too large to materialize (CelebA). Wraps PyTorch DataLoader, yields JAX NHWC arrays. Includes `streaming_train_epoch()` and `streaming_eval_epoch()` using Python for-loops instead of `lax.scan`.
- **Pretrained weight loading** (`src/models/pretrained.py`): Converts PyTorch ResNet-50 ImageNet weights to Flax params. Caches as .npz for reuse.
- **BERT wrapper** (`src/models/bert.py`): Flax BERT classifier using HuggingFace `FlaxBertModel`. Returns `(logits, {"z": [CLS] embedding})`.

### Phase 2: Dataset Classes
- **cCIFAR10** (`src/datasets/ccifar10.py`): 10 corruption types mapped to 10 classes (LfF mapping). Uses `imagecorruptions` package. Cached to disk after generation. Supports train/val/test splits.
- **Waterbirds** (`src/datasets/waterbirds.py`): Binary bird classification with background spurious correlation. Loads from Sagawa et al. pre-composited dataset + metadata.csv.
- **CelebA** (`src/datasets/celeba.py`): Blond Hair classification with Male spurious attribute. Uses torchvision CelebA.
- **CivilComments** (`src/datasets/civilcomments.py`): Toxicity classification with 8 demographic identity groups. BERT tokenization cached to disk.

### Phase 3: Integration
- **Unified entry point** (`train.py`): Supports all materializable datasets (cMNIST, cCIFAR10, Waterbirds) with ResNet models.
- **Config files**: `config/ccifar10/`, `config/waterbirds/`, `config/celeba/`, `config/civilcomments/`.
- **Data README** (`data/README.md`): Download and setup instructions for all datasets.

## Test Results

All 25 existing tests pass. New infrastructure verified manually (ResNet init, train step, eval step, val split).

## Verification Results

**Oracle training with fixed palette**: Train 100.0%, Test 99.86% (up from ~96% with old palette). Confirms all 10 DFA hue-wheel colors are fully distinguishable. Oracle checkpoint updated in `oracle_pair.yaml`.

## Completed in Follow-up
- Pair mode wiring in `train.py` (`make_train_step_resnet_pair` + `run_train_eval_pair`)
- Streaming training loop in `train.py` (`_run_streaming` using `StreamingLoader`)
- BERT model registration in `train.py`
- Oracle training verification (99.86% test accuracy)
