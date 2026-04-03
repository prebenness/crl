# CBA-OM Implementation Session (2026-04-03)

## What was done

Implemented CBA-OM (Causal Backdoor Adjustment via Oracle Marginalization) as a new experiment mode in the codebase, based on Proposal 5 (section 4.2) from the `cba-om.MD` design document.

### CBA-OM method summary

- **Training**: colour-conditioned classifier `p(y|x,s)` with standard cross-entropy. Colour `s` provided via learned embedding concatenated with image features (late fusion).
- **Inference**: backdoor marginalisation over all 10 colours. No oracle needed at test time.
- **Architecture**: 3x100 MLP (ULA protocol) + 16-dim colour embedding.

### Files modified

- `src/config.py` — added `CBAOMConfig(embed_dim=16, num_colors=10)`
- `src/models/classifiers.py` — added `CBAOMMlp` Flax module
- `src/datasets/datasets.py` — extended `make_epoch_batches` with `*aux_arrays`
- `src/training/steps.py` — added `make_train_step_cba_om`, `make_eval_step_cba_om` (marginalised via `jax.vmap`)
- `src/training/epochs.py` — added `make_train_epoch_cba_om` (3-element batch scan)
- `src/training/train_state.py` — added `create_state_cba_om`
- `src/training/runners.py` — added `run_train_eval_cba_om`
- `colored_mnist.py` — added `cba_om` mode dispatch

### Files created

- `config/colored_mnist/cba_om.yaml`

## Results

### CBA-OM with ground truth colour oracle (Phase 1)

Config: `cba_om.yaml`, lr=1e-3, batch_size=128, wd=1e-4, early_stopping_patience=15.

| Metric | Value |
|--------|-------|
| Best test accuracy | **0.9047** (epoch 26) |
| Final test accuracy | 0.9022 |
| Train accuracy | 1.0000 |
| Training time | 7.7s (41 epochs, early stopped) |

### Comparison

| Method | Test Acc |
|--------|----------|
| ERM | ~0.10 |
| VIB Pair | 0.8846 |
| MDL Pair | 0.8755 |
| Oracle + HSIC | 0.884 |
| **CBA-OM (gt oracle)** | **0.9047** |
| Oracle + MMD | 0.9224 |
| CCDB target | ~0.96 |

## Observations

- CBA-OM beats HSIC-based methods and VIB pair with a far simpler pipeline (single-stage CE training, no reweighting, no distribution matching).
- Does not yet beat Oracle + MMD (0.9224). Gap is ~2 points.
- The model converges extremely fast (near-perfect train by epoch 5) and the test accuracy plateaus around 0.90.
- Early stopping at epoch 41 suggests diminishing returns from more training.
- The 0.90 plateau may be related to the 3x100 architecture ceiling (same issue observed with MMD at ~0.92).

## Next steps

- Phase 2: CBA-OM with model oracle (use stored OracleMLP checkpoint instead of ground truth colours).
- Hyperparameter search: try different learning rates, batch sizes, longer training with no early stopping, LR scheduling.
- Architecture experiments: try the proposal's (256,256,100) architecture to see if higher capacity helps.
- Combine CBA-OM with regularisation (e.g. weight decay sweep, dropout).
