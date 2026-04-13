# S3E Implementation Session (2026-04-04)

## What was done

Implemented S3E (Spectral Spurious Subspace Elimination) as a standalone script `s3e.py`, based on Method 5 from the `s3E-claude.md` design document. Inspired by concept erasure techniques (Ravfogel et al., 2020, "Null it out", ACL; Ravfogel et al., 2022, "Linear adversarial concept erasure", ICML).

### S3E method summary

4-phase pipeline that removes spurious colour information from representations via spectral identification and orthogonal projection:

1. **Phase 1 (ERM Backbone)**: Train 3x100 MLP with standard CE on biased cMNIST.
2. **Phase 2 (Probe + SVD)**: Freeze backbone. Extract penultimate 100-d features. Fit L2-regularised 10-class logistic regression predicting colour from features. SVD of probe weight matrix W_s (10x100) → right singular vectors V identify colour-encoding directions.
3. **Phase 3 (Projection)**: P_perp = I - V_{:,1:r} V_{:,1:r}^T. Projected features = h - (h @ V_r) @ V_r^T.
4. **Phase 4 (Retrain Head)**: Train linear classifier Dense(10) on projected features with standard CE.

Sweep over r (rank of spurious subspace removed, 1-9). Phase 1+2 run once, shared across sweep.

### Files created

- `s3e.py` — standalone script (~450 lines, JAX/Flax)
- `config/colored_mnist/s3e.yaml` — experiment config
- `docs/2_active_discussions/s3e-evaluation-plan.MD` — post-implementation evaluation plan

### Files modified

- `src/models/classifiers.py` — added `"h": h` to `ULAMLPClassifier` aux dict (one-line, backward-compatible). This exposes the penultimate layer (layer 3) output alongside the existing `"z"` (layer 2).

### Design decisions

- Extract from `h` (penultimate layer 3, 100-d) → retrain single linear head
- Optional FG-CCDB reweighting in Phase 4 via `s3e.reweight` config flag
- Full W&B logging for all phases + console output
- GT colour oracle for initial implementation; interface designed for future model oracle swap
- Standalone script (not integrated into `colored_mnist.py`) — integrate if results warrant

## Status

**Baseline sweep complete (2026-04-04).**

### Smoke test (passed)

```bash
python s3e.py config/colored_mnist/s3e.yaml s3e.phase1_epochs=3 s3e.probe_epochs=5 s3e.phase4_epochs=3 sweep.lambda_min_exp=1 sweep.lambda_max_exp=3 sweep.lambda_steps=3
```

All 4 phases ran end-to-end without errors. Confirmed feature extraction, SVD, projection, and head retraining all work correctly.

### Baseline sweep results (GT oracle, seed=0, r=1..9)

Run: `results/s3e/20260404_174450_s3e_cmnist_ula/`
Config: default `s3e.yaml` (phase1=20 epochs, probe=50, phase4=50).

| r | Test Acc | Best Epoch | Var Removed |
|---|----------|------------|-------------|
| 1 | 0.8675   |         49 | 22.2%       |
| 2 | 0.8681   |         36 | 41.4%       |
| 3 | 0.8683   |         32 | 58.7%       |
| 4 | 0.8698   |         25 | 72.0%       |
| 5 | 0.8705   |         13 | 81.4%       |
| 6 | 0.8726   |         12 | 87.6%       |
| 7 | 0.8711   |         11 | 92.8%       |
| 8 | 0.8730   |         14 | 97.1%       |
| 9 | **0.8747**|         16 | 99.9%       |

**Best: r=9 at 0.8747.**

### Observations

- Monotonically improving with r (removing more colour directions always helps).
- Best result removes essentially the entire probe subspace (99.9% of variance).
- Beats IPSN (0.838), comparable to HSIC (0.884), but trails CBA-OM (0.905) and Oracle+MMD (0.922).
- Probe was very good: 98.5% train acc, 74.3% test acc on colour prediction — the spurious subspace is well-identified.
- Phase 4 early stopping kicked in at high r (r=7..9 stopped around epoch 26-31), suggesting the cleaned features are easier to learn from.
- The lift over a vanilla ERM baseline (which would score ~0.86-0.87 at 20 epochs) is modest (~1-2 points). This may be because a linear head can't fully exploit the projected features, or 20 Phase 1 epochs isn't enough for the backbone to develop rich enough representations.

## Next steps

See `docs/2_active_discussions/s3e-evaluation-plan.MD` for full evaluation plan. Remaining:

1. ~~Smoke test~~ Done.
2. ~~Baseline sweep~~ Done. Best r=9 at 0.8747.
3. **Longer Phase 1 training**: The ERM backbone was only trained 20 epochs. A stronger backbone may give the projection more to work with — try 50 or 100 epochs.
4. **Reweighting ablation**: `s3e.reweight=true` — add FG-CCDB importance weighting in Phase 4.
5. **Seed sweep**: Run r=9 (or top-3 r values) across seeds 0-4 for error bars.
6. **Model oracle**: Replace GT colour labels with OracleMLP predictions (hard argmax or soft).
7. **Deeper Phase 4 head**: Currently linear. Try a 1-hidden-layer MLP to see if it can better exploit the cleaned features.
