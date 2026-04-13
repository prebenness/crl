# S3E Evaluation Plan

## Method

S3E (Spectral Spurious Subspace Elimination) removes spurious colour information from ERM-learned representations via spectral identification and orthogonal projection. 4-phase pipeline: ERM backbone -> colour probe + SVD -> orthogonal projection -> retrain linear head.

Inspired by concept erasure (Ravfogel et al., 2020, "Null it out", ACL; Ravfogel et al., 2022, "Linear adversarial concept erasure", ICML).

## Evaluation Runs (post smoke test)

### 1. Baseline Sweep (GT oracle, r=1..9)

```bash
python s3e.py config/colored_mnist/s3e.yaml
```

Default config: phase1_epochs=20, probe_epochs=50, phase4_epochs=50. ~10-15 min total.

**Expected diagnostics:**
- Phase 1 ERM: ~95% train acc, ~10% test acc (colour shortcut on biased data)
- Phase 2 probe: ~95% colour prediction accuracy (colour is linearly predictable from ERM features)
- SVD: clear drop-off in singular values (first few capture most colour variance)
- Sweep: test_acc should increase with r (more colour removed), plateau, then possibly drop at high r (digit info lost)

### 2. Compare Against Current Best

| Method | Test Acc |
|--------|----------|
| Oracle + MMD | 0.9224 |
| CBA-OM | 0.9047 |
| VIB Pair | 0.8846 |
| Oracle + HSIC | 0.884 |
| IPSN | 0.838 |
| **S3E (best r)** | **???** |

### 3. Reweighting Ablation

```bash
python s3e.py config/colored_mnist/s3e.yaml s3e.reweight=true
```

Does FG-CCDB importance reweighting during Phase 4 help on top of the projection?

### 4. Seed Sweep

Run best r with 5 seeds to get mean +/- std:

```bash
for seed in 0 1 2 3 4; do
  python s3e.py config/colored_mnist/s3e.yaml training.seed=$seed sweep.lambda_min_exp=BEST_R sweep.lambda_max_exp=BEST_R sweep.lambda_steps=1
done
```

### 5. Model Oracle (Phase 2)

Replace GT colours with trained OracleMLP checkpoint. Compare:
- Hard labels (argmax of oracle predictions)
- Soft labels (full probability distribution — requires soft-target CE in probe, future work)

```bash
python s3e.py config/colored_mnist/s3e.yaml model.oracle_checkpoint=results/.../best.npz
```

(Not yet implemented — requires adding oracle model loading to s3e.py)
