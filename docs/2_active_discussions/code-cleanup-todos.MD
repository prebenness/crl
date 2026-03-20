# Code Cleanup & Engineering TODOs

Updated 2026-03-20. Ordered by estimated effort (smallest first). Research-roadmap items are excluded — see `docs/3_future_roadmap/research-roadmap.MD` for those.

---

### M — half day

**1. Improve shared-MDL reporting semantics**
- Source: former `engineering-roadmap.MD` item 7.
- In shared mode, report both discrete `|H|` and shared-objective compression terms clearly. Avoid implying a single `|H|` fully describes the shared objective.

**2. Add best-epoch selection for outer accuracy (cMNIST)**
- Source: `colored-mnist-current-status.MD` next-work item 1.
- Peak-vs-final behaviour is a bottleneck. Save and select best outer checkpoints during training.

**3. Resume interrupted ANBN `mode_forward` experiments**
- Source: `20260315-experiment-progress-note.MD`.
- Two runs saved mid-warmup (epoch 500 and epoch 2500). Need resuming and evaluation.

### L — 1+ day

**4. `lax.fori_loop` epoch fusion for full-batch ANBN training**
- Source: former `engineering-roadmap.MD` item 1.
- Reduce Python overhead in long ANBN runs. Keep chunk boundaries aligned with logging/eval.

**5. Staged temperature annealing with discretization checkpoints (P2)**
- Source: quantization session log (Change 3, ~150 lines across 6 files).
- Fixed-temperature stages with freeze masks for converged weights. Gated by `anneal_mode: "staged"` config.

**6. DiffQ-style continuous relaxation for cMNIST (P2)**
- Source: quantization session log (Change 4, new module ~200 lines).
- Replaces full M-dimensional categorical per weight with continuous noise + ST rounding. Addresses 38M-float memory bottleneck on cMNIST.

### XL — multi-day, defer

**7. RELAX estimator (P3)**
- Source: quantization session log (Change 7, ~300 lines).
- Learned control variate for unbiased low-variance gradients. Only needed if P0-P2 still show problematic gradient variance.
