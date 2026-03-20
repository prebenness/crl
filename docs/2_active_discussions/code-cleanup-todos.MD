# Code Cleanup & Engineering TODOs

Updated 2026-03-19. Ordered by estimated effort (smallest first). Research-roadmap items are excluded — see `docs/3_future_roadmap/research-roadmap.MD` for those.

---

## Done (this session)

- ~~Unify φ initialization in ANBN shared mode~~ — `shared_weights.py` now uses P_base-informed init, matching cMNIST path.
- ~~Rename `mdl_lambda` → `lambda1` in cMNIST shared paths~~ — `_mdl_shared_loss`, `make_train_step_mdl_shared`, `make_train_step_mdl_shared_pair` in `steps.py`.

---

## Outstanding

### XS — < 30 min

**1. Entropy variable naming (docs only)**
- Ensure any future docstrings/comments refer to "entropy bonus (subtracted)" rather than mixing `−(1/β) H` and `τ * H` framings.
- No code change needed — just a convention note for future work.

**2. Short-circuit `n_samples` when `mode_forward=True`**
- Source: quantization session log (P1, Change 6).
- When mode_forward is on, forward pass is deterministic — only one sample needed. Short-circuit in `_mdl_loss` and `_mdl_shared_loss` (~20 lines).

**3. Vectorize `_compute_discrete_hyp_bits`**
- Source: `engineering-roadmap.MD` item 2.
- Replace Python loops over discrete weights with lookup-based indexing. Low risk, modest runtime improvement.

### S — 1–2 hours

**4. Use full fixed test sets during evaluation**
- Source: `engineering-roadmap.MD` item 3.
- Stop dropping remainder batches. Keep test iteration deterministic and complete.

**5. Improve train tail-batch handling**
- Source: `engineering-roadmap.MD` item 4.
- Stop dropping remainder samples. Either mask the tail batch or rotate offsets across epochs.

**6. Clarify `p_test` semantics for `cmnist_ula`**
- Source: `engineering-roadmap.MD` item 5.
- Either ignore it explicitly with a warning or wire it into alternate dataset builders.

**7. Enforce or truly support `rep_dim` width configuration**
- Source: `engineering-roadmap.MD` item 6.
- `ULAMLPClassifier` should either require `rep_dim == 100` or use the config value consistently.

**8. Unify batch scaling convention (ANBN vs cMNIST)**
- Source: original code-cleanup-todos item 3.
- ANBN uses `B/N` scaling on reg terms with summed data NLL; cMNIST uses averaged data NLL with `1/N` scaling. Both equivalent, but confusing. Low priority — align to cMNIST convention if touching these files.

### M — half day

**9. Improve shared-MDL reporting semantics**
- Source: `engineering-roadmap.MD` item 7.
- In shared mode, report both discrete `|H|` and shared-objective compression terms clearly. Avoid implying a single `|H|` fully describes the shared objective.

**10. Add best-epoch selection for outer accuracy (cMNIST)**
- Source: `colored-mnist-current-status.MD` next-work item 1.
- Peak-vs-final behaviour is a bottleneck. Save and select best outer checkpoints during training.

**11. Resume interrupted ANBN `mode_forward` experiments**
- Source: `20260315-experiment-progress-note.MD`.
- Two runs saved mid-warmup (epoch 500 and epoch 2500). Need resuming and evaluation.

### L — 1+ day

**12. `lax.fori_loop` epoch fusion for full-batch ANBN training**
- Source: `engineering-roadmap.MD` item 1.
- Reduce Python overhead in long ANBN runs. Keep chunk boundaries aligned with logging/eval.

**13. Staged temperature annealing with discretization checkpoints (P2)**
- Source: quantization session log (Change 3, ~150 lines across 6 files).
- Fixed-temperature stages with freeze masks for converged weights. Gated by `anneal_mode: "staged"` config.

**14. DiffQ-style continuous relaxation for cMNIST (P2)**
- Source: quantization session log (Change 4, new module ~200 lines).
- Replaces full M-dimensional categorical per weight with continuous noise + ST rounding. Addresses 38M-float memory bottleneck on cMNIST.

### XL — multi-day, defer

**15. RELAX estimator (P3)**
- Source: quantization session log (Change 7, ~300 lines).
- Learned control variate for unbiased low-variance gradients. Only needed if P0-P2 still show problematic gradient variance.
