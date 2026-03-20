# Code Cleanup & Engineering TODOs

Updated 2026-03-20. Ordered by estimated effort (smallest first). Research-roadmap items are excluded — see `docs/3_future_roadmap/research-roadmap.MD` for those.

---

## Done

### Session 2026-03-19
- ~~Unify φ initialization in ANBN shared mode~~ — `shared_weights.py` now uses P_base-informed init, matching cMNIST path.
- ~~Rename `mdl_lambda` → `lambda1` in cMNIST shared paths~~ — `_mdl_shared_loss`, `make_train_step_mdl_shared`, `make_train_step_mdl_shared_pair` in `steps.py`.

### Session 2026-03-20
- ~~Entropy variable naming~~ — All docstrings/comments now consistently use "entropy bonus (subtracted): tau * H(pi)" framing.
- ~~Short-circuit `n_samples` for `mode_forward`~~ — All 4 cMNIST step factories and both ANBN train-step creation sites now set `n_samples=1` when `mode_forward=True`.
- ~~Vectorize `_compute_discrete_hyp_bits`~~ — Replaced Python loop with `grid_codelengths[idx]` lookup.
- ~~Full fixed test sets during evaluation~~ — New `make_eval_batches` pads last batch + returns counts. `make_eval_epoch` weights by counts. All 4 cMNIST runners updated.
- ~~Train tail-batch handling~~ — Documented in `make_epoch_batches`: random permutation means different samples dropped each epoch; acceptable for training.
- ~~Clarify `p_test` for `cmnist_ula`~~ — Warning emitted when `p_test != 0.1` since `CMNISTuLA` always uses unbiased test per uLA protocol.
- ~~Enforce `rep_dim` for ULA~~ — `ULAMLPClassifier` now warns if `rep_dim != 100`; hidden widths fixed at 100 for paper compatibility.
- ~~Unify batch scaling convention~~ — ANBN now uses averaged data NLL + `1/N` reg scaling, matching cMNIST. **Note: existing ANBN hyperparameters (lambda, lr) will need recalibration.**

---

## Outstanding

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
