# MMD Performance Push: Diagnosis and Improvement Plan

**Created:** 2026-04-03
**Status:** Active — Phase 3 complete. All three batch-support fixes (bank, EMA, stratified) failed to break 92% plateau. Revised diagnosis: bottleneck is not cell coverage. Regrouping.
**Goal:** Break the 0.92 Oracle+MMD plateau and push toward the ~96% FG-CCDB target on cMNIST (p_train=0.95)

---

## Context

The current best Oracle+MMD result is **0.9224** (batch=256, mmd_weight=0.5, lr=1e-3, wd=1e-4, 200 epochs). All MMD runs plateau around 0.92 regardless of hyperparameters. The theoretical framework (conditional invariance note) guarantees that weighted CE + class-conditional MMD should close the cMNIST-MNIST gap under the right assumptions. The gap between theory and practice means one or more of: (a) the representation still leaks color, (b) the representation lost digit information, or (c) the empirical estimators are too noisy.

### Why 0.92 is probably not an architecture ceiling

FG-CCDB (Zhao & Zhang, 2026) achieves **96.18%** at beta=2% and **98.21%** at beta=5% using the **same 3x100 MLP** with simple sample reweighting and no invariance penalty. CBA-OM hits 0.9047 with the same architecture. The architecture can clearly do better than 0.92 — the bottleneck is in our training procedure.

### Current implementation facts

- 10 classes, 10 colors → 100 (y,s) cells
- Batch size 256 → average 2.56 samples/cell, many cells empty per batch
- At p_train=0.95: majority p(s=y|y)=0.95, minority p(s≠y|y)=0.05/9≈0.0056
- Importance weights: w_majority ≈ 1/0.95 ≈ 1.05, w_minority ≈ 1/(0.0056) ≈ 178 (clipped to w_max=10.0)
- MMD applied to 100-d penultimate representation z (2nd hidden layer)
- Loss: (1-λ)*WeightedCE + λ*MMD, current best at λ=0.5
- Kernel: RBF with median heuristic sigma, standardized features

---

## Phase 1: Diagnostics (measure before changing anything) — COMPLETE

All diagnostics implemented in `src/training/group_eval.py` and wired into `run_train_eval_oracle_pair` in `runners.py`. Diagnostics run automatically at end of training when `s_test_gt` is provided. Functions verified with standalone tests.

### 1.1 Frozen-feature probes

**What:** At the end of training (or at saved best checkpoint), extract z for all validation/test samples. Train two probes on frozen features.

**Label probe:**
- Logistic regression (or 1-layer linear) mapping z → y_hat
- Trained on test set features (colors randomized, so no shortcut)
- Reports: accuracy, per-class accuracy

**Conditional color probe:**
- For each digit y ∈ {0,...,9}, train a separate logistic regression z → s_hat using ONLY test samples with that digit
- Average probe accuracy across digits
- Chance level = 10% (10 colors, uniform on test set)

**Interpretation matrix:**

| Label probe | Color probe | Diagnosis |
|-------------|-------------|-----------|
| High (>95%) | High (>15%) | Failure A: color still leaking, λ too weak or MMD too noisy |
| Low (<90%)  | Low (~10%)  | Failure B: over-compression, invariance too aggressive |
| Low (<90%)  | High (>15%) | Failure C: poor optimization, both objectives failing |
| High (>95%) | Low (~10%)  | Representation is good — problem is elsewhere (head, weighting, evaluation) |

**Implementation:** New function `run_frozen_probes(outer_state, x, y, s, num_classes)` in `src/training/group_eval.py`. Use `sklearn.linear_model.LogisticRegression` on CPU (small, fast). Call at end of runner or on saved checkpoint.

**Files to modify:**
- `src/training/group_eval.py` — add `run_frozen_probes`
- `src/training/runners.py` — call probes at best epoch / end of training

---

### 1.2 Per-cell accuracy breakdown

**What:** For every (y, s) cell in the test set, compute accuracy and log the full 10×10 matrix.

**Implementation:** `compute_per_cell_accuracy(y_true, y_pred, s, num_classes)` returning a [K, K] matrix where entry [y, s] = accuracy of samples with true label y and color s.

**What to look for:**
- Pattern A (common high, rare low): still shortcutting
- Pattern B (all colors similarly mediocre within a class): over-compression
- Pattern C (specific digit pairs collapse, e.g. 3/5, 4/9): invariance destroying fine shape detail

**Also log:** Per-cell confusion — for each (y, s) cell, what is the most common wrong prediction? This distinguishes "predicting by color" from "lost shape detail."

**Files to modify:**
- `src/training/group_eval.py` — add `compute_per_cell_accuracy`, `compute_per_cell_confusions`
- `src/training/runners.py` — call at eval time

---

### 1.3 Weight diagnostics

**What:** Inspect the importance weight distribution computed by `compute_mmd_weights`.

**Metrics to log:**
1. Full p(s|y) table [10×10] — reveals cell sizes
2. w_max and w_max / median(w) ratio
3. Effective Sample Size: ESS = (Σw_i)² / Σw_i² (global and per-class)
4. Per-cell average gradient contribution (if feasible)

**Current state (can compute analytically):**
- At p_train=0.95: p(s=y|y)=0.95, p(s≠y|y)≈0.0056
- q(s) = 0.1 (uniform)
- w_aligned = 0.1/0.95 ≈ 0.105
- w_minority = 0.1/0.0056 ≈ 17.9 (before smoothing)
- With w_max=10.0, minority weights are clipped → biased estimator
- With 60k samples: ~57k aligned, ~333 per minority cell (60k × 0.05/9)
- ESS estimate: dominated by the ~3000 minority samples with w=10.0

**Implementation:** Print diagnostics in runner before training starts. Log ESS per epoch.

**Files to modify:**
- `src/training/runners.py` — add weight diagnostic logging after `compute_mmd_weights`

---

### 1.4 Batch support statistics

**What:** For each training batch, count how many samples fall in each (y, s) cell.

**Metrics:**
- Minimum n_{y,s} across cells (how many cells are empty?)
- Average n_{y,s}
- Number of empty cells (out of 100)
- Number of cells with < 2 samples (MMD requires ≥ 2)

**Expected values (batch=256, p_train=0.95):**
- Expected majority cell count: 256 × 0.095 ≈ 24.3 (there are 10 majority cells)
- Expected minority cell count: 256 × 0.00056 ≈ 0.14 (there are 90 minority cells)
- Expected empty minority cells per batch: ~87 out of 90
- **This means MMD can only compare ~10-13 cells per batch** (the 10 majority cells + maybe 3 lucky minority cells)
- The all-pairs MMD over 100 cells is almost entirely masked out

This is likely a primary failure mode. The MMD penalty barely sees any minority-color data per batch.

**Implementation:** Log batch statistics inside epoch function or as a diagnostic pass.

**Files to modify:**
- `src/training/epochs.py` or separate diagnostic function

---

### 1.5 Offline invariance metrics (full validation set)

**What:** After each epoch, compute stable full-dataset invariance metrics (not the noisy per-batch MMD).

**Metrics:**
1. **Mean mismatch:** D_mean = (1/|Y||S|) Σ_{y,s} ||μ_{y,s} - μ_y||² where μ_{y,s} is the cell mean and μ_y is the class mean of z
2. **Covariance mismatch:** D_cov = (1/|Y||S|) Σ_{y,s} ||Σ_{y,s} - Σ_y||_F² (optional, compute if cheap enough)
3. **Full-dataset MMD:** class_cond_mmd_rbf on the entire test set (not batched) — the "true" value of what we're optimizing

**Implementation:** New function `compute_offline_invariance(z, y, s, num_classes)`. Call on test features after each epoch (or every N epochs).

**Files to modify:**
- `src/training/group_eval.py` — add `compute_offline_invariance`
- `src/training/runners.py` — call periodically

---

### 1.6 Gradient conflict diagnostic

**What:** Every N steps, separately compute ∇L_cls and ∇L_inv, measure cosine similarity.

**Implementation:** Inside the train step, compute two separate `jax.grad` calls (one for weighted CE, one for MMD), flatten params, compute cos(g_cls, g_inv). Log as a metric.

**Cost:** Doubles backward pass cost. Run every ~50 steps, not every step.

**Interpretation:**
- cos > 0: objectives cooperate
- cos ≈ 0: objectives independent
- cos < 0: objectives fight — suggests λ warmup needed

**Files to modify:**
- `src/training/steps.py` — add diagnostic variant of train step (or a separate diagnostic function)
- Log to metrics dict

---

## Phase 2: Ablation suite (same encoder, same everything, only change loss)

Run all ablations with **identical** settings: UlaMlp outer, frozen oracle inner, batch=256, lr=1e-3, wd=1e-4, 200 epochs, 3 seeds each.

### Ablation A: Plain CE (no weighting, no MMD)
```
L = CE(logits, y)
```
Shortcut baseline. Expected: ~10% test acc (model learns color).

### Ablation B: Weighted CE only (no MMD)
```
L = WeightedCE(logits, y, w)
```
Isolates reweighting effect. If this is already much worse than plain CE on train acc, weights are too noisy.

### Ablation C: Unweighted CE + MMD
```
L = (1-λ)*CE(logits, y) + λ*MMD(z, y, s)
```
Isolates MMD effect without weight noise.

### Ablation D: Full method (current)
```
L = (1-λ)*WeightedCE(logits, y, w) + λ*MMD(z, y, s)
```
Current best config.

**Interpretation:**
- B bad → weight variance is a/the main problem
- C bad → MMD estimator/strength is a/the main problem
- B and C okay but D bad → interaction/gradient conflict is the problem

**Implementation:** All 4 configs created. All use `outer_loss: "mmd"` code path for consistency. Two binary factors: `mmd.uniform_weights` (on/off) × `mmd.weight` (0.0/0.5). Added `uniform_weights: bool` field to `MMDConfig` and guard in runner.

| Config | `mmd.weight` | `mmd.uniform_weights` | Loss |
|--------|-------------|----------------------|------|
| A | 0.0 | true | Plain CE |
| B | 0.0 | false | Weighted CE |
| C | 0.5 | true | CE + MMD |
| D | 0.5 | false | Weighted CE + MMD |

All share: batch=256, lr=1e-3, wd=1e-4, 200 epochs, seed=0. Run each with seeds {0, 1, 2} by overriding `training.seed`.

**Files created:**
- `config/colored_mnist/ablation_a_plain_ce.yaml`
- `config/colored_mnist/ablation_b_weighted_ce.yaml`
- `config/colored_mnist/ablation_c_ce_mmd.yaml`
- `config/colored_mnist/ablation_d_full.yaml`

**Code changes for ablation support:**
- `src/config.py` — added `uniform_weights: bool = False` to `MMDConfig`
- `src/training/runners.py` — when `uniform_weights=True`, sets `w_train = jnp.ones(...)`

### Ablation results (seed=0)

| Ablation | Loss | Best Test Acc | Label Probe | D_mean | ESS% |
|----------|------|--------------|-------------|--------|------|
| A | Plain CE | 85.92% | 92.3% | 77.5 | 100% |
| B | Weighted CE | 91.55% | 93.3% | 15.6 | 7.15% |
| C | CE + MMD | 86.33% | 88.7% | 59.4 | 100% |
| D | Weighted CE + MMD | 92.06% | 93.6% | 10.8 | 7.15% |

**Conclusion:** Weighting drives almost all gains (+5.6pp). MMD alone adds +0.4pp and actually degrades the label probe. MMD on top of weighting adds only +0.5pp. With 79/100 cells empty per batch, the MMD penalty is near-vacuous. **Batch support is the primary bottleneck.**

### Ablation interpretation

The 2×2 factor design cleanly isolates contributions:

- **Weighting effect** (A→B): +5.6pp. Importance reweighting alone gets to 91.5%.
- **MMD effect** (A→C): +0.4pp. MMD without weighting is nearly useless — it can't enforce invariance when ~79/100 cells are empty per batch. Worse, it degrades the label probe from 92.3% to 88.7% (over-regularising on a near-vacuous signal).
- **MMD on top of weighting** (B→D): +0.5pp. Marginal improvement. The 0.92 plateau is essentially the ceiling of importance weighting with w_max=10 and ESS=7%.
- **D_mean progression**: 77.5 → 59.4 (MMD alone barely helps) → 15.6 (weighting) → 10.8 (full). Weighting reduces color leakage far more than MMD does.

---

## Phase 3: Fixes (implement based on diagnostic findings)

Each fix keeps the same theoretical objective but improves the empirical estimator.

### 3.1 Balanced (y,s) batch sampling

**Replaces:** Raw importance weighting with a lower-variance estimator of the same debiased risk.

**How:** Build each batch by drawing equal samples from each (y,s) cell that has data.

**Concrete design for 10×10 cMNIST:**
- 100 cells, but 90 minority cells have very few samples (~333 each at p_train=0.95)
- Option A (full balance): Draw k samples from each of the 100 cells. With k=2, batch=200. Many minority cells have enough (333 samples). Shuffle within batch, use unweighted CE.
- Option B (hybrid): Draw k_maj from majority cells, k_min from minority cells. E.g. k_maj=16, k_min=4 → batch = 10×16 + 90×4 = 520. Apply residual weights to correct for remaining imbalance.
- Option C (stratified by class, uniform within class): For each class y, draw samples with replacement weighted by 1/p(s|y). Effective sample = uniform over colors within class. Batch = 10 × B_per_class.

**Recommended:** Option A with k=2, batch=200 (or k=3, batch=300). Simple, clean, exact target distribution within each batch. Unweighted CE inside the batch.

**Files to modify:**
- `src/datasets/datasets.py` — add `make_balanced_epoch_batches(x, y, s, samples_per_cell, seed)`
- `src/training/runners.py` — use balanced batches when enabled
- `src/config.py` — add `balanced_batches: bool = False` and `samples_per_cell: int = 2`

---

### 3.2 Queue-based mean/covariance matching (replace minibatch MMD)

**Replaces:** Noisy per-batch pairwise kernel MMD with a stable cross-batch moment-matching penalty.

**How:** Maintain a FIFO queue of recent (z, y, s) tuples for each (y,s) cell. Compute cell means and (optionally) covariances from the queue. Penalize deviation from class mean.

**Loss:**
```
L_inv = Σ_y Σ_s ||μ_{y,s} - μ_y||² + β Σ_y Σ_s ||Σ_{y,s} - Σ_y||_F²
```

**Queue design:**
- Queue length Q per cell (e.g., Q=512)
- Updated every step: push new batch features, pop oldest
- Means/covariances computed from queue, not just current batch
- Queue stored as JAX arrays: `queue_z[K, K, Q, D]`, `queue_count[K, K]`
- For 10×10×512×100 = 51.2M floats ≈ 200MB — feasible

**Alternative (simpler, likely sufficient):** Exponential moving average (EMA) of cell means.
```
μ_{y,s} ← α * μ_{y,s} + (1-α) * μ_{y,s}^{batch}
```
With α=0.99. Much less memory. Only tracks first moments.

**Recommended:** Start with EMA mean matching (simplest). Add covariance if mean-only is insufficient.

**Files to modify:**
- `src/loss_fns/reg_loss_fns.py` — add `mean_matching_loss(cell_means, class_means, cell_counts)` and `cov_matching_loss(...)`
- `src/training/steps.py` — add train step variant that uses mean matching instead of MMD
- `src/training/epochs.py` — add epoch function that maintains/updates EMA state
- `src/config.py` — add `InvarianceConfig(mode="mmd"|"mean"|"mean_cov", ema_alpha=0.99, beta_cov=0.1)`

---

### 3.3 Lambda warmup and ramp

**Replaces:** Constant λ from epoch 0 with a schedule that lets the encoder learn digit structure before invariance pressure is applied.

**Schedule:**
```
epoch 0-W:  λ_eff = 0          (warmup, classification only)
epoch W-R:  λ_eff = λ * (ep - W) / (R - W)  (linear ramp)
epoch R+:   λ_eff = λ          (full invariance)
```

**Suggested defaults:** W=10, R=30 (out of 200 total epochs). These can also be adaptive — start ramp when label probe accuracy exceeds a threshold.

**Files to modify:**
- `src/config.py` — add `warmup_epochs: int = 0`, `ramp_epochs: int = 0` to MMDConfig
- `src/training/runners.py` — compute `lambda_eff` per epoch and pass to train step

---

### 3.4 Invariance stopping / adaptive λ

**What:** Stop increasing invariance pressure once the conditional color probe is near chance (~10%). Continuing to push invariance below the finite-sample noise floor destroys digit information.

**Implementation:** After each probe evaluation, if color_probe_acc < threshold (e.g., 12%), freeze λ_eff at current value or set to 0.

**Files to modify:**
- `src/training/runners.py` — conditional λ adjustment based on probe results

---

### Phase 3 design discussion (2026-04-04)

**Key constraint:** The fix must work at p_train=0.995, not just 0.95. At p_train=0.995, each minority cell has only ~33 samples (60k × 0.005/9). Balanced batch sampling (§3.1) breaks down — drawing k=2 per cell exhausts minority data in ~16 epochs. We need a method that doesn't require seeing every cell in every batch.

**How CCDB and FG-CCDB solve this (for comparison):**

Neither uses cross-batch accumulation. Both are offline multi-stage pipelines:

- **CCDB** (Zhao, Li & Zhang, 2026, CVPR, arXiv:2504.17314): Stage 1 trains a biased feature extractor with intra-class compactness. Stage 2 extracts all features, computes global μ̄ and Σ̄ on the full training set, then optimises per-sample weights to minimise W₂(p(ζ|y,w) || p(ζ)) across classes. Stage 3 retrains with those weights. The batch support problem is avoided because weight optimisation happens offline on the full dataset.

- **FG-CCDB** (Zhao & Zhang, 2025, NeurIPS, arXiv:2505.06831): Replaces CCDB's per-sample Gaussian approximation with group-level counts. Closed-form weights: W_{i,j} = q_i / P_{i,j}, then w_{i,j} = W_{i,j} / N_{i,j}. Instant computation, no optimisation. Also three-stage.

Both methods fundamentally require a separate biased model to partition groups and a retrain step. Our oracle_pair setup already has the oracle inner model providing s predictions, so the BEO stage is unnecessary — but we'd still need to decide whether to adopt their offline reweighting or keep an end-to-end single loop.

**Three candidate approaches for our setup:**

**Option 1: CCDB/FG-CCDB-style offline reweighting (multi-stage)**
- Use our existing oracle to predict s for all training samples (already done).
- Compute FG-CCDB closed-form group weights: W_{i,j} = q_i / P_{i,j}.
- Retrain outer model with weighted sampling (WeightedRandomSampler).
- Pro: Simple, proven, closed-form. Con: Two-pass pipeline, doesn't use an invariance penalty at all — the conditional invariance comes entirely from the reweighting. Also our current importance weights are already the same formula (w = q(s) / p(s|y)), just applied per-sample rather than per-group — the difference is FG-CCDB uses WeightedRandomSampler instead of per-sample loss weighting.

**Option 2: EMA mean matching (single-stage, online)**
- Keep the end-to-end training loop.
- Replace per-batch kernel MMD with an EMA of per-cell feature means: μ_{y,s} ← α·μ_{y,s} + (1-α)·μ_{y,s}^{batch} for cells present in the current batch.
- Invariance penalty: L_inv = (1/|Y|) Σ_y Σ_s ||μ_{y,s}^{EMA} - μ_y^{EMA}||².
- z is the 100-d penultimate representation (aux["z"] from the outer model).
- Pro: Single training loop, accumulates signal across batches so even rare cells build up reliable means over many epochs. Works at p_train=0.995 since a cell with 33 samples will be seen ~33 times per epoch, updating the EMA each time. Con: First-moment only (no covariance matching). EMA introduces a lag — penalty reflects slightly stale representations. Gradients flow through the current batch's contribution to the EMA, not the full EMA.
- Note: D_mean (our diagnostic) already measures exactly this quantity. We'd be turning it into a loss term.

**Option 3: Hybrid — FG-CCDB weights + EMA mean matching**
- Use FG-CCDB closed-form group weights for the CE term (replaces our clipped importance weights, giving exact debiasing without w_max truncation).
- Add EMA mean matching as an explicit invariance penalty.
- Pro: Gets exact CE debiasing (no clipping) plus an invariance signal that accumulates across batches. Con: Most complex option.

**Open question:** Is the per-batch kernel MMD salvageable at all, or should we abandon it entirely in favour of one of these?

**Decision pending.**

---

### Phase 3 experiments (2026-04-04)

#### Experiment 3a: Stop-gradient FIFO memory bank for kernel MMD

**Idea:** Maintain a FIFO queue of representations per (y,s) cell. At MMD time, concatenate bank z (stop-gradient) with batch z to augment cell support. Three variants tested:

**Bank v1 — store z directly:**
- Bank stores z representations, updates via FIFO scatter.
- Result: D_mean exploded to 85.6 (vs 10.8 baseline). Bank z becomes stale as encoder changes.
- **Abandoned.**

**Bank v2 — store x, recompute z per epoch:**
- Bank stores raw x. At epoch start, forward-pass all bank x through current encoder to get fresh bank_z.
- Result: D_mean exploded to 524.7 over 200 epochs. Bank z goes stale within the epoch as encoder is updated.
- **Abandoned.**

**Bank v3 — store x, recompute z every N steps:**
- Added `bank_refresh_steps` parameter. Python loop over JIT step fn with periodic bank z refresh.
- Configs tested: Q=8 with N=1 (every step), batch_size=512.
- Result: Best 91.4% (vs 91.0% kernel MMD baseline). D_mean=40 (vs 6.7 baseline) — actually worse.
- **Fundamental limitation:** Gradients only flow through the batch z, not the bank z (stop-gradient). The bank improves the MMD *estimate* but doesn't improve the *gradient signal* — only cells present in the current batch get gradient, regardless of bank contents.
- **Abandoned.**

**Implementation files:** `init_mmd_bank`, `update_mmd_bank` in `reg_loss_fns.py`; `make_train_step_oracle_pair_mmd_banked` in `steps.py`; `make_train_epoch_pair_mmd_banked` in `epochs.py`; wired through `runners.py` and `colored_mnist.py`. Config: `bank_size`, `bank_refresh_steps` in MMDConfig.

#### Experiment 3b: EMA mean matching

**Idea:** Replace kernel MMD with a first-moment penalty. Maintain exponential moving averages of per-cell feature means. Penalize deviation of each cell mean from the class mean (averaged over all colors). Gradient flows through current batch's cell means toward a stop-gradient EMA class mean target.

**Implementation:** `init_ema_state`, `mean_match_loss` in `reg_loss_fns.py`; `make_train_step_oracle_pair_mean_match` in `steps.py`; `make_train_epoch_pair_mean_match` in `epochs.py`; wired through `runners.py` and `colored_mnist.py`. Config: `mode: "mean_match"`, `ema_alpha` in MMDConfig.

**Results:**

| mmd.weight | ema_alpha | Best Test Acc | Final Test Acc | D_mean | Notes |
|-----------|-----------|--------------|----------------|--------|-------|
| 0.5 | 0.99 | **89.4% (ep 18)** | 84.5% | 0.005 | Overfits after penalty collapses |
| 1.0 | 0.9 | 11.2% | 10.1% | 0.000004 | Collapsed to chance — invariance dominates |
| 1.0 | 0.95 | 10.2% | 9.9% | 0.000001 | Collapsed to chance |
| 2.0 | 0.9 | 8.9% | 8.9% | 380.2 | Diverged — negative CE coeff in (1-w)*CE + w*MM |
| 2.0 | 0.95 | 8.9% | 8.9% | 286.2 | Diverged |

**Diagnosis:** Mean matching is fundamentally weaker than kernel MMD. It only constrains first moments — once cell means match the class mean (which happens fast, by epoch ~15), the penalty goes to near-zero and the model overfits. Kernel MMD constrains higher-order moments through the kernel and provides ongoing regularization. The best mean match result (89.4%) is *worse* than kernel MMD baseline (91.0%).

The EMA cross-batch accumulation works as designed (D_mean is excellent at 0.005), but a first-moment penalty is too easy to satisfy while still allowing color information in higher moments.

**Possible extensions (not yet tried):**
1. Add variance matching (second moment) — more like a moment-based surrogate for MMD
2. Use additive loss form CE + λ*MM instead of convex combination (1-λ)*CE + λ*MM
3. Early stopping to capture the peak at epoch 18 (but 89.4% is still worse than 91%)

#### Summary of Phase 3 findings

Both cross-batch approaches failed to beat the kernel MMD baseline:
- **Memory bank:** Correct MMD estimate but gradient only through batch → no improvement.
- **EMA mean matching:** Good cross-batch accumulation but first-moment penalty too weak → overfits.

#### Experiment 3c: Stratified minibatch sampling (2026-04-13)

**Idea:** Before training, index all samples by (y,s) cell. For each minibatch, guarantee 1 sample from each cell, fill remaining slots randomly. This gives the kernel MMD full cell support (0 empty cells) every step.

**Implementation:** `make_stratified_epoch_batches` in `datasets.py`. Config: `mmd.stratified: true`. For each of the K²=100 populated cells, draws 1 sample (with replacement across batches), then fills remaining `batch_size - n_cells` slots randomly from the full dataset.

**Results (seed=0, batch=256):**

| Variant | Epochs | Best Test Acc | Best Epoch | Color Probe | D_mean |
|---------|--------|--------------|------------|-------------|--------|
| Stratified + uniform weights | 200 | 90.8% | 65 | 0.435 | 3.23 |
| Stratified + importance weights | 500 | **91.85%** | 60 | 0.316 | 22.2 |
| Ablation D baseline (random batch + weights) | 200 | 92.06% | 112 | 0.360 | 10.8 |

**Progression (stratified + importance weights, 500 epochs):**

| Epoch | Test Acc | MMD loss |
|-------|----------|----------|
| 50 | 91.53% | 0.036 |
| 60 | **91.85%** | — |
| 100 | 90.99% | 0.015 |
| 200 | 90.12% | 0.008 |
| 300 | 89.68% | 0.006 |
| 500 | 86.93% | 0.018 |

**Diagnosis:** Stratified batching did not break the plateau. Without importance weights it's worse (90.8%) because the CE gradient is dominated by the ~156 random-fill majority samples. With importance weights it peaks at 91.85% — slightly below baseline (92.06%). The MMD loss still collapses quickly (~0.015 by epoch 100), then the model overfits with no regularization pressure. Full cell coverage helps the MMD *estimate* but doesn't prevent the penalty from becoming vacuous once means roughly align.

**Key insight from Phase 3 overall:** The ~92% plateau is not primarily a batch support problem. Three different fixes for cell coverage (memory bank, EMA accumulation, stratified batching) all failed to break past it. The bottleneck appears to be deeper: either (a) the RBF kernel MMD penalty is too easily satisfied once first moments approximately match, or (b) importance weighting with w_max clipping is the binding constraint, or (c) the 3×100 MLP overfits before the invariance penalty can drive sufficient generalization.

**Remaining options to explore:**
1. **Variance matching** alongside mean matching (second-moment constraint)
2. **FG-CCDB offline pipeline** — closed-form group weights, no invariance penalty at all. Proven to hit 96%.
3. **Stronger regularization** — higher weight decay, dropout, or learning rate scheduling to control overfitting
4. **Kernel choice / bandwidth** — the RBF median heuristic may give a bandwidth that makes MMD insensitive to higher-order distribution differences

---

## Phase 4: Experiment matrix

After implementing diagnostics (Phase 1) and fixes (Phase 3), run the following models. Same encoder, optimizer, training length, 3 seeds each.

| # | Model | Batching | CE weighting | Invariance | λ schedule | Key question |
|---|-------|----------|-------------|------------|------------|--------------|
| 1 | Plain CE | random | none | none | — | Shortcut baseline |
| 2 | Weighted CE only | random | importance | none | — | Is reweighting alone useful? |
| 3 | Balanced-batch CE | balanced | none (implicit) | none | — | Is balanced batching better than weighting? |
| 4 | Current best | random | importance | pairwise MMD | constant | Reproduce 0.9224 |
| 5 | Balanced + MMD | balanced | none | pairwise MMD | constant | Does balanced batching help MMD? |
| 6 | Balanced + mean match | balanced | none | EMA mean match | constant | Does mean matching beat minibatch MMD? |
| 7 | Balanced + mean match + warmup | balanced | none | EMA mean match | warmup+ramp | Does warmup help? |
| 8 | Balanced + mean/cov match + warmup | balanced | none | EMA mean+cov | warmup+ramp | Does covariance matching add value? |

**Per-model logging (all from Phase 1):**
1. Test accuracy (best epoch, final epoch)
2. Per-cell accuracy matrix [10×10]
3. Label probe accuracy
4. Conditional color probe accuracy (per-digit average)
5. Offline D_mean (full test set mean mismatch)
6. Batch support statistics
7. ESS (for weighted models)
8. Gradient cosine (for models with invariance term)

---

## Implementation order

```
Step 1: Frozen-feature probes (§1.1)              — most informative single diagnostic
Step 2: Per-cell accuracy (§1.2)                   — cheap, immediately actionable
Step 3: Weight + batch diagnostics (§1.3, §1.4)    — confirms the noise hypothesis
Step 4: Offline invariance metrics (§1.5)           — validates whether MMD is actually working
Step 5: Run ablation suite (Phase 2)                — localizes the failure mode
Step 6: Balanced batching (§3.1)                    — likely biggest single fix
Step 7: Mean matching loss (§3.2)                   — replaces noisy MMD estimator
Step 8: Lambda warmup (§3.3)                        — optimization improvement
Step 9: Full experiment matrix (Phase 4)            — systematic comparison
Step 10: Gradient conflict (§1.6)                   — run if needed for debugging
```

---

## Current hypothesis (to be confirmed or refuted by diagnostics)

Based on the analysis in mmd-perf-push.md and the batch arithmetic:

1. **Batch MMD is nearly vacuous.** With 100 cells and batch=256, ~87 minority cells are empty per batch. The MMD penalty only sees majority-color examples, so it enforces invariance among colors that are already well-represented — not where it's needed.

2. **Importance weights are clipped and biased.** Minority weights should be ~178 but are clipped to 10.0. This means the debiased CE under-weights minority samples by ~18x relative to the theoretical target. The weighted estimator is both high-variance (large weights) and biased (clipping).

3. **The 0.92 plateau is the performance achievable from the majority cells alone.** Since ~95% of training data is bias-aligned and the MMD mostly operates on those, the model effectively learns to classify the 95% well but hasn't been pushed to handle the other 5%.

4. **FG-CCDB's success with the same architecture confirms this.** FG-CCDB uses exact group-level reweighting (closed-form W_{i,j} = q_i / P_{i,j}) which correctly upweights every minority cell. No invariance penalty needed — just proper weighting. This suggests our method's primary bottleneck is estimation quality, not the objective.

---

## Change log

- 2026-04-03: Initial plan created from mmd-perf-push.md analysis
- 2026-04-04: Phase 2 ablation results added. Confirmed batch support as primary bottleneck.
- 2026-04-04: Phase 3 — memory bank (3 variants) tested and abandoned. Gradient-only-through-batch limitation.
- 2026-04-04: Phase 3 — EMA mean matching tested. Best 89.4% (worse than kernel MMD 91%). First-moment penalty too weak, collapses and overfits. Both cross-batch approaches failed to beat baseline.
- 2026-04-13: Phase 3 — Stratified minibatch sampling tested. Best 91.85% (slightly below baseline 92.06%). Full cell coverage doesn't break the plateau. Revised diagnosis: bottleneck is not batch support alone.
