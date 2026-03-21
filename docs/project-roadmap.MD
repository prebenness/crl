# Project Roadmap

Consolidated from `code-cleanup-todos.MD`, `research-roadmap.MD`, and `experiment-and-metrics-roadmap.MD`. Updated 2026-03-21.

**Reference papers:**
- Lan et al. (2024), "MDL Regularization of LSTM Language Models for Formal Language Learning", ACL 2024: `agent-context/arxiv/arXiv-2402.10013v2_contents/main-acl.tex`
- Abudy et al. (2025), "Learning Formal Languages with Small RNNs Using MDL Regularization", NeurIPS 2025 submission: `agent-context/arxiv/arXiv-2505.13398v2_contents/main.tex`
- Lan, Chemla & Katzir (2022), "Minimum description length recurrent neural networks", TACL 10:785–799: `agent-context/arxiv/arXiv-2111.00600v4_contents/main-3649-Lan.tex`
- Weiss, Goldberg & Yahav (2018), "On the Practical Computational Power of Finite Precision RNNs for Language Recognition", ACL 2018 Short: `agent-context/arxiv/arXiv-1805.04908v1_contents/`

**Fair comparison analysis:** `docs/comparisons.md` — detailed comparison targets, golden network discrepancies, threats to validity, and implementation tasks for apples-to-apples reporting.

---

## Section 1: Code Fixes & Optimization

Code cleanup, notation fixes, and performance optimization of existing functionality.

### ~~1.1 Improve shared-MDL reporting semantics~~ — Done (2026-03-20)

Shared training log, validation print, and final evaluation now show both `|H|_Lan` (Lan-style discrete codelength) and `shared_complexity` (lambda1 * code_ce + lambda2 * kl_dict). New helper `_compute_shared_discrete_complexity_bits` in `differentiable_mdl.py`.

### ~~1.2 `lax.fori_loop` epoch fusion for full-batch ANBN training~~ — Done (2026-03-21)

Implemented `lax.scan`-based epoch fusion in `src/mdl/training.py`. For full-batch training (bs >= N), contiguous same-phase epoch runs are fused into single `lax.scan` calls, with chunk boundaries at log/eval intervals and phase transitions. New functions: `anneal_tau_traceable` (JAX-traceable tau schedule using `jnp.where`), `make_fused_epoch_fn` (creates JIT'd scan runner). `make_train_step` gains `jit=False` option for use inside scan. Mini-batch fallback preserved via `_run_epoch`. ~2x speedup on debug runs.

### ~~1.3 Verify smoothing convention for evaluation~~ — Done (2026-03-20)

Training keeps `log_softmax` (optax, numerically stable). New `compute_data_nll_bits_smoothed` in `src/mdl/training.py` applies Abudy et al. (2025) convention (additive 1e-10, no re-normalization) for |D:H| evaluation. 3 tests added in `test_mdl.py`.

### ~~1.4 Fix n=0 inclusion in grammar-weighted test evaluation~~ — Done (2026-03-21)

Changed `compute_anbn_grammar_weights` default `min_n` from 0 to 1 and `compute_grammar_weighted_nll_bits` loop from `range(0, ...)` to `range(1, ...)`, matching Abudy et al. (2025) n≥1 convention. Updated all tests with new expected values (weights sum to 0.7, analytical |D:H| ≈ 2.42).

### ~~1.5 Verify smoothing consistency between training and evaluation on golden network~~ — Done (2026-03-21)

Already correct by construction: training uses optax log_softmax (numerically stable optimizer path), all reported metrics go through `compute_per_string_nll_bits` / `compute_data_nll_bits_smoothed` with additive 1e-10 matching Abudy et al. (2025). No code change needed.

---

## Section 2: Experiments

Achievable by running existing code — parameter sweeps, resumed runs, hyperparameter tuning. No code changes needed.

### 2.1 Resume interrupted ANBN `mode_forward` experiments (M — half day)

- Source: `20260315-experiment-progress-note.MD`.
- Two runs saved mid-warmup (epoch 500 and epoch 2500). Need resuming and evaluation.

### 2.2 Seed stability on best schedule family

- Once the new transition is validated, run the strongest schedule across multiple seeds.
- Quantify mean/std for `gen_n` and `total_mdl_bits`.
- **Seed count (ref: `docs/comparisons.md` §6.8):** Single seed suffices for Abudy-comparable experiments (they ran seed=100 only). For FLaRe/Deletang-comparable claims or any assertion of consistent improvement, need ≥5 seeds with mean ± std (FLaRe uses 10 seeds × 4 loss variants; Deletang et al. (2023) uses 10 seeds).

### 2.3 Transition and regularization ablations

- Sweep `mdl_lambda`, grid size, warmup duration, bridge duration, annealing schedule, and `n_samples`.
- Main goal: reduce the threat to validity from the current narrow good-`lambda` region.
- **LR ablation (ref: `docs/comparisons.md` §6.3):** Our lr=0.05 is 500× Abudy et al. (2025) Exp 3 (lr=1e-4) and 50× Lan et al. (2024) (lr=0.001). Defensible because we optimize categorical logits, but a reviewer will check. Run ablation over lr ∈ {0.001, 0.01, 0.05, 0.1} to show results are not cherry-picked.

### 2.4 Shared-MDL validation

- Objective mismatch is fixed (CE_2 + P_base correction, P_base-informed φ init).
- Test whether shared mode can match or exceed basic MDL under matched settings.
- Compare corrected shared mode against basic MDL under matched schedules and seeds.
- Sweep `lambda2` around the viable band (`20` to `100`) to measure how broad that anchor-stable region is.

### 2.5 Push ULA-comparable models toward ~96% cMNIST target

- Continue with ULA-style architectures only when making SOTA-facing comparisons.
- Use VIB as the current stronger baseline while MDL continues to improve.

---

## Section 3: New Features, Experiments, Metrics & Research

Items requiring new code, new experimental infrastructure, or new research directions.

### 3.A New Code Features

#### Best-epoch selection and outer-model robustness tracking — cMNIST (M — half day)

- Source: `colored-mnist-current-status.MD` next-work item 1; former `research-roadmap.MD` item 6.
- Peak-vs-final behaviour is a bottleneck. Save and select best outer checkpoints during training.
- Add group-balanced and worst-group metrics for `cmnist_ula`.

#### ~~Staged temperature annealing with discretization checkpoints~~ — Done (2026-03-21)

Implemented in `src/mdl/training.py` and `differentiable_mdl.py`. Extended `MDLTrainState` with `freeze_mask` (n_params,) float32. New functions: `compute_confidence_mask`, `freeze_confident_weights` (one-hot × 100 logits), `apply_freeze_mask_to_grads` (gradient masking). `make_train_step` gains `use_freeze_mask` parameter. CLI args: `--anneal_mode staged`, `--staged_tau_levels`, `--staged_epoch_fracs`, `--freeze_threshold`. Tests in `tests/test_staged_annealing.py`.

#### DiffQ-style continuous relaxation for cMNIST — P2 (L — 1+ day)

- Source: quantization session log (Change 4, new module ~200 lines).
- Replaces full M-dimensional categorical per weight with continuous noise + ST rounding. Addresses 38M-float memory bottleneck on cMNIST.

#### RELAX estimator — P3 (XL — multi-day, defer)

- Source: quantization session log (Change 7, ~300 lines).
- Learned control variate for unbiased low-variance gradients. Only needed if P0-P2 still show problematic gradient variance.

### 3.B New Formal Language Tasks

We currently implement only **aⁿbⁿ** (in `src/mdl/data.py`). Abudy et al. (2025) evaluate on six tasks. We need five more. The goal is to replicate their full experimental battery using our differentiable MDL relaxation — and demonstrate something they could not: gradient-based MDL training on all tasks. Their Experiment 3 (gradient descent) excluded MDL because it is non-differentiable. Our relaxation fills that gap.

#### aⁿbⁿcⁿ (two counters)

- **Language:** strings `a^n b^n c^n` for n ≥ 1.
- **Computational requirement:** two independent counters (count a's, decrement through b's and c's).
- **Grammar:** sample n ~ Geometric(p=0.3). Produce string `# a^n b^n c^n #`.
  - This is the same geometric distribution used for aⁿbⁿ in `src/mdl/data.py:generate_anbn_strings`.
- **Alphabet:** `{#, a, b, c}` → 4 symbols (vs 3 for aⁿbⁿ). This means `NUM_SYMBOLS = 4` and the LSTM input/output size must change.
- **Test set:** all strings n=1 to n_max_train + 1000. For example, if training sees up to n=20, test is n=1..1020.
  - Ref: Abudy et al. (2025) Appendix A.1 (line 491).
- **Golden network:** discovered by Lan et al. (2022). Network diagram in Abudy et al. (2025) Figure 9 (line 597). The golden network has |H| = 241 bits (line 809).
  - The golden network uses a free-form RNN, not an LSTM. We need to either: (a) reimplement this as a free-form RNN and add that architecture to our codebase, or (b) construct an equivalent LSTM. Option (a) is closer to the paper but requires a new model class. Option (b) is simpler if an LSTM equivalent exists.
  - **Decision needed:** which architecture to use.
- **Implementation:**
  - New file: `src/mdl/data_anbncn.py` (or extend `data.py` with a task selector).
  - Generator: `generate_anbncn_strings(num_strings, p=0.3, seed)` → list of `[#, a, ..., a, b, ..., b, c, ..., c, #]`.
  - Test set builder: `make_anbncn_test_set(n_max)` → all strings n=1..n_max.

#### Dyck-1 (matched parentheses, single type)

- **Language:** well-balanced parentheses `(`, `)`.
- **Computational requirement:** depth counter (single bracket type).
- **Grammar (Abudy et al. (2025) Table 6, line 497–509):**
  ```
  S → ( S ) S    with probability 1/3
  S → ε          with probability 2/3
  ```
  Training strings: max length 200. Wrap with `#` delimiters: `# ( ( ) ) #`.
- **Alphabet:** `{#, (, )}` → 3 symbols.
- **Test set:** all valid Dyck-1 strings of length ≤ 10 (Abudy et al. (2025) line 494).
  - Exhaustive enumeration, not a sweep over n. The number of Dyck-1 strings of length 2k is the Catalan number C(k). For length ≤ 10: C(1)+C(2)+C(3)+C(4)+C(5) = 1+2+5+14+42 = 64 strings (plus the empty string).
  - Must be weighted by grammar probabilities for |D:H| computation.
- **Golden network:** found during a GA run under MDL (Abudy et al. (2025) line 255). Network diagram in Figure 11 (line 608). |H| = 113 bits (line 815).
  - Differentiable variant in Figure 12 (line 617).
- **Implementation:**
  - New file: `src/mdl/data_dyck.py`.
  - PCFG sampler: `generate_dyck1_strings(num_strings, max_length=200, seed)`.
  - Test set: `enumerate_dyck1(max_length=10)` → all valid strings ≤ length 10.
  - Note: unlike aⁿbⁿ, the test set has variable-length strings with different structures, so batched evaluation needs padding and masking (similar to what `evaluate_deterministic_accuracy` already does).

#### Dyck-2 (matched parentheses and brackets)

- **Language:** well-balanced `(`, `)`, `[`, `]`.
- **Computational requirement:** stack (must track which bracket type to close).
- **Grammar (Abudy et al. (2025) Table 7, line 512–525):**
  ```
  S → ( S ) S    with probability 1/6
  S → [ S ] S    with probability 1/6
  S → ε          with probability 2/3
  ```
  Training strings: max length 200.
- **Alphabet:** `{#, (, ), [, ]}` → 5 symbols.
- **Test set:** all valid Dyck-2 strings of length ≤ 10 (line 494).
- **Golden network:** from Lan et al. (2022). Figure 13 (line 624). |H| = 579 bits (line 821).
  - This is the hardest task — Lan et al. (2022) failed on it with their GA. Abudy et al.'s (2025) GA found a solution with |H| = 327 (better than the golden, line 822), but it still had Δ_test = 2.0%.
  - Uses special activation functions: Floor, Modulo-3, Unsigned Step, plus Multiplication units (line 730). This is far beyond what a standard LSTM can do.
  - **Decision needed:** do we attempt Dyck-2 with our LSTM architecture, or do we implement a more expressive architecture? This task may require a different approach for us.
- **Implementation:**
  - Extend `src/mdl/data_dyck.py` with `generate_dyck2_strings` and `enumerate_dyck2`.

#### Arithmetic Syntax (nested addition)

- **Language:** nested addition formulas over `1`s: `(1+1)`, `((1+1)+1)`, `(1+((1+1)+(1+1)))`, etc.
- **Computational requirement:** stack (track argument positions in nested expressions).
- **Grammar (Abudy et al. (2025) Table 8, line 532–545):**
  ```
  S → ( E + E )    with probability 1.0
  E → 1            with probability 0.67
  E → ( E + E )    with probability 0.33
  ```
  Training strings: length ≤ 200.
- **Alphabet:** `{#, (, ), +, 1}` → 5 symbols.
- **Test set:** all valid expressions of length ≤ 40 (line 529).
- **Golden network:** manually constructed by Abudy et al. (2025) Figure 15 (line 631). |H| = 967 bits (line 827).
  - Uses Floor, Modulo-4, Absolute Value, Unsigned Step activations, plus Multiplication units (line 731).
  - **Same architecture concern as Dyck-2:** requires non-standard activations.
- **Implementation:**
  - New file: `src/mdl/data_arithmetic.py`.
  - PCFG sampler + exhaustive enumerator for expressions ≤ length 40.

#### Toy-English (minimal English fragment)

- **Language:** a minimal English fragment with nouns, transitive/intransitive/complement verbs, and object relative clauses.
- **Computational requirement:** recursive syntactic structure.
- **Grammar (Abudy et al. (2025) Table 9, line 552–573):**
  ```
  S   → NP VP           (p=1.0)
  NP  → N               (p=0.75)
  NP  → N RC            (p=0.25)
  RC  → NP Vtr          (p=1.0)
  VP  → Vtr NP          (p=0.34)
  VP  → Vi              (p=0.33)
  VP  → Vcp S           (p=0.33)
  N   → "dogs"          (p=1.0)
  Vtr → "chase"         (p=1.0)
  Vi  → "sleep"         (p=1.0)
  Vcp → "think that"    (p=1.0)
  ```
  Training strings: length ≤ 200.
- **Alphabet:** `{#, dogs, chase, sleep, think, that}` → 6 word-level tokens (or 7 if `think that` is two tokens).
  - **Clarification needed:** Abudy et al. (2025) list `Vcp → "think that"` as a single production. Determine whether the network sees this as one token or two. Looking at the example strings ("dogs think that dogs sleep"), it appears to be two tokens.
- **Test set:** all grammatical strings of length ≤ 20 (line 549).
- **Golden network:** manually constructed. Figure 16 (line 637). |H| = 870 bits (line 833).
  - Uses Unsigned Step activation + Multiplication units (line 732).
- **Implementation:**
  - New file: `src/mdl/data_toy_english.py`.
  - PCFG sampler at word level + exhaustive enumerator ≤ 20 tokens.

#### Architecture considerations

Abudy et al. (2025) use "free-form RNNs" (directed graphs with units, weighted connections, and per-unit activation functions), not LSTMs. Their golden networks use activations like Floor, Modulo-3, Modulo-4, Unsigned Step, and Multiplication units — far beyond what a standard LSTM provides.

**For our differentiable MDL work, we have two paths:**

1. **Stick with LSTMs** for all tasks. This is simpler and matches Lan et al. (2024) who used LSTM hidden_size=3 for aⁿbⁿ. We may not achieve the same |H| as their golden networks (which are free-form RNNs), but we can still compare MDL vs L1/L2 under a fixed LSTM architecture. The key claim is that MDL regularisation under gradient descent improves generalisation — this holds regardless of architecture.

2. **Implement free-form RNNs with categorical weight parameterisation.** This would allow direct comparison to their golden networks and |H| values. Significantly more work.

**Recommendation:** start with LSTMs (path 1) for aⁿbⁿ, aⁿbⁿcⁿ, and Dyck-1 (which only need counting, achievable with LSTM cell states). Defer Dyck-2, Arithmetic, and Toy-English until we decide on architecture. These three require stack-like computation that standard LSTMs struggle with.

#### Per-task implementation checklist

For each new task, create:
1. `src/mdl/data_{task}.py` — data generator (PCFG sampler) + test set constructor.
2. Golden network implementation (or stub if architecture is TBD).
3. YAML config: `config/anbn_mdl/{task}_train.yaml` (copy from `basic_train.yaml`, adjust symbols/architecture).
4. Evaluation function: deterministic accuracy adapted for the task's prediction structure.
5. Tests: `tests/test_{task}_data.py` — verify grammar sampling, test set completeness, string validity.

### 3.C New Metrics

We track the raw ingredients but don't combine them into the formats used by the papers.

#### ~~Analytical optimal |D:H|~~ — Done (2026-03-20)

Implemented in `src/mdl/evaluation.py`: `compute_optimal_dh_test(max_n, p)` returns golden network's grammar-weighted test NLL (data term) and |H|. Validated: golden test |D:H| ≈ 2.94 bits (matching Abudy et al. 2025, arXiv:2505.13398v2, line 803). Note: our LSTM golden |H| = 1137 bits (not 139 — that's Abudy et al.'s free-form RNN golden).

#### ~~Composite |D:H| score for trained networks~~ — Done (2026-03-20)

Implemented in `src/mdl/evaluation.py`: `evaluate_trained_network_dh(apply_fn, params, grid_codelengths, hidden_size, test_max_n, p)` computes grammar-weighted test NLL + Lan-style |H| for argmax-discretised trained networks.

#### ~~Δ% — relative gap to optimum~~ — Done (2026-03-20)

Implemented in `src/mdl/evaluation.py`: `compute_delta_pct(score, optimal)`. Also `format_abudy_comparison_table(...)` for paper-style output with train/test Δ% columns. Convention: Δ% operates on the data term |D:H|, not total.

#### ~~Golden network reference scores under each regulariser~~ — Done (2026-03-20)

Implemented in `src/mdl/evaluation.py`: `evaluate_golden_under_regularisers(max_n, p)` returns CE, L1 norm, L2 norm squared, and MDL |H|. Also `format_golden_regulariser_table(result)`. Note: norms are for our LSTM golden (Lan et al. 2024), not directly comparable to Abudy et al.'s free-form RNN.

#### ~~Grammar-weighted test set evaluation~~ — Done (2026-03-20)

Implemented in `src/mdl/evaluation.py`: `compute_anbn_grammar_weights(max_n, p, min_n=0)` returns raw PCFG weights P(n) = p*(1-p)^n. `compute_grammar_weighted_nll_bits(forward_fn, max_n, p)` computes |D:H|_test = Σ P(n) × NLL_total(n) including n=0. Uses Abudy et al. (2025) smoothing convention (additive 1e-10). Also `compute_train_dh(...)` for raw NLL sum over training corpus, and `compute_optimal_dh_train(...)` for golden's train baseline.

#### ~~Recognition accuracy (FLaRe / Deletang comparison)~~ — Done (2026-03-21)

Implemented in `src/mdl/evaluation.py`: `compute_full_string_accuracy` (per-string boolean, batched with padding), `generate_negative_anbn` (4 types: wrong counts, interleaved, single-symbol, random), `compute_recognition_accuracy` (returns accuracy, tp_rate, tn_rate). Tests in `tests/test_evaluation.py::TestRecognitionAccuracy`. Note: golden LM gets 0% on valid strings (memoryless PCFG predicts P(a)=0.7 at a→b transition).

### 3.D Baseline Comparisons

#### ~~Standard LSTM baselines (no-reg, L1, L2)~~ — Metrics integrated (2026-03-21)

Baseline experiment script (`scripts/experiments/baseline_lstm_experiments.py`) now computes grammar-weighted |D:H|_test, |D:H|_train, and Δ% using `src/mdl/evaluation.py`. Output includes Abudy et al. (2025)-style comparison table via `format_abudy_comparison_table`. Results saved to `results.json` with full metric breakdown. Long production runs (1000+ epochs, multi-seed) deferred.

#### Verify LSTM golden |H| decomposition against Lan et al. (2024)

**Ref:** `docs/comparisons.md` §7.1.

Lan et al. (2024) Table 1 reports 3920 as the golden network's total MDL objective (L_D + λ·|H|), not |H| alone. Our `golden.py` computes |H| = 1137 bits. Decompose their 3920 into data term + hypothesis term and verify our |H| matches their weight-encoding component exactly.

**Steps:**
1. Run `golden_mdl_score()` and record full breakdown (architecture bits, per-weight bits, total |H|).
2. Cross-check against Lan et al. (2024) Appendix B: gate order, bias convention, output layer construction (ε = (2¹⁴−1)⁻¹, `limit_denominator(1000)`, tanh(1) scaling).
3. Compute data term L_D on matching training set (1000 strings, Geometric(p=0.3)) and check |H| + L_D ≈ 3920 under their λ convention.
4. Document the exact decomposition.

#### Free-form RNN implementation (Stages 1–2)

**Ref:** `docs/comparisons.md` §7.2.

**Stage 1 — Golden construction & |H| verification:**
Build the Lan et al. (2022, TACL 10:785–799, arXiv:2111.00600) / Abudy et al. (2025) free-form RNN architecture and verify the aⁿbⁿ golden network gives |H| = 139 bits exactly.

- Create `src/mdl/freeform_rnn.py` — `FreeFormRNN` data structure + `freeform_forward` function.
- Create `src/mdl/freeform_coding.py` — `freeform_codelength(network)` following Lan et al. (2022) encoding (unit count, activation type, bias, adjacency, weights).
- Create `src/mdl/golden_freeform.py` — hard-code aⁿbⁿ golden topology from Abudy et al. (2025) Figure 7.
- Create `tests/test_freeform_golden.py` — verify |H| = 139 bits, 100% deterministic accuracy on n=1..1500.
- Also build the differentiable variant (Abudy et al. Figure 8, Linear+ReLU only) and verify accuracy.

**Stage 2 — Trainable free-form RNN with differentiable MDL (after Stage 1):**
Apply categorical relaxation to free-form RNN weights. Fixed-topology approach: fix graph structure to golden/supergraph, learn weights via Gumbel-Softmax over rational grid.

- Create `src/mdl/freeform_training.py` — adapt categorical MDL training loop for free-form forward pass.
- Enables direct comparison to Abudy et al. (2025) Exp 3 (GD from golden init) AND from-scratch training (our novel contribution).

#### Differentiable MDL vs. non-differentiable MDL — the central comparison

- **The key comparison:** Abudy et al. (2025) Experiment 3 (gradient descent, line 194) uses only L1/L2/none because MDL is non-differentiable. We add the missing row: "Differentiable MDL (GD)".
- **Format:** produce a table with columns matching theirs, adding our method:

  ```
  Task | Method              | |H| | Train |D:H| | Test |D:H| | Δ_train(%) | Δ_test(%)
  -----|---------------------|------|-------------|------------|------------|----------
  aⁿbⁿ | MDL-GA (theirs)     | 139  | 1529.39     | 2.94       | -0.2       | 0.1
  aⁿbⁿ | L1-GD (theirs)      | ...  | ...         | ...        | ...        | ...
  aⁿbⁿ | L2-GD (theirs)      | ...  | ...         | ...        | ...        | ...
  aⁿbⁿ | Diff-MDL-GD (ours)  | ?    | ?           | ?          | ?          | ?
  ```

- This is the central result of the paper. If our Diff-MDL-GD achieves Δ_test close to 0 (matching or approaching the GA-MDL result), it validates the approach.

### 3.E Analysis & Visualization

#### Learned-weight / mechanism analysis

- Compare learned networks against the handcrafted golden network.
- Characterize what distinguishes very high-generalization runs from brittle ones.

#### Loss landscape visualization (LOW PRIORITY)

- **What:** 2D/3D plots of the loss surface around the golden network.
- **Why:** Lan et al. (2024) Figure 3 shows that CE+MDL has the golden network at a minimum while CE+L1/L2 do not. Reproducing this with our relaxed objective is a strong visual argument.
- **How:**
  - Choose two perturbation directions in logit space (e.g. random orthogonal directions, or the direction toward a trained solution).
  - Evaluate J_β on a 2D grid of perturbation magnitudes.
  - Plot as a surface or contour.
- **Implementation:** new script `scripts/analysis/loss_landscape.py`.

#### Output probability / memory visualization (LOW PRIORITY)

- **What:** per-timestep output probability distributions and LSTM cell state traces.
- **Why:** Lan et al. (2024) Figures 5–6 show that trained networks develop (imperfect) counting mechanisms. Showing our MDL-trained networks develop clean counting is compelling.
- **How:** run forward pass on selected test strings (e.g. n=5, n=20, n=100), record softmax outputs and (h, c) at each timestep, plot.
- **Implementation:** new script `scripts/analysis/network_traces.py`.

#### General training diagnostics

- Training dynamics plots
- Output probability heatmaps
- Hidden-state / cell-state traces
- Loss-surface or schedule-comparison plots where useful

### 3.F Publication

- Connect proposal theory and the strongest empirical evidence.
- Be explicit about optimizer path dependence, schedule effects, and current limits.

#### Fair comparison documentation checklist

**Ref:** `docs/comparisons.md` §5–6. Each item below must be addressed in the paper to withstand reviewer scrutiny.

- [ ] **Architecture mismatch (§6.4):** State clearly in all comparison tables that Abudy et al. (2025) GD baselines use free-form RNNs, not LSTMs. |H| comparison across architectures is invalid; Δ_test comparison is valid.
- [ ] **Grid cannot represent golden (§6.5):** Acknowledge that the LSTM golden net (LARGE=127) is outside our rational grid (n_max=10). Frame any |H| result honestly: "solutions in restricted grid S."
- [ ] **Training vs eval loss asymmetry (§6.6):** Document that training loss averages over positions while |D:H| sums then grammar-weights. Standard practice but not identical to the reported metric.
- [ ] **Report effective λ (§6.7):** State effective λ_eff = mdl_lambda / N (e.g., 1000/1000 = 1.0), not just the config value. Abudy's GA uses pure MDL (λ=1 implicitly).
- [ ] **LR difference (§6.3):** State our lr and the 500× gap vs Abudy. Include ablation results (item 2.3).
- [ ] **Test range (§6.2):** Document that our test_max_n=1500 vs Abudy's n_max_train+1000. Numerically negligible for Geometric(0.3) but should be stated.
- [ ] **Golden-init vs random-init (§6.10):** Present both where applicable. Abudy Exp 3 starts from golden weights; we start from random init — complementary experiments.
- [ ] **n=0 convention (§6.1):** After fixing item 1.4, state which min_n convention each reported number uses.
- [ ] **Seed count (§6.8):** Match or exceed comparison papers' seed counts per item 2.2 notes.

#### Progression goals

**Ref:** `docs/comparisons.md` §4.

- **Level 1 (minimum publishable):** Across aⁿbⁿ, aⁿbⁿcⁿ, Dyck-1: Δ_test substantially below GD+L1/L2 baselines. |H| ≥ order of magnitude smaller. Generalization to 10×+ training length with >95% det. acc.
- **Level 2 (strong paper):** Perfect/near-perfect generalization on aⁿbⁿ (done). Strong on aⁿbⁿcⁿ and Dyck-1. Δ_test ≈ 0%. Recognition accuracy vs FLaRe.
- **Level 3 (excellent paper):** Perfect generalization on all counter languages. Meaningful Dyck-2/stack-language results. FLaRe comparison showing MDL-LSTM beats all standard LSTMs.

---

## Appendix A: Suggested Implementation Order

**Phase A — paper-ready numbers for aⁿbⁿ (no new tasks needed):**
1. ~~Compute analytical optimal |D:H| for aⁿbⁿ using `src/mdl/golden.py`.~~ Done
2. ~~Add composite |D:H| evaluation for trained networks.~~ Done
3. ~~Compute Δ% for our best existing runs.~~ Done
4. ~~Add grammar weights for aⁿbⁿ test set (geometric distribution).~~ Done
5. ~~Golden network scores under CE, L1, L2, MDL.~~ Done
6. ~~Run standard LSTM with CE/L1/L2 on aⁿbⁿ.~~ (metrics integrated; long runs deferred)
7. ~~Produce the main results table for aⁿbⁿ.~~ (table infrastructure done; production numbers need long runs)
8. ~~Fix n=0 inclusion in grammar-weighted evaluation (item 1.4)~~ Done
9. ~~Verify smoothing consistency on golden network (item 1.5)~~ Done
10. Verify LSTM golden |H| decomposition against Lan et al. (2024) (item 3.D)
11. LR ablation over {0.001, 0.01, 0.05, 0.1} (item 2.3)
12. Production runs: 1000+ epochs, multi-seed baselines + MDL (items 2.1, 2.2)
13. ~~Implement recognition accuracy metric (item 3.C)~~ Done

**Phase B — second and third tasks:**
14. Implement aⁿbⁿcⁿ (data, golden network, evaluation).
15. Implement Dyck-1 (data, golden network, evaluation).
16. Run differentiable MDL + baselines on both new tasks.

**Phase C — free-form RNN & harder tasks (ref: `docs/comparisons.md` §7):**
17. Free-form RNN Stage 1: architecture, coding scheme, aⁿbⁿ golden net, verify |H|=139 (item 3.D).
18. Free-form RNN Stage 2: categorical relaxation training, Abudy Exp 3 comparison (item 3.D).
19. Decide on architecture for Dyck-2, Arithmetic, Toy-English (LSTM vs free-form RNN).
20. Implement remaining tasks.

**Phase D — visualizations, polish, and write-up:**
21. Loss landscape plots.
22. Output probability / memory traces.
23. Complete fair comparison documentation checklist (item 3.F).
24. Write-up targeting progression Level 1 → Level 2 (ref: `docs/comparisons.md` §4).

---

## Appendix B: Reference Numbers from Abudy et al. (2025)

Golden network |H| and optimal test |D:H| for each task (from their full results table, lines 803–837):

| Task | Golden |H| (bits) | Optimal test |D:H| | Golden L1 | Golden L2 |
|------|---------|---------------------|------------|-----------|
| aⁿbⁿ | 139 | 2.94 | 7.00 | 15.00 |
| aⁿbⁿcⁿ | 241 | 2.94 | 11.33 | 16.44 |
| Dyck-1 | 113 | 1.77 | 7.00 | 15.00 |
| Dyck-2 | 579 | 2.32 | 27.33 | 35.11 |
| Arithmetic | 967 | 3.96 | 47.75 | 54.67 |
| Toy-English | 870 | 4.49 | 49.00 | 61.00 |

Golden train |D:H| values:

| Task | Golden train |D:H| |
|------|---------------------|
| aⁿbⁿ | 1531.77 |
| aⁿbⁿcⁿ | 1483.91 |
| Dyck-1 | 1544.48 |
| Dyck-2 | 2121.53 |
| Arithmetic | 2581.29 |
| Toy-English | 2232.74 |

Backpropagation hyperparameters (their Experiment 3, Appendix Table 10, line 745–762):
- Optimizer: Adam
- Learning rate: 1e-4
- Epochs: 1000
- Regularisation: None, L1 (λ=1), L2 (λ=1)
- Seeds: 100
- Corpus seed: 100

These numbers are our targets and comparison points.
