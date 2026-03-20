# Experiment & Metrics Roadmap

Written 2026-03-20 after reviewing Lan et al. (2024) and Abudy et al. (2025) against our current codebase.

The goal is to replicate the full experimental battery from Abudy et al. (2025) using our differentiable MDL relaxation — and in doing so demonstrate something they could not: gradient-based MDL training on all tasks. Their Experiment 3 (gradient descent) excluded MDL because it is non-differentiable. Our relaxation fills that gap.

**Reference papers:**
- Lan et al. (2024): `agent-context/arxiv/arXiv-2402.10013v2_contents/main-acl.tex`
- Abudy et al. (2025): `agent-context/arxiv/arXiv-2505.13398v2_contents/main.tex`

**Reference tables for target numbers:**
- Abudy et al. Table 1 (main results): lines ~300–340 of `main.tex`
- Abudy et al. full results: Appendix tables starting at line ~794 (GA arch search), ~844 (GA weights only), and Exp 3 GD results
- Golden network |H| values per task: lines 803, 809, 815, 821, 827, 833

---

## Part 1: Missing formal language tasks

We currently implement only **aⁿbⁿ** (in `src/mdl/data.py`). Abudy et al. evaluate on six tasks. We need five more. Each subsection below specifies exactly what to build.

### 1.1 aⁿbⁿcⁿ (two counters)

- **Language:** strings `a^n b^n c^n` for n ≥ 1.
- **Computational requirement:** two independent counters (count a's, decrement through b's and c's).
- **Grammar:** sample n ~ Geometric(p=0.3). Produce string `# a^n b^n c^n #`.
  - This is the same geometric distribution used for aⁿbⁿ in `src/mdl/data.py:generate_anbn_strings`.
- **Alphabet:** `{#, a, b, c}` → 4 symbols (vs 3 for aⁿbⁿ). This means `NUM_SYMBOLS = 4` and the LSTM input/output size must change.
- **Test set:** all strings n=1 to n_max_train + 1000. For example, if training sees up to n=20, test is n=1..1020.
  - Ref: Abudy et al. Appendix A.1 (line 491).
- **Golden network:** discovered by Lan et al. (2022). Network diagram in Abudy et al. Figure 9 (line 597). The golden network has |H| = 241 bits (line 809).
  - The golden network uses a free-form RNN, not an LSTM. We need to either: (a) reimplement this as a free-form RNN and add that architecture to our codebase, or (b) construct an equivalent LSTM. Option (a) is closer to the paper but requires a new model class. Option (b) is simpler if an LSTM equivalent exists.
  - **Decision needed:** which architecture to use.
- **Implementation:**
  - New file: `src/mdl/data_anbncn.py` (or extend `data.py` with a task selector).
  - Generator: `generate_anbncn_strings(num_strings, p=0.3, seed)` → list of `[#, a, ..., a, b, ..., b, c, ..., c, #]`.
  - Test set builder: `make_anbncn_test_set(n_max)` → all strings n=1..n_max.

### 1.2 Dyck-1 (matched parentheses, single type)

- **Language:** well-balanced parentheses `(`, `)`.
- **Computational requirement:** depth counter (single bracket type).
- **Grammar (Abudy et al. Table 6, line 497–509):**
  ```
  S → ( S ) S    with probability 1/3
  S → ε          with probability 2/3
  ```
  Training strings: max length 200. Wrap with `#` delimiters: `# ( ( ) ) #`.
- **Alphabet:** `{#, (, )}` → 3 symbols.
- **Test set:** all valid Dyck-1 strings of length ≤ 10 (Abudy et al. line 494).
  - Exhaustive enumeration, not a sweep over n. The number of Dyck-1 strings of length 2k is the Catalan number C(k). For length ≤ 10: C(1)+C(2)+C(3)+C(4)+C(5) = 1+2+5+14+42 = 64 strings (plus the empty string).
  - Must be weighted by grammar probabilities for |D:H| computation.
- **Golden network:** found during a GA run under MDL (Abudy et al. line 255). Network diagram in Figure 11 (line 608). |H| = 113 bits (line 815).
  - Differentiable variant in Figure 12 (line 617).
- **Implementation:**
  - New file: `src/mdl/data_dyck.py`.
  - PCFG sampler: `generate_dyck1_strings(num_strings, max_length=200, seed)`.
  - Test set: `enumerate_dyck1(max_length=10)` → all valid strings ≤ length 10.
  - Note: unlike aⁿbⁿ, the test set has variable-length strings with different structures, so batched evaluation needs padding and masking (similar to what `evaluate_deterministic_accuracy` already does).

### 1.3 Dyck-2 (matched parentheses and brackets)

- **Language:** well-balanced `(`, `)`, `[`, `]`.
- **Computational requirement:** stack (must track which bracket type to close).
- **Grammar (Abudy et al. Table 7, line 512–525):**
  ```
  S → ( S ) S    with probability 1/6
  S → [ S ] S    with probability 1/6
  S → ε          with probability 2/3
  ```
  Training strings: max length 200.
- **Alphabet:** `{#, (, ), [, ]}` → 5 symbols.
- **Test set:** all valid Dyck-2 strings of length ≤ 10 (line 494).
- **Golden network:** from Lan et al. (2022). Figure 13 (line 624). |H| = 579 bits (line 821).
  - This is the hardest task — Lan et al. (2022) failed on it with their GA. Abudy et al.'s GA found a solution with |H| = 327 (better than the golden, line 822), but it still had Δ_test = 2.0%.
  - Uses special activation functions: Floor, Modulo-3, Unsigned Step, plus Multiplication units (line 730). This is far beyond what a standard LSTM can do.
  - **Decision needed:** do we attempt Dyck-2 with our LSTM architecture, or do we implement a more expressive architecture? This task may require a different approach for us.
- **Implementation:**
  - Extend `src/mdl/data_dyck.py` with `generate_dyck2_strings` and `enumerate_dyck2`.

### 1.4 Arithmetic Syntax (nested addition)

- **Language:** nested addition formulas over `1`s: `(1+1)`, `((1+1)+1)`, `(1+((1+1)+(1+1)))`, etc.
- **Computational requirement:** stack (track argument positions in nested expressions).
- **Grammar (Abudy et al. Table 8, line 532–545):**
  ```
  S → ( E + E )    with probability 1.0
  E → 1            with probability 0.67
  E → ( E + E )    with probability 0.33
  ```
  Training strings: length ≤ 200.
- **Alphabet:** `{#, (, ), +, 1}` → 5 symbols.
- **Test set:** all valid expressions of length ≤ 40 (line 529).
- **Golden network:** manually constructed by Abudy et al. Figure 15 (line 631). |H| = 967 bits (line 827).
  - Uses Floor, Modulo-4, Absolute Value, Unsigned Step activations, plus Multiplication units (line 731).
  - **Same architecture concern as Dyck-2:** requires non-standard activations.
- **Implementation:**
  - New file: `src/mdl/data_arithmetic.py`.
  - PCFG sampler + exhaustive enumerator for expressions ≤ length 40.

### 1.5 Toy-English (minimal English fragment)

- **Language:** a minimal English fragment with nouns, transitive/intransitive/complement verbs, and object relative clauses.
- **Computational requirement:** recursive syntactic structure.
- **Grammar (Abudy et al. Table 9, line 552–573):**
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
  - **Clarification needed:** Abudy et al. list `Vcp → "think that"` as a single production. Determine whether the network sees this as one token or two. Looking at the example strings ("dogs think that dogs sleep"), it appears to be two tokens.
- **Test set:** all grammatical strings of length ≤ 20 (line 549).
- **Golden network:** manually constructed. Figure 16 (line 637). |H| = 870 bits (line 833).
  - Uses Unsigned Step activation + Multiplication units (line 732).
- **Implementation:**
  - New file: `src/mdl/data_toy_english.py`.
  - PCFG sampler at word level + exhaustive enumerator ≤ 20 tokens.

### 1.6 Architecture considerations

Abudy et al. use "free-form RNNs" (directed graphs with units, weighted connections, and per-unit activation functions), not LSTMs. Their golden networks use activations like Floor, Modulo-3, Modulo-4, Unsigned Step, and Multiplication units — far beyond what a standard LSTM provides.

**For our differentiable MDL work, we have two paths:**

1. **Stick with LSTMs** for all tasks. This is simpler and matches Lan et al. (2024) who used LSTM hidden_size=3 for aⁿbⁿ. We may not achieve the same |H| as their golden networks (which are free-form RNNs), but we can still compare MDL vs L1/L2 under a fixed LSTM architecture. The key claim is that MDL regularisation under gradient descent improves generalisation — this holds regardless of architecture.

2. **Implement free-form RNNs with categorical weight parameterisation.** This would allow direct comparison to their golden networks and |H| values. Significantly more work.

**Recommendation:** start with LSTMs (path 1) for aⁿbⁿ, aⁿbⁿcⁿ, and Dyck-1 (which only need counting, achievable with LSTM cell states). Defer Dyck-2, Arithmetic, and Toy-English until we decide on architecture. These three require stack-like computation that standard LSTMs struggle with.

### 1.7 Per-task implementation checklist

For each new task, create:
1. `src/mdl/data_{task}.py` — data generator (PCFG sampler) + test set constructor.
2. Golden network implementation (or stub if architecture is TBD).
3. YAML config: `config/anbn_mdl/{task}_train.yaml` (copy from `basic_train.yaml`, adjust symbols/architecture).
4. Evaluation function: deterministic accuracy adapted for the task's prediction structure.
5. Tests: `tests/test_{task}_data.py` — verify grammar sampling, test set completeness, string validity.

---

## Part 2: Missing metrics

We track the raw ingredients but don't combine them into the formats used by the papers.

### 2.1 Analytical optimal |D:H| (HIGH PRIORITY)

- **What:** the total MDL score achieved by the golden network on the true test distribution: `|D:H|_optimal = |D:H_golden|_test + |H_golden|`.
  - `|D:H_golden|_test` = cross-entropy of the golden network's output distribution against the true grammar distribution, summed over the exhaustive test set. In bits.
  - `|H_golden|` = discrete hypothesis codelength of the golden network under Lan-style encoding.
- **Why:** this is the denominator of Δ%, the primary metric in Abudy et al.'s results tables (line 795).
- **Current state:** `src/mdl/golden.py` has `golden_mdl_score()` which computes |H| for the aⁿbⁿ golden network. We need to also compute the data term on the test set.
- **Reference values from Abudy et al. (line 803–833):**
  - aⁿbⁿ: |H|=139, test |D:H|=2.94
  - aⁿbⁿcⁿ: |H|=241, test |D:H|=2.94
  - Dyck-1: |H|=113, test |D:H|=1.77
  - Dyck-2: |H|=579, test |D:H|=2.32
  - Arithmetic: |H|=967, test |D:H|=3.96
  - Toy-English: |H|=870, test |D:H|=4.49
- **Implementation:**
  - Add `compute_golden_data_codelength(golden_forward_fn, test_inputs, test_targets, grammar_weights)` to `src/mdl/golden.py` (or a new `src/mdl/evaluation.py`).
  - This runs the golden network on each test string, computes per-position CE in bits, and sums weighted by grammar probability.
  - Add `compute_optimal_mdl_score(task)` that returns `|H_golden| + |D:H_golden|_test` for each task.

### 2.2 Composite |D:H| score for trained networks (HIGH PRIORITY)

- **What:** `|D:H|_trained = data_nll_bits_test + |H|_argmax` as a single number.
  - `data_nll_bits_test` = CE loss of the argmax-discretised trained network on the test set, in bits.
  - `|H|_argmax` = hypothesis codelength of the argmax-discretised weights under Lan-style encoding.
- **Why:** makes our results directly comparable to both papers' table format.
- **Current state:**
  - `data_nll_bits` is logged during training but on train batches, not the test set.
  - `complexity_argmax_bits` is logged as a monitoring quantity (see `docs/1_closed_topics/mdl-metrics-and-logging.MD` line 45).
  - **Missing:** evaluation of both terms on the full test set with argmax-discretised weights.
- **Implementation:**
  - Add `evaluate_mdl_score(apply_fn, params, grid_values, grid_codelengths, test_inputs, test_targets, grammar_weights)` to `src/mdl/evaluation.py`.
  - Steps: (1) discretise weights via argmax, (2) compute |H| = sum of `ell(grid_values[argmax(logits)])`, (3) run forward pass with discrete weights on full test set, (4) compute CE in bits weighted by grammar probabilities, (5) return `|D:H| = |H| + data_nll_bits_test`.

### 2.3 Δ% — relative gap to optimum (HIGH PRIORITY)

- **What:** `Δ(%) = (|D:H|_trained - |D:H|_optimal) / |D:H|_optimal × 100`.
- **Why:** headline metric in Abudy et al. Tables 1–2 (lines ~300–340). Reported separately for train and test.
- **Depends on:** 2.1 and 2.2.
- **Implementation:**
  - Compute both train and test Δ%: `Δ_train = (train_score - optimal_train) / optimal_train × 100` and `Δ_test = (test_score - optimal_test) / optimal_test × 100`.
  - Note: Abudy et al. report Δ_train and Δ_test separately (columns in their tables).
  - Report in evaluation logs and summary tables.

### 2.4 Golden network reference scores under each regulariser (MEDIUM PRIORITY)

- **What:** evaluate the golden network's total loss under CE, CE+L1, CE+L2, and CE+MDL.
- **Why:** the core argument of Lan et al. (2024) is that only MDL has the golden network at or near the optimum. Reproducing this with our differentiable objective strengthens the paper.
- **Reference:** Lan et al. Table 4 (appendix), and Abudy et al. Table in lines 803–837 where golden values are in parentheses.
- **Implementation:**
  - For each regulariser r ∈ {none, L1, L2, MDL}:
    - Set the network weights to the golden network's values.
    - Compute `total_loss_r = CE(golden, test_set) + r(golden_weights)`.
    - For MDL: `r = lambda * sum_i ell(golden_weight_i)`.
    - For L1: `r = lambda * sum_i |golden_weight_i|`.
    - For L2: `r = lambda * sum_i golden_weight_i^2`.
  - Report as a table matching Lan et al. Table 4 format.

### 2.5 Grammar-weighted test set evaluation (MEDIUM PRIORITY)

- **What:** weight each test string by its probability under the true grammar when computing aggregate metrics.
- **Why:** Abudy et al. compute |D:H| = Σ_string P(string) × (-log2 P(string | model)), i.e. the expected codelength under the true distribution. Our current `evaluate_deterministic_accuracy` treats all strings uniformly.
- **How for aⁿbⁿ:** string of length n has probability P(n) = p × (1-p)^n where p=0.3. Weight each test string's CE contribution by P(n) / Z where Z normalises over the test range.
- **Implementation:**
  - Add `grammar_weights(task, n_values)` function returning a weight vector.
  - For aⁿbⁿ and aⁿbⁿcⁿ: geometric distribution P(n) = 0.3 × 0.7^n.
  - For Dyck-1/2, Arithmetic, Toy-English: compute P(string) from the PCFG by multiplying production probabilities along the derivation. This requires tracking the derivation during generation or computing it from the string.
  - Pass weights to `evaluate_mdl_score` (metric 2.2).

### 2.6 Smoothing for zero-probability prevention (MEDIUM PRIORITY)

- **What:** Abudy et al. add 10^-10 to output probabilities before computing log-loss to avoid log(0) when a network assigns zero probability to the correct next token (marked with * in their tables, line 795).
- **Current state:** our `_compute_data_nll_bits` in `src/mdl/training.py` uses `optax.softmax_cross_entropy_with_integer_labels`, which computes `log_softmax` (numerically stable, never exactly 0 unless logits are -inf). This is probably fine for training but should be verified for evaluation.
- **Implementation:** ensure evaluation code applies the same smoothing convention as Abudy et al. when computing |D:H| scores for comparison.

### 2.7 Loss landscape visualisation (LOW PRIORITY)

- **What:** 2D/3D plots of the loss surface around the golden network.
- **Why:** Lan et al. Figure 3 shows that CE+MDL has the golden network at a minimum while CE+L1/L2 do not. Reproducing this with our relaxed objective is a strong visual argument.
- **How:**
  - Choose two perturbation directions in logit space (e.g. random orthogonal directions, or the direction toward a trained solution).
  - Evaluate J_β on a 2D grid of perturbation magnitudes.
  - Plot as a surface or contour.
- **Implementation:** new script `scripts/analysis/loss_landscape.py`.

### 2.8 Output probability / memory visualisation (LOW PRIORITY)

- **What:** per-timestep output probability distributions and LSTM cell state traces.
- **Why:** Lan et al. Figures 5–6 show that trained networks develop (imperfect) counting mechanisms. Showing our MDL-trained networks develop clean counting is compelling.
- **How:** run forward pass on selected test strings (e.g. n=5, n=20, n=100), record softmax outputs and (h, c) at each timestep, plot.
- **Implementation:** new script `scripts/analysis/network_traces.py`.

---

## Part 3: Baseline comparisons

### 3.1 Standard LSTM baselines (no-reg, L1, L2)

Abudy et al. compare MDL against L1, L2, and no-regularisation across all tasks and all three experimental settings.

- **Current state:** `src/mdl/baseline_lstm.py` and `scripts/experiments/baseline_lstm_experiments.py` exist but are not fully integrated.
- **What to implement:**
  - Standard LSTM training (regular float weights, no categorical parameterisation) with:
    - CE only (no regularisation)
    - CE + L1 (λ=1, matching Abudy et al. Appendix Table 10, line 756)
    - CE + L2 (λ=1)
  - Optimizer: Adam, lr=1e-4, 1000 epochs (matching Abudy et al. Table 10).
  - Evaluation: same metrics as for MDL networks (composite |D:H|, Δ%, deterministic accuracy).
  - The discrete |H| for baseline networks: discretise to nearest rational on the grid, compute Lan-style codelength. (This is how Abudy et al. compute |H| for L1/L2 networks.)
- **Output format:** a results table matching Abudy et al. Table 1:

  ```
  Task | Regulariser | |H| | L1 | L2 | Train |D:H| | Test |D:H| | Δ_train(%) | Δ_test(%)
  ```

### 3.2 Our differentiable MDL vs. their non-differentiable MDL

- **The key comparison:** Abudy et al. Experiment 3 (gradient descent, line 194) uses only L1/L2/none because MDL is non-differentiable. We add the missing row: "Differentiable MDL (GD)".
- **Format:** produce a table with columns matching theirs, adding our method:

  ```
  Task | Method          | |H| | Train |D:H| | Test |D:H| | Δ_train(%) | Δ_test(%)
  -----|-----------------|------|-------------|------------|------------|----------
  aⁿbⁿ | MDL-GA (theirs) | 139  | 1529.39     | 2.94       | -0.2       | 0.1
  aⁿbⁿ | L1-GD (theirs)  | ...  | ...         | ...        | ...        | ...
  aⁿbⁿ | L2-GD (theirs)  | ...  | ...         | ...        | ...        | ...
  aⁿbⁿ | Diff-MDL-GD (ours) | ?  | ?          | ?          | ?          | ?
  ```

- This is the central result of the paper. If our Diff-MDL-GD achieves Δ_test close to 0 (matching or approaching the GA-MDL result), it validates the approach.

---

## Part 4: Implementation order

**Phase A — paper-ready numbers for aⁿbⁿ (no new tasks needed):**
1. Metric 2.1: compute analytical optimal |D:H| for aⁿbⁿ using `src/mdl/golden.py`.
2. Metric 2.2: add composite |D:H| evaluation for trained networks.
3. Metric 2.3: compute Δ% for our best existing runs.
4. Metric 2.5: add grammar weights for aⁿbⁿ test set (geometric distribution).
5. Metric 2.4: golden network scores under CE, L1, L2, MDL.
6. Baselines 3.1: run standard LSTM with CE/L1/L2 on aⁿbⁿ.
7. Comparison 3.2: produce the main results table for aⁿbⁿ.

**Phase B — second and third tasks:**
8. Experiment 1.1: implement aⁿbⁿcⁿ (data, golden network, evaluation).
9. Experiment 1.2: implement Dyck-1 (data, golden network, evaluation).
10. Run differentiable MDL + baselines on both new tasks.

**Phase C — harder tasks (architecture decision needed):**
11. Decide on architecture for Dyck-2, Arithmetic, Toy-English (LSTM vs free-form RNN).
12. Experiments 1.3–1.5: implement remaining tasks.

**Phase D — visualisations and polish:**
13. Metric 2.7: loss landscape plots.
14. Metric 2.8: output probability / memory traces.
15. Metric 2.6: verify smoothing convention.

---

## Part 5: Reference numbers from Abudy et al.

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
