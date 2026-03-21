# Comparison Targets & Metrics

What we need to show, what numbers to hit, and where the comparisons come from.

Updated 2026-03-21.

---

## 1. Metrics

### 1.1 Test |D:H| and Δ% (Abudy et al. comparison)

**|D:H|** is the cross-entropy of the model on the grammar-weighted exhaustive test set, in bits. It measures how well the model approximates the true probability distribution of the grammar — not just accept/reject, but the right *probabilities* at every step.

**Δ%** = (test |D:H| − optimal |D:H|) / optimal |D:H| × 100. A value of 0% means the model has perfectly learned the grammar distribution.

**Optimal |D:H|** is computed analytically from the PCFG for each task.

**Smoothing convention:** Abudy et al. add 10⁻¹⁰ to output probabilities before computing log-loss (marked with * in their tables when a network assigns zero probability). Our code uses `log_softmax` which is numerically stable but not identical. Verify this for apples-to-apples reporting.

### 1.2 |H| (hypothesis codelength)

The codelength in bits needed to encode the network hypothesis. Smaller = simpler model in the MDL sense. Computed via the Lan et al. rational encoding scheme: each weight w = ±n/m gets `ℓ(w) = 1 + |E(n)| + |E(m)|` bits.

### 1.3 Deterministic Accuracy (Lan et al. comparison)

Fraction of correct argmax predictions at positions where the next token is fully determined (from the first `b` onward in a^nb^n, including the final `#`). Evaluated on strings n=1 to n=1500 (Lan et al.) or much longer for length generalization claims.

### 1.4 Recognition Accuracy (FLaRe / Deletang comparison)

Binary classification: given a string, does it belong to the language? A string is "accepted" if and only if the model's argmax prediction is correct at *every* position; otherwise "rejected." This is equivalent to per-string deterministic accuracy = 100% vs < 100%.

**To compare against FLaRe:** also need to test on *negative* examples (invalid strings). FLaRe uses 50/50 positive/negative split. Negative examples = random strings + perturbations of valid strings.

**Why implement it:** It puts our results directly in the same table as Deletang et al. (ICLR 2023, ~4000 citations) and FLaRe (ICLR 2025). Low implementation cost, high payoff.

---

## 2. Comparison Targets per Task

### 2.1 a^nb^n

**Optimal test |D:H|:** 2.94 bits

| Method | Source | |H| (bits) | Δ_test (%) | Det. Acc. (n≤1500) |
|--------|--------|-----------|-----------|---------------------|
| GD + No reg | Abudy Exp 3 | ~5524 | 0.2% | — |
| GD + L1 (λ=0.1) | Abudy Exp 3 | ~5854 | 0.5% | — |
| GD + L2 (λ=0.1) | Abudy Exp 3 | ~5610 | 0.3% | — |
| GA + MDL (free-form RNN) | Abudy Exp 1 | 139 | 0.1% | — |
| Golden LSTM (Lan 2024) | Lan Exp | ~3920 total MDL | — | 100% |
| Best GD LSTM (Lan 2024) | Lan Exp | — | — | 77.33% (fail@n=73) |
| **Our golden LSTM** | This work | **1137** | — | 100% (to n=2²⁴) |
| **Our best trained** | This work | **701 total MDL** | TBD | 100% (to n=10⁶) |

**Targets for us:**
- Δ_test ≈ 0% (matching GA+MDL upper bound)
- |H| dramatically smaller than GD baselines (~5500)
- Deterministic accuracy 100% well beyond n=1500
- Recognition accuracy 100% on length-500 strings (beating FLaRe LSTM ~0.91)

### 2.2 a^nb^nc^n

**Optimal test |D:H|:** 2.94 bits

| Method | Source | |H| (bits) | Δ_test (%) |
|--------|--------|-----------|-----------|
| GD + No reg | Abudy Exp 3 | ~20043 | 0.1% |
| GD + L1 | Abudy Exp 3 | ~20541 | 0.9% |
| GD + L2 | Abudy Exp 3 | ~19909 | 2.1% |
| GA + MDL (free-form RNN) | Abudy Exp 1 | 241 | 0.0% |
| Golden (free-form RNN) | Lan 2022 | 241 | 0.0% |

**Targets:** Δ_test ≈ 0%, |H| << 20000. Perfect generalization beyond training length.

### 2.3 Dyck-1

**Optimal test |D:H|:** 1.77 bits

| Method | Source | |H| (bits) | Δ_test (%) |
|--------|--------|-----------|-----------|
| GD + No reg | Abudy Exp 3 | ~4876 | 1.0% |
| GD + L1 | Abudy Exp 3 | ~5138 | 5.0% |
| GD + L2 | Abudy Exp 3 | ~4874 | 4.4% |
| GA + MDL (free-form RNN) | Abudy Exp 1 | 113 | 0.1% |
| Golden (free-form RNN) | Abudy | 113 | 0.8% |

**Targets:** Δ_test < 1%, |H| << 4800.

### 2.4 Dyck-2 (stretch)

**Optimal test |D:H|:** 2.32 bits

| Method | Source | |H| (bits) | Δ_test (%) |
|--------|--------|-----------|-----------|
| GA + MDL | Abudy Exp 1 | 327 | 2.0% |
| Golden (free-form RNN) | Lan 2022 | 579 | — |

No GD baseline exists (Abudy couldn't run GD on this task). Any GD result is novel.
Requires stack-like computation. Standard LSTMs may not suffice.

### 2.5 Arithmetic Syntax (stretch)

**Optimal test |D:H|:** 3.96 bits

| Method | Source | |H| (bits) | Δ_test (%) |
|--------|--------|-----------|-----------|
| GA + MDL | Abudy Exp 1 | 431 | −0.4% |
| Golden (free-form RNN) | Abudy | 967 | — |

No GD baseline. Golden net uses Floor, Modulo-4, Abs, Unsigned Step activations.

### 2.6 Toy-English (stretch)

**Optimal test |D:H|:** 4.49 bits

| Method | Source | |H| (bits) | Δ_test (%) |
|--------|--------|-----------|-----------|
| GA + MDL | Abudy Exp 1 | 414 | 2.0% |
| Golden (free-form RNN) | Abudy | 870 | — |

No GD baseline. Golden net uses Unsigned Step + Multiplication units.

---

## 3. Broader Length-Generalization Benchmarks

These use recognition accuracy (binary accept/reject), not |D:H|.

### Deletang et al. (ICLR 2023)

Trained up to length 40, tested on longer strings. Key results:
- LSTMs generalize on counter languages (a^nb^n type) but fail on context-free (Dyck, stack)
- Only stack/tape-augmented RNNs generalize on CF and CS languages
- 20,910 models tested across 15 tasks

### FLaRe (ICLR 2025)

Trained on length [0,40], tested on length [0,500]. Recognition accuracy (mean ± std, 10 runs):

| Task | Transformer | RNN | LSTM |
|------|-------------|-----|------|
| Modular Arithmetic | 0.69±0.11 | **1.00±0.00** | 0.98±0.03 |
| Parity | 0.56±0.03 | 0.71±0.24 | **0.90±0.20** |
| Dyck-(2,3) | 0.70±0.09 | **0.95±0.05** | 0.91±0.10 |
| Stack Manipulation | 0.66±0.14 | **0.84±0.16** | 0.75±0.17 |

Nobody achieves 100% recognition accuracy on anything beyond regular languages with standard architectures.

---

## 4. Progression Goals

**Level 1 (minimum publishable):**
Across a^nb^n, a^nb^nc^n, Dyck-1: Δ_test substantially below GD+L1/L2 baselines. |H| at least an order of magnitude smaller. Generalization to 10x+ training length with >95% deterministic accuracy. The story: "differentiable MDL dramatically improves compression and generalization over standard regularizers."

**Level 2 (strong paper):**
Perfect or near-perfect generalization on a^nb^n (done). Strong generalization on a^nb^nc^n and Dyck-1 (>95% det. acc. at 10–100x training length). Δ_test close to 0%. Recognition accuracy implemented and compared against FLaRe. The story: "first gradient-based method approaching GA+MDL quality."

**Level 3 (excellent paper):**
Perfect generalization on all counter languages. Meaningful Dyck-2 or stack-language results (even imperfect). FLaRe comparison showing MDL-LSTM beats all standard LSTMs. The story: "MDL regularization makes gradient descent find the right algorithm."

---

## 5. Golden Network Discrepancies

### 5.1 The source papers

Three papers form the chain of golden network constructions:

1. **Weiss, Goldberg & Yahav (2018)**: "On the Practical Computational Power of Finite Precision RNNs for Language Recognition" (ACL 2018 Short Papers). Provides the general recipe for implementing a counting mechanism in an LSTM — saturated gates, unit increments/decrements in the cell state. This is the recipe our LSTM golden net follows via Lan et al. (2024).

2. **Lan, Chemla & Katzir (2022)**: "Minimum description length recurrent neural networks" (TACL, 10:785–799). Constructs golden networks as *free-form RNNs* (directed graphs with units, weighted connections, per-unit activation functions) rather than LSTMs. These are much more compact. This is what Abudy et al. use.

3. **Lan, Chemla & Katzir (2024)**: "Bridging the empirical-theoretical gap..." (ACL 2024). Uses the Weiss et al. recipe to construct an LSTM golden net with hidden_size=3 and LARGE=127 for saturated gates. This is what our code follows.

### 5.2 Architecture mismatch: LSTM vs free-form RNN

**Our golden net:** LSTM with hidden_size=3, following Lan et al. (2024) / Weiss et al. (2018).
- 108 total parameters (4×3×3 input weights + 4×3×3 hidden weights + 4×3 + 4×3 biases + 3×3 + 3 output layer)
- 83 of these are structural zeros (gates that are unused but must still be encoded)
- **|H| = 1137 bits** (415 bits wasted on encoding zeros)

**Abudy's golden net for a^nb^n:** Free-form RNN following Lan et al. (2022 TACL).
- Minimal directed graph — only the connections that are actually needed
- **|H| = 139 bits**

This is an 8× difference in |H|, entirely due to architecture, not algorithm quality. The LSTM has structural overhead (unused gates, hidden-to-hidden weights that are all zero) that free-form RNNs avoid.

### 5.3 Implications for our comparisons

When comparing against Abudy et al.'s results:
- Their GD baselines (Exp 3) also use free-form RNNs with |H| ~5000–20000. These are not LSTMs.
- Comparing our LSTM |H| against their free-form RNN |H| is not apples-to-apples.
- The *relative improvement* (our MDL vs our baselines, all using the same LSTM architecture) is the clean comparison.
- The Δ_test comparison is fair regardless of architecture — it only depends on how well the model approximates the grammar distribution, not on |H|.

### 5.4 Output layer construction

Our golden LSTM output layer uses transcendental weights (log probabilities divided by tanh(1)), approximated as rationals via `Fraction.limit_denominator(1000)`. This produces weights like 4443/362 and -5033/395, costing 45–49 bits each. The output layer alone costs 393 bits (9 weights + 3 biases), about 35% of total |H|.

Lan et al. (2024) use the same construction and the same `limit_denominator(1000)` convention.

Abudy et al.'s free-form RNN golden nets may use simpler output constructions since the architecture is more flexible.

### 5.5 Zero encoding cost

Each zero-valued parameter costs 5 bits: 1 (sign) + 1 (E(0)) + 3 (E(1)). Our LSTM has 83 zeros, costing 415 bits — more than the entire Abudy golden net. This is an inherent cost of using a fixed LSTM architecture vs a minimal free-form graph.

### 5.6 |H| discrepancy with Lan et al. (2024)

Lan et al. (2024) Table 1 reports 3920 as the golden network's MDL loss. This appears to be the *total* MDL objective value L_MDL = L_D + λ·Σℓ(θ_i), not just |H|. Their λ value and the data term L_D are folded in.

Our separate computation gives |H| = 1137 bits for the same LSTM golden net. This is consistent — the total MDL of 3920 includes the data term on the training set.

### 5.7 Recommendations

1. **For Δ_test comparisons:** Safe to compare directly against Abudy et al. regardless of architecture. The metric only depends on output quality, not model size.
2. **For |H| comparisons:** Only compare our LSTM |H| against other LSTM |H| values. Do not compare against free-form RNN |H| values.
3. **For deterministic accuracy:** Safe to compare against any architecture.
4. **For recognition accuracy:** Safe to compare against FLaRe/Deletang (they also test LSTMs).
5. **If we implement free-form RNNs later:** Then direct |H| comparison against Abudy becomes valid.

---

## 6. Threats to Fair Comparison

Concrete issues found by comparing our code (`src/mdl/`, `config/anbn_mdl/basic_train.yaml`) against the experimental details in Abudy et al. (2025, Appendix D) and Lan et al. (2024).

### 6.1 Test set starts at n=0, Abudy starts at n=1

**Status: MUST FIX before reporting Δ_test.**

`compute_grammar_weighted_nll_bits` in `evaluation.py` iterates `for n in range(0, max_n + 1)` — includes the empty string (n=0). Abudy et al. (2025, Appendix "Task-specific probabilistic grammars") explicitly state: "test includes all n = 1, ..., 1020". The empty string has PCFG weight P(0) = 0.3, which is the largest single weight in the distribution. Including or excluding it will shift the absolute |D:H| and therefore Δ_test.

**Fix:** add a `min_n` parameter to `compute_grammar_weighted_nll_bits` and default it to 1 for Abudy-comparable reporting. Keep the n=0 variant available for information-theoretic completeness, but clearly label which convention each number uses.

### 6.2 Test set range differs: we use fixed n=1..1500, Abudy uses n=1..(n_max_train + 1000)

**Status: SHOULD VERIFY whether this matters numerically.**

Our `test_max_n: 1500` is hardcoded in the config. Abudy's test set goes to n_max_train + 1000 — so if training samples go up to n=20, the test set covers n=1..1020. With Geometric(p=0.3) weighting, P(1500) ≈ 5.6×10⁻²³⁵ — effectively zero. In practice this likely makes no numerical difference to the grammar-weighted |D:H|, but it should be documented. The more important difference is that our grammar weights are raw P(n) = p(1−p)^n (unnormalized, summing to ~1 for large max_n), while Abudy says "weighted by their true probabilities" — confirm whether they normalize.

### 6.3 Learning rate is 500× higher than Abudy's GD baseline

**Status: DEFENSIBLE but must be stated explicitly.**

Our config: `lr: 0.05` (Adam). Abudy Experiment 3 (Appendix D): Adam with lr = 10⁻⁴. Lan et al. (2024): Adam with lr = 0.001, β₁=0.9, β₂=0.999.

This is a 500× difference vs Abudy and 50× vs Lan. It's defensible because:
- We optimize categorical logits (softmax parameterization), not continuous weights directly. The gradient magnitudes and loss landscape are structurally different.
- Abudy's GD experiment starts from golden weights and trains 1000 epochs with a free-form RNN. We train from random init for 10000 epochs with an LSTM. These are fundamentally different experimental setups.

**Nevertheless**, a reviewer will check this. The paper must state our lr and justify the difference. Consider running a small ablation over lr ∈ {0.001, 0.01, 0.05, 0.1} to show results are not cherry-picked.

### 6.4 Abudy's GD baselines use free-form RNNs, not LSTMs

**Status: MUST BE STATED clearly in any comparison table.**

Confirmed from Abudy et al. (2025, Section 4.3): "we extend the analysis to new tasks and free-form RNNs." Their Experiment 3 GD baselines all use free-form RNNs. Our experiments use LSTMs.

Consequences:
- **Δ_test comparison is still valid** — it only measures output quality, not model size.
- **|H| comparison is NOT valid** across architectures (1137 vs 139 for golden; ~5500 vs our LSTM |H| for trained).
- The cleanest argument is within-architecture: "our MDL-LSTM achieves |H|=X vs baseline-LSTM |H|=Y, a Z× improvement." Cross-architecture comparisons should be presented as context, not direct comparisons.

### 6.5 Rational grid cannot represent the golden solution

**Status: BY DESIGN, but must be understood and stated.**

Our config: `n_max: 10, m_max: 10`, producing ~68 grid values with numerators ≤ 10 and denominators ≤ 10. The LSTM golden network uses `LARGE = 127` for saturated gates. The value 127 = 127/1 requires n_max ≥ 127, which is far beyond our grid. Even `shared_train.yaml` with n_max=15 cannot represent 127.

This means: the trained network **cannot** discover the exact Lan et al. golden solution by construction. It must find an alternative solution. This is actually fine and arguably interesting — if our method finds a *different* network with similar |D:H| but smaller |H| (because it uses simpler weights), that's a positive result. But we must acknowledge that the golden LSTM is not in our search space.

**Impact on reporting:** Our trained |H| and the golden |H| = 1137 are computed over the same coding scheme, so comparison is fair. But saying "our method found a solution with |H| = X < 1137" would be misleading if the golden solution was unreachable anyway. The honest framing is: "our method finds solutions in the restricted grid S with |H| = X, while the golden LSTM requires weights outside S."

### 6.6 Training loss averages over positions; |D:H| sums then grammar-weights

**Status: NOT A BUG, but an asymmetry to document.**

`_compute_data_nll_bits` in `training.py` divides by `sum(mask)` — per-position average. The reported test |D:H| (`compute_grammar_weighted_nll_bits`) sums over positions per string, then grammar-weights across strings. The training objective thus weights all positions equally regardless of string length, while the evaluation metric implicitly weights positions in longer strings more (since NLL is summed per string, and longer strings have more terms).

This is standard practice (most LM papers train on per-token loss but report per-sequence or per-dataset metrics), but it means we're not directly optimizing the exact metric we report.

### 6.7 λ/N scaling makes effective λ depend on corpus size

**Status: DOCUMENT clearly.**

Our training objective is: `data_nll_avg + (lambda/N) × E[|H|] - (1/N) × tau × H(pi)`. With `lambda=1000` and `N=1000`, the effective weight on |H| is 1.0 per string. But Abudy's GA optimizes `|D:H|_total + |H|` directly — pure MDL with no λ. Their GD experiment uses `lambda=1` for L1/L2 on continuous weights, which is a different regularizer entirely.

The paper should report the effective regularization strength, not just the config value. State explicitly: "effective λ_eff = mdl_lambda / N = 1.0."

### 6.8 Number of seeds

**Status: MUST RUN multiple seeds for submission.**

Deletang et al. (2023): 10 seeds. FLaRe (2025): 10 seeds × 4 loss variants. Abudy Experiment 3: single seed (seed=100, corpus_seed=100). Our config: single seed (seed=0, data_seed=0).

For Abudy-comparable experiments, a single seed may suffice (they only ran one). For FLaRe/Deletang comparison or any claim of consistent improvement, we need at least 5 seeds with mean ± std.

### 6.9 Smoothing convention: training vs evaluation

**Status: ALREADY HANDLED, verify consistency.**

Training uses `optax.softmax_cross_entropy_with_integer_labels` (numerically stable log_softmax). Evaluation uses additive 1e-10 smoothing matching Abudy (implemented in `compute_data_nll_bits_smoothed` and `compute_per_string_nll_bits`). This split is correct — train with stable numerics, report with Abudy's convention. But verify the two give similar values on the golden network; any significant gap would indicate a bug.

### 6.10 Abudy GD starts from golden weights; we start from random init

**Status: DIFFERENT EXPERIMENTAL QUESTION, not a bug.**

Abudy Experiment 3 trains free-form RNNs "starting from the golden weights" with L1/L2/none for 1000 epochs. Their question: "does GD preserve the golden solution under different regularizers?" Our question: "can GD + differentiable MDL *discover* good solutions from scratch?"

These are complementary experiments. We should present both:
1. From-scratch training → demonstrates our method's discovery capability.
2. From-golden-init training (if we implement free-form RNN) → directly comparable to Abudy Exp 3.

Neither alone tells the whole story.

---

## 7. Implementation Tasks: Architecture Verification & Free-Form RNN

### 7.1 Task A: Verify LSTM golden network |H| against Lan et al. (2024)

**Goal:** Confirm our LSTM golden network's |H| matches Lan et al. (2024) exactly.

**Context:** Our `golden.py` already builds the LSTM golden net and computes |H| = 1137 bits. Lan et al. (2024) report a total MDL objective value of 3920 for the golden net (Table 1), but this is L_MDL = L_D + λ·Σℓ(θ_i), not |H| alone. We need to decompose their number and verify our |H| matches their weight-encoding component.

**Steps:**

1. Run `golden_mdl_score()` and record the breakdown:
   - Architecture encoding bits (should be `|E(hidden_size=3)|` = 5 bits)
   - Per-weight bits for all 108 parameters
   - Total |H|

2. Cross-check against Lan et al. (2024) Appendix B:
   - Verify the LSTM equations match (gate order, bias convention, transpose convention)
   - Verify the golden weight matrices match exactly (W_ig, W_io, b_ii, b_if, b_io)
   - Verify the output layer construction: ε = (2¹⁴ − 1)⁻¹, `limit_denominator(1000)`, tanh(1) scaling
   - Verify we encode ALL 108 parameters including zeros (Lan encodes every weight sequentially)
   - **Key paper reference:** Lan et al. (2024) Appendix B (lines 517–687 of `agent-context/arxiv/arXiv-2402.10013v2_contents/main-acl.tex`)

3. Compute the data term L_D on the Lan et al. training set (1000 strings from Geometric(p=0.3), train/val split 950/50) and check whether |H| + L_D ≈ 3920 under their λ convention.

4. **If there's a mismatch:** compare our `integer_code_length` and `rational_codelength` against the encoding scheme described in Lan et al. (2024) Section 3.2 (see `agent-context/arxiv/arXiv-2402.10013v2_contents/main-acl.tex`). The self-delimiting code is E(n) = 1^{k(n)} 0 bin_{k(n)}(n), |E(n)| = 2k(n)+1, k(n) = ceil(log2(n+1)).

**Expected result:** Our |H| = 1137 bits is correct for the LSTM golden net. The 3920 from Lan's table includes the data term. Document the exact decomposition.

---

### 7.2 Task B: Implement free-form RNN golden network for a^nb^n

**Goal:** Implement the Lan et al. (2022 TACL) / Abudy et al. (2025) free-form RNN architecture, build the a^nb^n golden network, and verify |H| = 139 bits exactly.

This is split into two stages: Stage 1 gets the architecture and coding scheme working (verified by bit count), Stage 2 makes it trainable with our differentiable MDL objective.

#### Stage 1: Golden network construction & |H| verification

**What is a free-form RNN?** A directed graph of computational units. Each unit computes:
```
output_t = activation(bias + Σ_j w_j · input_j_t + Σ_k v_k · hidden_k_{t-1})
```
Connections are explicit — only edges that exist in the graph are computed and encoded. Connections are typed as **forward** (from input or same-timestep units) or **recurrent** (from previous-timestep hidden units). Each unit has an independently chosen activation function.

**The a^nb^n golden network (Abudy et al. 2025, Figure 7; originally from Lan et al. 2022):**

The golden network is a 6-unit free-form RNN. The exact topology is specified in the figure (agent-context/arxiv/arXiv-2505.13398v2_contents/, look for an_bn.pdf or the figure reference at lines 580–584 of main.tex). There is also a **differentiable variant** (Figure 8, diff_an_bn.pdf, lines 588–591) that replaces non-differentiable activations (Step) with differentiable ones (Linear, ReLU) for use with gradient descent.

Activations used in the golden net: Linear, ReLU, Sigmoid, Unsigned Step.
Activations in the differentiable variant: Linear, ReLU only.

Weights are simple rationals — values like 2, -3, 2.33 (=7/3), -15, 1. These should be encodable with short codelengths.

**Encoding scheme (Lan et al. 2022 TACL — CRITICAL: read this paper):**

The encoding for a free-form RNN encodes:
1. Number of units (prefix-free integer code)
2. For each unit: activation function type (from a finite set: {linear, ReLU, tanh, sigmoid, step, ...})
3. For each unit: bias value (rational, prefix-free coded)
4. Connection topology: which connections exist, each marked as forward or recurrent
5. For each connection: weight value (rational, prefix-free coded: 1 sign bit + |E(numerator)| + |E(denominator)|)

**The crucial difference from LSTM encoding:** only connections that *exist* are encoded. No bits are wasted on zero-weight connections. This is why |H| = 139 for the free-form RNN vs 1137 for the LSTM — the LSTM must encode 83 zeros at 5 bits each.

**The exact encoding details are in Lan et al. (2022)** (see `agent-context/arxiv/arXiv-2111.00600v4_contents/main-3649-Lan.tex`). Abudy et al. reference it (line 259) but do not reproduce the full encoding scheme. The Lan 2022 paper is essential for verifying the 139-bit count.

**Implementation steps:**

1. **Read Lan et al. (2022) encoding scheme** from the TACL paper (arXiv:2111.00600). Extract the exact bit-level encoding for: unit count, activation type per unit, bias per unit, connection adjacency, connection type (forward/recurrent), weight per connection.

2. **Create `src/mdl/freeform_rnn.py`** with:
   - A `FreeFormRNN` data structure: list of units (activation, bias), list of connections (source_unit, target_unit, weight, is_recurrent)
   - A `freeform_forward(network, x)` function that executes the graph on batched input sequences. Must handle topological ordering of forward connections and recurrent connections.
   - Input convention: one-hot encoded symbols, same as LSTM (`{#, a, b}` = 3 symbols for a^nb^n)
   - Output: logits over next symbol (same as LSTM output layer)

3. **Create `src/mdl/freeform_coding.py`** with:
   - `freeform_codelength(network)` — compute |H| for a free-form RNN following Lan et al. (2022) encoding
   - Must handle: unit count encoding, activation type encoding, bias encoding, adjacency encoding, weight encoding

4. **Build the a^nb^n golden network** as a `FreeFormRNN` instance. Hard-code the topology from Abudy et al. Figure 7.

5. **Verify |H| = 139 bits exactly.** If it doesn't match, debug the encoding scheme against Lan et al. (2022). Common pitfalls:
   - Off-by-one in the unit count encoding
   - Different enumeration of activation types
   - Different convention for encoding adjacency (bitmap vs explicit edge list)
   - Different convention for encoding the sign of zero biases

6. **Run the golden network on a^nb^n strings** (n=1..1500) and verify 100% deterministic accuracy, matching Lan et al. (2024) and Abudy et al. (2025).

7. **Also build the differentiable variant** (Figure 8) which uses only Linear and ReLU. Verify it also achieves perfect accuracy. Compute its |H| — it may differ from 139 since different activations have different costs.

**Files to create:**
- `src/mdl/freeform_rnn.py` — architecture definition and forward pass
- `src/mdl/freeform_coding.py` — encoding scheme and |H| computation
- `src/mdl/golden_freeform.py` — golden network construction for a^nb^n (and later other tasks)
- `tests/test_freeform_golden.py` — verification: |H| = 139, accuracy = 100%

**Reference files:**
- `agent-context/arxiv/arXiv-2505.13398v2_contents/main.tex` — Abudy et al. (2025), Figure 7, Table 1
- `agent-context/arxiv/arXiv-2402.10013v2_contents/main-acl.tex` — Lan et al. (2024), for comparison
- `agent-context/arxiv/arXiv-2111.00600v4_contents/` — Lan et al. (2022 TACL), encoding scheme

#### Stage 2: Make the free-form RNN trainable with differentiable MDL

**Goal:** Apply our categorical relaxation (Section 3 of the paper) to the free-form RNN, so we can train it with gradient-based MDL and compare directly against Abudy et al.

**This is a larger task and should only be started after Stage 1 is verified.**

**Key design decisions:**

1. **Fixed topology vs learnable topology:** The simplest approach is to fix the graph topology (e.g., use the golden network's topology) and only learn the weights via categorical relaxation over the rational grid. This avoids the combinatorial problem of learning which connections exist. The topology can be fixed to the golden network's topology, or to a slightly larger "supergraph" that includes the golden net as a subgraph.

2. **Activation functions:** If the topology is fixed, the activation functions can also be fixed. Alternatively, make activation choice a discrete variable optimized alongside weights — but this adds significant complexity.

3. **Categorical relaxation:** Each weight in the free-form RNN becomes a categorical distribution over the rational grid S, exactly as in our LSTM implementation. The forward pass, Gumbel-Softmax, temperature annealing, etc. all transfer directly.

4. **MDL objective for training:** The coding cost changes — instead of summing over all LSTM parameters, sum over the free-form RNN's connections. The architecture prefix (unit count, activation types, topology) is constant if the topology is fixed, so it drops out of the gradient.

5. **Comparison with Abudy Experiment 3:** Their GD experiment starts from golden weights and trains with L1/L2/none. We should do the same (start from golden, train with our differentiable MDL) AND also train from random initialization (which Abudy doesn't do with GD).

**Implementation steps:**

1. Create `src/mdl/freeform_training.py` — adapt the existing categorical MDL training loop to work with the free-form RNN forward pass instead of the LSTM forward pass.

2. Define the rational grid S for the free-form RNN. The golden net weights (2, -3, 7/3, -15, 1) are all simple rationals, so the grid doesn't need to be large.

3. Implement the per-connection coding cost: for connection with weight w, cost = `rational_codelength(w)`. For the fixed-topology variant, this is the only trainable cost.

4. Train and evaluate:
   - From golden initialization → verify we stay near the golden solution
   - From random initialization → the real test: can gradient-based MDL discover the golden solution?
   - Compare against L1/L2/no-reg baselines on the same architecture

5. Report |H|, Δ_test, and deterministic accuracy. These are directly comparable to Abudy et al. Table 1.

---

## 8. Key Reference Papers

Papers available in `agent-context/arxiv/` that are essential for the implementation tasks above.

### For Task A (LSTM golden net verification)

- **Lan, Chemla & Katzir (2024)** — "Bridging the Empirical-Theoretical Gap..." (ACL 2024). Appendix B has the exact LSTM golden network construction, weight matrices, output layer derivation, and encoding scheme. See `agent-context/arxiv/arXiv-2402.10013v2_contents/main-acl.tex`.

- **Weiss, Goldberg & Yahav (2018)** — "On the Practical Computational Power of Finite Precision RNNs for Language Recognition" (ACL 2018 Short). Source of the LSTM counting recipe (saturated gates, unit increment/decrement in cell state). Cited by Lan et al. (2024) as the basis for the golden LSTM. See `agent-context/arxiv/arXiv-1805.04908v1_contents/`.

### For Task B (free-form RNN)

- **Lan, Geyer, Chemla & Katzir (2022)** — "Minimum Description Length Recurrent Neural Networks" (TACL 10:785–799). **This is the critical reference.** Contains the complete free-form RNN encoding scheme: how to encode unit count, activation types, biases, connection topology (adjacency), and weights. The 139-bit golden network for a^nb^n originates here. See `agent-context/arxiv/arXiv-2111.00600v4_contents/main-3649-Lan.tex`.

- **Abudy et al. (2025)** — "A Minimum Description Length Approach to Regularization in Neural Networks" (NeurIPS 2025). Figure 7 has the a^nb^n golden network diagram, Figure 8 has the differentiable variant. Table 1 has the |H| = 139 and Δ_test values. Experiment 3 has the GD baselines. See `agent-context/arxiv/arXiv-2505.13398v2_contents/main.tex`.

### For recognition accuracy comparisons

- **Deletang et al. (2023)** — "Neural Networks and the Chomsky Hierarchy" (ICLR 2023). The 20,910-model benchmark across 15 formal language tasks with recognition accuracy. Establishes that standard LSTMs fail to generalize on counter and context-free languages. See `agent-context/arxiv/arXiv-2207.02098v3_contents/`.

- **FLaRe / Svete et al. (2025)** — "Training Neural Networks as Recognizers of Formal Languages" (ICLR 2025). Current benchmark for formal language recognition. Trained on [0,40], tested on [0,500]. Provides the LSTM/RNN/Transformer baselines we compare against. See `agent-context/arxiv/arXiv-2411.07107v3_contents/main.tex`.
