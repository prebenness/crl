# Colored MNIST Training Setups

This document describes the four training modes available in `colored_mnist.py`. Each mode runs a lambda sweep, training a fresh model for each lambda value to explore the accuracy-regularisation tradeoff.

All modes use Colored MNIST: binary digit labels (digit < 5), with digit colour correlated to label at rate `p_train` in train and `p_test` in test. The distribution shift (`p_train=0.9`, `p_test=0.1`) means a model relying on colour will perform well on train but poorly on test.

---

## 1. `single` — VIB with ControlVAE

A single Variational Information Bottleneck model. The encoder maps input to a stochastic latent `z ~ N(mu, diag(exp(logvar)))`, from which a classifier head and a reconstruction decoder both operate.

**Inner model loss:**

```
L_inner = (1 - beta) * L_task + beta * KL
```

where:

- `L_task = alpha * CE(y, f(z)) + (1 - alpha) * BCE(x, decoder(z))`
  - `CE` — softmax cross-entropy between classifier logits and labels, averaged over K MC samples of z
  - `BCE` — sigmoid binary cross-entropy between reconstruction logits and input pixels, averaged over K MC samples
  - `alpha` — mixing weight (default 0.01, so reconstruction dominates the task loss)
- `KL = D_KL(q(z|x) || N(0, I)) = 0.5 * mean_over_batch[ sum_d (mu_d^2 + exp(logvar_d) - 1 - logvar_d) ]`
- `beta` — adaptive weight from ControlVAE dual-ascent:
  - `beta(t) = clip(beta(t-1) + ctrl_ki * (sg(KL) - lambda), beta_min, beta_max)`
  - `lambda` is the swept capacity target

**Uses reconstruction loss:** Yes. The `SimpleDecoder` (2-layer MLP) reconstructs the input from z, and `BCE` is part of the task loss.

**What lambda controls:** Information capacity target. The ControlVAE controller adjusts beta to push KL toward lambda. Low lambda = tight bottleneck (less information passes through z), high lambda = loose bottleneck.

```bash
# Edit configs/default.yaml: set model.mode to "single"
python colored_mnist.py configs/default.yaml
```

---

## 2. `pair` — VIB inner + HSIC outer

Two models trained jointly. The inner model is the same VIB as in `single` mode. The outer model is a standard (non-variational) classifier whose representation is decorrelated from the inner model's representation via class-conditional HSIC.

**Inner model loss:** Identical to `single` mode above.

**Outer model loss:**

```
L_outer = (1 - w_hsic) * CE(y, g(z2)) + w_hsic * HSIC(sg(mu1), z2 | y)
```

where:

- `CE` — softmax cross-entropy for the outer classifier
- `HSIC(sg(mu1), z2 | y)` — class-conditional HSIC (Hilbert-Schmidt Independence Criterion) with RBF kernels, measuring statistical dependence between the inner model's bottleneck mean `mu1` (stop-gradiented) and the outer model's representation `z2`, conditioned on class label
  - Computed per-class then averaged over classes with >= 2 members
  - Both representations are standardized before kernel computation
- `w_hsic` — HSIC loss weight (default 0.50)
- `sg()` — stop_gradient (inner model gradients don't flow through the outer loss)

The idea: the inner model captures shortcuts (colour) via its bottleneck; the outer model is forced to find features independent of the inner representation, which pushes it toward shape-based classification that generalises under distribution shift.

**Uses reconstruction loss:** Yes, in the inner model (same as `single`). The outer model does not use reconstruction.

**What lambda controls:** Same as `single` — VIB capacity target for the inner model.

```bash
python colored_mnist.py configs/default.yaml
```

---

## 3. `mdl` — MDL-regularised MLP

A single MLP where every weight and bias is parameterized as a categorical distribution over a finite grid of rational numbers `S = {+/-n/m : 0<=n<=n_max, 1<=m<=m_max, gcd(n,m)=1}`. Training uses Gumbel-Softmax straight-through to select discrete rational weights while maintaining gradient flow.

**Inner model loss:**

```
L = CE(y, f(theta)) + lambda * (1/N) * L_hyp - (1/N) * tau * H
```

where:

- `CE` — softmax cross-entropy between classifier logits and labels, **averaged** over the batch
- `L_hyp = sum_i sum_m pi_{i,m} * l(s_m)` — expected hypothesis codelength
  - `pi_{i,m} = softmax(alpha_i)_m` — probability that weight i takes grid value s_m (computed from the categorical logits, no Gumbel noise)
  - `l(s_m)` — description length in bits of rational value s_m under the Li & Vitanyi self-delimiting code: `l(+/-n/m) = 1 + |E(n)| + |E(m)|` where `|E(k)| = 2*ceil(log2(k+1)) + 1`
  - Simpler rationals (small numerator/denominator) have shorter codes, so the penalty biases toward simple weights
- `H = sum_i H(pi_i) = sum_i [ -sum_m pi_{i,m} * log2(pi_{i,m}) ]` — total entropy of the categorical weight distributions
  - `tau * H` is the entropy bonus: at high temperature it encourages exploration (uniform distributions over the grid), at low temperature it vanishes and weights crystallize to point masses
- `tau` — Gumbel-Softmax temperature, annealed exponentially from `tau_start` to `tau_end` over training
- `1/N` — scaling factor (N = total training set size). Since CE is averaged over the batch, the hypothesis and entropy terms use `1/N` (not `B/N`) so that over a full epoch each term accumulates correctly
- `lambda` — the swept MDL penalty weight

**Training phases:**
1. **Warmup** (first `warmup_epochs` epochs): continuous relaxation (each weight = expected value under softmax), zero-variance gradients for stable initialization
2. **Main training** (remaining epochs): Gumbel-Softmax straight-through (discrete weights in forward pass, soft gradients in backward)

**Uses reconstruction loss:** No. The model has no decoder. Regularisation comes entirely from the weight description length penalty.

**What lambda controls:** Weight complexity penalty strength. Low lambda = almost no MDL pressure (model free to use any weights), high lambda = strong pressure toward simple rational weights (short description).

```bash
python colored_mnist.py configs/mdl.yaml
```

---

## 4. `mdl_pair` — MDL inner + HSIC outer

Two models trained jointly. The inner model is the same MDL-regularised MLP as in `mdl` mode. The outer model is a standard classifier decorrelated via class-conditional HSIC, identical to the outer model in `pair` mode.

**Inner model loss:** Identical to `mdl` mode above. The 100-dimensional bottleneck layer activations serve as the representation `z1` for HSIC (analogous to `mu` in VIB).

**Outer model loss:**

```
L_outer = (1 - w_hsic) * CE(y, g(z2)) + w_hsic * HSIC(sg(z1), z2 | y)
```

Identical in form to the `pair` outer loss, except `sg(z1)` comes from the MDL inner model's bottleneck rather than a VIB encoder's mean.

**Uses reconstruction loss:** No. Neither the inner nor the outer model has a decoder.

**What lambda controls:** MDL penalty weight for the inner model (same as `mdl`).

```bash
python colored_mnist.py configs/mdl_pair.yaml
```

---

## Summary

| Mode | Inner regularisation | Outer model | Reconstruction | Lambda meaning |
|------|---------------------|-------------|----------------|----------------|
| `single` | VIB (KL capacity) | — | Yes | KL capacity target |
| `pair` | VIB (KL capacity) | HSIC decorrelation | Yes (inner only) | KL capacity target |
| `mdl` | MDL (weight codelength) | — | No | Weight complexity penalty |
| `mdl_pair` | MDL (weight codelength) | HSIC decorrelation | No | Weight complexity penalty |
