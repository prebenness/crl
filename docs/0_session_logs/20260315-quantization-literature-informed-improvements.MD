# Quantization-literature-informed MDL improvements

## Motivation

External review identified that the differentiable MDL implementation uses the same technical machinery (categorical distributions over a grid + Gumbel-Softmax + temperature annealing) as the neural network quantization literature — specifically Louizos et al. (2019, "Relaxed Quantization for Discretized Neural Networks," arXiv:1810.01875) — without engaging with lessons learned from that line of work. This session addresses that gap by surveying the relevant quantization literature and implementing the most impactful changes.

## Literature surveyed

| Paper | Key takeaway for this project |
|-------|-------------------------------|
| Louizos et al., ICLR 2019 (arXiv:1810.01875) — Relaxed Quantization | Categorical over grid + Gumbel-Softmax. Temperature management is the critical challenge; initialization strongly affects which grid region the optimizer explores. |
| Lee et al., ICLR 2021 (arXiv:1911.12990) — Semi-Relaxed Quantization / DropBits | Gumbel argmax can select grid points far from the mode, causing catastrophic forward-pass variance. Fix: use mode of π in forward pass, Gumbel soft sample in backward pass only. |
| Défossez et al., TMLR 2022 (arXiv:2104.09987) — DiffQ | Avoids full M-dimensional categorical per weight; uses continuous noise + ST rounding + differentiable code length. Dramatic memory savings. |
| Grathwohl et al., ICLR 2018 (arXiv:1711.00123) — RELAX | Learned control variate for unbiased, low-variance gradients through discrete variables. Best of REINFORCE (unbiased) and ST (low variance). |

## Implementation plan

Seven changes were proposed, prioritized by effort/impact/risk:

| Priority | Change | Effort | Impact | Risk | Status |
|----------|--------|--------|--------|------|--------|
| **P0** | 1. Mode-based forward pass | 1-line × 2 files | High (variance reduction) | Very low | **Done** |
| **P0** | 5. P_base init for shared prior φ | 3 lines × 1 file | Medium (faster shared convergence) | Very low | **Done** |
| **P1** | 2. Codelength-informed logit init | New initializer + config | Medium (better starting point) | Low | **Done** |
| **P1** | 6. n_samples optimization for mode forward | ~20 lines | Medium (compute savings) | Low | Remaining |
| **P2** | 3. Staged annealing with discretization checkpoints | ~150 lines across 6 files | High (addresses premature collapse) | Medium | Remaining |
| **P2** | 4. DiffQ-style continuous relaxation | New module ~200 lines | High for CMNIST (memory) | Medium | Remaining |
| **P3** | 7. RELAX estimator | New module ~300 lines | Marginal if P0-P2 work | High | Remaining |

## What was implemented

### P0 — Change 1: Mode-based forward pass (Lee et al. 2021)

**Problem:** In the Gumbel ST branch, `idx = argmax(y_soft)` selects the argmax of Gumbel-perturbed logits. If π puts 95% mass on `1/2` but residual mass on `-3/1`, a Gumbel noise realization can select `-3/1`, producing a catastrophic loss spike and high-variance gradient.

**Fix:** When `mode_forward=True`, the forward pass uses `idx = argmax(all_logits)` (mode of π, no Gumbel noise) while the backward pass still flows through the Gumbel-Softmax soft sample.

**Files modified:**
- `src/mdl/lstm.py:141-150` — `jax.lax.cond` on `mode_fwd` in the vmapped `sample_one`
- `src/models/mdl_classifiers.py:183-188` — `jnp.where` on `self.mode_forward` (vectorized)

### P0 — Change 5: P_base initialization for shared prior φ

**Problem:** `phi_logits` was initialized to zeros (uniform softmax). Training wasted epochs "discovering" that simple rationals should have higher prior mass.

**Fix:** Initialize `phi_logits = -l(s_m) * ln(2)` so `softmax(phi_logits) ≈ P_base`.

**File modified:**
- `src/training/train_state.py:76-79` — `create_state_mdl_shared`

### P1 — Change 2: Codelength-informed logit initialization

**Problem:** LSTM logits initialized with `normal(0.1)` give near-uniform softmax over all ~350 grid points, including expensive ones like 127/128 (19 bits). MLP logits had Kaiming peaks but zero background.

**Fix:** New `codelength_informed_init` sets logits to `-scale * l(s_m)` + small noise, concentrating initial mass on cheap rationals (0, ±1, ±1/2). For MLP, the Kaiming peak is preserved on top of the codelength-informed background.

**Files modified:**
- `src/mdl/lstm.py:19-38` — new `codelength_informed_init()` function
- `src/mdl/lstm.py:112-118` — conditional init selection based on `init_cl_scale`
- `src/models/mdl_classifiers.py:17-76` — `kaiming_categorical_init` extended with `grid_codelengths` and `cl_scale` params

### Config & CLI additions

- `src/config.py` — `MDLConfig.mode_forward: bool = False`, `MDLConfig.init_cl_scale: float = 0.0`
- `differentiable_mdl.py` — `--mode_forward` flag, `--init_cl_scale` float arg, both in config serialization
- `colored_mnist.py` — threaded to `GumbelSoftmaxMLP` constructor

All defaults preserve exact backward compatibility with existing experiments.

## What remains

### P1 — Change 6: n_samples optimization for mode forward (~20 lines)

When `mode_forward=True`, the forward pass is deterministic (same mode every sample). Running K forward passes wastes compute — only one is needed; Gumbel noise only affects the backward path. Short-circuit in `_mdl_loss` and `_mdl_shared_loss`.

### P2 — Change 3: Staged temperature annealing (~150 lines, 6 files)

Replace smooth exponential decay with fixed-temperature stages:
1. Train at `tau_start` until convergence → discretize confident weights (max(π_i) > threshold) → freeze via gradient mask
2. Continue uncertain weights at `tau_mid`
3. Final stage at `tau_end`

Requires: `freeze_mask` in `MDLTrainState`, gradient masking in step functions, stage transition logic in training loops. Gated by `anneal_mode: "staged"` config.

### P2 — Change 4: DiffQ-style continuous relaxation (new module, ~200 lines)

Parallel implementation path for ColoredMNIST where the full categorical (108K × 350 = 38M floats) is a memory bottleneck. One continuous weight + noise scale per parameter instead of M logits. Conforms to existing `(logits, aux)` interface.

### P3 — Change 7: RELAX estimator (new module, ~300 lines)

Defer until P0-P2 results are evaluated. Only needed if mode-forward + staged annealing still show problematic gradient variance.

## Validation

```
./venv/bin/pytest tests/ --ignore=tests/test_mdl_inner_parity.py -v
# 113 passed in 86.68s
```

The 2 failures in `test_mdl_inner_parity.py` are pre-existing (exact float comparison `atol=0.0, rtol=0.0` fails on both old and new code due to JIT nondeterminism).

## How to use the new features

```bash
# ANBN with mode forward + codelength-informed init
python3.12 differentiable_mdl.py config/anbn_mdl/basic_train.yaml \
    --mode_forward --init_cl_scale 1.0

# ColoredMNIST via YAML config
mdl:
  mode_forward: true
  init_cl_scale: 0.5
```

## Recommended next steps

1. Run A/B experiments on ANBN: `--mode_forward` vs baseline, `--init_cl_scale 1.0` vs `0.0`, across existing best schedules.
2. Run ColoredMNIST sweeps with `mode_forward: true` and compare variance/convergence.
3. Based on results, decide whether P2 changes (staged annealing, DiffQ) are needed.
