# ColoredMNIST Current Status

## Benchmark Context

The near-term target is to approach the `~96%` cMNIST result range discussed in the CCDB paper, while keeping comparisons meaningful to uLA-style setups.

## Current Best Results (`cmnist_ula`, ULA-comparable models)

### Best Oracle Pair + MMD Result (2026-04-01)

- Run:
  - `results/colored_mnist/20260401_2*_oracle_pair_mmd_*` (multiple runs, see session log)
- Best outer test accuracy:
  - **`0.9224`** (MMD weight=0.5, batch=256, lr=1e-3, wd=1e-4, 200 epochs)
- Oracle: frozen color-predicting inner model (train acc ~95%), outer trained with weighted CE + MMD.
- All MMD runs plateau around 0.92 — likely architecture ceiling for 3x100 MLP.

### Best Oracle Pair + HSIC Result (2026-04-01)

- Best outer test accuracy:
  - `0.884` (HSIC weight=0.5, 100 epochs)
- HSIC now uses `one_hot(argmax(inner_logits))` as color signal (same as MMD).
- MMD outperforms HSIC by ~4 points under identical conditions.

### Best VIB Pair Result (2026-02-28)

- Run:
  - `results/colored_mnist/20260228_220856_cmnist_pair_ula_lam1e-3_alpha1e-3_wd1e-4_e40_cmnist_ula_pair`
- Final outer test accuracy:
  - `0.8846`
- Inner model behavior:
  - `test_acc1 = 0.1133`
  - At chance, but `train_acc1` was also ~11% — the inner model learned nothing, not color-only.
  - See inner model investigation notes below.

### Best Observed VIB Pair Peak (2026-02-28)

- Run:
  - `results/colored_mnist/20260228_220952_cmnist_pair_ula_lam3e-3_alpha1e-3_wd1e-4_e40_cmnist_ula_pair`
- Best observed outer test accuracy during training:
  - `0.8896`
- Final value dropped after the peak.

### Best MDL Pair Result (2026-02-28)

- Run:
  - `results/colored_mnist/20260228_215807_cmnist_mdl_pair_g10_lam1e-1_hsic0p8_softtau_cmnist_ula_mdl_pair`
- Final outer test accuracy:
  - `0.8755`

### Best CBA-OM Result (2026-04-03)

- Best test accuracy:
  - `0.9047` (wd=1e-4, lr=1e-3, batch=128, 50 epochs, gt oracle)
- Single-stage CE training on colour-conditioned classifier p(y|x,s), backdoor
  marginalisation at test time. No distribution matching or reweighting.
- Beats HSIC and VIB pair with far simpler pipeline; ~2 points below Oracle+MMD.
- See session log `20260403-182900-cba-om-implementation.MD`.

### Best IPSN Result (2026-04-03)

- Best test accuracy:
  - `0.838` (gamma_adv=0.05, nu_cycle=0, rho_recon=0.1, gt oracle)
- Disentangled encoder with gradient-reversed adversary for colour invariance.
  Adversary converges to chance (ce_adv ≈ ln(10)), confirming content code is colour-free.
- Adversarial training is fragile: gamma_adv ≥ 0.2 causes training collapse.
  Stable range is gamma_adv ∈ [0.05, 0.1].
- Reconstruction weight and cycle consistency not yet tuned.
- See session log `20260403-195000-ipsn-implementation.MD`.

## Interpretation

- Oracle pair + MMD is the current best path at **0.9224**.
- CBA-OM (0.9047) is a strong second with a much simpler pipeline.
- IPSN (0.838) works but underperforms simpler methods; adversarial instability limits it.
- MMD consistently beats HSIC (~0.92 vs ~0.88) with the same oracle inner model.
- The gap to the `~96%` target is narrowing but still significant.
- The 0.92 ceiling may be fundamental to the 3x100 MLP — needs investigation.
- Best-checkpoint saving implemented (2026-03-30), summary table reports best (2026-04-01).

## Inner Model Investigation (2026-03-30)

The inner VIB model (`ula_mlp_var`, 50-dim bottleneck) must learn to capture the color
shortcut for the pair mechanism to work: target ~95% train acc, ~10% test acc on `cmnist_ula`.

### Key findings

- On **binary** `colored_mnist` with Jan 28 hyperparams (`lr=1e-3, wd=1e-2, alpha=0.01,
  epochs=5`), the ULA MLP achieves a clean color-only regime at lambda 0.5–7:
  train ~86–89%, test ~10%.
- On **10-class** `cmnist_ula` with identical hyperparams, the color-only regime does not
  appear. The model transitions directly from "learns nothing" to "learns both color and shape".
- The transition is too sharp: at every lambda where train acc is high enough to indicate
  color was learned, test acc is well above 10% (shape has leaked through).

### Best inner-model-only results so far (10-class `cmnist_ula`)

Best train/test gap (most color-dominant) achieved with:
- `lr=3e-4, wd=1e-2, alpha=0.01, epochs=20, lambda~5–9`
- Train ~91–93%, test ~25–27%
- ~20 points of shape information consistently leaks through.

### Hyperparameters explored

| Lever | Values tried | Effect |
|-------|-------------|--------|
| Alpha (CE vs recon) | 0.01, 0.1, 0.5, 0.9 | Higher alpha = worse (more shape leaks). Best at 0.01. |
| Weight decay | 1e-4, 1e-2 | 1e-2 better (matches working binary setup) |
| Learning rate | 1e-4, 3e-4, 1e-3 | 3e-4 gives best gap; lower LR doesn't help further |
| Epochs | 5, 10, 20, 40 | More epochs helps train converge but shape leaks equally |
| Lambda | 1e-3 to 1e3 | Sweet spot around 5–9 nats for 10-class |

### Observations on binary vs 10-class

- Binary `colored_mnist` produces a clean color-only regime with the same model and
  hyperparams. 10-class `cmnist_ula` does not, under any hyperparams tried so far.
- One known difference: binary test set has anti-correlated colors (p_test=0.1 flips the
  label), making the color-only regime clearly visible as a dip below chance. 10-class test
  set has uniform random colors, so a color-only model scores at chance (10%) —
  indistinguishable from having learned nothing.
- It is unclear whether the regime truly doesn't exist for 10-class or whether we haven't
  found the right hyperparameters yet.

### Open questions

- Why does shape leak through so readily on 10-class? Is it the wider capacity gap,
  the test correlation structure, the bottleneck width (50 vs 16), or something else?
- Would a fixed KL penalty (instead of ControlVAE) give tighter control?
- Could architectural changes (e.g. removing the post-bottleneck hidden layer) help
  without violating ULA comparability?
- Is the pair mechanism with HSIC effective even when the inner model isn't purely
  color-dominant?

## Immediate Next Work

1. ~~Add or use explicit best-epoch selection for outer accuracy.~~ Done (2026-03-30).
2. ~~Implement MMD conditional invariance outer loss.~~ Done (2026-04-01). See session log `20260401-200000-conditional-invariance-mmd.MD`.
3. ~~Implement CBA-OM (causal backdoor adjustment).~~ Done (2026-04-03). Best 0.9047.
4. ~~Implement IPSN (interventional path surgery network).~~ Done (2026-04-03). Best 0.838.
5. Break the 0.92 MMD plateau: try different seeds, stronger regularization, LR scheduling.
6. Run VIB pair (non-oracle) with MMD — the real use case where inner model learns jointly.
7. IPSN tuning: sweep rho_recon, enable cycle consistency (nu_cycle), try gradient clipping for higher gamma_adv.
8. Resolve the inner model color-only regime problem for 10-class.
9. Continue tuning only on ULA-style models (`ula_mlp_var` + `ula_mlp`, and `mdl_mlp` + `ula_mlp`).
10. Improve the ANBN MDL method further before leaning harder on MDL pair for cMNIST.
