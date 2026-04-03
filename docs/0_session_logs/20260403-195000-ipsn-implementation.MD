# IPSN Implementation Session (2026-04-03)

## What was done

Implemented IPSN (Interventional Path Surgery Network) as a new experiment mode in the codebase, based on Proposal 5 from the `ipsn-chatgpt.md` design document. Inspired by domain-adversarial training (Ganin et al., 2016, "Domain-Adversarial Training of Neural Networks", JMLR 17) applied as path-specific causal intervention.

### IPSN method summary

- **Encoder**: Shared trunk E(x) splits into content code c and colour code b.
- **Training**: 5-term loss:
  - CE(y, g(c)): digit classification from content code
  - lambda * CE(s, h_s(b)): colour classification from colour code
  - gamma * CE(s, a_s(grad_rev(c))): adversarial colour prediction from content (gradient reversal makes this push c away from colour info)
  - rho * L1(x, D(c,s)): reconstruction / recolouring
  - nu * sum_s' KL(g(c) || g(E_c(D(c,s')))): cycle consistency (decode to each colour, re-encode, verify digit prediction is stable)
- **Inference**: g(E_c(x)) only. No oracle needed at test time.
- **Architecture**: 3x100 MLP encoder trunk + auxiliary heads (all smaller than trunk).

### Files modified

- `src/config.py` -- added `IPSNConfig` dataclass
- `src/training/train_state.py` -- added `create_state_ipsn`
- `src/training/steps.py` -- added `make_train_step_ipsn`, `make_eval_step_ipsn`
- `src/training/epochs.py` -- added `make_train_epoch_ipsn`
- `src/training/runners.py` -- added `run_train_eval_ipsn`
- `colored_mnist.py` -- added `ipsn` mode dispatch, imports, model registry entry

### Files created

- `src/models/ipsn_classifier.py` -- `IPSNMlp` Flax module with `setup()` for sub-method calls
- `config/colored_mnist/ipsn.yaml` -- experiment config
- `docs/0_session_logs/20260403-195000-ipsn-implementation.MD` -- this file

## Results

### IPSN with default config (gamma_adv=0.5, nu_cycle=0.1)

Training collapses at epoch 8 due to adversarial instability. ce_adv explodes from ~3 to millions. Best test acc before collapse: 0.3885 (epoch 4).

### IPSN with stabilised config (gamma_adv=0.1, nu_cycle=0)

| Metric | Value |
|--------|-------|
| Best test accuracy | **0.8366** (epoch 34) |
| Final test accuracy | 0.8035 |
| Train accuracy | 0.9971 |
| Training time | 16.3s (49 epochs, early stopped) |

### Comparison

| Method | Test Acc |
|--------|----------|
| ERM | ~0.10 |
| MDL Pair | 0.8755 |
| **IPSN (gt oracle, gamma=0.1)** | **0.8366** |
| VIB Pair | 0.8846 |
| Oracle + HSIC | 0.884 |
| CBA-OM (gt oracle) | 0.9047 |
| Oracle + MMD | 0.9224 |
| CCDB target | ~0.96 |

### gamma_adv sweep (nu_cycle=0, rho_recon=0.1, gt oracle)

| gamma_adv | Best Test Acc | Notes |
|-----------|--------------|-------|
| 0.05 | **0.838** | Stable, best result |
| 0.10 | 0.818 | Stable |
| 0.20 | 0.113 | Collapses |
| 0.30 | 0.113 | Collapses |

### Soft colour label refactor

After initial results, refactored IPSN to accept soft colour distributions (B, num_colors)
instead of hard integer labels. `nn.Embed` replaced with `nn.Dense(embed_dim, use_bias=False)`.
Colour and adversary losses use soft-target CE. Hard labels are one-hot encoded at the boundary.
This enables Phase 2: using full predicted distributions from a trained inner model, not just argmax.

## Observations

- Gradient reversal works: adversary converges to ce_adv ~2.3 (= ln(10), chance level), confirming content code c is colour-invariant.
- Colour classifier also works: ce_color converges to ~0, confirming colour code b captures colour.
- Adversarial training is fragile: gamma_adv >= 0.2 collapses. Stable range is [0.05, 0.1].
- Best result (0.838) is below VIB pair (0.8846) and CBA-OM (0.9047). The method works but doesn't yet outperform simpler approaches.
- Reconstruction weight, cycle consistency, and learning rate schedule remain unexplored.

## Next steps

- Sweep rho_recon with gamma_adv=0.05
- Enable cycle consistency (nu_cycle > 0) once base is tuned
- Try gradient clipping to allow higher gamma_adv without collapse
- Try learning rate warmup or adversarial weight scheduling
- Phase 2: model oracle (argmax and soft predictions) instead of ground truth colours
