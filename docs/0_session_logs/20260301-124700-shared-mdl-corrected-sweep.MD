# Shared MDL Corrected Sweep

## Scope

Ran the first focused shared-MDL experiments after aligning the implementation to the revised proposal.

## Main Result

Corrected shared mode can now produce strong ANBN solutions.

Best confirmed run:
- `results/anbn_mdl/20260301_122915_shared_e900_lr0.03_lam0.144_g10x10_s0`
- `lambda1 = 0.144`
- `lambda2 = 100`
- `total_mdl_bits = 907`
- direct float64 evaluation: perfect through `n = 1,000,000`

This is the first shared-mode result in the same empirical generalization regime as the current best basic-MDL run, though it is still worse on compression than the basic `701`-bit frontier.

## Additional Result

Secondary viable run:
- `results/anbn_mdl/20260301_123700_shared_e900_lr0.03_lam0.144_g10x10_s0`
- `lambda1 = 0.144`
- `lambda2 = 20`
- `total_mdl_bits = 945`
- direct float64 evaluation: perfect through `n = 1,000,000`

This run reached full structured validation during warmup and then degraded in the ST tail, with the saved best checkpoint preserving the useful warmup basin.

## Negative Result

- `lambda1 = 0.05`, `lambda2 = 100` failed the structured validation set early in the short sweep.
- The earlier pre-fix style setting with weaker anchoring (`lambda2 = 5`) also failed badly.

## Interpretation

1. The objective fix was essential. Shared mode became viable only after switching to `CE_2(pi_i, phi)` and `P_base(s) ∝ 2^{-ell(s)}`.
2. A strong anchor to `P_base` matters. The viable region currently appears to require a moderate-to-strong `lambda2`.
3. Shared mode now shows the same qualitative training pattern as basic MDL: the best checkpoint appears during warmup and later ST can damage it.

## Next Work

1. Sweep `lambda2` more systematically in the viable band (`20` to `100`).
2. Compare shared vs basic under matched seeds.
3. Consider initializing `phi` at `P_base` rather than uniform, since the current start likely still makes shared optimization harder than necessary.
