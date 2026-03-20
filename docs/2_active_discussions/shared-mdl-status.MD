# Shared MDL Status

## Current State

Shared-weight MDL is now viable under the corrected objective, though it still trails the best basic-MDL frontier.

## Best Confirmed Shared Result

- Run:
  - `results/anbn_mdl/20260301_122915_shared_e900_lr0.03_lam0.144_g10x10_s0`
- Result:
  - `gen_n = 1,000,000`
  - `n_perfect = 1,000,000`
  - `total_mdl_bits = 907`

This is substantially better than the pre-fix shared runs and is now in the same empirical generalization regime as the current best basic-MDL checkpoint. It is still worse on compression than the best basic run (`701` bits through `1e6`).

## What Changed

- The old shared objective was structurally misaligned with the proposal.
- After correcting the objective to use `CE_2(pi_i, phi)` and setting `P_base(s) ∝ 2^{-ell(s)}`, shared mode began producing strong solutions.
- A strong anchor to `P_base` matters: `lambda2 = 100` worked well with `lambda1 = 0.144`.

## Current Theory / Code Alignment

The shared-MDL implementation has now been updated to match the revised proposal:

- shared code term: `sum_i CE_2(pi_i, phi)`
- base prior: `P_base(s) ∝ 2^{-ell(s)}`
- dictionary term: `KL(phi || P_base)`

That removes the earlier objective mismatch. Pre-fix shared-mode experiment results should not be used as evidence about the corrected method.

## Immediate Empirical Picture

1. `lambda1 = 0.144`, `lambda2 = 100`
- `907` bits
- perfect through direct evaluation to `n = 1,000,000`

2. `lambda1 = 0.144`, `lambda2 = 20`
- `945` bits
- perfect through direct evaluation to `n = 1,000,000`
- also reaches full structured validation during warmup before degrading in the ST tail

3. `lambda1 = 0.05`, `lambda2 = 100`
- did not enter the good basin in the short sweep
- failed the structured validation set early

## Immediate Next Work

1. Compare corrected shared mode against basic MDL under matched schedules and seeds.
2. Sweep `lambda2` around the viable band (`20` to `100`) to measure how broad that anchor-stable region is.
