# Shared MDL Status

## Current State

Shared-weight MDL is not currently competitive with basic MDL.

## Best Observed Shared Probe

- Run:
  - `results/anbn_mdl/20260228_205956_shared_e500_lr0.05_lam0.001_g10x10_s0`
- Result:
  - `gen_n = 3`
  - `n_perfect = 3`
  - `total_mdl_bits = 1443`

This is far below the current basic-MDL frontier.

## Why Tuning Alone Was Not Enough

A zero-KL diagnostic (`lambda1 = 0`, `lambda2 = 0`) still failed to recover basic-MDL-like behavior. That indicated the problem was structural, not just bad hyperparameters.

## Current Theory / Code Alignment

The shared-MDL implementation has now been updated to match the revised proposal:

- shared code term: `sum_i CE_2(pi_i, phi)`
- base prior: `P_base(s) ∝ 2^{-ell(s)}`
- dictionary term: `KL(phi || P_base)`

That removes the earlier objective mismatch. Existing shared-mode experiment results were obtained before that fix, so they should not be used as evidence about the corrected method.

## Immediate Next Work

1. Re-run shared mode under the corrected objective.
2. Compare against basic MDL under matched schedules and seeds.
3. Only then judge whether shared mode is viable or whether it still needs additional tuning.
