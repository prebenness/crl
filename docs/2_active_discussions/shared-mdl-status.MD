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

## Why Tuning Alone Is Not Enough

A zero-KL diagnostic (`lambda1 = 0`, `lambda2 = 0`) still failed to recover basic-MDL-like behavior. That suggests the problem is structural, not just bad hyperparameters.

## Important Theory / Code Mismatch

The proposal defines the shared prior base distribution as:

- `P_base(s) ∝ 2^{-ell(s)}`

The current implementation instead uses:

- `P_base(s) ∝ 1 / (|s| + 1)`

Until that mismatch is resolved, shared-mode outcomes should not be treated as a clean test of the proposed shared-MDL theory.

## Immediate Next Work

1. Align the implemented `P_base` with the proposal.
2. Re-evaluate shared mode only after that objective mismatch is fixed.
3. Then compare basic vs shared under matched schedules and seeds.
