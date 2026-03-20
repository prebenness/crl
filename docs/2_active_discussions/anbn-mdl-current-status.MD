# ANBN MDL Current Status

## Best Confirmed Result

- Best checkpoint:
  - `results/anbn_mdl/20260301_011753_basic_e12000_lr0.03_lam0.144_g10x10_s0`
- Confirmed by post-run evaluation to `n = 1,000,000`:
  - `total_mdl_bits = 701`
  - `gen_n = 1,000,000`
  - `n_perfect = 1,000,000`

This is the current frontier point.

## What Worked

- Long warmup helped substantially.
- The strongest checkpoint was found early, during warmup, before later training degraded the run.
- This suggests the optimizer can reach a very strong basin from scratch.

## Current Risks / Open Questions

1. The result is empirically strong, but not yet proven golden.
2. Performance appears sensitive to regularization strength (`mdl_lambda`) and training schedule.
3. The historical baseline schedule near `mdl_lambda ~= 0.144` is unstable and can collapse into trivial solutions.
4. **Batch scaling convention changed (2026-03-20)**: ANBN now uses averaged data NLL + `1/N` reg scaling (matching cMNIST). The 701-bit result above was obtained under the old summed NLL + `B/N` convention. Hyperparameters (`mdl_lambda`, `lr`) will need recalibration under the new convention.

## Latest Code Change

- The ANBN training transition has been refactored in commit `53880d2`.
- Current transition is now:
  - `warmup` (soft, fixed `tau`)
  - `bridge` (deterministic ST, fixed `tau`)
  - tail (`GST` or `DST`, with annealing)
- Adam state is reset automatically at estimator boundaries.

## Immediate Next Work

1. Re-run the long-warm schedule under the new transition logic.
2. Sweep a broader `mdl_lambda` band to test whether the good region widens.
3. Check seed stability once the schedule is no longer cliff-like.
