# MDL Tuning Summary (2026-02-28)

## Goal

Push the ANBN `differentiable_mdl.py` runs toward a golden / near-golden solution before using the same MDL method in ColoredMNIST.

## Key Result

The strongest result came from **softening** the MDL penalty, not from more seed sweeps:

- Run: `results/anbn_mdl/20260228_131511_basic_e10000_lr0.05_lam0.1_g10x10_s0`
- Mode: `basic`
- Config:
  - `epochs=10000`
  - `warmup_epochs=1000`
  - `lr=0.05`
  - `mdl_lambda=0.1`
  - `n_max=10`, `m_max=10`
  - `n_samples=16`
  - `tau_start=1.0`, `tau_end=0.01`
  - `batch_size=0`
  - `eval_every=200`
  - `log_every=1000`
  - `seed=0`, `data_seed=0`
  - `val_min_n=22`, `val_max_n=5000`
- Behavior:
  - hit `4979/4979` perfect on the structured validation set at the **first** evaluation
  - saved a best checkpoint immediately
- Eval to `n=100000`:
  - `gen_n = 100000`
  - `|H| = 969 bits`
  - verdict: `EMPIRICALLY_GOLDEN`
- Eval to `n=1000000`:
  - `gen_n = 1000000`
  - `|H| = 969 bits`
  - verdict: `EMPIRICALLY_GOLDEN`
- Eval to `n=10000000`:
  - started as an additional stress test
  - intentionally stopped before completion because the float64 boundary search was much slower at that scale
  - no larger-`n` failure was observed, but this run did **not** finish and should not be treated as a validated result

This is the first run in this sweep that completely broke the old `~1.8e4` plateau.

## What Mattered

1. `mdl_lambda` was the decisive lever.
2. Wider validation still matters, but once `val_max_n=5000` is in place, the main bottleneck for `lambda=1.0` appears to be **overcompression**.
3. A useful reminder: the hand-constructed golden network is `1137 bits`, so a learned model at `~780 bits` can be **too simple** to represent the real counting mechanism.

## Targeted Experiments

### 1. `lambda=1.0`, `val_max_n=5000`, finer checkpointing

- Run: `results/anbn_mdl/20260228_124122_basic_e10000_lr0.05_lam1.0_g10x10_s0`
- Change: `eval_every=100` instead of `200`
- Result:
  - still converged to the same best checkpoint quality as the previous `lambda=1.0` best
  - `gen_n = 18674` at `n=100000`
  - `|H| = 787 bits`
- Conclusion:
  - finer checkpoint timing did **not** move the ceiling for `lambda=1.0`

### 2. `lambda=1.0`, `val_max_n=10000`

- Run: `results/anbn_mdl/20260228_124435_basic_e10000_lr0.05_lam1.0_g10x10_s0`
- Result before stopping:
  - eventually reached `gen_n = 2809` on the `n=22..10000` validation set
  - checkpoint was saved
- Practical issue:
  - much slower than the `5000`-range runs
  - expensive enough that it was not the best use of the slot once the softer-regularization direction became clearly better
- Conclusion:
  - promising but too slow for the marginal gain we were likely to get

### 3. `lambda=0.5`

- Run: `results/anbn_mdl/20260228_130635_basic_e10000_lr0.05_lam0.5_g10x10_s0`
- Result before stopping:
  - by the warmup/ST transition it still had not produced meaningful validation progress
  - first few ST evaluations remained at `gen_n = 0`
- Conclusion:
  - softer than `1.0`, but not soft enough to change the qualitative behavior quickly

### 4. `lambda=0.1`

- Run: `results/anbn_mdl/20260228_131511_basic_e10000_lr0.05_lam0.1_g10x10_s0`
- Result:
  - immediate jump to full validation perfection
  - empirical generalization through `n=10^5` and `n=10^6`
- Conclusion:
  - this is the current best direction by a wide margin

### 5. Shared mode quick probe

- Run: `results/anbn_mdl/20260228_133450_shared_e3000_lr0.05_lam0.1_g10x10_s0`
- Config:
  - `mode=shared`
  - `epochs=3000`
  - `warmup_epochs=500`
  - `lambda1=0.1`
  - `lambda2=0.1`
  - other settings matched the successful basic runs
- Result before stopping:
  - stayed at `gen_n = 0` through the end of warmup on the `n=22..5000` validation set
- Conclusion:
  - shared mode is still not the fastest path to a strong solution, even after softening the KL terms

## Model Notes For The Winning `lambda=0.1` Checkpoint

- `64` non-zero weights
- `964` weight bits (`969` total including architecture)
- Analysis detected:
  - a clean `COUNTER` component in cell dimension `2`
  - very sharp `+1` / `-1` counter increments
- Caveat:
  - still marked `EMPIRICALLY_GOLDEN`, not `PROVEN_GOLDEN`
  - hidden-to-hidden weights are non-zero and the analyzer reports some gate leakage, so there is still a theoretical possibility of failure at larger `n`

## Infrastructure Fixes Made During The Sweep

1. `src/utils/checkpointing.py`
   - run directories are now collision-safe
   - if a timestamped name already exists, a `_r1`, `_r2`, ... suffix is added
   - this prevents parallel launches in the same second from corrupting the same output directory
2. `src/utils/checkpointing.py`
   - `TeeLogger` now flushes on every write
   - `train.log` can now be tailed while a run is still active

## Recommended Next Steps

1. Stress-test the winning checkpoint at even larger `n` (for example `10^7` and `10^8`).
2. Sweep a narrow band around the new winner:
   - `mdl_lambda = 0.05`
   - `mdl_lambda = 0.2`
3. If the ANBN result stays stable, port the same softer-regularization setting into the ColoredMNIST MDL runs as the new default starting point.
