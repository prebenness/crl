# Session Log: MDL Phase Transition Refactor

## Scope

Refactored the ANBN `differentiable_mdl.py` training transition between soft warmup and straight-through training to better preserve strong warmup basins.

## Changes Made

1. Replaced ANBN's global temperature anneal with phase-aware annealing.
   - `tau` is now held at `tau_start` throughout warmup and bridge.
   - Annealing to `tau_end` begins only in the tail phase.

2. Added a new bridge phase to ANBN training.
   - New CLI flag: `--bridge_epochs`
   - If omitted, defaults to `max(1, ceil(0.1 * warmup_epochs))` when warmup is enabled.
   - If explicitly set to `0`, the bridge is disabled.
   - Bridge uses deterministic straight-through (hard argmax forward, soft gradients).

3. Updated both `basic` and `shared` training loops to use a 3-stage schedule.
   - `warmup` -> `bridge` -> `GST` / `DST` tail

4. Made Adam-state reset mandatory at estimator boundaries.
   - Reset on:
     - `warmup -> bridge`
     - `warmup -> GST` / `DST` when bridge is disabled
     - `bridge -> GST`
     - `GST -> DST` when `--det_st_after_tau` triggers
   - No disable flag was added; this is now the supported behavior.

5. Updated resume override handling so `--bridge_epochs` survives resume/eval config restoration.

6. Improved console phase descriptions.
   - Logs now describe warmup, bridge, and tail separately.
   - The printed tau schedule now reflects the actual hold-then-anneal behavior.

## Verification

- `python -m py_compile differentiable_mdl.py` passed.
- Ran a small smoke test:
  - confirmed default bridge inference (`warmup_epochs=4` -> `bridge_epochs=1`)
  - confirmed `tau` stayed fixed through warmup and bridge
  - confirmed optimizer reset messages appeared at phase boundaries

## Commit

- Commit created: `53880d2` (`Refine MDL phase transitions`)

## Why This Matters

The previous ANBN implementation sharpened the relaxation during warmup and then switched abruptly into ST training. The strongest recent checkpoint (`701` bits through `n=1e6`) was found during warmup and then degraded later, so the transition itself is a plausible source of instability. This refactor is intended to make the transition gentler and reduce sensitivity to a narrow `mdl_lambda` band.
