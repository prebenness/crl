# ColoredMNIST MDL Phase Alignment

## Scope

Applied the ANBN MDL phase-transition changes to the ColoredMNIST MDL code paths (`mdl` and `mdl_pair`).

## Important Scope Note

ColoredMNIST currently does not implement a shared-weight MDL objective (no shared `phi`, no `P_base`, no shared-code term). Its MDL paths use the basic expected-codelength objective only. That means the ANBN shared-objective correction has no direct ColoredMNIST analogue to port.

The relevant change here was the training-phase mechanics.

## Changes

1. Added `mdl.bridge_epochs` to the typed config (`src/config.py`).
2. ColoredMNIST MDL runners now resolve bridge length the same way as ANBN:
   - default `bridge_epochs = max(1, ceil(0.1 * warmup_epochs))` when warmup > 0
   - default `0` when warmup = 0
   - explicit `0` disables the bridge
   - clipped to remaining epochs
3. `tau` is now held constant through `warmup + bridge`, then annealed over the remaining ST tail.
4. The bridge is now a real multi-epoch deterministic-ST phase instead of the old one-epoch special case.
5. Adam/AdamW state is now reset automatically at MDL phase boundaries:
   - `warmup -> bridge`
   - `warmup -> st` when bridge is disabled
   - `bridge -> st`
6. For `mdl_pair`, both inner and outer optimizers are reset at phase boundaries so momentum does not carry across the inner representation regime change.

## Verification

- `python -m py_compile src/config.py src/training/runners.py colored_mnist.py`
- `./venv/bin/python -m pytest tests/test_training_runners.py -q` -> `2 passed`
- `WANDB_MODE=offline ./venv/bin/python colored_mnist.py config/colored_mnist/smoke_mdl_single_sweep.yaml`
  - observed default `bridge_epochs=1` for `warmup_epochs=1`
  - observed optimizer resets at `warmup -> bridge` and `bridge -> st`
  - observed `tau` held constant through warmup and bridge

## Notes

The smoke config has only one tail epoch, so `tau` does not visibly decrease there. On longer runs, annealing now starts only after the bridge phase.
