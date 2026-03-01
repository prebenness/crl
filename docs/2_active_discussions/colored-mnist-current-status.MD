# ColoredMNIST Current Status

## Benchmark Context

The near-term target is to approach the `~96%` cMNIST result range discussed in the CCDB paper, while keeping comparisons meaningful to uLA-style setups.

## Current Best Results (`cmnist_ula`, ULA-comparable models)

### Best Final VIB Pair Result

- Run:
  - `results/colored_mnist/20260228_220856_cmnist_pair_ula_lam1e-3_alpha1e-3_wd1e-4_e40_cmnist_ula_pair`
- Final outer test accuracy:
  - `0.8846`
- Inner model behavior:
  - `test_acc1 = 0.1133`
  - effectively near chance, which means the inner model is strongly shortcut-dominated

### Best Observed VIB Pair Peak

- Run:
  - `results/colored_mnist/20260228_220952_cmnist_pair_ula_lam3e-3_alpha1e-3_wd1e-4_e40_cmnist_ula_pair`
- Best observed outer test accuracy during training:
  - `0.8896`
- Final value dropped after the peak.

### Best Final MDL Pair Result

- Run:
  - `results/colored_mnist/20260228_215807_cmnist_mdl_pair_g10_lam1e-1_hsic0p8_softtau_cmnist_ula_mdl_pair`
- Final outer test accuracy:
  - `0.8755`

## Interpretation

- The VIB pair path is currently stronger than the MDL pair path on outer accuracy.
- The gap to the `~96%` target is still large.
- Peak-vs-final behavior suggests model selection / best-checkpoint saving is still a meaningful bottleneck.

## Immediate Next Work

1. Add or use explicit best-epoch selection for outer accuracy.
2. Continue tuning only on ULA-style models (`ula_mlp_var` + `ula_mlp`, and `mdl_mlp` + `ula_mlp`).
3. Improve the ANBN MDL method further before leaning harder on MDL pair for cMNIST.
