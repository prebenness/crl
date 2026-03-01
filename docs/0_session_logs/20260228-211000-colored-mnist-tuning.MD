# ColoredMNIST Tuning Summary (2026-02-28)

## Goal

Two immediate goals:

1. Find an MDL-regularized inner model that is clearly dominated by the spurious color signal.
2. Get the strongest outer held-out test accuracy we can with `mdl_pair`.

Dataset used throughout:

- `cmnist_ula`
- train bias: `p_train = 0.95`
- test bias: `p_test = 0.1`

Interpretation:

- A color-dominated inner model should have train accuracy near the train color correlation (`~0.95`) and much lower test accuracy.

## 1. Inner MDL Model (`mdl`)

### Run

- Sweep dir:
  - `results/colored_mnist/20260228_211013_cmnist_mdl_single_lambda_1e-1_cmnist_ula_mdl`
- Config:
  - single lambda
  - `lambda = 1e-1`
  - `epochs = 20`
  - `warmup_epochs = 5`
  - `lr = 1e-3`
  - `n_max = 5`, `m_max = 5`
  - `tau_start = 2.0`, `tau_end = 0.1`
  - `n_samples = 4`
  - `batch_size = 128`

### Final result

- `train_acc = 0.9537`
- `test_acc = 0.2407`

### Interpretation

This is strongly color-biased:

- train accuracy is almost exactly at the train color correlation level
- test accuracy is far below what a digit-semantic model should achieve

It is not pure “color only” in the strictest sense (test is above random-chance `0.1`), but it is clearly dominated by the spurious color shortcut and is a useful inner model for paired experiments.

## 2. MDL Pair (`mdl_pair`)

### Best saved final result

- Sweep dir:
  - `results/colored_mnist/20260228_211407_cmnist_mdl_pair_lambda_1e-1_cmnist_ula_mdl_pair`
- Config:
  - `lambda = 1e-1`
  - `hsic.weight = 0.5`
  - `epochs = 20`
  - same MDL inner settings as above

### Final result

- inner:
  - `train_acc1 = 0.9538`
  - `test_acc1 = 0.2411`
- outer:
  - `train_acc2 = 0.9987`
  - `test_acc2 = 0.8719`

This is the strongest saved final run from this sweep block.

## 3. Comparison Runs

### Lower MDL penalty (`lambda = 1e-2`)

- Sweep dir:
  - `results/colored_mnist/20260228_211812_cmnist_mdl_pair_lambda_1e-2_cmnist_ula_mdl_pair`
- Final outer result:
  - `test_acc2 = 0.8451`

This is worse than `lambda = 1e-1`.

### Higher HSIC weight (`hsic.weight = 0.8`)

- Sweep dir:
  - `results/colored_mnist/20260228_212213_cmnist_mdl_pair_lambda_1e-1_hsic_0p8_cmnist_ula_mdl_pair`
- Final outer result:
  - `test_acc2 = 0.8588`

This is worse than the `hsic=0.5` final run.

### Important note

The `hsic=0.8` run had a slightly better *observed* mid-training peak:

- `outer_test_acc = 0.8725` at epoch 17

but it did not hold that by the final epoch, so the best *saved* final result remains the `hsic=0.5` run at `0.8719`.

## Current Best

### Best inner-spurious model

- `results/colored_mnist/20260228_211013_cmnist_mdl_single_lambda_1e-1_cmnist_ula_mdl`
- final:
  - `train_acc = 0.9537`
  - `test_acc = 0.2407`

### Best saved outer test accuracy

- `results/colored_mnist/20260228_211407_cmnist_mdl_pair_lambda_1e-1_cmnist_ula_mdl_pair`
- final:
  - `outer test_acc = 0.8719`

## Next Steps

1. Keep `lambda = 1e-1` as the first MDL-pair baseline.
2. Add explicit outer-model checkpointing / best-epoch saving.
   - The higher-HSIC run briefly beat the best final result mid-training, but that point was not saved.
3. If continuing the sweep, vary one thing at a time:
   - `hsic.weight` around `0.5`
   - training epochs / early stopping
   - then MDL grid size (`5x5` to `10x10`) if runtime budget allows
