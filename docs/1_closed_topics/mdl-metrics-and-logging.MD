# MDL Metrics And Logging

This document defines the unified MDL metric names used across the ANBN and ColoredMNIST code paths, and explains what each term means.

### 1) One identity to keep in mind

Every MDL run logs terms so this is always true:

`objective_total_* = data_nll_* + reg_complexity_weighted_* - reg_entropy_bonus_*`

So if a run looks confusing, read it as:

- `data_nll_*` pushes fit to labels.
- `reg_complexity_weighted_*` penalizes model complexity.
- `reg_entropy_bonus_*` is subtracted, so it rewards higher posterior entropy over weight values.

### 2) Why there are multiple entropy terms

You will see two entropy-related quantities:

- `entropy_weights_*`: raw entropy of categorical weight distributions.
  - This is `sum_i H(pi_i)`.
  - It is a descriptive quantity (how diffuse the weight posteriors are).
- `reg_entropy_bonus_*`: the entropy contribution after optimization scaling.
  - This is what actually enters the objective as a subtraction term.
  - It includes factors like `tau` and train-size scaling.

So:

- `entropy_weights_*` tells you how uncertain weights are.
- `reg_entropy_bonus_*` tells you how much that uncertainty changes the loss.

### 3) Term-by-term definitions

- `objective_total_*`: exact scalar optimized by gradient descent.
- `data_nll_*`: negative log-likelihood / cross-entropy data term.
- `complexity_expected_*`:
  - Basic MDL (`a^n b^n`): `E[|H|] = sum_i sum_m pi_{i,m} * l(s_m)`.
  - Shared MDL (`a^n b^n`): `lambda1 * code_cross_entropy_bits + lambda2 * KL(phi||P_base)` before batch scaling.
  - ColoredMNIST MDL: expected code-length penalty for MLP weights.
- `reg_complexity_weighted_*`: complexity contribution after train-size/batch scaling (and lambda weighting).
- `entropy_weights_*`: raw weight entropy (`sum_i H(pi_i)`).
- `reg_entropy_bonus_*`: weighted entropy bonus actually used in objective.
- `reg_net_*`: `reg_complexity_weighted_* - reg_entropy_bonus_*`.
- `complexity_argmax_bits`: complexity of current argmax-discretized model. Monitoring only, not the optimized expectation.

Shared-MDL extra terms:

- `code_cross_entropy_bits`: `sum_i CE_2(pi_i, phi) = -sum_i sum_m pi_{i,m} log2 phi_m`.
- `kl_pi_phi_bits`: `sum_i KL(pi_i || phi)`.
- `kl_phi_pbase_bits`: `KL(phi || P_base)`.
- `phi_entropy_bits`: entropy of shared prior `phi`.
- `phi_min_prob`, `phi_max_prob`: min/max entry of `phi` (helps detect collapse).

### 4) Exact per-mode formulas

Notation: `N` = train set size, `tau = 1/beta`.  All paths use averaged data NLL + `1/N` scaling on regularization terms.

ANBN basic mode (`differentiable_mdl.py`, bits):

`objective_total_bits = mean_data_nll_bits + (1/N) * mdl_lambda * complexity_expected_bits - (1/N) * tau * entropy_weights_bits`

ANBN shared mode (`differentiable_mdl.py`, bits):

`objective_total_bits = mean_data_nll_bits + (1/N) * (lambda1 * code_cross_entropy_bits + lambda2 * kl_phi_pbase_bits) - (1/N) * tau * entropy_weights_bits`

ColoredMNIST MDL (`colored_mnist.py`, nats):

`objective_total_nats = mean_data_nll_nats + (1/N) * mdl_lambda * complexity_expected_nats - (1/N) * tau * entropy_weights_nats`

### 5) Units

- ANBN logs are in bits.
- ColoredMNIST MDL logs both:
  - `*_nats` (native optimization units there).
  - `*_bits` (converted by dividing by `ln(2)`).

### 6) Naming policy

The MDL code paths now emit the unified names above directly.
Old MDL metric aliases have been removed.
