# Prime-Exponent Relaxation: Deferred Extensions

Tracking future improvements to the prime-exponent continuous relaxation for MDL-style rational regularization.

## Sign Handling

- **Hard sign with STE**: Replace `tanh(u_i)` with binary `sign(u_i)` in the forward pass, using a straight-through estimator for gradients. This would make the sign discrete during training, which may help convergence to clean rational weights. The current smooth `tanh` sign was chosen for simplicity in v1.

## Initialization

- **Xavier/He-informed initialization**: Instead of `N(0, 0.01)` for all exponent coordinates, compute a per-layer target weight scale (e.g., He init: `sqrt(2/fan_in)`) and set `z` values so that `exp(z^T log(primes))` matches that scale. May require a per-layer scalar scale factor `rho`.

## Sparsity & Zero Handling

- **Zero gates (`g_i`)**: Add a learnable gate parameter `q_i` per weight, with `g_i = sigmoid(q_i)`, so that `s_i = g_i * sigma_i * exp(a_i)`. Include a gate cost `(1 - g_i) * c_0` in the regularizer to allow exact zeros without infinite exponent penalty.

## Scale Factors

- **Per-layer scalar scale factor `rho`**: Factor each layer as `W = rho * W_tilde` where `rho` is a learned real scalar and `W_tilde` uses the prime-exponent structure. Lets the network set overall magnitude easily while MDL shapes the fine structure.

## Alternative Parameterizations

- **Positive/negative exponent split**: Replace single real exponent `z_{i,r}` with two nonneg variables: `z_{i,r} = u_{i,r} - v_{i,r}` where `u, v >= 0`. Regularizer becomes `lambda * sum(u + v) * log(p_r) + gamma * sum(u * v)`. Makes numerator/denominator mass explicit and penalizes simultaneous use at the same prime.

## Architecture

- **Layerwise prime basis**: Allow different layers to use different numbers of primes `P` or different prime sets. E.g., early layers might need only P=4 while later layers benefit from P=8.

## Training Schedule

- **Lambda annealing**: Start with low `lambda_mdl` to let the model learn the task, then increase to encourage simplification. Analogous to tau annealing in the Gumbel-Softmax approach.
