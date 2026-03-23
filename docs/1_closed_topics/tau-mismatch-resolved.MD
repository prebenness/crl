# Resolved: all_probs uses softmax(α), not softmax(α/τ)

## The question

`src/mdl/lstm.py:164` computes `all_probs = jax.nn.softmax(all_logits, axis=-1)` at temperature 1, while the DST/GST forward pass uses `softmax(all_logits / tau)`. Should these match?

## Decision

No change needed. The current code is correct.

## Reasoning

The objective J_β has three additive terms. The complexity term (Σᵢ Σₘ πᵢ,ₘ ℓ(sₘ)) and the entropy bonus (−τ · H(π)) are computed exactly under π = softmax(α). Their gradients are known in closed form. No estimation is needed for these terms.

The data term E[L_D(θ)] is intractable and requires an estimator. The DST estimator uses `softmax(α/τ)` in the backward pass as a smooth gradient proxy. The τ there is a hyperparameter of the gradient estimator controlling smoothing radius—a property of the optimizer, not the objective.

The total gradient per step is: `(DST estimate of ∂E[L_D]/∂α) + (exact ∂[complexity + entropy]/∂α)`. The first contribution flows through `softmax(α/τ)`, the second through `softmax(α)`. These are two separate contributions summed into one update. One is estimated with a particular tool, the other doesn't need estimation.

Computing complexity and entropy under `softmax(α/τ)` would make the coding cost of the hypothesis depend on the optimizer's smoothing radius, which is coding-theoretically incoherent. At τ < 1.0, it would also double-suppress the entropy bonus—once by the τ coefficient, again by the low entropy of the peaked distribution.
