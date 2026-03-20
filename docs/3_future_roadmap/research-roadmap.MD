# Research Roadmap

## ANBN / MDL

1. Seed stability on the best schedule family
- Once the new transition is validated, run the strongest schedule across multiple seeds.
- Quantify mean/std for `gen_n` and `total_mdl_bits`.

2. Baseline comparisons against standard regularizers
- Reproduce standard LSTM baselines (no-reg, L1, L2, etc.) and compare against differentiable MDL.
- This is central to the research claim.

3. Transition and regularization ablations
- Sweep `mdl_lambda`, grid size, warmup duration, bridge duration, annealing schedule, and `n_samples`.
- Main goal: reduce the threat to validity from the current narrow good-`lambda` region.

4. Learned-weight / mechanism analysis
- Compare learned networks against the handcrafted golden network.
- Characterize what distinguishes very high-generalization runs from brittle ones.

5. Shared-MDL validation
- Objective mismatch is fixed (CE_2 + P_base correction, P_base-informed φ init).
- Test whether shared mode can match or exceed basic MDL under matched settings.

## ColoredMNIST / Robustness

6. Improve outer-model selection and robustness tracking
- Save/select best outer checkpoints, not just final epochs.
- Add group-balanced and worst-group metrics for `cmnist_ula`.

7. Push ULA-comparable models toward the cMNIST target range
- Continue with ULA-style architectures only when making SOTA-facing comparisons.
- Use VIB as the current stronger baseline while MDL continues to improve.

## Publication

8. Visualizations and diagnostics
- Training dynamics plots
- Output probability heatmaps
- Hidden-state / cell-state traces
- Loss-surface or schedule-comparison plots where useful

9. Paper draft
- Connect proposal theory and the strongest empirical evidence.
- Be explicit about optimizer path dependence, schedule effects, and current limits.
