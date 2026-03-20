# Code Cleanup TODOs — 2026-03-19

## What was done

Implemented the two actionable code-cleanup items from `code-cleanup-todos.MD`:

### 1. Unified φ initialization in ANBN shared mode

- `src/mdl/shared_weights.py` — `create_shared_mdl_state` now accepts `grid_codelengths` and initializes `phi_logits = -cl * ln(2)` (P_base-informed), matching the cMNIST path in `train_state.py:77-81`.
- `differentiable_mdl.py` — both call sites updated to pass `grid_codelengths`.
- Docstring updated to describe the P_base-informed init.

### 2. Renamed `mdl_lambda` → `lambda1` in cMNIST shared paths

- `src/training/steps.py` — renamed the parameter in `_mdl_shared_loss`, `make_train_step_mdl_shared`, and `make_train_step_mdl_shared_pair`.
- The epoch-level parameter name (`mdl_lambda` in `epochs.py`) stays unchanged because it's shared with the basic MDL path and is just a positional pass-through.

### Items not changed

- **Batch scaling convention** (TODO #3): deferred per original doc ("low priority, if touching these files").
- **Entropy variable naming** (TODO #4): no code change needed, docs-only note.

## Verification

All 82 tests pass (`test_mdl.py` + `test_math.py`).

## Updated TODO list

Rewrote `docs/2_active_discussions/code-cleanup-todos.MD` with a consolidated, effort-ordered list of all outstanding engineering work drawn from:
- Original cleanup todos
- `engineering-roadmap.MD`
- `research-roadmap.MD` (engineering-adjacent items only)
- Quantization session log (remaining P1/P2/P3 items)
- Active discussion status docs

15 items total, from XS (~30 min) to XL (multi-day).
