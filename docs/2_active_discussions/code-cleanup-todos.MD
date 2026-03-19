# Code Cleanup TODOs

Identified during proposal-vs-code review, 2026-03-19. These are deferred code changes — do not change theory or paper to accommodate; the paper takes precedence on notation.

## 1. Unify φ initialization in ANBN shared mode

**File:** `src/mdl/shared_weights.py`, line 185
**Issue:** `phi_logits` is initialized to `jnp.zeros`, giving a uniform prior. The cMNIST path (`src/training/train_state.py:77-81`) already uses `phi_init = -cl * ln(2)` (P_base-informed). The ANBN path should match.
**Fix:** Replace `phi_logits = jnp.zeros((M,), ...)` with the same P_base-informed init used in `create_state_mdl_shared`. Update the docstring to remove "uniform adaptive prior" language.

## 2. Rename `mdl_lambda` → `lambda1` in cMNIST shared paths

**File:** `src/training/steps.py` — `_mdl_shared_loss`, `make_train_step_mdl_shared`, `make_train_step_mdl_shared_pair`
**Issue:** The shared-weight code-term weight is called `mdl_lambda` in the cMNIST path but `lambda1` in the ANBN path (`src/mdl/shared_weights.py`). Should be `lambda1` everywhere for consistency with the proposal.
**Fix:** Rename the parameter in the cMNIST shared loss functions. Config key (`sweep mdl_lambda`) can stay as-is since it's user-facing and serves double duty in basic vs shared mode.

## 3. Unify batch scaling convention

**Files:** `src/mdl/training.py` (ANBN) vs `src/training/steps.py` (cMNIST)
**Issue:** ANBN uses `B/N` scaling on regularization terms with summed data NLL. cMNIST uses averaged data NLL with `1/N` scaling. Both are equivalent over a full epoch, but having two conventions complicates reading the code.
**Fix:** Low priority. If touching these files for other reasons, consider aligning both to the cMNIST convention (averaged data + `1/N` scaling), which is more standard in deep learning.

## 4. Entropy variable naming

**Issue:** The proposal uses `−(1/β) H` notation (negative term). The code computes `τ * H` and subtracts. The code convention is fine — just ensure any future docstrings or comments refer to "entropy bonus (subtracted)" rather than mixing the two framings.
**Fix:** No code change needed. Just a note for future docstring consistency.
