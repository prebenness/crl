# 2026-03-31 — cCIFAR10 Severity Resolution

## What happened

Continued from a previous session that was investigating the exact corruption severity level used in the DFA pre-built cCIFAR10 data (used by uLA and CCDB baselines).

## Finding

**DFA cCIFAR10 uses Hendrycks severity 5 (maximum).** This corresponds to LfF's loop parameter `severity=4` (since LfF calls `protocol[method_name](img, severity+1)`).

### Evidence

Deterministic proof from the Pixelate corruption (Class 8):

- LfF's `pixelate()` downscales to `int(32 * c)` pixels then upscales back to 32x32
- Severity parameter `c = [0.95, 0.9, 0.85, 0.75, 0.65]` at severity [1..5]
- Every Class 8 image in the DFA data has **exactly 20 unique rows**
- `int(32 * 0.65) = 20` matches severity 5 uniquely (severity 4 gives 24)
- This is lossless and deterministic; no statistical estimation needed

Supporting evidence from other corruption classes:
- Class 4 (Contrast): within-image std ~16, consistent with c=0.15 (severity 5)
- Class 3 (Brightness): mean pixel ~0.67, consistent with c=0.2-0.3 (severity 4-5)

## Files updated

- `agent-context/additional-crl-datasets.tex`: Updated severity section from "partially resolved" to "resolved" with empirical determination details. Removed severity from remaining open questions.

## Remaining open questions (across all datasets)

These cannot be resolved without CCDB's (unpublished) code:
- CCDB's MLP hidden dimension for cMNIST
- CCDB's exact BERT variant for CivilComments
- CCDB's toxicity threshold
- CCDB's 4-group breakdown for CivilComments
- cCIFAR10 augmentation details for uLA (beyond normalisation)
