# ColoredMNIST Overnight Batch (2026-03-02 00:45 CET)

Current best pre-batch result:
- `mdl_pair`
- warmup-only (`bridge_epochs=0`, `warmup_epochs=epochs`)
- `lambda=1e-2`
- `hsic=0.9`
- `epochs=100`
- final outer test accuracy: `0.8397`
- run: `results/colored_mnist/20260301_234155_cmnist_mdl_pair_warmup100_lam1e2_hsic09_cmnist_ula_mdl_pair`

Batch objective:
1. Extend warmup-only training length for `mdl_pair`.
2. Measure robustness to HSIC and lambda around the working regime.
3. Compare `mdl_pair` against `mdl_shared_pair` under matched settings.
4. Add a small seed sweep at longer training lengths.

Batch files:
- Script: `/tmp/cmnist_mdl_overnight_20260302.sh`
- Summary: `/tmp/cmnist_mdl_overnight_20260302_0045/summary.tsv`
- Log: `/tmp/cmnist_mdl_overnight_20260302_0045/batch.log`
- Nohup: `/tmp/cmnist_mdl_overnight_20260302_0045/nohup.out`

Execution policy:
- Each run is warmup-only (`epochs == warmup_epochs`, `bridge_epochs=0`).
- The script checks for a GPU before every run and retries if CUDA is unavailable.
- Failed runs are recorded in the TSV and the batch continues.
