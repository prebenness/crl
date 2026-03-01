# ANBN MDL Overnight Batch (2026-03-01 UTC)

- Started: 2026-02-28T23:23:53+01:00
- Host cwd: /home/prebenness/workspace/crl
- Source checkpoints:
  - results/anbn_mdl/20260228_200521_basic_e40_lr0.001_lam0.144_g10x10_s0_resume_best
  - results/anbn_mdl/20260228_183438_basic_e10000_lr0.05_lam0.144_g10x10_s0

## Planned Runs
1. 833-bit source soft-only tail: lr=5e-4, 20 epochs, tau_end=0.90
2. 833-bit source soft-only tail: lr=7.5e-4, 20 epochs, tau_end=0.85
3. 833-bit source soft-only tail: lr=2.5e-4, 40 epochs, tau_end=0.95
4. 841-bit source soft-only tail: lr=1e-3, 40 epochs, tau_end=0.80
5. From scratch: lambda=0.1438 baseline schedule + long-val probes
6. From scratch: lambda=0.1440 baseline schedule + long-val probes
7. From scratch: lambda=0.1442 baseline schedule + long-val probes
8. From scratch: lambda=0.1440, longer warmup / colder schedule / more samples


### 833-tail-lr5e-4-e20-tau0.90
- Started: 2026-02-28T23:23:53+01:00
- Ended: 2026-02-28T23:35:14+01:00
- Exit status: 0
- Output dir: results/anbn_mdl/20260228_222354_basic_e20_lr0.0005_lam0.144_g10x10_s0_resume_best
- 833-tail-lr5e-4-e20-tau0.90: [20260228_222354_basic_e20_lr0.0005_lam0.144_g10x10_s0_resume_best](results/anbn_mdl/20260228_222354_basic_e20_lr0.0005_lam0.144_g10x10_s0_resume_best)
  - bits: 833
  - gen_n: 100000
  - n_perfect: 100000
  - first_failure_n: None

### 833-tail-lr7.5e-4-e20-tau0.85
- Started: 2026-02-28T23:35:14+01:00
- Ended: 2026-02-28T23:46:47+01:00
- Exit status: 0
- Output dir: results/anbn_mdl/20260228_223515_basic_e20_lr0.00075_lam0.144_g10x10_s0_resume_best
- 833-tail-lr7.5e-4-e20-tau0.85: [20260228_223515_basic_e20_lr0.00075_lam0.144_g10x10_s0_resume_best](results/anbn_mdl/20260228_223515_basic_e20_lr0.00075_lam0.144_g10x10_s0_resume_best)
  - bits: 831
  - gen_n: 100000
  - n_perfect: 100000
  - first_failure_n: None

### 833-tail-lr2.5e-4-e40-tau0.95
- Started: 2026-02-28T23:46:47+01:00
- Ended: 2026-02-28T23:58:18+01:00
- Exit status: 0
- Output dir: results/anbn_mdl/20260228_224648_basic_e40_lr0.00025_lam0.144_g10x10_s0_resume_best
- 833-tail-lr2.5e-4-e40-tau0.95: [20260228_224648_basic_e40_lr0.00025_lam0.144_g10x10_s0_resume_best](results/anbn_mdl/20260228_224648_basic_e40_lr0.00025_lam0.144_g10x10_s0_resume_best)
  - bits: 833
  - gen_n: 100000
  - n_perfect: 100000
  - first_failure_n: None

### 841-tail-lr1e-3-e40-tau0.80
- Started: 2026-02-28T23:58:18+01:00
- Ended: 2026-03-01T00:09:31+01:00
- Exit status: 0
- Output dir: results/anbn_mdl/20260228_225819_basic_e40_lr0.001_lam0.144_g10x10_s0_resume_best
- 841-tail-lr1e-3-e40-tau0.80: [20260228_225819_basic_e40_lr0.001_lam0.144_g10x10_s0_resume_best](results/anbn_mdl/20260228_225819_basic_e40_lr0.001_lam0.144_g10x10_s0_resume_best)
  - bits: 837
  - gen_n: 100000
  - n_perfect: 100000
  - first_failure_n: None

### scratch-lam0.1438-baseline
- Started: 2026-03-01T00:09:31+01:00
- Ended: 2026-03-01T00:54:05+01:00
- Exit status: 0
- Output dir: results/anbn_mdl/20260228_230932_basic_e10000_lr0.05_lam0.1438_g10x10_s0
- scratch-lam0.1438-baseline: [20260228_230932_basic_e10000_lr0.05_lam0.1438_g10x10_s0](results/anbn_mdl/20260228_230932_basic_e10000_lr0.05_lam0.1438_g10x10_s0)
  - bits: 785
  - gen_n: 0
  - n_perfect: 0
  - first_failure_n: 1

### scratch-lam0.1440-baseline
- Started: 2026-03-01T00:54:05+01:00
- Ended: 2026-03-01T01:36:57+01:00
- Exit status: 0
- Output dir: results/anbn_mdl/20260228_235405_basic_e10000_lr0.05_lam0.144_g10x10_s0
- scratch-lam0.1440-baseline: [20260228_235405_basic_e10000_lr0.05_lam0.144_g10x10_s0](results/anbn_mdl/20260228_235405_basic_e10000_lr0.05_lam0.144_g10x10_s0)
  - bits: 785
  - gen_n: 0
  - n_perfect: 0
  - first_failure_n: 1

### scratch-lam0.1442-baseline
- Started: 2026-03-01T01:36:57+01:00
- Ended: 2026-03-01T02:17:52+01:00
- Exit status: 0
- Output dir: results/anbn_mdl/20260301_003658_basic_e10000_lr0.05_lam0.1442_g10x10_s0
- scratch-lam0.1442-baseline: [20260301_003658_basic_e10000_lr0.05_lam0.1442_g10x10_s0](results/anbn_mdl/20260301_003658_basic_e10000_lr0.05_lam0.1442_g10x10_s0)
  - bits: 785
  - gen_n: 0
  - n_perfect: 0
  - first_failure_n: 1

### scratch-lam0.1440-longwarm
- Started: 2026-03-01T02:17:52+01:00
- Ended: 2026-03-01T03:02:38+01:00
- Exit status: 0
- Output dir: results/anbn_mdl/20260301_011753_basic_e12000_lr0.03_lam0.144_g10x10_s0
- scratch-lam0.1440-longwarm: [20260301_011753_basic_e12000_lr0.03_lam0.144_g10x10_s0](results/anbn_mdl/20260301_011753_basic_e12000_lr0.03_lam0.144_g10x10_s0)
  - bits: 701
  - gen_n: 100000
  - n_perfect: 100000
  - first_failure_n: None

## Current Frontier Snapshot
- [20260228_200521_basic_e40_lr0.001_lam0.144_g10x10_s0_resume_best](results/anbn_mdl/20260228_200521_basic_e40_lr0.001_lam0.144_g10x10_s0_resume_best)
  - bits: 833
  - gen_n: 1000000
  - n_perfect: 1000000
  - first_failure_n: None
- [20260228_183438_basic_e10000_lr0.05_lam0.144_g10x10_s0](results/anbn_mdl/20260228_183438_basic_e10000_lr0.05_lam0.144_g10x10_s0)
  - bits: 841
  - gen_n: 1000000
  - n_perfect: 1000000
  - first_failure_n: None
- [20260228_131511_basic_e10000_lr0.05_lam0.1_g10x10_s0](results/anbn_mdl/20260228_131511_basic_e10000_lr0.05_lam0.1_g10x10_s0)
  - bits: 969
  - gen_n: 1000000
  - n_perfect: 1000000
  - first_failure_n: None
- [20260228_185026_basic_e800_lr0.005_lam0.144_g10x10_s0_resume_best](results/anbn_mdl/20260228_185026_basic_e800_lr0.005_lam0.144_g10x10_s0_resume_best)
  - bits: 823
  - gen_n: 120170
  - n_perfect: 120170
  - first_failure_n: 120171
- [20260301_011753_basic_e12000_lr0.03_lam0.144_g10x10_s0](results/anbn_mdl/20260301_011753_basic_e12000_lr0.03_lam0.144_g10x10_s0)
  - bits: 701
  - gen_n: 100000
  - n_perfect: 100000
  - first_failure_n: None
- [20260228_184818_basic_e800_lr0.005_lam0.144_g10x10_s0_resume_best](results/anbn_mdl/20260228_184818_basic_e800_lr0.005_lam0.144_g10x10_s0_resume_best)
  - bits: 823
  - gen_n: 100000
  - n_perfect: 100000
  - first_failure_n: None
- [20260228_223515_basic_e20_lr0.00075_lam0.144_g10x10_s0_resume_best](results/anbn_mdl/20260228_223515_basic_e20_lr0.00075_lam0.144_g10x10_s0_resume_best)
  - bits: 831
  - gen_n: 100000
  - n_perfect: 100000
  - first_failure_n: None
- [20260228_224648_basic_e40_lr0.00025_lam0.144_g10x10_s0_resume_best](results/anbn_mdl/20260228_224648_basic_e40_lr0.00025_lam0.144_g10x10_s0_resume_best)
  - bits: 833
  - gen_n: 100000
  - n_perfect: 100000
  - first_failure_n: None
- [20260228_222354_basic_e20_lr0.0005_lam0.144_g10x10_s0_resume_best](results/anbn_mdl/20260228_222354_basic_e20_lr0.0005_lam0.144_g10x10_s0_resume_best)
  - bits: 833
  - gen_n: 100000
  - n_perfect: 100000
  - first_failure_n: None
- [20260228_225819_basic_e40_lr0.001_lam0.144_g10x10_s0_resume_best](results/anbn_mdl/20260228_225819_basic_e40_lr0.001_lam0.144_g10x10_s0_resume_best)
  - bits: 837
  - gen_n: 100000
  - n_perfect: 100000
  - first_failure_n: None

- Finished: 2026-03-01T03:02:38+01:00

## Post-Run Re-evaluation

- After the batch finished, I re-ran evaluation on the saved `best` checkpoint from:
  - [20260301_011753_basic_e12000_lr0.03_lam0.144_g10x10_s0](/home/prebenness/workspace/crl/results/anbn_mdl/20260301_011753_basic_e12000_lr0.03_lam0.144_g10x10_s0)
- Using:
  - `--eval --ckpt_select best --test_max_n 1000000`
- Result:
  - `total_mdl_bits = 701`
  - `gen_n = 1000000`
  - `n_perfect = 1000000`

This is the strongest confirmed checkpoint so far:

- same tested generalization range as the previous best (`1e6`)
- substantially smaller than the previous best (`833` bits)
