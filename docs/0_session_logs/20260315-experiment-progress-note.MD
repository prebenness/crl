# Experiment progress note — resume 2026-03-16

## Runs to resume

Both runs were interrupted mid-training. Checkpoints are saved and can be resumed with `--resume --ckpt <dir>`.

### 1. mode_forward only
- **Dir:** `results/anbn_mdl/20260315_213538_basic_e12000_lr0.03_lam0.144_g10x10_s42_cfgbasic_train`
- **Last checkpoint:** epoch 500 (eval checkpoint; train log shows progress to ~epoch 5000)
- **Resume:**
```bash
python3.12 differentiable_mdl.py config/anbn_mdl/basic_train.yaml \
    --mdl_lambda 0.144 --lr 0.03 --epochs 12000 --warmup_epochs 5000 \
    --seed 42 --mode_forward --eval_every 500 --log_every 200 \
    --resume --ckpt results/anbn_mdl/20260315_213538_basic_e12000_lr0.03_lam0.144_g10x10_s42_cfgbasic_train
```

### 2. mode_forward + init_cl_scale=1.0
- **Dir:** `results/anbn_mdl/20260315_213545_basic_e12000_lr0.03_lam0.144_g10x10_s42_cfgbasic_train`
- **Last checkpoint:** epoch 2500 (eval checkpoint; train log shows progress to ~epoch 4000)
- **Resume:**
```bash
python3.12 differentiable_mdl.py config/anbn_mdl/basic_train.yaml \
    --mdl_lambda 0.144 --lr 0.03 --epochs 12000 --warmup_epochs 5000 \
    --seed 42 --mode_forward --init_cl_scale 1.0 --eval_every 500 --log_every 200 \
    --resume --ckpt results/anbn_mdl/20260315_213545_basic_e12000_lr0.03_lam0.144_g10x10_s42_cfgbasic_train
```

## Completed baseline
- **Dir:** `results/anbn_mdl/20260315_212850_basic_e12000_lr0.03_lam0.144_g10x10_s42_cfgbasic_train`
- **Result:** gen_n=0, total_mdl_bits=769 — weak run, not golden

## Early observations
- Both interrupted runs were still in warmup (pre-ST phase); the mode_forward effect won't be visible until the ST phase (epoch 5000+).
- The init_cl_scale=1.0 run showed slightly higher `complexity_argmax_bits` during warmup (858 vs 804) — may flip after ST transition.
- The baseline (no new flags) produced a weak result (gen_n=0), consistent with known schedule sensitivity at lambda=0.144.

## Next steps on resume
1. Resume both runs to completion (12000 epochs).
2. Compare gen_n, total_mdl_bits, and training dynamics across all three conditions.
3. If mode_forward shows improvement, run multi-seed stability (seeds 0-4).
