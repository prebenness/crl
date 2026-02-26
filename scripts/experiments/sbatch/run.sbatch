#!/bin/bash
#SBATCH --account=prebenmn
#SBATCH --job-name=lambda-sweep
#SBATCH --partition=a40q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/home/prebenmn/D1/workspace/crl/logs/lambda_sweep_%A_%a.log

set -euo pipefail

REPO=/home/prebenmn/D1/workspace/crl
PY=$REPO/venv/bin/python
N_LAM=8
SEED=42
EPOCHS=300
N_SAMPLES=10

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
LIDX=$TASK_ID

if (( LIDX < 0 || LIDX >= N_LAM )); then
  echo "LIDX=$LIDX out of range (N_LAM=$N_LAM)"
  exit 1
fi

LAM_EXP=$($PY - <<PY
import numpy as np
print(np.linspace(-3, 1, $N_LAM)[$LIDX])
PY
)

WORK=${SLURM_TMPDIR:-/tmp}/cmnist_${SLURM_JOB_ID}_${TASK_ID}
mkdir -p "$WORK"
cd "$WORK"
ln -s "$REPO/data" data || true

CFG=$WORK/cfg.yaml
$PY - <<PY
import yaml
cfg = yaml.safe_load(open("$REPO/configs/mdl_pair.yaml"))
cfg["training"]["epochs"] = $EPOCHS
cfg["mdl"]["n_samples"] = $N_SAMPLES
cfg["training"]["seed"] = $SEED
cfg["sweep"]["lambda_min_exp"] = float("$LAM_EXP")
cfg["sweep"]["lambda_max_exp"] = float("$LAM_EXP")
cfg["sweep"]["lambda_steps"] = 1
yaml.safe_dump(cfg, open("$CFG", "w"), sort_keys=False)
PY

export WANDB_MODE=offline
export JAX_COMPILATION_CACHE_DIR=${SCRATCH:-$HOME/.cache}/jax_cache
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

$PY "$REPO/colored_mnist.py" "$CFG"

DEST="$REPO/results/slurm_${SLURM_JOB_ID}_${TASK_ID}"
mkdir -p "$DEST"
cp -r "$WORK/results" "$DEST/" || true
cp -r "$WORK/wandb" "$DEST/" 2>/dev/null || true
cp "$CFG" "$DEST/"
