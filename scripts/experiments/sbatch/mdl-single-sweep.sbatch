#!/bin/bash
#SBATCH --account=prebenmn
#SBATCH --job-name=mdl-single-sweep
#SBATCH --output=logs/%x_%j.log     # job-name_job-ID.log
#SBATCH --partition a40q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --chdir=/global/D1/homes/prebenmn/workspace/crl

set -euo pipefail

echo "Starting job at time:" && date +%Y-%m-%d_%H:%M:%S

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
CONFIG_PATH="${CONFIG_PATH:-config/colored_mnist/ex3_mdl_single_sweep.yaml}"

# Avoid W&B's stdout wrapping in non-interactive batch jobs.
export WANDB_CONSOLE=off
# The offline backend still hangs on long ex3 batch runs; disable W&B unless
# explicitly overridden for interactive debugging.
export WANDB_MODE=${WANDB_MODE:-disabled}

# Set venv for partition
source /global/D1/homes/prebenmn/workspace/crl/venvs/a40q/bin/activate
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

echo "Using config: $CONFIG_PATH"
python -u colored_mnist.py "$CONFIG_PATH"
