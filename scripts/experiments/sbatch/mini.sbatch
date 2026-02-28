#!/bin/bash
#SBATCH --account=prebenmn
#SBATCH --job-name=lambda-sweep
#SBATCH --output=logs/%x_%j.log     # job-name_job-ID.log
#SBATCH --partition a40q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

echo "Starting job at time:" && date +%Y-%m-%d_%H:%M:%S

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set venv for partition
source /global/D1/homes/prebenmn/workspace/crl/venvs/a40q/bin/activate
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python -u colored_mnist.py config/colored_mnist/mdl_single_sweep.yaml
