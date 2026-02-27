#!/bin/bash
#SBATCH --account=prebenmn
#SBATCH --job-name=lambda-sweep
#SBATCH --output=logs/%x_%j.log     # job-name_job-ID.log
#SBATCH --partition hgx2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

echo "Starting job at time:" && date +%Y-%m-%d_%H:%M:%S

# Set venv for partition
source /global/D1/homes/prebenmn/workspace/crl/hgx2q/bin/activate
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

srun python colored_mnist.py configs/mdl.yaml