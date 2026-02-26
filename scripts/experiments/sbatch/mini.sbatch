#!/bin/bash
#SBATCH --account=prebenmn
#SBATCH --job-name=lambda-sweep
#SBATCH --output=logs/sbatch-test.log   
#SBATCH --partition a40q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

echo "Starting job at time:" && date +%Y-%m-%d_%H:%M:%S

# Replace this with your venv path, and also --account with your username #SBATCH --gres=gpu:1
source /global/D1/homes/prebenmn/workspace/crl/venv/bin/activate

# Set the number of OpenMP threads.  Depends on OpenMP library used (pip list ).
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK      # dgx2q = 6 max,  hgx2q = 16 max, a100q = 128 per GPU
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK       # dgx2q = 6 max,  hgx2q = 16 max, a100q = 128 per GPU
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python colored_mnist.py configs/mdl.yaml