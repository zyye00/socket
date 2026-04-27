#!/bin/bash
#SBATCH --job-name=RPC-driver
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1%2
#SBATCH --time=00:30:00
#SBATCH --output=driver.%A.%a.out

set -euo pipefail

module load  orca/6.1.1

echo "Starting driver ${SLURM_ARRAY_TASK_ID}..."
python -u driver.py
