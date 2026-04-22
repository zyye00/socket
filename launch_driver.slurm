#!/bin/bash
#SBATCH --job-name=socket-driver
#SBATCH --partition=CPU
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:01:00
#SBATCH --output=driver.%j.%t.out

set -euo pipefail

echo "Starting drivers..."
srun python -u my_driver.py
