#!/bin/bash
#SBATCH --job-name=socket-master
#SBATCH --partition=CPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=master.%j.out

set -euo pipefail

echo "Starting master..."
python -u my_master.py
