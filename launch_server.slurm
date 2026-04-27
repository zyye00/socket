#!/bin/bash
#SBATCH --job-name=RPC-server
#SBATCH --partition=CPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=server.%j.out

set -euo pipefail

echo "Starting server..."
python -u server.py
