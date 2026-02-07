#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:20:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -J hpc_hw

set -euo pipefail


for i in {10..29}; do
  n=$((2**i))

  srun ./task3 "$n"
done
