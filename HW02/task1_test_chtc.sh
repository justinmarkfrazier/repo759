#!/bin/bash
#SBATCH --partition=wright
#SBATCH --time=0-01:00:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -J hpc_hw

n=$((2**20))
srun ./task1 "$n"