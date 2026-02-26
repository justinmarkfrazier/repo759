#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:45:00
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -J hpc_hw

srun ./task3