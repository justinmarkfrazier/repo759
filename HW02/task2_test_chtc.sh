#!/bin/bash
#SBATCH --partition=wright
#SBATCH --time=0-01:00:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -J hpc_hw

srun ./task2 50 9