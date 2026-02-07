#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:20:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -J hpc_hw

module load nvidia/cuda/13.0.0

nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

n=$((2**10))
srun ./task3 3