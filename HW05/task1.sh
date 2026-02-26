#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:10:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -J hpc_hw

module load nvidia/cuda/13.0.0

nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

# Run a few cases (block_dim must be <= 32)
echo "---- RUN n=256 block_dim=16 ----"
srun ./task1 256 16

echo "---- RUN n=1024 block_dim=16 ----"
srun ./task1 1024 16

echo "---- RUN n=1024 block_dim=32 ----"
srun ./task1 1024 32