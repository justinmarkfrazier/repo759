#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:10:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -J hpc_hw

module load nvidia/cuda/13.0.0

nvcc task2.cu reduce.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

# Run a few cases
echo "---- RUN n=257 block_dim=16 ----"
srun ./task2 257 16

echo "---- RUN n=1024 block_dim=16 ----"
srun ./task2 1024 16

echo "---- RUN n=1024 block_dim=32 ----"
srun ./task2 1024 32

echo "---- RUN n=2^30 block_dim=32 ----"
srun ./task2 1073741824 32