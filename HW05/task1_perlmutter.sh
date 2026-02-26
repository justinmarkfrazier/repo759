#!/bin/bash
#SBATCH -A m4295
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH -J hpc_gpu_test
#SBATCH -t 0:10:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jmfrazier2@wisc.edu

nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

# Run a few cases (block_dim must be <= 32)
echo "---- RUN n=256 block_dim=16 ----"
srun ./task1 256 16

echo "---- RUN n=1024 block_dim=16 ----"
srun ./task1 1024 16

echo "---- RUN n=1024 block_dim=32 ----"
srun ./task1 1024 32