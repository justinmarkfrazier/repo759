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

nvcc task2.cu reduce.cu -Wno-deprecated-gpu-targets -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

# Run a few cases (block_dim must be <= 32)
echo "---- RUN n=257 block_dim=16 ----"
srun ./task2 257 16

echo "---- RUN n=1024 block_dim=16 ----"
srun ./task2 1024 16

echo "---- RUN n=1024 block_dim=32 ----"
srun ./task2 1024 32

echo "---- RUN n=2^30 block_dim=32 ----"
srun ./task2 1073741824 32