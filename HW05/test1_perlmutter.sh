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

nvcc test1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o test1

srun ./test1 16
srun ./test1 7
srun ./test1 31
srun ./test1 13
srun ./test1 32

compute-sanitizer --tool memcheck ./test1 16
compute-sanitizer --tool racecheck ./test1 16
compute-sanitizer --tool synccheck ./test1 16