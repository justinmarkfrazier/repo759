#!/bin/bash
#SBATCH -A m4295
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH -J hpc_gpu_test
#SBATCH -t 0:20:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jmfrazier2@wisc.edu

nvcc task2.cu scan.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -arch=sm_80 -lcublas -std=c++17 -o task2

srun ./task2 1000 512
srun ./task2 1024 512
srun ./task2 1024 1024
srun ./task2 10000 1024
srun ./task2 65536 1024

compute-sanitizer --tool memcheck ./task2 1024 1024