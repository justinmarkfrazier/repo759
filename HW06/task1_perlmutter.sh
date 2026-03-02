#!/bin/bash
#SBATCH -A m4295
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH -J hpc_gpu_test
#SBATCH -t 0:02:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jmfrazier2@wisc.edu

nvcc task1.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -arch=sm_80 -lcublas -std=c++17 -o task1

./task1 32 1
./task1 32 10
./task1 32 100
./task1 32 1000
./task1 32 10000

./task1 2048 1
./task1 2048 10
./task1 2048 100
./task1 2048 1000
./task1 2048 10000