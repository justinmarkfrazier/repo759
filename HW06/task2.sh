#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:10:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -J hpc_hw

module load nvidia/cuda/13.0.0

nvcc task2.cu scan.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std=c++17 -o task2

srun ./task2 1000 512
srun ./task2 1024 512
srun ./task2 1024 1024
srun ./task2 10000 1024
srun ./task2 65536 1024

compute-sanitizer --tool memcheck ./task2 1024 1024