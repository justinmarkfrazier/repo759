#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:10:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -J hpc_hw

module load nvidia/cuda/13.0.0

nvcc task1.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std=c++17 -o task1

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