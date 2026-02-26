#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:05:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -J hpc_hw

module load nvidia/cuda/13.0.0

nvcc test1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o test1

srun ./test1 16
srun ./test1 7
srun ./test1 31
srun ./test1 13
srun ./test1 32

compute-sanitizer --tool memcheck ./test1 16
compute-sanitizer --tool racecheck ./test1 16
compute-sanitizer --tool synccheck ./test1 16