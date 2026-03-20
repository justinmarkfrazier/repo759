#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:00:20
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH -J hpc_hw

g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

ts=$((2**10))
./task3 1000000 8 $ts