#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --exclusive
#SBATCH --time=0-00:10:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH -J hpc_hw

g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

for t in $(seq 1 20)
do
    ./task1 1024 $t | awk -v t="$t" 'NR==3 {print t, $1}' >> results1.txt
done