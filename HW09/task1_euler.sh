#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:10:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH -J hpc_hw

g++ task1.cpp cluster.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

export OMP_PLACES=cores
export OMP_PROC_BIND=spread

for t in $(seq 1 10)
do
    ts=$((2**10))
    ./task1 5040000 $t | awk -v t="$t" 'NR==3 {print t, $1}' >> results1.txt
done