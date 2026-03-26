#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:10:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH -J hpc_hw

g++ task2.cpp montecarlo_nosimd.cpp -Wall -O3 -std=c++17 -o task2_nosimd -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec
g++ task2.cpp montecarlo.cpp -Wall -O3 -std=c++17 -o task2_simd -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec

export OMP_PLACES=cores
export OMP_PROC_BIND=spread

> results2a.txt

for t in $(seq 1 10)
do
    avg=$(
        for run in $(seq 1 10)
        do
            ./task2_nosimd 1000000 $t | awk 'NR==2 {print $1}'
        done | awk '{sum += $1} END {print sum / NR}'
    )

    echo "$t $avg" >> results2a.txt
done

> results2b.txt

for t in $(seq 1 10)
do
    avg=$(
        for run in $(seq 1 10)
        do
            ./task2_simd 1000000 $t | awk 'NR==2 {print $1}'
        done | awk '{sum += $1} END {print sum / NR}'
    )

    echo "$t $avg" >> results2b.txt
done