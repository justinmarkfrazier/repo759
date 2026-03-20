#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:10:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH -J hpc_hw

g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

for t in $(seq 1 20)
do
    ts=$((2**10))
    ./task3 1000000 $t $ts | awk -v t="$t" 'NR==3 {print t, $1}' >> results3a.txt
done