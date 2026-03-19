#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:10:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH -J hpc_hw

g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

for t in $(seq 1 20)
do
    ./task2 1024 $t | awk -v t="$t" 'NR==3 {print t, $1}' >> results2.txt
done