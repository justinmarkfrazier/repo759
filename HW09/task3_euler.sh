#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:10:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH -J hpc_hw

mpicxx task3.cpp -Wall -O3 -o task3

> results3.txt

for i in $(seq 1 25)
do
    n=$((2**i))
    srun -n 2 task3 $n | awk -v t="$i" 'NR==1 {print t, $1}' >> results3.txt
done