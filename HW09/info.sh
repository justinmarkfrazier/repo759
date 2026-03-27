#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:05:00
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH -J task3_info

> task3_info.txt

echo "=== hostnames seen by the 2 MPI ranks ===" >> task3_info.txt
srun -n 2 hostname >> task3_info.txt

echo "" >> task3_info.txt
echo "=== lscpu ===" >> task3_info.txt
lscpu >> task3_info.txt

echo "" >> task3_info.txt
echo "=== numactl -H ===" >> task3_info.txt
numactl -H >> task3_info.txt