#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:05:00
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH -J task3_info

> task3_info.txt

echo "=== rank placement / hostname / allowed CPUs ===" >> task3_info.txt
srun -n 2 bash -c '
echo "HOST=$(hostname)"
echo "CPUS_ALLOWED=$(grep Cpus_allowed_list /proc/self/status | awk "{print \$2}")"
echo
' >> task3_info.txt

echo "=== lscpu ===" >> task3_info.txt
lscpu >> task3_info.txt
echo >> task3_info.txt

echo "=== numactl -H ===" >> task3_info.txt
numactl -H >> task3_info.txt
echo >> task3_info.txt