#!/bin/bash
#SBATCH -A m4295
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH -J hpc_gpu_test
#SBATCH -t 0:10:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jmfrazier2@wisc.edu

set -euo pipefail

# compile once
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

block_dim=32

data_int="results_int.txt"
data_float="results_float.txt"
data_double="results_double.txt"

: > "$data_int"
: > "$data_float"
: > "$data_double"

for i in {5..14}; do
  n=$((2**i))

  # run under slurm (inside an sbatch allocation)
  out=$(srun ./task1 "$n" "$block_dim")

  # task1 prints 9 lines:
  # 1 C1[0]
  # 2 C1[last]
  # 3 time1_ms
  # 4 C2[0]
  # 5 C2[last]
  # 6 time2_ms
  # 7 C3[0]
  # 8 C3[last]
  # 9 time3_ms
  t1=$(printf "%s\n" "$out" | sed -n '3p')
  t2=$(printf "%s\n" "$out" | sed -n '6p')
  t3=$(printf "%s\n" "$out" | sed -n '9p')

  printf "%s %s\n" "$i" "$t1" >> "$data_int"
  printf "%s %s\n" "$i" "$t2" >> "$data_float"
  printf "%s %s\n" "$i" "$t3" >> "$data_double"

  echo "n=2**$i (n=$n), block_dim=$block_dim"
  echo "$out"
  echo
done

echo "Wrote:"
echo "  $data_int"
echo "  $data_float"
echo "  $data_double"