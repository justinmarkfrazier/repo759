#!/bin/bash
#SBATCH -A m4295
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH -J hpc_gpu_test
#SBATCH -t 0:00:30
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jmfrazier2@wisc.edu

nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

datafile="results.txt"
: > "$datafile"

for i in {5..14}; do
  n=$((2**i))

  out=$(srun ./task1 "$n" 32)

  time_ms=$(printf "%s\n" "$out" | sed -n '2p')

  printf "%s %s\n" "$i" "$time_ms" >> "$datafile"

  echo "n=2**$i"
  echo "$out"
  echo
done

echo "Wrote $datafile:"
cat "$datafile"