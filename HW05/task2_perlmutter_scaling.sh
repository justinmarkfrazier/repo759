#!/bin/bash
#SBATCH -A m4295
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH -J hpc_gpu_test
#SBATCH -t 0:20:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jmfrazier2@wisc.edu

nvcc -O3 -std=c++17 -arch=sm_80 task2.cu reduce.cu -o task2

datafile="results2_1024_perlmutter.txt"
: > "$datafile"

for i in {10..30}; do
  n=$((2**i))

  out=$(srun ./task2 "$n" 1024)

  time_ms=$(printf "%s\n" "$out" | sed -n '2p')

  printf "%s %s\n" "$i" "$time_ms" >> "$datafile"

  echo "n=2**$i"
  echo "$out"
  echo
done

echo "Wrote $datafile:"
cat "$datafile"

datafile="results2_256_perlmutter.txt"
: > "$datafile"

for i in {10..30}; do
  n=$((2**i))

  out=$(srun ./task2 "$n" 256)

  time_ms=$(printf "%s\n" "$out" | sed -n '2p')

  printf "%s %s\n" "$i" "$time_ms" >> "$datafile"

  echo "n=2**$i"
  echo "$out"
  echo
done

echo "Wrote $datafile:"
cat "$datafile"