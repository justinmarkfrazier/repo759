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

nvcc task1.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -arch=sm_80 -lcublas -std=c++17 -o task1

# --- parameters ---
n_tests=100000
datafile="results_task1_${n_tests}_perlmutter.txt"
: > "$datafile"

# --- warm-up to avoid first-call overhead ---
echo "Running warm-up to prevent cold start"
srun ./task1 1024 10 > /dev/null

# --- sweep n = 2^5 ... 2^11 ---
for i in {5..11}; do
  n=$((2**i))

  out=$(srun ./task1 "$n" "$n_tests")     # task1 prints ONE line: ms_avg
  time_ms=$(printf "%s\n" "$out" | tail -n 1)

  printf "%s %s\n" "$i" "$time_ms" >> "$datafile"

  echo "n=2**$i ($n): ${time_ms} ms"
done

echo "Wrote $datafile:"
cat "$datafile"