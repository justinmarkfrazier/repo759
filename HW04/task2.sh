#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:10:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -J hpc_hw

module load nvidia/cuda/13.0.0

nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

datafile="results32_t2_euler.txt"
: > "$datafile"

for i in {10..29}; do
  n=$((2**i))

  out=$(srun ./task2 "$n" 128 2048)

  time_ms=$(printf "%s\n" "$out" | sed -n '2p')

  printf "%s %s\n" "$i" "$time_ms" >> "$datafile"

  echo "n=2**$i"
  echo "$out"
  echo
done

echo "Wrote $datafile:"
cat "$datafile"

datafile="results1024_t2_euler.txt"
: > "$datafile"

for i in {10..29}; do
  n=$((2**i))

  out=$(srun ./task2 "$n" 128 2048)

  time_ms=$(printf "%s\n" "$out" | sed -n '2p')

  printf "%s %s\n" "$i" "$time_ms" >> "$datafile"

  echo "n=2**$i"
  echo "$out"
  echo
done

echo "Wrote $datafile:"
cat "$datafile"