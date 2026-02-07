#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:20:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -J hpc_hw

module load nvidia/cuda/13.0.0

nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

set -euo pipefail

datafile="results.txt"
: > "$datafile"

for i in {10..29}; do
  n=$((2**i))

  out=$(srun ./task3 "$n")

  scan_ms=$(printf "%s\n" "$out" | head -n 1)

  printf "%s %s\n" "$i" "$scan_ms" >> "$datafile"

  echo "i=$i n=$n"
  echo "$out"
  echo
done

echo "Wrote $datafile:"
cat "$datafile"