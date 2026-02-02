#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:20:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -J hpc_hw
#SBATCH -o task1.out -e task1.err

set -euo pipefail

datafile="results.txt"
: > "$datafile"

for i in {10..30}; do
  n=$((2**i))

  out=$(srun ./task1 "$n")

  scan_ms=$(printf "%s\n" "$out" | head -n 1)

  printf "%s %s\n" "$i" "$scan_ms" >> "$datafile"

  echo "i=$i n=$n"
  echo "$out"
  echo
done

echo "Wrote $datafile:"
cat "$datafile"