#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:20:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -J hpc_hw

set -euo pipefail

datafile="results.txt"
: > "$datafile"

for i in {10..29}; do
  n=$((2**i))

  out=$(srun --quiet ./task3 "$n")

  # Grab the first line that *starts* with a number (handles scientific notation too),
  # then take its first field.
  scan_ms=$(printf "%s\n" "$out" \
    | tr -d '\r' \
    | awk '
        $1 ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/ { print $1; exit }
      ')

  # If parsing fails, stop with a helpful message
  if [[ -z "${scan_ms:-}" ]]; then
    echo "ERROR: Could not parse timing from task3 output for i=$i (n=$n)" >&2
    echo "----- raw output -----" >&2
    echo "$out" >&2
    echo "----------------------" >&2
    exit 1
  fi

  printf "%s %s\n" "$i" "$scan_ms" >> "$datafile"

  echo "i=$i n=$n timing_ms=$scan_ms"
done

echo "Wrote $datafile:"
cat "$datafile"