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

python3 - <<'PY'
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("results.txt")
i = data[:, 0].astype(int)
t_ms = data[:, 1]

plt.figure()
plt.plot(i, t_ms, marker='o')

plt.yscale('log')

plt.xlabel(r"$2^i$")
plt.ylabel("scan time (ms)")
plt.xticks(i, [fr"$2^{{{k}}}$" for k in i])

plt.tight_layout()
plt.savefig("resultsv2.png", dpi=200)
PY