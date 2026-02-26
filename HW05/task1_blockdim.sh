#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-01:00:00
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -J hpc_hw

module load nvidia/cuda/13.0.0

n=$((2**14))

# Compile once
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

out_i="best_int_euler.txt"
out_f="best_float_euler.txt"
out_d="best_double_euler.txt"
: > "$out_i"
: > "$out_f"
: > "$out_d"

echo "Sweeping block_dim=32..1 at n=$n"

best_bd_i=0; best_t_i=1e30
best_bd_f=0; best_t_f=1e30
best_bd_d=0; best_t_d=1e30

for bd in $(seq 32 -1 1); do
  echo "block_dim=$bd"

  # optional: print wall time per run
  start=$(date +%s)

  out=$(srun ./task1 "$n" "$bd")

  end=$(date +%s)
  echo "wall_seconds=$((end-start))"

  t1=$(printf "%s\n" "$out" | sed -n '3p')  # int time (ms)
  t2=$(printf "%s\n" "$out" | sed -n '6p')  # float time (ms)
  t3=$(printf "%s\n" "$out" | sed -n '9p')  # double time (ms)

  printf "%d %s\n" "$bd" "$t1" >> "$out_i"
  printf "%d %s\n" "$bd" "$t2" >> "$out_f"
  printf "%d %s\n" "$bd" "$t3" >> "$out_d"

  if awk "BEGIN{exit !($t1 < $best_t_i)}"; then best_t_i="$t1"; best_bd_i="$bd"; fi
  if awk "BEGIN{exit !($t2 < $best_t_f)}"; then best_t_f="$t2"; best_bd_f="$bd"; fi
  if awk "BEGIN{exit !($t3 < $best_t_d)}"; then best_t_d="$t3"; best_bd_d="$bd"; fi
done

echo
echo "Wrote:"
echo "  $out_i"
echo "  $out_f"
echo "  $out_d"
echo
echo "Best int:    block_dim=$best_bd_i  time_ms=$best_t_i"
echo "Best float:  block_dim=$best_bd_f  time_ms=$best_t_f"
echo "Best double: block_dim=$best_bd_d  time_ms=$best_t_d"