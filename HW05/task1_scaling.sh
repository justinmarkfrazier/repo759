#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --time=0-00:30:00
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -J hpc_hw

module load nvidia/cuda/13.0.0

nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

block_dim=32

data_int="results_int_euler.txt"
data_float="results_float_euler.txt"
data_double="results_double_euler.txt"

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