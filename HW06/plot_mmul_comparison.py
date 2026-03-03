import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("results_task1_100000_euler.txt")
i = data[:, 0].astype(int)
matmul1 = data[:, 1]

data = np.loadtxt("../HW05/results_float_euler.txt")
matmul2 = data[0:7, 1]

plt.figure()
plt.plot(i, matmul1, marker='o', label="cuBLAS implementation")
plt.plot(i, matmul2, marker='s', label="Tiled implementation")

plt.yscale('log')

plt.grid(True, which="both", ls="--", lw=0.5)

plt.title("Matrix Multiplication Time vs Array Side Length")
plt.xlabel("n")
plt.ylabel("time (ms)")
plt.xticks(i, [fr"$2^{{{k}}}$" for k in i])
plt.legend()

plt.tight_layout()
plt.savefig("task1_compare_euler.png", dpi=400)