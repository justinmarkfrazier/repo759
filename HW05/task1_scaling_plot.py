import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("results_int.txt")
i = data[:, 0].astype(int)
matmul1 = data[:, 1]

data = np.loadtxt("results_float.txt")
matmul2 = data[:, 1]

data = np.loadtxt("results_double.txt")
matmul3 = data[:, 1]

plt.figure()
plt.plot(i, matmul1, marker='o', label="matmul_1")
plt.plot(i, matmul2, marker='o', label="matmul_2")
plt.plot(i, matmul3, marker='o', label="matmul_3")

plt.yscale('log')

plt.grid(True, which="both", ls="--", lw=0.5)

plt.title("Matrix Multiplication Time vs Array Side Length")
plt.xlabel("n")
plt.ylabel("time (ms)")
plt.xticks(i, [fr"$2^{{{k}}}$" for k in i])
plt.legend()

plt.tight_layout()
plt.savefig("task1.pdf", dpi=400)