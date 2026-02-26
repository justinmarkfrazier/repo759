import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("best_int.txt")
bd = data[:, 0].astype(int)
matmul1 = data[:, 1]

data = np.loadtxt("best_float.txt")
matmul2 = data[:, 1]

data = np.loadtxt("best_double.txt")
matmul3 = data[:, 1]

plt.figure()
plt.plot(bd, matmul1, marker='o', label="matmul_1")
plt.plot(bd, matmul2, marker='o', label="matmul_2")
plt.plot(bd, matmul3, marker='o', label="matmul_3")

plt.yscale('log')

plt.grid(True, which="both", ls="--", lw=0.5)

plt.title(r"Matrix Multiplication Time vs block\_dim ($n=2^{14}$)")
plt.xlabel("block_dim")
plt.ylabel("time (ms)")
plt.xticks(bd)

plt.legend()
plt.tight_layout()
plt.savefig("blockdim_sweep.pdf", dpi=400)