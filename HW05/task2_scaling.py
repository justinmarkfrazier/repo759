import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("results2_1024.txt")
i = data[:, 0].astype(int)
matmul1 = data[:, 1]

data = np.loadtxt("results2_256.txt")
matmul2 = data[:, 1]

plt.figure()
plt.plot(i, matmul1, marker='o', label="1024 threads per block")
plt.plot(i, matmul2, marker='o', label="256 threads per block")

plt.yscale('log')

plt.grid(True, which="both", ls="--", lw=0.5)

plt.title("Parallel Reduction Time vs Array Length")
plt.xlabel("n")
plt.ylabel("time (ms)")
plt.xticks(i, [fr"$2^{{{k}}}$" for k in i])
plt.legend()

plt.tight_layout()
plt.savefig("task2_euler.png", dpi=400)