import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("results_task1_100000_euler.txt")
i = data[:, 0].astype(int)
matmul1 = data[:, 1]

plt.figure()
plt.plot(i, matmul1, marker='o')

plt.yscale('log')

plt.grid(True, which="both", ls="--", lw=0.5)

plt.title("CudaBLAS Matrix Multiplication Time vs Array Side Length")
plt.xlabel("n")
plt.ylabel("time (ms)")
plt.xticks(i, [fr"$2^{{{k}}}$" for k in i])

plt.tight_layout()
plt.savefig("task1_euler.png", dpi=400)