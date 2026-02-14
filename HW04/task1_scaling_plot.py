import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("results32_euler.txt")
i = data[:, 0].astype(int)
t_ms_32 = data[:, 1]

data = np.loadtxt("results1024_euler.txt")
t_ms_1024 = data[:, 1]

plt.figure()
plt.plot(i, t_ms_32, marker='o', label="32 threads per block")
plt.plot(i, t_ms_1024, marker='s', label="1024 threads per block")

plt.yscale('log')

plt.grid(True, which="both", ls="--", lw=0.5)

plt.title("Matrix Multiplication Time vs Array Side Length")
plt.xlabel("n")
plt.ylabel("time (ms)")
plt.xticks(i, [fr"$2^{{{k}}}$" for k in i])
plt.legend()

plt.tight_layout()
plt.savefig("task1.pdf", dpi=400)