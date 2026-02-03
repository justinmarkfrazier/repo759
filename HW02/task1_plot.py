import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("results_euler2.txt")
i = data[:, 0].astype(int)
t_ms = data[:, 1]

plt.figure()
plt.plot(i, t_ms, marker='o')

plt.yscale('log')

plt.grid(True, which="both", ls="--", lw=0.5)

plt.title("Inclusive Scan Time vs Array Size")
plt.xlabel("n")
plt.ylabel("scan time (ms)")
plt.xticks(i, [fr"$2^{{{k}}}$" for k in i])

plt.tight_layout()
plt.savefig("task1.pdf", dpi=400)