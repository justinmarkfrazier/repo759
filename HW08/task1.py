import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("results1.txt")
i = data[:, 0].astype(int)
t = data[:, 1]

plt.figure()
plt.plot(i, t, marker='o')

plt.grid(True, which="both", ls="--", lw=0.5)

plt.title("Task 1 Matrix Multiplication Time vs. Number of Threads")
plt.xlabel("threads")
plt.ylabel("time (ms)")
plt.xticks(i, [fr"{k}" for k in i])

plt.tight_layout()
plt.savefig("task1.png", dpi=400)