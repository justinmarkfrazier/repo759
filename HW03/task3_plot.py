import numpy as np
import matplotlib.pyplot as plt

data16 = np.loadtxt("results16.txt")
i16 = data16[:, 0].astype(int)
t_ms16 = data16[:, 1]

data512 = np.loadtxt("results512.txt")
i512 = data512[:, 0].astype(int)
t_ms512 = data512[:, 1]

plt.figure()
plt.plot(i16, t_ms16, marker='o', label="Block size 16")
plt.plot(i512, t_ms512, marker='s', label="Block size 512")

plt.yscale('log')

plt.grid(True, which="both", ls="--", lw=0.5)

plt.title("vscale time vs n for different block sizes")
plt.legend()
plt.xlabel("n")
plt.ylabel("vscale time (ms)")
plt.xticks(i16, [fr"$2^{{{k}}}$" for k in i16])

plt.tight_layout()
plt.savefig("task3.pdf", dpi=400)