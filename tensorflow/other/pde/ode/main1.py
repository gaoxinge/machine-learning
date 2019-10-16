import numpy as np
import matplotlib.pyplot as plt


h = 0.01
steps = 1000
xs = [0 for _ in range(steps)]
ys = [0 for _ in range(steps)]
for i in range(steps):
    xs[i] = i * h
    ys[i] = np.exp(3 * xs[i] + 2)
fig = plt.figure()
plt.plot(xs, ys)
fig.savefig("tmp1.png")
