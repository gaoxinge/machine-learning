import numpy as np
import matplotlib.pyplot as plt


h = 0.01
steps = 1000
xs = [0 for _ in range(steps)]
ys = [0 for _ in range(steps)]
for i in range(steps):
    if i == 0:
        xs[0] = 0
        ys[0] = np.exp(2)
    else:
        xs[i] = xs[i - 1] + h
        ys[i] = ys[i - 1] + 3 * ys[i - 1] * h + 2 * h
fig = plt.figure()
plt.plot(xs, ys)
fig.savefig("tmp2.png")
