import numpy as np
import matplotlib.pyplot as plt


def f1(x, y):  # guess limit is 0.5
    return (x**2-x*y)/(x**2-y**2)


res = 50
a, b = 1, 1
np.vectorize(f1)
X = np.outer(np.linspace(0, 2, res), np.ones(res))
Y = X.copy().T
Z = (f1(X, Y))

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z)
ax.plot(1, 1, 1)

plt.grid()
plt.show()