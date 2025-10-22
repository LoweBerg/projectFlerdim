import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from math import e


def F_implicit(x, y, z):
    return x + 2*y + z + e**(2*z) - 1

X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)

roots = np.zeros((50, 50))     #loop that iterates over every possible combination on the lists X and Y
for i in range(np.size(X)):
    for j in range(np.size(X)):
        roots[i, j] = fsolve(lambda z: X[i] + 2*Y[j] + z + e**(2*z) - 1, 0)


#Surface plot

X_, Y_ = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z_ = roots

ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(X_, Y_, Z_)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

