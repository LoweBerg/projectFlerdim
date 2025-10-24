import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from math import e, floor, sqrt

X = np.linspace(-10**(-5), 10**(-5), 51)
Y = np.linspace(-10**(-5), 10**(-5), 51)
roots = np.zeros((51, 51))#loop that iterates over every possible combination on the lists X and Y

for i in range(np.size(X)):
    for j in range(np.size(X)):
        roots[i, j] = fsolve(lambda z: X[i] + 2*Y[j] + z + e**(2*z) - 1, 0)

Mid = floor(51/2)
H = (2*10**-5)/51

# coefficients

DDX = (roots[Mid+1, Mid]-roots[Mid, Mid])/H
DDY = (roots[Mid, Mid+1]-roots[Mid, Mid])/H
DFXX = (roots[Mid+2, Mid] - 2*roots[Mid+1, Mid] + roots[Mid, Mid])/(H**2)
DFXY = (roots[Mid+1, Mid+1]-roots[Mid, Mid+1]-roots[Mid+1, Mid]-roots[Mid, Mid])/(H**2)
DFYY = (roots[Mid, Mid+2] - 2*roots[Mid, Mid+1] + roots[Mid, Mid])/(H**2)


def P_2(x, y):
    return roots[Mid, Mid] + DDX*x + DDY*y + (DFXX*x**2 + 2*DFXY*x*y + DFYY*y**2)/2


PVZ_2 = np.zeros((51, 51))  # loop that iterates over every possible combination on the lists X and Y

for i in range(np.size(X)):
    for j in range(np.size(X)):
        PVZ_2[i, j] = P_2(X[i], Y[j])

#Surface plot of Z(x, y)

X_, Y_ = np.meshgrid(np.linspace(-10**(-5), 10**(-5), 51), np.linspace(-10**(-5), 10**(-5), 51))
Z_ = roots

ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(X_, Y_, Z_)
plt.title('Z(x, y)')
plt.xlabel('X')
plt.ylabel('Y')

# surface plot of Z and Taylor fcn

ax = plt.figure().add_subplot(projection='3d', xlabel='x', ylabel='y', zlabel='z')
ax.plot_surface(X_, Y_, P_2(X_, Y_), color='purple', lw=0.4, alpha=0.7, edgecolors='purple')            # plotting taylor
ax.plot_surface(X_, Y_, Z_, color='blue', lw=0.4, alpha=0.7, edgecolors='b')                    # plotting Z
plt.title('Taylor fcn (purple) & Z fcn (blue)')

# calculating & plotting absolute error
err = abs(Z_ - PVZ_2)

ax = plt.figure().add_subplot(projection='3d', xlabel = 'x', ylabel = 'y', zlabel = 'z')
plt.contourf(X_, Y_, err, levels = 150, cmap = 'inferno')
plt.title('Absolute error')
plt.colorbar()
plt.show()
