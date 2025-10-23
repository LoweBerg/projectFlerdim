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

DDX = (roots[Mid+1, Mid]-roots[Mid, Mid])/H
DDY = (roots[Mid, Mid+1]-roots[Mid, Mid])/H
DFXX = (roots[Mid+2, Mid] - 2*roots[Mid+1, Mid] + roots[Mid, Mid])/(H**2)
DFXY = (roots[Mid+1, Mid+1]-roots[Mid, Mid+1]-roots[Mid+1, Mid]-roots[Mid, Mid])/(H**2)
DFYY = (roots[Mid, Mid+2] - 2*roots[Mid, Mid+1] + roots[Mid, Mid])/(H**2)


def P_2(x, y):
    return roots[Mid, Mid] + DDX*x + DDY*y + (DFXX*x**2 + 2*DFXY*x*y + DFYY*y**2)/2


print(roots[Mid+1, Mid+1])
print(roots[Mid+1, Mid])
print(DDX)
print(DFXY)


#Surface plot

X = np.linspace(-1, 1, 51)
Y = np.linspace(-1, 1, 51)
roots = np.zeros((51, 51))#loop that iterates over every possible combination on the lists X and Y

for i in range(np.size(X)):
    for j in range(np.size(X)):
        roots[i, j] = fsolve(lambda z: X[i] + 2*Y[j] + z + e**(2*z) - 1, 0)

X_, Y_ = np.meshgrid(np.linspace(-1, 1, 51), np.linspace(-1, 1, 51))
Z_ = roots


ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(X_, Y_, P_2(X_, Y_), color = 'pink', alpha = 0.7)            # plotting taylor
ax.plot_surface(X_, Y_, Z_, color= 'orange', alpha = 0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.show()



