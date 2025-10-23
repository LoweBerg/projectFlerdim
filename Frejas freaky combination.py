import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from math import e


def ddx(f, x, y, h):
    return (f(x+h, y)-f(x, y))/h


def ddy(f, x, y, h):
    return (f(x, y+h)-f(x, y))/h

def hessian(f, x, y, h):
    """
    Calculates the numerical hessian of a generic function around a point (x, y)
    :param f: function to be analyzed
    :param x: x-value of point
    :param y: y-value of point
    :param h: small value to approximate derivative
    :return: hessian matrix for functio
    """
    dfxx = (ddx(f, x+h, y, h) - ddx(f, x, y, h))/h
    dfxy = (ddx(f, x, y+h, h) - ddx(f, x, y, h))/h
    # not necessary but can be nice to have anyway
    # dfyx = (ddy(f, x+h, y, h) - ddy(f, x, y, h))/h
    dfyy = (ddy(f, x, y+h, h) - ddy(f, x, y, h))/h

    return np.array([[dfxx, dfxy], [dfxy, dfyy]])


def F_implicit(x, y, z):
    return x + 2*y + z + e**(2*z) - 1

X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)




roots = np.zeros((50, 50))#loop that iterates over every possible combination on the lists X and Y
DDX = np.zeros((50, 50))
DDY = np.zeros((50, 50))
DFXX = np.zeros((50, 50))
DFXY = np.zeros((50, 50))
DFYY = np.zeros((50, 50))

H = 10**-6
for i in range(np.size(X)):
    for j in range(np.size(X)):
        roots[i, j] = fsolve(lambda z: X[i] + 2*Y[j] + z + e**(2*z) - 1, 0)
        DDX[i, j] = (fsolve(lambda z: X[i]+H + 2*Y[j] + z + e**(2*z) - 1, 0) - roots[i, j]) / H
        DDY[i, j] = (fsolve(lambda z: X[i] + 2 * Y[j] + H + z + e ** (2 * z) - 1, 0) - roots[i, j]) / H
        DFXX[i, j] = (((fsolve(lambda z: X[i]+2*H + 2*Y[j] + z + e**(2*z) - 1, 0) - roots[i, j]) / H) - DDX[i, j]) / 2*H
        DFXY[i, j] = (((fsolve(lambda z: X[i]+H + 2*(Y[j]+H) + z + e**(2*z) - 1, 0) - roots[i, j]) / H) - DDX[i, j]) / H
        DFYY[i, j] = (((fsolve(lambda z: X[i] + 2*(Y[j]+2*H) + z + e**(2*z) - 1, 0) - roots[i, j]) / H) - DDY[i, j]) / 2*H
#Surface plot

X_, Y_ = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z_ = DFYY

print("wa")
print(DDX[0,0])
print(DDY[0,0])
print(DFXX[0,0])
print(DFXY[0,0])
print(DFYY[0,0])

ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(X_, Y_, DFYY)
plt.xlabel('x')
plt.ylabel('y')
plt.show()