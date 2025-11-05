import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from math import e


# Had to copy in the functions since importing also seemed to import plots from previous tasks
def ddx(f, x, y, h):
    return (f(x+h, y)-f(x, y))/h


def ddy(f, x, y, h):
    return (f(x, y+h)-f(x, y))/h


def gen_z(f, x, y, width, res) -> np.ndarray:
    X_, Y_ = np.meshgrid(np.linspace(x - width, x + width, res), np.linspace(y - width, y + width, res))

    Z_ = (f(X_, Y_))

    return X_, Y_, Z_


def grad(f, x, y, h):  # returns the gradient of a function f at a point x, y by comparing with x+h, y+h
    return np.array([ddx(f, x, y, h), ddy(f, x, y, h)])


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


def z(x, y):
    # Necessary steps to vectorize z
    shape = np.shape(x)
    x_ravel = np.ravel(x)
    y_ravel = np.ravel(y)

    res_ravel = np.zeros(np.shape(x_ravel))

    for i in range(np.size(x_ravel)):
        res_ravel[i] = fsolve(lambda z_: x_ravel[i] + 2*y_ravel[i] + z_ + e**(2*z_) - 1, x0=0)[0]

    res = np.reshape(res_ravel, shape)

    return res


G = grad(z, 0, 0, 10**-4)

H = hessian(z, 0, 0, 10**-4)


def p_2(x, y):
    return z(0, 0) + G[0] * x + G[1] * y + (H[0, 0] * x ** 2 + 2 * H[0, 1] * x * y + H[1, 1] * y ** 2) / 2


resolution = 50
half_width = 0.5

X, Y, Z = gen_z(z, 0, 0, half_width, resolution)
P_2 = gen_z(p_2, 0, 0, half_width, resolution)[2]


# Surface plot of Z(x, y)
ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(X, Y, Z)
plt.title('Z(x, y)')
plt.xlabel('X')
plt.ylabel('Y')

# surface plot of Z and Taylor fcn
ax = plt.figure().add_subplot(projection='3d', xlabel='x', ylabel='y', zlabel='z')
ax.plot_surface(X, Y, P_2, color='purple', lw=0.4, alpha=0.7, edgecolors='purple')            # plotting taylor
ax.plot_surface(X, Y, Z, color='blue', lw=0.4, alpha=0.7, edgecolors='b')                    # plotting Z
plt.title('Taylor fcn (purple) & Z fcn (blue)')

# calculating & plotting absolute error
err = abs(Z - P_2)

ax = plt.figure().add_subplot(projection='3d', xlabel='x', ylabel='y', zlabel='z')
plt.contourf(X, Y, err, levels=150, cmap='inferno')
plt.title('Absolute error')
plt.colorbar()
plt.show()
