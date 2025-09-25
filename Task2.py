import numpy as np
import matplotlib.pyplot as plt


def ddx(f, x, y, h):
    return (f(x+h, y)-f(x, y))/h


def ddy(f, x, y, h):
    return (f(x, y+h)-f(x, y))/h


def grad(f, x, y, h):
    return np.array([ddx(f, x, y, h), ddy(f, x, y, h)])


def func(x, y):
    return np.sin(x+y)


def analytic_sine(x, y):
    return np.cos(x+y), np.cos(x+y)


def gen_z(f, x, y, width, res):
    X_, Y_ = np.meshgrid(np.linspace(x - width, x + width, res), np.linspace(y - width, y + width, res))

    Z_ = (f(X_, Y_))

    return X_, Y_, Z_


# setup
ax = plt.figure().add_subplot(projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Error")
ax.set_zlim(-10**-5, 10**-5)
np.vectorize(grad)
np.vectorize(ddx)
np.vectorize(ddy)

X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))

Z1 = grad(func, X, Y,  10**-5)

Z2 = np.array(analytic_sine(X, Y))

Z3 = np.array([Z2[0] - Z1[0], Z2[1] - Z2[1]])

Err = np.array([np.sqrt(Z3[0]**2+Z3[1]**2)])

print("yay")
Err = np.squeeze(Err, axis=0)

ax.plot_surface(X, Y, Err)
plt.show()
