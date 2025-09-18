import numpy as np
import matplotlib.pyplot as plt


def f1(x, y):  # guess limit is 0.5
    return (x**2-x*y)/(x**2-y**2)


def f2(x, y):
    return (x**2 + y**2)/(x**2 + x*y + y**2)


def gen_z(f, a, b, res):
    X_, Y_ = np.meshgrid(np.linspace(a - 1, a + 1, res), np.linspace(b - 1, b + 1, res))

    Z_ = (f(X_, Y_))

    return X_, Y_, Z_


# setup
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


X, Y, Z = gen_z(f2, 0, 0, 50)

ax.plot_surface(X, Y, Z)

plt.show()
