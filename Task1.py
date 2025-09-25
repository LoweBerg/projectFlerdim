import numpy as np
import matplotlib.pyplot as plt


def f1(x, y):  # guess limit is 0.5
    return (x**2-x*y)/(x**2-y**2)


def f2(x, y):  # guess no limit
    return (x**2 + y**2)/(x**2 + x*y + y**2)


def f3(x, y):
    return (np.sin(x+x*y)-x-x*y)/(x**3*(y+1)**3)


def f4(x, y):
    return 8*x*y-4*x**2*y-2*x*y**2+x**2*y**2


def f5(x, y):
    return (x**2+3*y**2)*np.e**(-x**2-y**2)


def gen_z(f, x, y, width, res):
    X_, Y_ = np.meshgrid(np.linspace(x - width, x + width, res), np.linspace(y - width, y + width, res))

    Z_ = (f(X_, Y_))

    return X_, Y_, Z_


# setup
fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[0].set_zlabel("Z")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
axs[1].set_zlabel("Z")
resolution = 100

X1, Y1, Z1 = gen_z(f1, 1, 1, 1, resolution)
X2, Y2, Z2 = gen_z(f2, 0, 0, 1, resolution)
X3, Y3, Z3 = gen_z(f3, 0, -1, 1, resolution)

X4, Y4, Z4 = gen_z(f4, 1, 2, 5, resolution)
Z4[Z4 > 10] = np.nan  # matplotlib limits are shit
Z4[Z4 < 0] = np.nan

X5, Y5, Z5 = gen_z(f5, 0, 0, 5, resolution)

axs[0].plot_surface(X5, Y5, Z5)
pcm = axs[1].contourf(X5, Y5, Z5, levels=20, cmap='rainbow')
fig.colorbar(pcm, ax=axs[1])
plt.show()
