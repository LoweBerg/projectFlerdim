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


def f5(x, y):  # (0, +-1.25)
    return (x**2+3*y**2)*np.e**(-x**2-y**2)


def gen_z(f, x, y, width, res):
    X_, Y_ = np.meshgrid(np.linspace(x - width, x + width, res), np.linspace(y - width, y + width, res))

    Z_ = (f(X_, Y_))

    return X_, Y_, Z_


# setup
#fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
#axs[0].set_xlabel("X")
#axs[0].set_ylabel("Y")
#axs[0].set_zlabel("Z")
#axs[1].set_xlabel("X")
#axs[1].set_ylabel("Y")
#axs[1].set_zlabel("Z")
resolution = 100 # for most of the plots 100 resolution gives good enough results
resolution_4 = 1000 # for the 4th plot a higher resolution more clearly distinguishes the local extrema

X1, Y1, Z1 = gen_z(f1, 1, 1, 1, resolution)
X2, Y2, Z2 = gen_z(f2, 0, 0, 1, resolution)
X3, Y3, Z3 = gen_z(f3, 0, -1, 1, resolution)

X4, Y4, Z4 = gen_z(f4, 1, 2, 5, resolution_4)
Z4[Z4 > 10] = np.nan  # matplotlib limits are shit
Z4[Z4 < 0] = np.nan

X5, Y5, Z5 = gen_z(f5, 0, 0, 2, resolution)


# switches; makes it easy to switch from one plot to another :D

plot1 = 1
plot2 = 1
plot3 = 1
plot4 = 1
plot5 = 1


# prepares the plots for plotting if a switch is turned on

if plot1:   # the plot shows that this limit equals 0.5 for x,y -> (1,1)
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    fig.suptitle("Plot 1")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_zlabel("Z")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].set_zlabel("Z")
    axs[0].plot_surface(X1, Y1, Z1)
    pcm = axs[1].contourf(X1, Y1, Z1, levels=20, cmap='rainbow')
    fig.colorbar(pcm, ax=axs[1])
    
if plot2:   # there is no limit here since there are multiple z-values for the same x,y
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    fig.suptitle("Plot 2")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_zlabel("Z")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].set_zlabel("Z")
    axs[0].plot_surface(X2, Y2, Z2)
    pcm = axs[1].contourf(X2, Y2, Z2, levels=20, cmap='rainbow')
    fig.colorbar(pcm, ax=axs[1])
    
if plot3:   # limit approaches -1/6
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    fig.suptitle("Plot 3")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_zlabel("Z")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].set_zlabel("Z")
    axs[0].plot_surface(X3, Y3, Z3)
    pcm = axs[1].contourf(X3, Y3, Z3, levels=20, cmap='rainbow')
    fig.colorbar(pcm, ax=axs[1])
    
if plot4:   # local maximum at x,y = 1,2 which is best seen in the contour graph
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    fig.suptitle("Plot 4")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_zlabel("Z")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].set_zlabel("Z")
    axs[0].plot_surface(X4, Y4, Z4)
    pcm = axs[1].contourf(X4, Y4, Z4, levels=20, cmap='rainbow')
    fig.colorbar(pcm, ax=axs[1])
    
if plot5:   # two local maxima at x=0 y=1,-1 and one minimum at the origin
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    fig.suptitle("Plot 5")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_zlabel("Z")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].set_zlabel("Z")
    axs[0].plot_surface(X5, Y5, Z5)
    pcm = axs[1].contourf(X5, Y5, Z5, levels=20, cmap='rainbow')
    fig.colorbar(pcm, ax=axs[1])


plt.show()

