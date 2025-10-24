import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from math import e, floor

# Please just look at the github. This format sucks so much
# https://github.com/LoweBerg/projectFlerdim

# Task 1


def f1(x, y):
    return (x ** 2 - x * y) / (x ** 2 - y ** 2)


def f2(x, y):
    return (x ** 2 + y ** 2) / (x ** 2 + x * y + y ** 2)


def f3(x, y):
    return (np.sin(x + x * y) - x - x * y) / (x ** 3 * (y + 1) ** 3)


def f4(x, y):
    return 8 * x * y - 4 * x ** 2 * y - 2 * x * y ** 2 + x ** 2 * y ** 2


def f5(x, y):
    return (x ** 2 + 3 * y ** 2) * np.e ** (-x ** 2 - y ** 2)


def gen_z(f, x, y, width, res):
    X_, Y_ = np.meshgrid(np.linspace(x - width, x + width, res), np.linspace(y - width, y + width, res))

    Z_ = (f(X_, Y_))

    return X_, Y_, Z_


# setup
resolution = 100  # for most of the plots 100 resolution gives good enough results
resolution_4 = 1000  # for the 4th plot a higher resolution more clearly distinguishes the local extrema

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

if plot1:  # no limit since function is undefined along x = y
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

if plot2:  # there is no limit here since there are multiple z-values for the same x,y
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

if plot3:  # looks like it has limit -1/6 but is actually undefined along y = x - 1
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

if plot4:  # local maximum at x,y = 1,2 which is best seen in the contour graph
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

if plot5:  # two local maxima at x=0 y=1,-1 and one minimum at the origin
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


# Task 2


def ddx(f, x, y, h):
    return (f(x+h, y)-f(x, y))/h


def ddy(f, x, y, h):
    return (f(x, y+h)-f(x, y))/h


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


def func(x, y):
    """
    function from assignment
    """
    return np.sin(x+y)


def analytic_grad(x, y):
    """
    calculates the analytical gradient of sin(x+y) at a point (x, y)
    """
    return np.array([np.cos(x+y), np.cos(x+y)])


def analytic_hess(x, y):
    """
    calculates the analytical hessian of sin(x+y) at a point (x, y)
    """
    return np.array([[-np.sin(x+y), -np.sin(x+y)], [-np.sin(x+y), -np.sin(x+y)]])


def gen_err(y1_, y2_):
    """
    generates an array of absolute error between two vector arrays
    by comparing distance between vectors
    """
    diff = np.array([y2_[0] - y1_[0], y2_[1] - y1_[1]])
    err = np.array([np.sqrt(diff[0]**2+diff[1]**2)])

    return err


def f5(x, y):  # (0, +-1.25)
    """
    Second function from Eq. 2
    """
    return (x**2+3*y**2)*np.e**(-x**2-y**2)


# switches
plot1 = 1
plot2 = 1
plot3 = 1

# setup
np.vectorize(grad)
np.vectorize(ddx)
np.vectorize(ddy)

X1, Y1 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))

Z1 = grad(func, X1, Y1, 10**-6)

Z2 = analytic_grad(X1, Y1)

Err = gen_err(Z1, Z2)

Err = np.squeeze(Err, axis=0)

print("yay") # Very neccessary. Program needs some validation

if plot1:
    fig = plt.figure()
    plt.title("Absolute error between numerical and analytic gradient")
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Error")
    ax.set_zlim(-10**-5, 10**-5)
    ax.plot_surface(X1, Y1, Err)

X2 = np.linspace(-9, -4)

Y2 = np.array([np.linalg.norm(grad(func, np.pi/4, np.pi/4, 10**h) - analytic_grad(np.pi / 4, np.pi / 4)) for h in X2])

if plot2:
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title(r"Absolute error at point ($\frac{\pi}{4}$, $\frac{\pi}{4}$) depending on h")
    ax.set_xlabel("exponent of h")
    ax.set_ylabel("Absolute error")
    plt.grid()
    plt.plot(X2, Y2)

X3, Y3 = np.meshgrid(np.linspace(-100, 100, 500), np.linspace(-100, 100, 500))
Z1 = hessian(func, X3, Y3, 10**-6)[0, 0]
Z2 = analytic_hess(X3, Y3)[0, 0]

if plot3:
    fig = plt.figure()
    plt.title("Error in first value of hessian matrix")
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("error")
    ax.plot_surface(X3, Y3, Z2-Z1)


print("Gradient of f5: \n", grad(f5, 0, 0, 10**-6))
print("Hessian of f5: \n", hessian(f5, 0, 0, 10**-6))

plt.show()

# Task 3


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

# Task 4

# taking ddx, ddy, gradient and hessian from earlier tasks
def ddx(f, x, y, h):
    return (f(x+h, y)-f(x, y))/h

def ddy(f, x, y, h):
    return (f(x, y+h)-f(x, y))/h

def gradient(x, y, func, h):
    return np.array([ddx(func, x, y, h), ddy(func, x, y, h)])

def hessian(f, x, y, h):
    dfxx = (ddx(f, x+h, y, h) - ddx(f, x, y, h))/h
    dfxy = (ddx(f, x, y+h, h) - ddx(f, x, y, h))/h
    dfyy = (ddy(f, x, y+h, h) - ddy(f, x, y, h))/h

    return np.array([[dfxx, dfxy], [dfxy, dfyy]])

def himmelblau(x, y):
    return (x**2+y-11)**2+(x+y**2-7)**2

def optimization(x_init, y_init, func, alpha, k):
    global iterates     # need to access this for the plot, probably not the cleanest/most efficient method but it works (?)
    point = np.array([[x_init, y_init]])    # make the initial point into an array
    iterates = []   # a list of every point we get throughout the algorithm, index 1-20
    for i in range(0, k):
        newpoint = point - alpha*gradient(point[0][0], point[0][1], func, 10**(-6))
        point = newpoint
        iterates.append(point)
    print(iterates)
    print('\n\n point ', point)
    return point    # = point**(k+1)

# plotting himmelblau on given interval using same methoud as prev tasks
X = np.outer(np.linspace(-5, 5, 50), np.ones(50))
Y = X.copy().T
Z = himmelblau(X, Y)

ax1 = plt.subplot(1, 2, 1, projection = '3d', xlabel = 'x', ylabel = 'y', zlabel='z')
ax2 = plt.subplot(1,2,2, projection = '3d', xlabel = 'x', ylabel = 'y', zlabel='z')
ax1.plot_surface(X, Y, Z, color = 'orange')
plt.contourf(X, Y, Z, 100, cmap='inferno')

ax1.set_title('Plot using .plot_surface')
ax2.set_title('Plot using .contourf')
plt.colorbar(shrink=0.3)
plt.suptitle("Himmelblau's fcn")
plt.show()
# can see that we have 4 minimas & one maxima locally!

# using our algorithm to find minimas around the two points
optimization(0, 0, himmelblau, 0.01, 20)                # algorithm run 1
X1 = np.array([item[0][0] for item in iterates])
Y1 = np.array([item[0][1] for item in iterates])
Z1 = himmelblau(X1, Y1)

optimization(0.2, -4, himmelblau, 0.01, 20)             # algorithm run 2
X2 = np.array([item[0][0] for item in iterates])
Y2 = np.array([item[0][1] for item in iterates])
Z2 = himmelblau(X2, Y2)

# plotting the iteration points computed by the algorithm as a function
ax1 = plt.subplot(1,2,1, projection = '3d', xlabel = 'x', ylabel = 'y', zlabel='z')
ax2 = plt.subplot(1,2,2, projection = '3d', xlabel = 'x', ylabel = 'y', zlabel='z')
ax1.scatter3D(X1, Y1, Z1)
ax1.plot_surface(X, Y, Z, alpha = 0.5)
ax2.plot_surface(X, Y, Z, alpha = 0.5)
ax2.scatter3D(X2, Y2, Z2)
ax1.set_title('Algorithm applied to point (0,0)')
ax2.set_title('Algorithm applied to point (1/5, -4)')

plt.suptitle("Himmelblau's fcn")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# checking if we found minima around (0,0)
x20 = X1[19]
print(x20)
y20 = Y1[19]
print(y20)

print('\n Gradient (0,0)', gradient(x20, y20, himmelblau, 10**(-6)))
print('\n Hessian (0,0)', hessian(himmelblau, x20, y20, 10**(-6)))

# around (1/5, -4)
x20 = X2[19]
print(x20)
y20 = Y2[19]
print(y20)

print('\n Gradient (1/5, -4)', gradient(x20, y20, himmelblau, 10**(-6)))
print('\n Hessian (1/5, -4)', hessian(himmelblau, x20, y20, 10**(-6)))

# the gradient is approximately zero at both points, and both hessians are
# positive definite, which means both points are near the local minimas!
