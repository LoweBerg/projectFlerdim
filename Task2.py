import numpy as np
import matplotlib.pyplot as plt


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
