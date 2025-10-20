import numpy as np
import matplotlib.pyplot as plt

def ddx(f, x, y, h):
    return (f(x+h, y)-f(x, y))/h

def ddy(f, x, y, h):
    return (f(x, y+h)-f(x, y))/h

def gradient(x, y, func, h):
    return np.array([ddx(func, x, y, h), ddy(func, x, y, h)])

def hessian(f, x, y, h):
    dfxx = (ddx(f, x+h, y, h) - ddx(f, x, y, h))/h
    dfxy = (ddx(f, x, y+h, h) - ddx(f, x, y, h))/h
    # not necessary but can be nice to have anyway
    # dfyx = (ddy(f, x+h, y, h) - ddy(f, x, y, h))/h
    dfyy = (ddy(f, x, y+h, h) - ddy(f, x, y, h))/h

    return np.array([[dfxx, dfxy], [dfxy, dfyy]])

def himmelblau(x, y):
    return (x**2 + y - 11)**2+ (x + y**2 - 7)**2

def optimization(x_init, y_init, func, alpha, k):
    global iterates     # need to access this for the plot, probably not the cleanest/most efficient method but it works (?)
    point = np.array([[x_init, y_init]])    # make the initial point into an array
    iterates = []   # a list of every point we get throughout the algorithm, index 1-20
    for i in range(0, k):
        newpoint = point - alpha*gradient(x_init, y_init, func, 10**(-3))
        point = newpoint
        iterates.append(point)
    print(iterates)
    return point, iterates    # = point**(k+1)

# plotting himmelblau on given interval using same methoud as prev tasks
X = np.outer(np.linspace(-5, 5, 50), np.ones(50))
Y = X.copy().T                                  
Z = himmelblau(X, Y)

ax1 = plt.subplot(1, 2, 1, projection = '3d')
ax2 = plt.subplot(1,2,2, projection = '3d')
ax1.plot_surface(X, Y, Z, color = 'orange')
plt.contourf(X, Y, Z, 100, cmap='inferno')

plt.colorbar(shrink=0.5)
plt.suptitle('function 1')
plt.xlabel('x')
plt.ylabel('y')
plt.show()     


# using our algorithm to find minimas around the two points
optimization(0, 0, himmelblau, 0.01, 20)                # algorithm run 1
X1 = np.outer([item[0][0] for item in iterates], np.ones(20))
Y1 = np.outer([item[0][1] for item in iterates], np.ones(20)).T
Z1 = himmelblau(X1, Y1)

optimization(1/5, -4, himmelblau, 0.01, 20)             # algorithm run 2
X2 = np.outer([item[0][0] for item in iterates], np.ones(20))
Y2 = np.outer([item[0][1] for item in iterates], np.ones(20)).T
Z2 = himmelblau(X2, Y2)

# plotting the iteration points computed by the algorithm as a function 
ax1 = plt.subplot(1,2,1, projection = '3d')
ax2 = plt.subplot(1,2,2, projection = '3d')
ax1.plot_surface(X1, Y1, Z1)
ax2.plot_surface(X2, Y2, Z2)
ax1.set_title('point (0,0)')
ax2.set_title('point (1/5, -4)')

plt.suptitle("himmelblau's fcn")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# checking if we found minima around (0,0)
x20 = X1[19][0]
print(x20)
y20 = Y1[19][19]
print(y20)

print('\n Gradient (0,0)', gradient(x20, y20, himmelblau, 10**(-3)))
print('\n Hessian (0,0)', hessian(himmelblau, x20, y20, 10**(-3)))

# around (1/5, -4)
x20 = X2[19][0]
print(x20)
y20 = Y2[19][19]
print(y20)

print('\n Gradient (1/5, -4)', gradient(x20, y20, himmelblau, 10**(-3)))
print('\n Hessian (1/5, -4)', hessian(himmelblau, x20, y20, 10**(-3)))

"""
gradient should be zero, hessian should be positive definite

"""
