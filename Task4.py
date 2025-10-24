import numpy as np
import matplotlib.pyplot as plt

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
