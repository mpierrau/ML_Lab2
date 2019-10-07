import numpy
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#numpy.random.seed(100)
C = 0.8
classA = numpy.concatenate((numpy.random.randn( 10 , 2) * 0.2 + [ 1.5 , 0.5 ],numpy.random.randn( 10 , 2) * 0.2 + [ -1.5 , 0.5 ] ) )
classB = numpy.random.randn( 20 , 2) * 0.2 + [ 0.0 , -0.5]
x = numpy.concatenate((classA,classB))
t = numpy.concatenate((numpy.ones(classA.shape[0]),-numpy.ones(classB.shape[0])))
N = x.shape[0] # Number of rows (samples) (note that each training sample will have a corresponding α-value.)
α = numpy.zeros(N)

# These four lines randomly reorders the samples.
permute = list(range(N))
random.shuffle(permute)
x = x[permute,:]
t = t[permute]


"""
Linear Kernel
"""
def K1(x,y):
    return numpy.dot(x,y)

"""
Polynomial Kernel
"""
def K2(x,y,p):
    return ( numpy.dot(x,y) + 1 )**p

"""
Radial Basis Function (RBF) Kernel
"""
def K3(x,y,σ):
    return numpy.exp( -( ((x - y)**2)/(2*σ**2) ) )

"""
Takes a vector as an argument
Returns a scalar
"""
def objective(α):
    part1 = 0
    for i in range(N):
        for j in range(N):
            part1 += α[i]*α[j]*t[i]*t[j]*K1(x[i],x[j])
    return 1/2*numpy.sum(part1) - numpy.sum(α)




"""
Calculates a value which should be constrained to zero
Takes a vector as an argument
Returns a scalar
"""
def zerofun(α):
    return numpy.dot(α,t)


"""
Indicator Function
"""
def ind(α,s,ts,b):
    result = 0
    for i in range(N):
        result += α[i]*t[i]*K1(s,x[i])
    return result - b

"""
Equation (7)
b is threshold value
"""
def b(α,s,ts):
    result = 0
    for i in range(N):
        result += α[i]*t[i]*K1(s,x[i])
    return result - ts

## ===== HEART OF THE PROGRAM ===== ##




B=[(0, C) for b in range(N)]
XC = {'type':'eq' , 'fun':zerofun}

ret = minimize( objective, α, bounds = B, constraints = XC)
alpha = ret[ 'x' ]

# Extracting the non-zeros α
nonzero_α_indices = numpy.nonzero(alpha > 10e-5)
nonzero_x_values = x[nonzero_α_indices]
nonzero_t_values = t[nonzero_α_indices]



plt.plot( [p[0] for p in classA] , [p[1] for p in classA] , 'b.')
plt.plot( [p[0] for p in classB] , [p[1] for p in classB] , 'r.')
plt.plot( [p[0] for p in nonzero_x_values] , [p[1] for p in nonzero_x_values] , 'k.')
plt.axis('equal')
plt.legend(['classA','classB'])
plt.grid()
plt.show()

xgrid = numpy.linspace(-5,5)
ygrid = numpy.linspace(-4,4)

grid = numpy.array([[ind()]])
