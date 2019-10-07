import numpy
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

numpy.random.seed(100)
C = 0.8
classA = numpy.concatenate((numpy.random.randn( 10 , 2) * 0.2 + [ 1.5 , 0.5 ],numpy.random.randn( 10 , 2) * 0.2 + [ -1.5 , 0.5 ] ) )
classB = numpy.random.randn( 20 , 2) * 0.2 + [ 0.0 , -0.5]
x = numpy.concatenate((classA,classB))
t = numpy.concatenate((numpy.ones(classA.shape[0]),-numpy.ones(classB.shape[0])))
N = x.shape[0] # Number of rows (samples) (note that each training sample will have a corresponding α-value.)
α = numpy.zeros(N)
epsilon = 0.00001

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
Indicator Function (Equation (6))
Classifies new data point [new_x, new_y] given alpha, t and b.
"""
def ind(α,t,new_x,new_y,b):
    result = 0
    s = [new_x, new_y]
    for i in range(len(α)):
        result += α[i]*t[i]*K1(s,x[i])
    return result - b

"""
Equation (7)
b is threshold value
"""
def b(α_nz,sv,t_sv, x_nz, t_nz):
    result = 0
    for i in range(len(α_nz)):
        result += α_nz[i]*t_nz[i]*K1(sv,x_nz[i])
    return result - t_sv

## ===== HEART OF THE PROGRAM ===== ##




B=[(0, C) for b in range(N)]
XC = {'type':'eq' , 'fun':zerofun}

ret = minimize( objective, α, bounds = B, constraints = XC)
alpha = ret[ 'x' ]

# Extracting the non-zeros α
nonzero_α_indices = numpy.nonzero(alpha > 10e-5)
nonzero_α = alpha[nonzero_α_indices]
nonzero_x = x[nonzero_α_indices]
nonzero_t = t[nonzero_α_indices]

# Calculate b value using equation 7
# Must use a point ON the margin. This corresponds to a point with 0 < alpha < C
# Below we extract the indices of the alpha fulfill the above criteria
# Rounding error means that values like 0.7999999999 are still less than 0.8, but they
# shouldn't be... For now I solve it by subtracting some small epsilon

SV_indices = []

for i in range(len(nonzero_α)):
    if ((0 < nonzero_α[i]) & (nonzero_α[i] < (C - epsilon))):
        SV_indices.append(i)

# Pick one of these Support Vectors and corresponding t-value
SV_point = nonzero_x[SV_indices][0]
SV_t = nonzero_t[SV_indices][0]

# Calculate b using this SV and t_SV, as well as all alpha that fulfill the criteria
# and aren't 0.

b_val = b(nonzero_α[SV_indices],
  SV_point,
  SV_t,
  nonzero_x[SV_indices],
  nonzero_t[SV_indices])

# These are for plotting the decision boundary
xgrid = numpy.linspace(-5,5)
ygrid = numpy.linspace(-4,4)
grid = numpy.array([[ind(nonzero_α[SV_indices], nonzero_t[SV_indices], x, y,b_val)
                     for x in xgrid]
                    for y in ygrid])
plt.plot( [p[0] for p in classA] , [p[1] for p in classA] , 'b.')
plt.plot( [p[0] for p in classB] , [p[1] for p in classB] , 'r.')
plt.plot( [p[0] for p in nonzero_x] , [p[1] for p in nonzero_x] , 'k.')
plt.axis('equal')
plt.legend(['classA','classB'])
plt.grid()
plt.contour(xgrid, ygrid, grid,
            (-1.0, 0.0, 1.0),
            colors=('red', 'black', 'blue'),
            linewidths=(1,3,1))
plt.show()
