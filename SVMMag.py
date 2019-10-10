import numpy
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import SVM_DATA


numpy.random.seed(100)
C = 10000
x , t , classA , classB = SVM_DATA.input_sample4(100)
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
def K1(x_in,y_in):
        return numpy.dot(x_in,numpy.transpose(y_in))

"""
Polynomial Kernel
"""
def K01(x,y,p=6):
    return ( numpy.dot(x,numpy.transpose(y)) + 1 )**p

"""
Radial Basis Function (RBF) Kernel
"""
def K12(x,y,σ=1.0):
    return numpy.exp( -( (numpy.linalg.norm(x - y)**2)/(2*σ**2) ) )

"""
Takes a vector as an argument
Returns a scalar
"""
P = numpy.zeros([N,N])
for i in range(N):
    for j in range(N):
        P[i,j] = t[i]*t[j]*K1(x[i],x[j])

def objective(α):
    part1 = numpy.dot(α , numpy.dot(α.transpose() , P))/2
    return part1 - numpy.sum(α)


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
def ind(α_in,t_in,x_in,new_x,new_y,b):
    result = 0
    s = numpy.array([new_x, new_y])
    for i in range(len(α_in)):
        result += α_in[i]*t_in[i]*K1(s,x_in[i])
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




B=[(0, C) for index in range(N)]
XC = {'type':'eq' , 'fun':zerofun}

ret = minimize( objective, α, bounds = B, constraints = XC)
print(ret)
print(ret['message'])
print("Minimization successful? " + str(ret['success']))
alpha = ret[ 'x' ]

# Extracting the non-zeros α
NZ_indices = numpy.nonzero(alpha > 10e-5)
SV_indices = numpy.logical_and(alpha>10e-5,numpy.round(alpha,8) < C)

SV_α = alpha[SV_indices]
SV_x = x[SV_indices]
SV_t = t[SV_indices]
NZ_α = alpha[NZ_indices]
NZ_x = x[NZ_indices]
NZ_t = t[NZ_indices]

# Calculate b value using equation 7
# Must use a point ON the margin. This corresponds to a point with 0 < alpha < C
# Below we extract the indices of the alpha fulfill the above criteria
# Rounding error means that values like 0.7999999999 are still less than 0.8, but they


# Pick one of these Support Vectors and corresponding t-value
# Calculate b using this SV and t_SV, as well as all alpha that fulfill the criteria
# and aren't 0.
try:
    b_val = b(SV_α,SV_x[0],SV_t[0],SV_x,SV_t)
except IndexError:
    b_val = 0
print("b_val: " + str(b_val))

# These are for plotting the decision boundary
xgrid = numpy.linspace(-6,6)
ygrid = numpy.linspace(-6,6)
grid = numpy.array([[ind(SV_α, SV_t, SV_x, x, y,b_val) for x in xgrid] for y in ygrid])



plt.plot( [p[0] for p in classA] , [p[1] for p in classA] , 'b.')
plt.plot( [p[0] for p in classB] , [p[1] for p in classB] , 'r.')
plt.plot( [p[0] for p in SV_x] , [p[1] for p in SV_x] , 'k.')
plt.axis('equal')
plt.legend(['classA','classB','SV'])
plt.grid()

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))
plt.title("C = " + str(C))
plt.savefig(str(C) + ".png", dpi=200)
#plt.show()

plt.close()
