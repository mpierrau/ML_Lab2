import numpy as np
import  pylab as random, math #py -m install numpy & py -m install scipy & py -m install matplotlib
import sklearn.datasets as dt #py -m install sklearn

def input_sample0(seed=None):
    np.random.seed(seed)
    classA = np.concatenate((np.random.randn(10,2)*0.2+[1.5,0.5], np.random.randn(20,2)*0.2+[-1.7,-0.6]) )
    classB = np.concatenate( (np.random.randn(20,2)*0.2+[1.6,-0.8] , np.random.randn(10,2)*0.2+[-1.5,0.5] ) )
    inputs=np.concatenate((classA,classB))
    targets= np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))
    inputs=np.concatenate((classA,classB))
    return inputs,targets,classA,classB


def input_sample1(seed=None):
    np.random.seed(seed)
    classA = np.concatenate((np.random.randn(10,2)*0.2+[1.5,0.5], np.random.randn(10,2)*0.2+[-1.5,0.5]))
    classB = np.random.randn(20,2)*0.2+[0.0,-0.5]
    inputs=np.concatenate((classA,classB))
    targets= np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))
    inputs=np.concatenate((classA,classB))
    return inputs,targets,classA,classB

def input_sample2(seed=None):
    np.random.seed(seed)
    classA = np.concatenate((np.random.randn(10,2)*0.2+[2,0.5], np.random.randn(10,2)*0.2+[3,0.5]))
    classB = np.random.randn(20,2)*0.2+[4,0.5]
    inputs=np.concatenate((classA,classB))
    targets= np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))
    inputs=np.concatenate((classA,classB))
    return inputs,targets,classA,classB

def input_sample3(seed=None):
    np.random.seed(seed)
    listA = [(np.random.normal(-2, 1), np.random.normal(1.5,1)) for i in range(10)] +\
             [(np.random.normal(2, 1), np.random.normal(0.5,1)) for i in range(10)]
    listB = [(np.random.normal(2, 0.5), np.random.normal(-2, 0.5)) for i in range(10)]
    classA=np.asarray(listA)
    classB=np.asarray(listB)
    inputs=np.concatenate((classA,classB))
    targets= np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))
    inputs=np.concatenate((classA,classB))
    return inputs,targets,classA,classB

def input_sample4(seed=None):
    np.random.seed(seed)
    X, Y = dt.make_moons(100)
    
    classC = []
    listA = []
    listB = []
    for i in range(len(Y)):
        if(Y[i]==1):
            listA.append(X[i])
        else:
             listB.append(X[i])
    classA=np.asarray(listA)
    classB=np.asarray(listB)
    print(type(listA))
    print(type(classA))
    inputs=np.concatenate((classA,classB))
    targets= np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))
    
    return inputs, targets,classA,classB

def input_sample5(seed=None):
    # Make a large circle containing a smaller circle in 2d
    np.random.seed(seed)
    X, Y = dt.make_circles(100, factor=0.2, noise=0.1)
    classC = []
    classC = []
    listA = []
    listB = []
    for i in range(len(Y)):
        if(Y[i]==1):
            listA.append(X[i])
        else:
             listB.append(X[i])
    classA=np.asarray(listA)
    classB=np.asarray(listB)
    print(type(listA))
    print(type(classA))
    inputs=np.concatenate((classA,classB))
    targets= np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))
    
    return inputs, targets,classA,classB