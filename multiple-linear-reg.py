import numpy as np
import math, copy

xTrain = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
yTrain = np.array([460, 232, 178])

b_initial = 785.1811367994083
w_initial = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
m,n = xTrain.shape

def getPrediction(x, w, b):
    # wx + b
    return np.dot(x, w) + b

# squared error cost function
def getCost(x, y, w, b):
    # sum of each prediction - the true value, squared
    sum = 0

    for i in range(m):
        error = getPrediction(x[i], w, b) - y[i]
        sum += error**2

    return sum / (2*m)

def computeGradient(x, y, w, b):
    derivativeCostW = 0.0
    derivativeCostB = 0.0

    for i in range(m):
        error = (np.dot(x[i], w) + b) - y[i]

        for j in range(n):
            derivativeCostW = derivativeCostW + error * x[i, j]
        derivativeCostB = derivativeCostB + error

    return derivativeCostW, derivativeCostB

def gradientDescent(x, y, w_init, b_init, alpha, numIterations):
    w = copy.deepcopy(w_init)
    b = b_init


    for i in range(numIterations):
        derivativeCostW, derivativeCostB = computeGradient(x, y, w, b)
        w = w - alpha / m * derivativeCostW
        b = b - alpha / m * derivativeCostB

        if i% math.ceil(numIterations / 10) == 0:
            print(getCost(x, y, w, b))

    return w, b

globalNumIterations = 1000
globalAlpha = 5.0e-7

gradientDescent(xTrain, yTrain, w_initial, b_initial, globalAlpha, globalNumIterations)
