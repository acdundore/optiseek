import numpy as np

def booth(x1, x2):
    y = (x1 + 2 * x2 - 7)**2 + (2 * x1 + x2 - 5)**2
    return y

def rosenbrock(x1, x2):
    a = 1
    b = 5
    y = (a - x1)**2 + b * (x2 - x1**2)**2
    return y

def wheelers_ridge(x1, x2):
    a = 1.5
    y = -np.exp(-(x1 * x2 - a)**2 - (x2 - a)**2)
    return y

def ackley2D(x1, x2):
    y = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + np.exp(1) + 20
    return y





