import numpy as np
import scipy.optimize as opt
import math
from sigmoid import sigmoid

def compute_cost(thetas, X, y):
    cost = 0
    theta_T = thetas.transpose()
    m = X.shape[0]
    n = X.shape[1]

    for i in range(0, m):
        x_i = X[i: i + 1, 0: n].transpose()
        y_i = y[i, 0]
        theta_feature_product = np.dot(theta_T, x_i)[0]
        hypothesis_value = sigmoid(theta_feature_product)
        cost += (y_i * math.log(hypothesis_value)) + ((1 - y_i) * math.log(1 - hypothesis_value))
    cost = -(cost / m)
    return cost

def compute_cost_reg(thetas, X, y, lambda_t=1.0):
    m = X.shape[0]    
    cost = compute_cost(thetas, X, y)

    theta_it = iter(thetas)
    reduced_theta = 0
    next(theta_it) # Skip first theta, not needed
    for theta in theta_it:
        reduced_theta += theta ** 2
    reduced_theta = (reduced_theta) * (lambda_t / (2.0 * m))
    return cost + reduced_theta
    

def gradient(thetas, X, y):
    gradient = np.ndarray(thetas.shape[0])
    theta_T = thetas.transpose()
    m = X.shape[0]
    n = X.shape[1]

    for j in range(0, n):
        term = 0
        for i in range(0, m):
            x_i = X[i: i + 1, 0: n].transpose()
            y_i = y[i, 0]

            theta_feature_product = np.dot(theta_T, x_i)[0]
            hypothesis_value = sigmoid(theta_feature_product)

            term += (hypothesis_value - y_i) * X[i, j]
        
        gradient[j] = (term / m)
    return gradient

def gradient_reg(thetas, X, y, lambda_t=1.0):
    gradient = np.ndarray(thetas.shape[0])
    theta_T = thetas.transpose()
    m = X.shape[0]
    n = X.shape[1]

    for j in range(0, n):
        term = 0
        for i in range(0, m):
            x_i = X[i: i + 1, :].transpose()
            y_i = y[i, 0]

            theta_feature_product = np.dot(theta_T, x_i)[0]
            hypothesis_value = sigmoid(theta_feature_product)

            term += (hypothesis_value - y_i) * X[i, j]
        
        gradient[j] = (term / m) + ( (lambda_t / m) * thetas[j] )

    print(gradient)
    return gradient

# Uses the truncated Newton method to determine our theata values that minimize our overall cost
def optimize(thetas, X, y):
    optimized_values = opt.fmin_tnc(func=compute_cost, x0=thetas, fprime=gradient, args=(X, y))
    return optimized_values[0]

def optimize_reg(thetas, X, y, lamda_t=1.0):
    optimized_values = opt.fmin_tnc(func=compute_cost_reg, x0=thetas, fprime=gradient_reg, args=(X, y, lamda_t))
    return optimized_values[0]

# Takes in a 1D array of feature values, and predicts based on current parameters. Returns a value [0,1] 
def predict(thetas, X):
    predicted_probability = sigmoid(np.dot(X, thetas)[0])
    return predicted_probability

def map_feature(degree, X, y):
    feature_map = np.ones(1)
    feature_one = X[:, 1]
    feature_two = X[:, 2]

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            term_k = np.dot((np.power(feature_one, (i-j))), (np.power(feature_two, j)))
            feature_map = np.hstack((feature_map, term_k))

    return feature_map
