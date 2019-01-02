import numpy as np
import scipy.optimize as opt
import math
from sigmoid import sigmoid

class LogisticRegression:
    def __init__(self, X, y):
        self.features = self.__features_init(X)
        self.results = y
        self.m = X.shape[0]
        self.n = X.shape[1] + 1
        self.thetas = np.zeros((self.n, 1))

    def __features_init(self, X):
        column_zero = np.ones((X.shape[0], 1))
        return np.hstack((column_zero, X))

    # Compute our cost using paramater, feature values, and result values
    def compute_cost(self, thetas=None, X=None, y=None):
        if(X is None and y is None and thetas is None):
            X = self.features
            y = self.results
            thetas = self.thetas
        elif(not (X is not None and y is not None and thetas is not None) ):
            raise Exception('Must provide either all or none of the arguments')

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

    def gradient(self, thetas=None, X=None, y=None):
        if(X is None and y is None and thetas is None):
            X = self.features
            y = self.results
            thetas = self.thetas
        elif(not (X is not None and y is not None and thetas is not None) ):
            raise Exception('Must provide either all or none of the arguments')

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
    
    # Uses the truncated Newton method to determine our theata values that minimize our overall cost
    def optimize(self):
        optimized_values = opt.fmin_tnc(func=self.compute_cost, x0=self.thetas, fprime=self.gradient, args=(self.features, self.results))
        self.thetas = np.reshape(optimized_values[0], (self.n, 1))

    # Takes in a 1D array of feature values, and predicts based on current parameters. Returns a value [0,1] 
    def predict(self, X):
        if(X.shape[0] == self.n - 1):
            X = np.insert(X, 0, 1)
        
        predicted_probability = sigmoid(np.dot(X, self.thetas)[0])
        return predicted_probability