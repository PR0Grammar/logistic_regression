import numpy as np
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

    def compute_cost(self):
        cost = 0
        theta_T = self.thetas.transpose()

        for i in range(0, self.m):
            x_i = self.features[i: i + 1, 0: self.n].transpose()
            y_i = self.results[i, 0]

            theta_feature_product = np.dot(theta_T, x_i)[0, 0]
            hypothesis_value = sigmoid(theta_feature_product)
            cost += (y_i * math.log(hypothesis_value)) + ((1 - y_i) * math.log(1 - hypothesis_value))
        
        cost = -(cost / self.m)
        return cost

    def gradient(self, feature_index):
        if(feature_index > self.n or feature_index < 0):
            raise Exception('Feature index out of range.')
        gradient = 0
        theta_T = self.thetas.transpose()

        for i in range(0, self.m):
            x_i = self.features[i: i + 1, 0: self.n]
            y_i = self.results[i, 0]

            theta_feature_product = np.dot(theta_T, x_i)[0,0]
            hypothesis_value = sigmoid(theta_feature_product)
            gradient += (hypothesis_value - y_i) * self.features[i, feature_index]
        
        gradient = (gradient / self.m)
        return gradient