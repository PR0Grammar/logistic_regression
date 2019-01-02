import numpy as np
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

data_matrix = np.loadtxt('./exdata1.txt')

admitted = np.empty((0, 3))
not_admitted = np.empty((0,3))

for row in range(0, data_matrix.shape[0]):
    if(data_matrix[row, 2] == 1):
        admitted = np.vstack((admitted, data_matrix[row, :]))
    else:
        not_admitted = np.vstack((not_admitted, data_matrix[row, :]))


# Graph without decision boundary

no_decision_boundary_graph = plt.figure()

admitted_x1 = admitted[:, 0]
admitted_x2 = admitted[:, 1]

not_admitted_x1 = not_admitted[:, 0]
not_admitted_x2 = not_admitted[:, 1]

plt.scatter(admitted_x1, admitted_x2, c='green', label='Admitted')
plt.scatter(not_admitted_x1, not_admitted_x2, c='purple', label='Not Admitted')

plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")


# Logistic Regression

features = data_matrix[0: data_matrix.shape[0], 0:2]
results = data_matrix[0: data_matrix.shape[0], 2:3]

log_regression = LogisticRegression(features, results)

print(log_regression.compute_cost())


# Show all graphs
plt.show()

