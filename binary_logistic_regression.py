import numpy as np
import matplotlib.pyplot as plt
import logisticregression as lr

data_matrix = np.loadtxt('./exdata1.txt')

admitted = np.empty((0, 3))
not_admitted = np.empty((0,3))

X = data_matrix[0: data_matrix.shape[0], 0: 2]
y = data_matrix[0: data_matrix.shape[0], 2: 3]

for row in range(0, data_matrix.shape[0]):
    if(data_matrix[row, 2] == 1):
        admitted = np.vstack((admitted, data_matrix[row, :]))
    else:
        not_admitted = np.vstack((not_admitted, data_matrix[row, :]))


# Plot w/o decision boundary

no_decision_boundary_graph = plt.figure()

admitted_x1 = admitted[:, 0]
admitted_x2 = admitted[:, 1]

not_admitted_x1 = not_admitted[:, 0]
not_admitted_x2 = not_admitted[:, 1]

admit = plt.scatter(admitted_x1, admitted_x2, c='green', label='Admitted')
not_admit = plt.scatter(not_admitted_x1, not_admitted_x2, c='purple', label='Not Admitted')

plt.legend((admit, not_admit), ("Admitted", "Not Admitted"))
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")


# Logistic Regression
feature_zero = np.ones((data_matrix.shape[0], 1))
features = np.hstack((feature_zero, data_matrix[: , 0:2]))
results = data_matrix[: , 2:3]

init_thetas = np.zeros(3)
optimized_thetas = lr.optimize(init_thetas, features, results)

# # Plot for decision boundary
db_x = np.array([np.min(X[:, 0]) - 2, np.max(X[:, 1]) + 2])
db_y = (-1.0 / optimized_thetas[2]) * (optimized_thetas[1] * db_x + optimized_thetas[0  ])

dec_bound = plt.plot(db_x, db_y)

# Show all graphs
plt.show()

