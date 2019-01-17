import numpy as np
import matplotlib.pyplot as plt
import logisticregression as lr

data_matrix = np.loadtxt('./exdata2.txt')

X = data_matrix[:, 0:2]
y = data_matrix[:, 2]

y_0_training_set = np.empty((0, 3))
y_1_training_set = np.empty((0, 3))

for i in range(0, data_matrix.shape[0]):
    if(data_matrix[i, 2] == 1):
        y_1_training_set = np.vstack((y_1_training_set, data_matrix[i, :]))
    else:
        y_0_training_set = np.vstack((y_0_training_set, data_matrix[i, :]))

# Plot w/o decision boundary

fig = plt.figure()

y_0_x_0 = y_0_training_set[:, 0]
y_0_x_1 = y_0_training_set[:, 1]

y_1_x_0 = y_1_training_set[:, 0]
y_1_x_1 = y_1_training_set[:, 1]

y_0_plot = plt.scatter(y_0_x_0, y_0_x_1, c='orange')
y_1_plot = plt.scatter(y_1_x_0, y_1_x_1, c='black')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend((y_0_plot, y_1_plot), ('No', 'Yes'))

f = np.array([1,2,3,4])

# Logistic Regression

feature_zero = np.ones((data_matrix.shape[0], 1))
features = np.hstack((feature_zero, data_matrix[:, 0:2]))
results = data_matrix[: , 2:3]

init_theta = np.zeros(3)
optimized_thetas = lr.optimize_reg(init_theta, features, results)

# # Plot for decision boundary

