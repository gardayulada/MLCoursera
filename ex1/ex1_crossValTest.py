from ex1_trainModel import theta, X_crossval, y_crossval
import matplotlib.pyplot as plt
import numpy as np
from ex1_costfunction import costFunction as cf

# Predict profit based on cross validation dataset
y_predict = X_crossval * theta

# Print cost function based on cross validaton dataset
cost = cf.costFunction(X_crossval, theta, y_crossval)
print('Final cost function based on cross validation data: {}'.format(cost))

# Plot the y_act and y_predict based on cross validation dataset
x_ax = X_crossval[:, 1]
plt.figure()
plt.scatter(np.array(x_ax), np.array(y_crossval), marker='x', label='y_act_crossval')
plt.scatter(np.array(x_ax), np.array(y_predict), marker='o', label='y_pred_crossval')
plt.show()