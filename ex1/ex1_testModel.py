from ex1_trainModel import theta, X_test, y_test
import ex1_crossValTest
from ex1_costfunction import costFunction as cf
import matplotlib.pyplot as plt
import numpy as np

# Predict profit based on test dataset
y_predict = X_test * theta

# Print cost function based on test dataset
cost = cf.costFunction(X_test, theta, y_test)
print('Final cost function based on test data: {}'.format(cost))

# Plot y_pred and y_act based against test dataset
x_ax = X_test[:,1]
plt.figure()
plt.scatter(np.array(x_ax), np.array(y_test), marker='x', label='y_act_test')
plt.scatter(np.array(x_ax), np.array(y_predict), marker='o', label='y_pred_test')
plt.show()