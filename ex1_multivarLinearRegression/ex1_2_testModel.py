from ex1_2_costfunction import costFunction as cf
from ex1_2_trainModel import theta, X_test, y_test
import ex1_2_crossValTest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Predict house price based on test data
y_pred_test = np.dot(X_test, theta)

# Print cost function based on test data
cost = cf.costFunction(X_test, theta, y_test)
print('Cost function based on test data: {}'.format(cost))

# Plot house price prediction vs actual based on test data
plt.figure()
plt3d = plt.axes(projection='3d')
plt3d.scatter3D(X_test[:,1], X_test[:,2], y_test, color='green')
plt3d.scatter3D(X_test[:,1], X_test[:,2], y_pred_test, color='blue', marker='x')
plt3d.set_xlabel('size')
plt3d.set_ylabel('n_bedroom')
plt3d.set_zlabel('House price')
plt.title('House price prediction vs actual based on test data')
plt.show()