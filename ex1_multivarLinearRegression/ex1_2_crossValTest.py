from ex1_2_trainModel import theta, X_crossval, y_crossval
from ex1_2_costfunction import costFunction as cf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Predict house price based on cross validation dataset
y_pred_crossval = np.dot(X_crossval, theta)

# Print Cost Function based on cross validation dataset
cost = cf.costFunction(X_crossval, theta, y_crossval)
print('Cost function based on cross validation data: {}'.format(cost))

# Plot House price predicted vs actual based on cross validation dataset
plt3d = plt.axes(projection='3d')
plt3d.scatter3D(X_crossval[:,1], X_crossval[:,2], y_crossval, color='green')
plt3d.scatter3D(X_crossval[:,1], X_crossval[:,2], y_pred_crossval, color='blue', marker='x')
plt3d.set_xlabel('size')
plt3d.set_ylabel('n_bedroom')
plt3d.set_zlabel('house price')
plt.title('House prices prediction vs actual based on cross validation data')
plt.show()


