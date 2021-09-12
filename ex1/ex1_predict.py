from ex1_trainModel import theta
import numpy as np

x1 = np.array([1,3.5])
predict1 = x1 * theta
print('Predicted y for x = 3.5 is {}'.format(float(predict1)))

x2 = np.array([1,7])
predict2 = x2 * theta
print('Predicted y for x = 7 is {}'.format(float(predict2)))

