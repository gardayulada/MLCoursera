import numpy as np
import pandas as pd
from trainModel import theta1, theta2, X_test, y_test
from forwardProp import fwdProp as fw
import crossValTest
from sklearn.preprocessing import PolynomialFeatures

'''
Predict based on trained theta and X_test data
'''
a2 = fw.fwdProp(X_test, theta1)
a2 = PolynomialFeatures(degree=1, include_bias=True).fit_transform(a2)
a3 = fw.fwdProp(a2, theta2)
y_pred_max = np.reshape(np.max(a3, axis=1), (len(a3), 1))
y_pred_test = np.where(a3==y_pred_max,1,0)
y_pred_test = np.argwhere(y_pred_test==1)
y_pred_test = np.reshape(y_pred_test[:,1], (len(y_pred_test),1))

'''
Compare predicted with actual data based on test data
'''
y_act_test = np.argwhere(y_test==1)
y_act_test = np.reshape(y_act_test[:,1], (len(y_act_test),1))
pred_test_df = pd.DataFrame(y_pred_test, columns=['y_pred'])
act_test_df = pd.DataFrame(y_act_test, columns=['y_act'])
compare_test_df = pd.concat([pred_test_df, act_test_df], axis=1)
compare_test_df['check'] = np.where(compare_test_df['y_pred']==compare_test_df['y_act'],1,0)
print('Comparison between Prediction and Actual based on Test Data:')
print(compare_test_df)
acc_test = np.sum(compare_test_df['check']) / compare_test_df['check'].count()
print('Accuracy based on Test Data: {}'.format(acc_test))