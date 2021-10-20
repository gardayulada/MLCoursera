import numpy as np
import pandas as pd
from trainModel import theta1, theta2, X_crossval, y_crossval
from forwardProp import fwdProp as fw
from sklearn.preprocessing import PolynomialFeatures

'''
Predict based on trained theta and X_crossval data
'''
a2 = fw.fwdProp(X_crossval, theta1)
a2 = PolynomialFeatures(degree=1, include_bias=True).fit_transform(a2)
a3 = fw.fwdProp(a2, theta2)
y_pred_max = np.reshape(np.max(a3, axis=1), (len(a3),1))
y_pred_crossval = np.where(a3==y_pred_max,1,0)
y_pred_crossval = np.argwhere(y_pred_crossval==1)
y_pred_crossval = np.reshape(y_pred_crossval[:,1], (len(y_pred_crossval),1))

'''
Compare predicted data with actual data based on cross validation data
'''
y_act_crossval = np.argwhere(y_crossval==1)
y_act_crossval = np.reshape(y_crossval[:,1], (len(y_act_crossval),1))
pred_crossval_df = pd.DataFrame(y_pred_crossval, columns=['y_pred'])
act_crossval_df = pd.DataFrame(y_act_crossval, columns=['y_act'])
crossval_compare = pd.concat([pred_crossval_df, act_crossval_df], axis=1)
crossval_compare['check'] = np.where(crossval_compare['y_pred']==crossval_compare['y_act'], 1, 0)
print('Comparison prediction vs actual based on cross validation data:')
print(crossval_compare)
crossval_acc = np.sum(crossval_compare['check']) / crossval_compare['check'].count()
print('Accuracy based on cross validation data: {}'.format(crossval_acc))