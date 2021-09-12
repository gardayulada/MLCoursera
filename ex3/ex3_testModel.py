from ex3_trainModel import theta, X_test, y_test, lambd
from ex3_costFunction import costFunction as cf
from ex3_sigmoidFunction import sigmoidFunction as sf
from ex3_mapFeature import mapFeature as mf
import numpy as np
import pandas as pd

# Include bias and map features based on degPolyNom
degPolyNom = 1
X_test2 = mf.mapFeature(X_test, degPolyNom)

# Predict y based on learned theta and cross validation data
y_pred_test = sf.sigmoidFunction(X_test2, theta)

# Compute cost based on test data
cost_test = cf.costFunction(X_test2, theta, y_test, lambd)
print('Cost based on test data: {}'.format(cost_test))

# Check accuracy based on test data
## Transform y prediction
print('y prediction matrix 1:')
print(y_pred_test[:3,:])
y_pred_testmax = np.reshape(np.max(y_pred_test, axis=1), (len(y_pred_test),1))
y_pred_test2 = np.where(y_pred_test==y_pred_testmax,1,0)
print('y prediction matrix 2:')
print(y_pred_test2[:3,:])
y_pred_test3 = np.argwhere(y_pred_test2==1)
y_pred_test3 = np.reshape(np.add(y_pred_test3[:,1], 1), (len(y_pred_test2),1))
print('y prediction vector:')
print(y_pred_test3[:3])
## Transform y actual
y_act_test = np.argwhere(y_test==1)
y_act_test2 = np.reshape(np.add(y_act_test[:,1], 1), (len(y_test),1))
print('y act vector:')
print(y_act_test2[:3])
## Compare y prediction and actual based on test data
df_pred = pd.DataFrame(y_pred_test3, columns=['y_pred'])
df_act = pd.DataFrame(y_act_test2, columns=['y_act'])
df_compare = pd.concat([df_pred, df_act], axis=1)
df_compare['check'] = np.where(df_compare['y_pred']==df_compare['y_act'],1,0)
print('Compare y prediction and actual based on test data:')
print(df_compare.head())
## Compute accuracy based on test data
accuracy = sum(df_compare['check']) / df_compare['check'].count()
print('Accuracy based on test data: {}'.format(accuracy))