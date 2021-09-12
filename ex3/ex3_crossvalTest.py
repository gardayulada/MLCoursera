from ex3_trainModel import theta, X_crossval, y_crossval, lambd
import numpy as np
from ex3_sigmoidFunction import sigmoidFunction as sf
from ex3_mapFeature import mapFeature as mf
from ex3_costFunction import costFunction as cf
import pandas as pd

# Include bias and map the feature based on degPolyNom
degPolyNom = 1
X_crossval2 = mf.mapFeature(X_crossval, degPolyNom)

# Predict y based on cross validation data
y_pred_crossval = sf.sigmoidFunction(X_crossval2, theta)

# Cost function based on cross validation data
cost_crossval = cf.costFunction(X_crossval2, theta, y_crossval, lambd)
print('Cost based on cross validation data: {}'.format(cost_crossval))

# Check accuracy based on cross validation data
## Transform y prediction
print('y prediction matrix 1')
print(y_pred_crossval[:3,:])
y_pred_crossmax = np.reshape(np.max(y_pred_crossval, axis=1), (len(y_pred_crossval),1))
y_pred_crossval2 = np.where(y_pred_crossval==y_pred_crossmax,1,0)
print('y prediction matrix 2:')
print(y_pred_crossval2[:3,:])
y_pred_crossval3 = np.argwhere(y_pred_crossval2==1)
y_pred_crossval3 = np.reshape(np.add(y_pred_crossval3[:,1], 1), (len(y_pred_crossval3),1))
print('y prediction vector')
print(y_pred_crossval3[:3])
## Transform y actual
y_act_crossval = np.argwhere(y_crossval==1)
y_act_crossval2 = np.reshape(np.add(y_act_crossval[:,1],1), (len(y_act_crossval),1))
## Compare y prediction with actual
df_pred = pd.DataFrame(y_pred_crossval3, columns=['y_pred'])
df_act = pd.DataFrame(y_act_crossval2, columns=['y_act'])
df_compare = pd.concat([df_pred, df_act], axis=1)
df_compare['check'] = np.where(df_compare['y_pred']==df_compare['y_act'],1,0)
print('Comparison y_pred vs y_act based on cross validation data:')
print(df_compare.head())
## Compute accuracy
accuracy = sum(df_compare['check']) / df_compare['check'].count()
print('Accuracy based on cross validation data: {}'.format(accuracy))