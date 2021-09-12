import numpy as np
from ex2_trainModel import theta, X_crossval, y_crossval
from ex2_costfunction import costFunction as cf
from ex2_sigmoidFunction import sigmoidFunction as sf
import pandas as pd

# Predict y_pred_crossval
y_pred_crossval = sf.sigmoidFunction(X_crossval, theta)
y_pred_crossval = np.where(y_pred_crossval>=0.5,1,0)

# Check Precision, Recall, F1 Score, and Accuracy
df_pred = pd.DataFrame(y_pred_crossval, columns=['predicted_val'])
df_act = pd.DataFrame(y_crossval, columns=['actual_val'])
df_act_pred = pd.concat([df_pred, df_act], axis=1)
df_act_pred['check'] = np.where(df_act_pred['predicted_val']==df_act_pred['actual_val'],1,0)
df_act_pred['check2'] = np.where((df_act_pred['predicted_val']==1)&(df_act_pred['predicted_val']==df_act_pred['actual_val']), 'True Positive',
                                 np.where((df_act_pred['predicted_val']==0)&(df_act_pred['predicted_val']==df_act_pred['actual_val']), 'True Negative',
                                          np.where((df_act_pred['predicted_val']==1)&(df_act_pred['predicted_val']!=df_act_pred['actual_val']), 'False Positive',
                                                   np.where((df_act_pred['predicted_val']==0)&(df_act_pred['predicted_val']!=df_act_pred['actual_val']), 'False Negative',''))))
accuracy = sum(df_act_pred['check']) / df_act_pred['check'].count()
precision = sum(np.where(df_act_pred['check2']=='True Positive',1,0)) / (sum(np.where(df_act_pred['check2']=='True Positive',1,0)) + sum(np.where(df_act_pred['check2']=='False Positive',1,0)))
recall = sum(np.where(df_act_pred['check2']=='True Positive',1,0)) / (sum(np.where(df_act_pred['check2']=='True Positive',1,0)) + sum(np.where(df_act_pred['check2']=='False Negative',1,0)))
f1score = 2 * precision * recall / (precision + recall)
print(df_act_pred)
print('Accuracy based on cross validation data: {}'.format(accuracy))
print('Precision based on cross validation data: {}'.format(precision))
print('Recall based on cross validation data: {}'.format(recall))
print('F1 Score based on cross validation data: {}'.format(f1score))

# Print cost function based on cross validation data
cost = cf.costFunction(X_crossval, theta, y_crossval)
print('Cost function based cross validation data: {}'.format(cost))