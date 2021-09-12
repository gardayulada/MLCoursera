from ex2_2_trainModel import theta, X_crossval, y_crossval, degPolyNom, lambd
from ex2_2_mapFeature import mapFeature as mf
from ex2_2_sigmoidFunction import sigmoidFunction as sf
from ex2_2_costFunction import costFunction as cf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Predict y based on cross validation data
X_crossval2 = mf.mapFeaturePolyNomial(X_crossval, degPolyNom)
y_pred_crossval = sf.sigmoidFunction(X_crossval2, theta)
print(y_pred_crossval)

# Print cost function based on cross validation data
cost = cf.costFunctionLogRegress(X_crossval2, theta, y_crossval, lambd)
print('Cost based on cross validation data: {}'.format(cost))

# Check the performance based on the cross validation data
y_pred_crossval = sf.sigmoidFunction(X_crossval2, theta)
y_pred_crossval2 = np.where(y_pred_crossval>=0.5,1,0)
df_predict = pd.DataFrame(y_pred_crossval2, columns=['y_predict_crossval'])
df_act = pd.DataFrame(y_crossval, columns=['y_act_crossval'])
df_metrics_crossval = pd.concat([df_predict, df_act], axis=1)
df_metrics_crossval['check'] = np.where(df_metrics_crossval['y_predict_crossval']==df_metrics_crossval['y_act_crossval'],1,0)
print(df_metrics_crossval)
accuracy = sum(df_metrics_crossval['check']) / df_metrics_crossval['check'].count()
df_precision_crossval = df_metrics_crossval.loc[df_metrics_crossval['y_predict_crossval']==1].reset_index(drop=True)
df_recall_crossval = df_metrics_crossval.loc[df_metrics_crossval['y_act_crossval']==1].reset_index(drop=True)
df_true_positive = df_metrics_crossval.loc[(df_metrics_crossval['check']==1) & (df_metrics_crossval['y_predict_crossval']==1)].reset_index(drop=True)
precision = df_true_positive['check'].count() / df_precision_crossval['check'].count()
recall = df_true_positive['check'].count() / df_recall_crossval['check'].count()
print('Accuracy based on cross validation data: {}'.format(accuracy))
print('Precision based on cross validation data: {}'.format(precision))
print('Recall based on cross validation data: {}'.format(recall))

# Plot the cross validation data based on prediction
df_xcrossval = pd.DataFrame(X_crossval, columns=['test1','test2'])
df_metrics_crossval2 = pd.concat([df_xcrossval, df_metrics_crossval], axis=1)
df_pred_true = df_metrics_crossval2.loc[df_metrics_crossval2['y_predict_crossval']==1].reset_index(drop=True)
df_pred_false = df_metrics_crossval2.loc[df_metrics_crossval2['y_predict_crossval']==0].reset_index(drop=True)
plt.figure()
plt.scatter(df_pred_true['test1'], df_pred_true['test2'], marker='o', color='blue')
plt.scatter(df_pred_false['test1'], df_pred_false['test2'], marker='x', color='green')
plt.title('Predicted Pass Fail based on Test1 and Test2 Cross Validation Data')
plt.xlabel('test1')
plt.ylabel('test2')
plt.show()