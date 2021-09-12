from ex2_2_trainModel import theta, X_test, y_test, degPolyNom, lambd
from ex2_2_mapFeature import mapFeature as mf
from ex2_2_sigmoidFunction import sigmoidFunction as sf
from ex2_2_costFunction import costFunction as cf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ex2_2_crossValTest

# Predict y based on test data
X_test2 = mf.mapFeaturePolyNomial(X_test, degPolyNom)
y_pred_test = sf.sigmoidFunction(X_test2, theta)
print(y_pred_test)

# Print cost function based on test data
cost = cf.costFunctionLogRegress(X_test2, theta, y_test, lambd)
print('Cost based on test data: {}'.format(cost))

# Check the performance based on the test data
y_pred_test = sf.sigmoidFunction(X_test2, theta)
y_pred_test2 = np.where(y_pred_test>=0.5,1,0)
df_predict = pd.DataFrame(y_pred_test2, columns=['y_predict_test'])
df_act = pd.DataFrame(y_test, columns=['y_act_test'])
df_metrics_test = pd.concat([df_predict, df_act], axis=1)
df_metrics_test['check'] = np.where(df_metrics_test['y_predict_test']==df_metrics_test['y_act_test'],1,0)
print(df_metrics_test)
accuracy = sum(df_metrics_test['check']) / df_metrics_test['check'].count()
df_precision_test = df_metrics_test.loc[df_metrics_test['y_predict_test']==1].reset_index(drop=True)
df_recall_test = df_metrics_test.loc[df_metrics_test['y_act_test']==1].reset_index(drop=True)
df_true_positive = df_metrics_test.loc[(df_metrics_test['check']==1) & (df_metrics_test['y_predict_test']==1)].reset_index(drop=True)
precision = df_true_positive['check'].count() / df_precision_test['check'].count()
recall = df_true_positive['check'].count() / df_recall_test['check'].count()
print('Accuracy based on test data: {}'.format(accuracy))
print('Precision based on test data: {}'.format(precision))
print('Recall based on test data: {}'.format(recall))

# Plot the test data based on prediction
df_xtest = pd.DataFrame(X_test, columns=['test1','test2'])
df_metrics_test2 = pd.concat([df_xtest, df_metrics_test], axis=1)
df_pred_true = df_metrics_test2.loc[df_metrics_test2['y_predict_test']==1].reset_index(drop=True)
df_pred_false = df_metrics_test2.loc[df_metrics_test2['y_predict_test']==0].reset_index(drop=True)
plt.figure()
plt.scatter(df_pred_true['test1'], df_pred_true['test2'], marker='o', color='blue')
plt.scatter(df_pred_false['test1'], df_pred_false['test2'], marker='x', color='green')
plt.title('Predicted Pass Fail based on Test1 and Test2 of Test Data')
plt.xlabel('test1')
plt.ylabel('test2')
plt.show()