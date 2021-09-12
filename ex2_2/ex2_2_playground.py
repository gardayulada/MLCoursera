import pandas as pd
import numpy as np
from ex2_2_mapFeature import mapFeature as mf
from ex2_2_gradientDescent import gradientDescent as gd
from ex2_2_costFunction import costFunction as cf
from ex2_2_sigmoidFunction import sigmoidFunction as sf
import matplotlib.pyplot as plt


# Import the data to pandas dataframe
df = pd.read_csv('/Users/garda.asyuhur/Octave/ex2/ex2data2.txt', delimiter=',',
                 names=['test1','test2','pass_fail'])

X = df[['test1','test2']].to_numpy()
y = df[['pass_fail']].to_numpy()

# From the optimization above, we train the model again using the optimized deg_polynom = 7, n_iterations = 10k, and alpha = 0.001
degPolyNom = 6
n = 100000
alpha = 0.001
lambd = 1
X2 = mf.mapFeaturePolyNomial(X, degPolyNom)
theta = np.zeros((len(X2[0,:]), 1))
iter_li = []
cost_func_li = []
i = 1
while i <= n:
    theta = gd.gradDescentLogReg(X2, theta, y, alpha, lambd)
    cost_train = cf.costFunctionLogRegress(X2, theta, y, lambd)
    iter_li.extend([i])
    cost_func_li.extend([cost_train])
    i += 1
print('Cost based on training data: {}'.format(cost_train))

# Plot the cost function as function of iteration
plt.figure()
plt.scatter(iter_li, cost_func_li, marker='x', color='blue')
plt.title('Plot of cost function vs iteration')
plt.xlabel('iteration')
plt.ylabel('cost_function')
plt.show()

# Check the performance based on the training data
y_predict_train = sf.sigmoidFunction(X2, theta)
print(y_predict_train)
y_predict_train2 = np.where(y_predict_train>=0.5,1,0)
df_predict = pd.DataFrame(y_predict_train2, columns=['y_predict_train'])
df_act = pd.DataFrame(y, columns=['y_act_train'])
df_metrics_train = pd.concat([df_predict, df_act], axis=1)
df_metrics_train['check'] = np.where(df_metrics_train['y_predict_train']==df_metrics_train['y_act_train'],1,0)
accuracy = sum(df_metrics_train['check']) / df_metrics_train['check'].count()
df_precision_train = df_metrics_train.loc[df_metrics_train['y_predict_train']==1].reset_index(drop=True)
df_recall_train = df_metrics_train.loc[df_metrics_train['y_act_train']==1].reset_index(drop=True)
df_true_positive = df_metrics_train.loc[(df_metrics_train['check']==1) & (df_metrics_train['y_predict_train']==1)].reset_index(drop=True)
precision = df_true_positive['check'].count() / df_precision_train['check'].count()
recall = df_true_positive['check'].count() / df_recall_train['check'].count()
print('Accuracy based on training data: {}'.format(accuracy))
print('Precision based on training data: {}'.format(precision))
print('Recall based on training data: {}'.format(recall))

# Plot the training data based on prediction
df_xtrain = pd.DataFrame(X, columns=['test1','test2'])
df_metrics_train2 = pd.concat([df_xtrain, df_metrics_train], axis=1)
df_pred_true = df_metrics_train2.loc[df_metrics_train2['y_predict_train']==1].reset_index(drop=True)
df_pred_false = df_metrics_train2.loc[df_metrics_train2['y_predict_train']==0].reset_index(drop=True)
plt.figure()
plt.scatter(df_pred_true['test1'], df_pred_true['test2'], marker='o', color='blue')
plt.scatter(df_pred_false['test1'], df_pred_false['test2'], marker='x', color='green')
plt.title('Predicted Pass Fail based on Test1 and Test2 Training Data')
plt.xlabel('test1')
plt.ylabel('test2')
plt.show()


