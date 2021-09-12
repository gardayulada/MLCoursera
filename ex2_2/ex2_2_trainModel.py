import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ex2_2_mapFeature import mapFeature as mf
import numpy as np
from ex2_2_gradientDescent import gradientDescent as gd
from ex2_2_costFunction import costFunction as cf
from ex2_2_sigmoidFunction import sigmoidFunction as sf

# Import the data to pandas dataframe
df = pd.read_csv('/Users/garda.asyuhur/Octave/ex2/ex2data2.txt', delimiter=',',
                 names=['test1','test2','pass_fail'])

# Visualize the data to see the likely boundary decision, hence deciding whether linear or polynomial will fit
df_true = df.loc[df['pass_fail']==1].reset_index(drop=True)
df_false = df.loc[df['pass_fail']==0].reset_index(drop=True)
plt.figure()
plt.scatter(df_true['test1'], df_true['test2'], marker='o', color='blue')
plt.scatter(df_false['test1'], df_false['test2'], marker='x', color='green')
plt.xlabel('test1')
plt.ylabel('test2')
plt.title('Pass Fail based on Test1 and Test2 Score')
plt.show()

# Add on x0 = 1 and split the data into X matrix and y vector
X = df[['test1','test2']].to_numpy()
y = df[['pass_fail']].to_numpy()

# Split the training, cross validation, and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=33)
X_crossval, X_test, y_crossval, y_test = train_test_split(X_test, y_test, train_size=0.5, random_state=50)

'''
# To get the optimized #features, #iteration, and alpha
df_compare = pd.DataFrame(columns=['deg_polynom','n_iterations','alpha','cost_train','cost_crossval'])
degPolyNom = 1
maxDegPolyNom = 7
lambd = 1000
while degPolyNom <= maxDegPolyNom:
    X_train2 = mf.mapFeaturePolyNomial(X_train, degPolyNom)
    X_crossval2 = mf.mapFeaturePolyNomial(X_crossval, degPolyNom)
    n = 10000
    n_iter = 1000
    while n_iter <= n:
        min_alpha = 0.0001
        alpha_iter = 0.001
        while alpha_iter >= min_alpha:
            theta = np.zeros((len(X_train2[0,:]),1))
            i = 1
            while i <= n_iter:
                theta = gd.gradDescentLogReg(X_train2, theta, y_train, alpha_iter, lambd)
                cost_train = cf.costFunctionLogRegress(X_train2, theta, y_train, lambd)
                i += 1
            print(degPolyNom)
            print(n_iter)
            print(alpha_iter)
            cost_crossval = cf.costFunctionLogRegress(X_crossval2, theta, y_crossval, lambd)
            df_compare_int = pd.DataFrame({'deg_polynom':[degPolyNom], 'n_iterations':[n_iter],
                                           'alpha':[alpha_iter], 'cost_train':[cost_train], 'cost_crossval':[cost_crossval]})
            df_compare = pd.concat([df_compare, df_compare_int]).reset_index(drop=True)
            alpha_iter -= 0.0001
        n_iter += 1000
    degPolyNom += 1
df_compare.sort_values(by=['cost_crossval','cost_train'], ascending=True, inplace=True)
print(df_compare)
'''

# From the optimization above, we train the model again using the optimized deg_polynom = 7, n_iterations = 10k, and alpha = 0.001
degPolyNom = 6
n = 100000
alpha = 0.01
lambd = 1
X_train2 = mf.mapFeaturePolyNomial(X_train, degPolyNom)
theta = np.zeros((len(X_train2[0,:]), 1))
iter_li = []
cost_func_li = []
i = 1
while i <= n:
    theta = gd.gradDescentLogReg(X_train2, theta, y_train, alpha, lambd)
    cost_train = cf.costFunctionLogRegress(X_train2, theta, y_train, lambd)
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
y_predict_train = sf.sigmoidFunction(X_train2, theta)
y_predict_train2 = np.where(y_predict_train>=0.5,1,0)
df_predict = pd.DataFrame(y_predict_train2, columns=['y_predict_train'])
df_act = pd.DataFrame(y_train, columns=['y_act_train'])
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
df_xtrain = pd.DataFrame(X_train, columns=['test1','test2'])
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

# Plot the boundary condition based on the learning theta
