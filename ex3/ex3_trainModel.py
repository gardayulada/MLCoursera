from mat4py import loadmat
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ex3_mapFeature import mapFeature as mf
from ex3_gradientDescent import gradientDescent as gd
from ex3_costFunction import costFunction as cf
import matplotlib.pyplot as plt
from ex3_sigmoidFunction import sigmoidFunction as sf

# Import the data from .mat format and assign to X matrix and y vector.
data = loadmat('/Users/garda.asyuhur/Octave/ex3/ex3data1.mat')
df = pd.DataFrame(data)
X = np.vstack(df['X'])
y_init = np.vstack(df['y'])

# Then need to replicate y vector to be 10-columns matrix as we have 10 classes
i = 1
y = y_init.copy()
y = np.where(y==i,1,0)
i += 1
while i <= 10:
    y_insert = y_init.copy()
    y_insert = np.where(y_insert==i,1,0)
    y = np.concatenate((y, y_insert), axis=1)
    i += 1

# Split the data into training, cross validation and test data
X_train, X_crosstest, y_train, y_crosstest = train_test_split(X, y, train_size=0.6, random_state=34)
X_crossval, X_test, y_crossval, y_test = train_test_split(X_crosstest, y_crosstest, train_size=0.5, random_state=41)

# Include bias and map the features into certain degree of polynomial
degPolyNom = 1
X_train2 = mf.mapFeature(X_train, degPolyNom)
X_crossval2 = mf.mapFeature(X_crossval, degPolyNom)

'''
# Optimize iteration, alpha,and lambda that has the lowest cost cross validation and cost training data
df_compare = pd.DataFrame(columns=['n_iter','alpha_iter','lambd_iter','avg_costtrain','avg_costcrossval'])
n_iter = 2000
n = 10000
while n_iter <= n:
    alpha_iter = 0.001
    alpha = 0.0001
    while alpha_iter >= alpha:
        lambd_iter = 1
        lambd = 1000
        while lambd_iter <= lambd:
            theta = np.zeros((len(X_train2[0,:]), 10))
            i = 1
            while i <= n_iter:
                theta = gd.gradDescentLogReg(X_train2, theta, y_train, alpha_iter, lambd_iter)
                cost_train = cf.costFunction(X_train2, theta, y_train, lambd_iter)
                i += 1
            print(n_iter)
            print(alpha_iter)
            print(lambd_iter)
            cost_crossval = cf.costFunction(X_crossval2, theta, y_crossval, lambd_iter)
            avg_costtrain = np.average(cost_train[0,:])
            avg_costcrossval = np.average(cost_crossval[0,:])
            df_compareint = pd.DataFrame({'n_iter':[n_iter], 'alpha_iter':[alpha_iter], 'lambd_iter':[lambd_iter],
                                          'avg_costtrain':[avg_costtrain], 'avg_costcrossval':[avg_costcrossval]})
            df_compare = pd.concat([df_compare, df_compareint]).reset_index(drop=True)
            lambd_iter *= 10
        alpha_iter -= 0.0001
    n_iter += 1000
df_compare.sort_values(by=['avg_costcrossval','avg_costtrain'], ascending=True, inplace=True)
print(df_compare)
print(df_compare.iloc[0,:])
'''

# Learn again the parameter theta, with the optimized iteration = 10k, alpha = 0.001, and lambda = 1
n = 10000
alpha = 0.001
lambd = 1
theta = np.zeros((len(X_train2[0,:]), 10))
i = 1
iter_li = []
cost_li = []
while i <= n:
    theta = gd.gradDescentLogReg(X_train2, theta, y_train, alpha, lambd)
    cost_train = cf.costFunction(X_train2, theta, y_train, lambd)
    iter_li.extend([i])
    cost_li.extend([np.average(cost_train)])
    i += 1

# Print the final cost based on training data
print('Cost based on training data: {}'.format(cost_train))

# Plot cost vs iteration based on training data
plt.figure(1)
plt.scatter(iter_li, cost_li, marker='o', color='blue')
plt.title('Plot cost vs iteration based on training data')
plt.xlabel('iteration')
plt.ylabel('cost')
plt.show()


# Check accuracy based on training data
## Predict y and then do some transformations prior comparison
y_pred_train = sf.sigmoidFunction(X_train2, theta)
print('Predicted y matrix (1):')
print(y_pred_train[:3,:])
y_pred_max = np.reshape(np.max(y_pred_train, axis=1), (len(y_pred_train),1))
y_pred_train2 = np.where(y_pred_train==y_pred_max,1,0)
print('Predicted y matrix (2):')
print(y_pred_train2[:3,:])
y_pred_train3 = np.argwhere(y_pred_train2==1)
y_pred_train3 = np.reshape(np.add(y_pred_train3[:,1],1), (len(y_pred_train2),1))
print('Predicted y vector:')
print(y_pred_train3)
## Transform y actual prior comparison
y_act_train = np.argwhere(y_train==1)
y_act_train2 = np.reshape(np.add(y_act_train[:,1],1), (len(y_act_train),1))
## Compare y prediction and y actual
df_pred = pd.DataFrame(y_pred_train3, columns=['y_pred'])
df_act = pd.DataFrame(y_act_train2, columns=['y_act'])
df_compare = pd.concat([df_pred, df_act], axis=1)
df_compare['check'] = np.where(df_compare['y_pred']==df_compare['y_act'],1,0)
print('Comparison y_pred vs y_act based on training data:')
print(df_compare.head())
## Compute Accuracy
accuracy = sum(df_compare['check']) / df_compare['check'].count()
print('Accuracy based on training data: {}'.format(accuracy))
