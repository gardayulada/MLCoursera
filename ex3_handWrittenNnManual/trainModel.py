'''
This is Model Training using Neural Network
'''

'''
Import the necessary modules
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from mat4py import loadmat
from forwardProp import fwdProp as fw
from backwardProp import backProp as back
from costFunction import costFunction as cf

'''
Import the data
'''
raw_data = loadmat('/Users/garda.asyuhur/Octave/ex3/ex3data1.mat')
df = pd.DataFrame(raw_data)
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
X = PolynomialFeatures(degree=1, include_bias=True).fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=34)
X_crossval, X_test, y_crossval, y_test = train_test_split(X_test, y_test, train_size=0.5, random_state=35)

'''
Initialize theta
'''
eps1 = np.sqrt(6) / np.sqrt(401+801)
eps2 = np.sqrt(6) / np.sqrt(802+10)
theta1 = np.random.uniform(low=-eps1, high=eps1, size=(401,801))
theta2 = np.random.uniform(low=-eps2, high=eps2, size=(802,10))

'''
Iteration to do forward propagation and backward propagation to train the model
'''
iter_li = []
cost_li = []
alpha = 0.01
lambd = 1
i = 1
while i <= 10000:
    # Forward Propagation
    a2 = fw.fwdProp(X_train, theta1)
    a2 = PolynomialFeatures(degree=1, include_bias=True).fit_transform(a2)
    a3 = fw.fwdProp(a2, theta2)
    a3_max = np.reshape(np.max(a3, axis=1), (len(a3),1))
    a3 = np.where(a3==a3_max, 1, 0)
    # Backward Propagation
    gradDescent2 = back.backPropLayer2(theta2, a3, y_train, a2, lambd)
    gradDescent1 = back.backPropLayer1(theta1, theta2, a3, y_train, a2, X_train, lambd)
    theta2 = np.subtract(theta2, np.multiply(alpha, gradDescent2))
    theta1 = np.subtract(theta1, np.multiply(alpha, gradDescent1))
    # Compute cost
    cost = cf.costFunction(X_train, theta1, theta2, y_train, lambd)
    cost_li.extend([np.average(cost)])
    # Increase iteration count
    iter_li.extend([i])
    i += 1

'''
Plot cost function by iteration
'''
plt.figure()
plt.scatter(iter_li, cost_li)
plt.title('Plot cost function by iteration')
plt.xlabel('iteration')
plt.ylabel('cost')
plt.show()

'''
Predict based on trained theta and X_train data
'''
a2 = fw.fwdProp(X_train, theta1)
a2 = PolynomialFeatures(degree=1, include_bias=True).fit_transform(a2)
a3 = fw.fwdProp(a2, theta2)
y_pred_max_train = np.reshape(np.max(a3, axis=1), (len(a3),1))
y_pred_train = np.where(a3==y_pred_max_train,1,0)
y_pred_train = np.argwhere(y_pred_train==1)
y_pred_train = np.reshape(y_pred_train[:,1], (len(y_pred_train),1))

'''
Compare predicted data with actual data based on training data
'''
y_act_max_train = np.reshape(np.max(y_train, axis=1), (len(y_train),1))
y_act_train = np.where(y_train==y_act_max_train,1,0)
y_act_train = np.argwhere(y_act_train==1)
y_act_train = np.reshape(y_act_train[:,1], (len(y_act_train),1))
pred_train_df = pd.DataFrame(y_pred_train, columns=['y_pred'])
act_train_df = pd.DataFrame(y_act_train, columns=['y_act'])
y_train_compare = pd.concat([pred_train_df, act_train_df], axis=1)
y_train_compare['check'] = np.where(y_train_compare['y_pred']==y_train_compare['y_act'],1,0)
print('Comparison prediction vs actual based on training data:')
print(y_train_compare)
train_acc = np.sum(y_train_compare['check']) / y_train_compare['check'].count()
print('Accuracy based on training data: {}'.format(train_acc))