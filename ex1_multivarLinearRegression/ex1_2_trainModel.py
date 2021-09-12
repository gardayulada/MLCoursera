import pandas as pd
import numpy as np
from ex1_2_featureNormalization import featureNormalize as norm
from sklearn.model_selection import train_test_split
from ex1_2_costfunction import costFunction as cf
from ex1_2_linearRegression import linearRegression as linReg
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

df = pd.read_csv('/Users/garda.asyuhur/Octave/ex1/ex1data2.txt', delimiter=',',
                 names=['size','n_bedroom','price'])

# Adding new columns 'x0' with values all '1'
df['x0'] = 1

# Ensure all the parameters and target parameter to be in float datatype
df = df.astype({'x0':'float', 'size':'float', 'n_bedroom':'float'})

# Assign the features into matrix X, and outcome into vector y
X = df[['x0','size','n_bedroom']].to_numpy()
y = df[['price']].to_numpy()

# Split the data into training, cross validation, and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=45)
X_crossval, X_test, y_crossval, y_test = train_test_split(X_test, y_test, train_size=0.5, random_state=30)

# Do Feature Scaling based on training set's average and standard deviation
X_train, avg_Xtrain, stdev_Xtrain = norm.mean_std_normalize(X_train)
X_crossval = (X_crossval - avg_Xtrain) / stdev_Xtrain
X_test  = (X_test - avg_Xtrain) / stdev_Xtrain

# Do Gradient Descent to optimize iteration number and alpha
df_n_alpha = pd.DataFrame(columns=['n','alpha','cost_func_train','cost_func_crossval'])
n = 10000
n_iter = 1000
while n_iter <= n:
    alpha = 0.0001
    alpha_iter = 0.001
    while alpha_iter >= alpha:
        i = 1
        theta = np.array([[0.0,0.0,0.0]])
        theta = np.transpose(theta)
        while i <= n_iter:
            cost = cf.costFunction(X_train, theta, y_train)
            theta = linReg.linearRegression(X_train, theta, y_train, alpha)
            i += 1
        cost_train = cf.costFunction(X_train, theta, y_train)
        cost_crossval = cf.costFunction(X_crossval, theta, y_crossval)
        df_n_alpha_inp = pd.DataFrame({'n':[n_iter], 'alpha':[alpha_iter], 'cost_func_train':[cost_train],
                                       'cost_func_crossval':[cost_crossval]})
        df_n_alpha = pd.concat([df_n_alpha, df_n_alpha_inp]).reset_index(drop=True)
        alpha_iter -= 0.0001
    n_iter += 1000

# Obtain the optimum n and alpha which cost function based on cross validation data is the lowest
df_n_alpha.sort_values(by=['cost_func_crossval','cost_func_train'], ascending=True, inplace=True)

# Train the model again using the optimized n=10,000 and alpha=0.0005 to get the theta
i = 1
n = 10000
theta = np.array([[0.0,0.0,0.0]])
theta = np.transpose(theta)
alpha = 0.0005
cost_li = []
n_li = []
while i <= n:
    cost = cf.costFunction(X_train, theta, y_train)
    theta = linReg.linearRegression(X_train, theta, y_train, alpha)
    n_li.extend([i])
    cost_li.extend([cost])
    i += 1

# Check if the cost function decreases with number of iteration
plt.figure()
plt.scatter(n_li, cost_li)
plt.xlabel('n_iteration')
plt.ylabel('cost function')
plt.title('Cost Function vs n_iteration based on training model')
plt.show()

# Print final cost function based on training data
cost = cf.costFunction(X_train, theta, y_train)
print('Cost function based on training data: {}'.format(cost))

# Plot the figure of predicted value and actual value based on training data
y_predict_train = np.dot(X_train, theta)
plt3d = plt.axes(projection='3d')
plt3d.scatter3D(X_train[:,1], X_train[:,2], y_train, color='green')
plt3d.scatter3D(X_train[:,1], X_train[:,2], y_predict_train, color='blue', marker='x')
plt3d.set_xlabel('size')
plt3d.set_ylabel('n_bedroom')
plt3d.set_zlabel('house_price')
plt.title('House price predicted vs actual based on training data')
plt.show()

