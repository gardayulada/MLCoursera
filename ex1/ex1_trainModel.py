import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ex1_costfunction as cf
import ex1_linearRegression as linReg
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/garda.asyuhur/Octave/ex1/ex1data1.txt', delimiter=',',
                 names=['city_pop','profit_per_truck'])

# Put data into matrix
x_data = [[1,i] for i in df['city_pop']]
X = np.matrix(x_data)
y = np.array(df['profit_per_truck'])
y = np.vstack(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=43)
X_crossval, X_test, y_crossval, y_test = train_test_split(X_test, y_test, train_size=0.5, random_state=35)

# Do Gradient Descent while learning which n and alpha provides the lowest cost function
n = 10000
n_iter = 1000
df_main = pd.DataFrame(columns=['n','alpha','cost_function_train','cost_function_crossval'])
while n_iter <= n:
    alpha = 0.0001
    alpha_iter = 0.001
    while alpha_iter >= alpha:
        theta = np.array([0,0])
        theta = np.vstack(theta)
        i = 1
        while i <= n_iter:
            cost = cf.costFunction.costFunction(X_train, theta, y_train)
            theta = linReg.linearRegression.gradientDescent(X_train, theta, y_train, alpha_iter)
            i += 1
        cost_crossval = cf.costFunction.costFunction(X_crossval, theta, y_crossval)
        df = pd.DataFrame({'n':[n_iter], 'alpha':[alpha_iter], 'cost_function_train':[cost], 'cost_function_crossval':[cost_crossval]})
        df_main = pd.concat([df_main, df]).reset_index(drop=True)
        alpha_iter -= 0.0001
    n_iter += 1000

df_main.sort_values(by=['cost_function_crossval','cost_function_train'], ascending=True, inplace=True)
print(df_main)


# From above training, we know the optimum n and alpha giving the lowest cost function are 10,000 and 0.0002. Now we retrain the model again using these n and alpha
n = 10000
alpha = 0.0002
i = 1
theta = np.array([0,0])
theta = np.vstack(theta)
iteration_axis = []
cost_trend_axis = []
while i <= n:
    cost = cf.costFunction.costFunction(X_train, theta, y_train)
    theta = linReg.linearRegression.gradientDescent(X_train, theta, y_train, alpha)
    iteration_axis.extend([i])
    cost_trend_axis.extend([cost])
    i += 1

# Plot the cost function with iteration_axis
plt.figure()
plt.scatter(iteration_axis, cost_trend_axis)
plt.xlabel('iteration_axis')
plt.ylabel('cost_function')
plt.show()

# Print final theta
print('Final theta:')
print(theta)

# Print final cost function
print('Final cost function based on training data: {}'.format(cost))

# Plot the predicted value vs y_train based on X_train
y_predict = X_train * theta
x_ax = X_train[:, 1]
plt.scatter(np.array(x_ax), y_train, marker='x', label='y_act')
plt.scatter(np.array(x_ax), np.array(y_predict), marker='o', label='y_predict')
plt.show()