import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ex2_sigmoidFunction import sigmoidFunction as sf
from ex2_costfunction import costFunction as cf
from ex2_logisticRegression import logisticRegression as logisticRegression

# Import the data
df = pd.read_csv('/Users/garda.asyuhur/Octave/ex2/ex2data1.txt', delimiter=',',
                 names=['exam1_score','exam2_score','admitted'])
df = df.astype({'admitted':'float'})

# Plot the original training data based on its actual training data (for visualization only)
df_true = df.loc[df['admitted']==1].reset_index(drop=True)
df_false = df.loc[df['admitted']==0].reset_index(drop=True)
plt.figure()
plt.scatter(df_true['exam1_score'], df_true['exam2_score'], marker='o', color='blue')
plt.scatter(df_false['exam1_score'], df_false['exam2_score'], marker='x', color='black')
plt.xlabel('exam1_score')
plt.ylabel('exam2_score')
plt.title('Overall dataset')
plt.show()

# Split the data into X and y
df['x0'] = 1
X = df[['x0','exam1_score','exam2_score']].to_numpy()
y = df[['admitted']].to_numpy()

# Split the data into train, cross valiation, and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=30)
X_crossval, X_test, y_crossval, y_test = train_test_split(X_test, y_test, train_size=0.5, random_state=31)

# Do gradient descent to get the optimum iteration and alpha for learning theta
n = 10000
n_iter = 1000
df_n_alpha = pd.DataFrame(columns=['n','alpha','cost_function_train','cost_function_crossval'])
while n_iter <= n:
    alpha = 0.0001
    alpha_iter = 0.001
    while alpha_iter >= alpha:
        theta = np.array([[0.0,0.0,0.0]])
        theta = np.transpose(theta)
        i = 1
        while i <= n_iter:
            theta = logisticRegression.logisticRegression(X_train, theta, y_train, alpha_iter)
            cost_train = cf.costFunction(X_train, theta, y_train)
            i += 1
        cost_crossval = cf.costFunction(X_crossval, theta, y_crossval)
        df_n_alpha_inp = pd.DataFrame({'n':[n_iter], 'alpha':[alpha_iter], 'cost_function_train':[cost_train],
                                       'cost_function_crossval':[cost_crossval]})
        df_n_alpha = pd.concat([df_n_alpha, df_n_alpha_inp]).reset_index(drop=True)
        alpha_iter  -= 0.0001
    n_iter += 1000
df_n_alpha.sort_values(by=['cost_function_crossval','cost_function_train'], ascending=True, inplace=True)

# Train the model again using optimized n and alpha to get learning parameter theta
n = 100000
alpha = 0.001
theta = np.array([[0.0,0.0,0.0]])
theta = np.transpose(theta)
n_li = []
cost_li = []
i = 1
while i <= n:
    theta = logisticRegression.logisticRegression(X_train, theta, y_train, alpha)
    cost = cf.costFunction(X_train, theta, y_train)
    n_li.extend([i])
    cost_li.extend([cost])
    i += 1

# Ensure that cost function decreases with iteration by plotting
plt.scatter(n_li, cost_li)
plt.xlabel('n_iteration')
plt.ylabel('cost_function')
plt.title('Cost function with iteration based on training data')
plt.show()

# Check Precision, Recall, F1 Score, and Accuracy
y_pred_train = sf.sigmoidFunction(X_train, theta)
y_pred_train = np.where(y_pred_train>=0.5,1,0)
df_pred = pd.DataFrame(y_pred_train, columns=['predicted_val'])
df_act = pd.DataFrame(y_train, columns=['actual_val'])
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
print('Accuracy based on training data: {}'.format(accuracy))
print('Precision based on training data: {}'.format(precision))
print('Recall based on training data: {}'.format(recall))
print('F1 Score based on training data: {}'.format(f1score))

# Print final cost function based on training data
print('Cost function based on training data: {}'.format(cost))