import pandas as pd
from sklearn.model_selection import train_test_split
from

# Import the data to pandas dataframe
df = pd.read_csv('/Users/garda.asyuhur/Octave/ex2/ex2data2.txt', delimiter=',',
                 names=['test1','test2','pass_fail'])

# Add on x0 = 1 and split the data into X matrix and y vector
X = df[['test1','test2']].to_numpy()
y = df[['pass_fail']].to_numpy()

# Split the training, cross validation, and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=33)
X_crossval, X_test, y_crossval, y_test = train_test_split(X_test, y_test, train_size=0.5, random_state=50)
