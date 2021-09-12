import pandas as pd
import numpy as np

df = pd.read_csv('/Users/garda.asyuhur/Octave/ex1/ex1data1.txt', delimiter=',',
                 names=['city_pop','profit_per_truck'])

# Put data into matrix
x_data = [[1,i] for i in df['city_pop']]
X = np.matrix(x_data)
y = np.array(df['profit_per_truck'])
y = np.vstack(y)

# Initialize theta
theta = np.array([0,0])
theta = np.vstack(theta)
alpha = 0.1
m = len(X)
result = (alpha/m) * np.transpose(np.transpose(X * theta - y) * X)
print(result)