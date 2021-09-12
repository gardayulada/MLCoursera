'''
Import all the necessary modules
'''
import pandas as pd
from mat4py import loadmat
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

'''
Import the data and put it into X and y. Add bias as well by using PolynomialFeatures
'''
data = loadmat('/Users/garda.asyuhur/Octave/ex3/ex3data1.mat')
df = pd.DataFrame(data)
X = np.vstack(df['X'])
y = np.hstack(df['y'])
y = np.where(y==10,0,y)
X = PolynomialFeatures(degree=1, include_bias=True).fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=34)
X_crossval, X_test, y_crossval, y_test = train_test_split(X_test, y_test, train_size=0.2, random_state=32)

'''
Generate the Neural Network model using Tensorflow
'''
nn_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(401,)),
    tf.keras.layers.Dense(802, activation='sigmoid'),
    tf.keras.layers.Dense(10)
])

'''
Define the training model
'''
nn_model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

'''
Train the model
'''
nn_model.fit(X_train, y_train, epochs=10000)

'''
Test the model using crossvalidation data
'''
crossval_loss, crossval_acc = nn_model.evaluate(X_crossval, y_crossval, verbose=1)
print('Crossval Accuracy: {}'.format(crossval_acc))
'''
Test the model using test data
'''
test_loss, test_acc = nn_model.evaluate(X_test, y_test, verbose=1)
print('Test Accuracy: {}'.format(test_acc))
pred_test = nn_model.predict(X_test)
pred_label = []
for pred in pred_test:
    max_val = np.max(pred)
    indice = np.where(pred==max_val)
    indice = indice[0]
    pred_label.extend(indice)
pred_label = np.array(pred_label)
pred_df = pd.DataFrame(pred_label, columns=['pred_label'])
act_df = pd.DataFrame(y_test, columns=['act_label'])
df_compare = pd.concat([pred_df, act_df], axis=1)
df_compare['check'] = np.where(df_compare['pred_label']==df_compare['act_label'],1,0)
print('Compare y prediction and actual based on test data:')
print(df_compare.head())