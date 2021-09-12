import numpy as np

v = np.array([[4,5,2]])
v = np.transpose(v)
w = np.array([[1,8,3]])
w = np.transpose(w)
print(v)
print(w)

def sigmoidFunction(v):
    result = np.where(v%2==0,0,1)
    return result

import pandas as pd

df_v = pd.DataFrame(v, columns=['vector v'])
df_w = pd.DataFrame(w, columns=['vector w'])
df = pd.concat([df_v, df_w], axis=1)
print(df)
