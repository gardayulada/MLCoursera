import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/garda.asyuhur/Octave/ex1/ex1data1.txt', delimiter=',',
                 names=['city_pop','profit_per_truck'])

print(df)

plt.figure()
plt.scatter(df['city_pop'], df['profit_per_truck'])
plt.xlabel('city_pop')
plt.ylabel('profit_per_truck')
plt.show()



