import numpy as np

'''
X = np.array([[3,15,5], [14,12,9]])
y = np.reshape(np.sum(X, axis=0), (1,3))
print(y)
print(y.shape)

print(len(X[1,:]))
print(len(X))

z = np.power(X[1:,:], 2)
print(z)

k = X.copy()
k[1,:] = 0
print(k)
print(min(X[1,:]))

l = np.reshape(np.max(X, axis=1), (len(X),1))
m = np.where(X==l,1,0)
print(m)

a = np.array([[0,1,0], [1,0,0], [0,0,1], [0,0,1]])
b = np.argwhere(a==1)
print(b[:,1])
b = np.reshape(np.add(b[:,1],1), (len(a),1))
print(b)
'''

e = np.array([[1,3,4]])
f = np.transpose(np.array([[2,5,6]]))
print(e.shape)
print(f.shape)
g = np.dot(e, f)
print(g)
print(g.shape)

h = np.array([[2,5,6]])
h = np.reshape(h, (len(h[0,:]),1))
print(h.shape)
i = np.dot(e,h)
print(i)
print(i.shape)
