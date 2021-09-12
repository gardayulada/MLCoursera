class costFunction:

    def costFunction(X, theta, y):
        import numpy as np
        m = len(X)
        cost = (1/(2*m)) * np.dot(np.transpose(np.dot(X, theta) - y), np.dot(X, theta) - y)
        return float(cost)