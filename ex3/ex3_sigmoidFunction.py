class sigmoidFunction:

    def sigmoidFunction(X, theta):
        import numpy as np
        sigma = np.dot(X, theta)
        y = np.divide(1, np.add(1, np.exp(np.multiply(-1, sigma))))
        return y