class linearRegression:

    def gradientDescent(X, theta, y, alpha):
        import numpy as np
        m = len(X)
        theta = theta - (alpha/m) * np.transpose(np.dot(np.transpose(np.dot(X, theta) - y), X))
        return theta