class logisticRegression:

    def logisticRegression(X, theta, y, alpha):
        import numpy as np
        from ex2_sigmoidFunction import sigmoidFunction as sf
        m = len(X)
        theta -= np.multiply((alpha/m), np.transpose(np.dot(np.transpose(sf.sigmoidFunction(X,theta) - y), X)))
        return theta