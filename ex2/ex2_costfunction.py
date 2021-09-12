class costFunction:

    def costFunction(X, theta, y):
        import numpy as np
        from ex2_sigmoidFunction import sigmoidFunction as sf
        m = len(X)
        cost_vector = np.multiply(y, np.log10(sf.sigmoidFunction(X,theta))) + np.multiply(np.subtract(1,y), np.log10(np.subtract(1, sf.sigmoidFunction(X,theta))))
        cost = (-1/m) * sum(cost_vector)
        return float(cost)