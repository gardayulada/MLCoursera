class costFunction:

    def costFunctionLogRegress(X, theta, y, lambd):
        import numpy as np
        from ex2_2_sigmoidFunction import sigmoidFunction as sf
        m = len(X)
        sigmoid_y = sf.sigmoidFunction(X, theta)
        cost_logreg = (-1/m) * sum(np.multiply(y, np.log10(sigmoid_y)) + np.multiply(np.subtract(1, y), np.log10(np.subtract(1, sigmoid_y))))
        cost_regular = (lambd / (2*m)) * (np.dot(np.transpose(theta), theta) - theta[0,0]**2)
        total_cost = cost_logreg + cost_regular
        return float(total_cost)