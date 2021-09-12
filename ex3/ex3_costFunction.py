class costFunction:

    def costFunction(X, theta, y, lambd):
        import numpy as np
        from ex3_sigmoidFunction import sigmoidFunction as sf

        n_class = len(y[0,:])
        m = len(X)
        sigmoid_y = sf.sigmoidFunction(X, theta)
        cost_logReg = np.multiply(-1/m, np.reshape(np.sum(np.add(np.multiply(y, np.log10(sigmoid_y)), np.multiply(np.subtract(1, y), np.log10(np.subtract(1, sigmoid_y)))), axis=0), (1, n_class)))
        cost_reg = np.multiply(lambd/(2 * m), np.reshape(np.sum(np.power(theta[1:,:], 2), axis=0), (1, n_class)))
        total_cost = cost_logReg + cost_reg
        return total_cost # The shape will be matrix 1 x n_class
