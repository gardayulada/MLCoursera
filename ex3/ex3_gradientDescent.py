class gradientDescent:

    def gradDescentLogReg(X, theta, y, alpha, lambd):
        import numpy as np
        from ex3_sigmoidFunction import sigmoidFunction as sf

        m = len(X)
        sigmoid_y = sf.sigmoidFunction(X, theta)
        theta2 = theta.copy()
        theta2[0,:] = 0
        theta -= np.multiply(alpha, np.add(np.multiply(1/(m * np.log(10)), np.transpose(np.dot(np.transpose(np.subtract(sigmoid_y, y)), X))), np.multiply(lambd/m, theta2)))
        return theta