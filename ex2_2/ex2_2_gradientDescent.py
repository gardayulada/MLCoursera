class gradientDescent:

    def gradDescentLogReg(X, theta, y, alpha, lambd):
        from ex2_2_sigmoidFunction import sigmoidFunction as sf
        import numpy as np
        m = len(X)
        theta2 = theta.copy()
        theta2[0,0] = 0
        sigmoid_y = sf.sigmoidFunction(X, theta)
        theta -= np.multiply(alpha, np.divide(np.transpose(np.dot(np.transpose(sigmoid_y - y), X)), (m * np.log(10))) + np.multiply(lambd/m, theta2))
        return theta