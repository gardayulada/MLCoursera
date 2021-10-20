class costFunction:

    def costFunction(X, theta1, theta2, y, lambd):
        import numpy as np
        from forwardProp import fwdProp as fw
        from sklearn.preprocessing import PolynomialFeatures

        m = len(y)
        a2 = fw.fwdProp(X, theta1)
        a2 = PolynomialFeatures(degree=1, include_bias=True).fit_transform(a2)
        a3 = fw.fwdProp(a2, theta2)

        cost_logreg = np.multiply((-1/m), np.sum(np.add(np.multiply(y, np.log10(a3)), np.multiply(np.subtract(1, y), np.log10(np.subtract(1, a3)))), axis=0))
        cost_reg = (lambd / (2*m)) * (np.sum(np.multiply(theta1[1:,:], theta1[1:,:])) + np.sum(np.multiply(theta2[1:,:], theta2[1:,:])))
        total_cost = np.add(cost_logreg, cost_reg)

        return total_cost