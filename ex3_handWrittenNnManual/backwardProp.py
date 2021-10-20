class backProp:

    def backPropLayer2(theta2, a3, y_act, a2, lambd):
        import numpy as np
        m = len(a3)
        theta2_2 = theta2.copy()
        theta2_2[0,:] = 0
        gradDescent = np.add(np.multiply(-1/(m*np.log(10)), np.transpose(np.dot(np.transpose(np.subtract(y_act, a3)), a2))), np.multiply(lambd/m, theta2_2))
        return gradDescent

    def backPropLayer1(theta1, theta2, a3, y_act, a2, a1, lambd):
        import numpy as np
        m = len(a3)
        theta1_2 = theta1.copy()
        theta1_2[0,:] = 0
        gradDescent = np.multiply(-1/(m*np.log(10)), np.transpose(np.dot(np.transpose(np.multiply(np.dot(np.subtract(y_act, a3), np.transpose(theta2)), np.multiply(a2, np.subtract(1, a2)))), a1)))
        gradDescent = gradDescent[:,1:]
        gradDescent = np.add(gradDescent, np.multiply(lambd/m, theta1_2))
        return gradDescent