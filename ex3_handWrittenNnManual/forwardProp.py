class fwdProp:

    def fwdProp(neuron, theta):
        import numpy as np
        out = np.dot(neuron,theta)
        sig_out = np.divide(1, np.add(1, np.exp(np.multiply(-1, out))))
        return sig_out