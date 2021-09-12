class featureNormalize:

    def mean_std_normalize(array):
        import numpy as np
        avg = np.average(array)
        stdev = np.std(array)
        array = (array - np.average(array)) / np.std(array)
        return array, avg, stdev