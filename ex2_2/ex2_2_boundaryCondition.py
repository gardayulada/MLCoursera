class boundaryCondition:

    def boundaryCondition(theta, degPolyNom):
        from ex2_2_costFunction import costFunction as cf
        from ex2_2_sigmoidFunction import sigmoidFunction as sf
        from ex2_2_mapFeature import mapFeature as mf
        import numpy as np

        X = np.array([[0,0]])
        i = -2
        i_max = 2
        while i <= i_max:
            j = -2
            j_max = 2
            while j <= j_max:
                M = np.array([[i,j]])
                M2 = mf.mapFeaturePolyNomial(M, degPolyNom)
                y = sf.sigmoidFunction(M2, theta)
                if y == 0.5:

