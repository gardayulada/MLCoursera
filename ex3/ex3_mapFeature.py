class mapFeature:

    def mapFeature(X, degPolyNom):
        import numpy as np
        from sklearn.preprocessing import PolynomialFeatures
        X2 = PolynomialFeatures(degree=degPolyNom, include_bias=True).fit_transform(X)
        return X2