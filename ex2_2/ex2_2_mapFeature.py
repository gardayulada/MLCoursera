class mapFeature:

    def mapFeaturePolyNomial(X_matrix, degree_polynom):
        from sklearn.preprocessing import PolynomialFeatures
        return PolynomialFeatures(degree=degree_polynom, include_bias=True).fit_transform(X_matrix)
