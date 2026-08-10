from otter.test_files import test_case

OK_FORMAT = False

name = "q9"
points = 4

@test_case(points=None, hidden=False)
def test_svd_eigenvalues(lambdas_svd, pca_full, X):
    import numpy as np
    lambdas_svd = np.asarray(lambdas_svd)
    assert lambdas_svd.shape == (X.shape[1],), 'lambdas_svd must have one entry per feature'
    assert np.allclose(lambdas_svd, pca_full.explained_variance_), 'lambda_i should equal sigma_i^2 / (m - 1)'

@test_case(points=None, hidden=False)
def test_svd_projection(Y_svd, X_pca, X):
    import numpy as np
    Y_svd = np.asarray(Y_svd)
    assert Y_svd.shape == (X.shape[0], 2), 'Y_svd must be (n_samples, 2)'
    assert np.allclose(np.abs(Y_svd), np.abs(X_pca)), "the SVD projection must match sklearn's PCA projection up to sign"

