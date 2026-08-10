from otter.test_files import test_case

OK_FORMAT = False

name = "q3"
points = 3

@test_case(points=None, hidden=False)
def test_pca_shape(X_pca, pca2, X):
    assert X_pca is not None, 'X_pca is not defined'
    assert X_pca.shape == (X.shape[0], 2), 'X_pca must be (n_samples, 2)'
    assert pca2.n_components_ == 2, 'pca2 must keep exactly 2 components'

@test_case(points=None, hidden=False)
def test_explained_var(explained_var):
    assert explained_var[0] > explained_var[1], 'components must come out in descending order of explained variance'
    assert 0.5 < explained_var.sum() < 0.6, 'PC1 + PC2 should explain ~55% of the standardized wine data'

