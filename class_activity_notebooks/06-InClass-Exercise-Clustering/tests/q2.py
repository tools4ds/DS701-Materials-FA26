from otter.test_files import test_case

OK_FORMAT = False

name = "q2"
points = 4

@test_case(points=None, hidden=False)
def test_my_kmeans_output(my_centroids, my_labels, my_wcss, my_iters, X, k):
    import numpy as np
    my_centroids = np.asarray(my_centroids)
    my_labels = np.asarray(my_labels)
    assert my_centroids.shape == (k, X.shape[1]), 'my_centroids must be (k, d)'
    assert my_labels.shape == (len(X),), 'my_labels must have one label per row of X'
    assert 1 < my_iters < 100, f'suspicious iteration count: {my_iters}'
    assert my_wcss > 0, 'WCSS should be a positive float'

