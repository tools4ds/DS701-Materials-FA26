from otter.test_files import test_case

OK_FORMAT = False

name = "q1"
points = 6

@test_case(points=None, hidden=False)
def test_building_blocks(assign_clusters, update_centroids, compute_wcss):
    import numpy as np
    X_t = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]])
    C_t = np.array([[0.0, 0.5], [10.0, 0.5]])
    lab_t = np.asarray(assign_clusters(X_t, C_t))
    assert lab_t.shape == (4,), 'assign_clusters must return one label per row of X'
    assert np.array_equal(lab_t, np.array([0, 0, 1, 1])), f'got {lab_t}'
    cen_t = np.asarray(update_centroids(X_t, lab_t, 2))
    assert cen_t.shape == (2, 2), f'update_centroids returned shape {cen_t.shape}'
    assert np.allclose(cen_t, C_t), f'got {cen_t}'
    w_t = compute_wcss(X_t, lab_t, C_t)
    assert np.isclose(w_t, 1.0), f'expected WCSS 1.0, got {w_t}'

@test_case(points=None, hidden=False)
def test_one_iteration(labels_1, centroids_1, wcss_before, wcss_after, X, init_centroids, k):
    import numpy as np
    labels_1 = np.asarray(labels_1)
    centroids_1 = np.asarray(centroids_1)
    assert labels_1.shape == (len(X),), 'labels_1 must have one label per row of X'
    assert centroids_1.shape == (k, X.shape[1]), 'centroids_1 must be (k, d)'
    assert wcss_after <= wcss_before + 1e-09, 'the update step can never increase WCSS'
    assert not np.allclose(centroids_1, init_centroids), 'the centers should have moved'

