from otter.test_files import test_case

OK_FORMAT = False

name = "q3"
points = 4

@test_case(points=None, hidden=False)
def test_k_sweep(wcss_by_k, sil_by_k):
    assert wcss_by_k is not None and sil_by_k is not None
    assert len(wcss_by_k) == 10, 'wcss_by_k needs one entry per k in range(1, 11)'
    assert len(sil_by_k) == 9, 'sil_by_k needs one entry per k in range(2, 11)'
    assert all((wcss_by_k[i] >= wcss_by_k[i + 1] - 1e-06 for i in range(9))), 'WCSS must be non-increasing in k'
    assert max(sil_by_k) <= 1.0 and min(sil_by_k) >= -1.0, 'silhouette lives in [-1, 1]'

