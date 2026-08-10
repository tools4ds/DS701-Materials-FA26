from otter.test_files import test_case

OK_FORMAT = False

name = "q6"
points = 4

@test_case(points=None, hidden=False)
def test_evr_basic(evr, cum_evr, X):
    import numpy as np
    evr = np.asarray(evr)
    cum_evr = np.asarray(cum_evr)
    assert evr.shape == (X.shape[1],), 'evr must have one entry per original feature'
    assert np.isclose(evr.sum(), 1.0), 'explained variance ratios must sum to 1'
    assert np.all(np.diff(evr) <= 1e-12), 'evr must be in descending order'
    assert np.allclose(cum_evr, np.cumsum(evr)), 'cum_evr must be the cumulative sum of evr'

@test_case(points=None, hidden=False)
def test_n_components_90(n_components_90, cum_evr, X):
    import numpy as np
    cum_evr = np.asarray(cum_evr)
    assert isinstance(n_components_90, (int, np.integer)), 'n_components_90 must be an int'
    assert 1 <= n_components_90 <= X.shape[1]
    assert cum_evr[n_components_90 - 1] >= 0.9, 'that many components does not reach 90%'
    assert n_components_90 == 1 or cum_evr[n_components_90 - 2] < 0.9, 'you can reach 90% with fewer components -- it must be the SMALLEST such k'

