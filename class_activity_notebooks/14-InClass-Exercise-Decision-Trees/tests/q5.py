from otter.test_files import test_case

OK_FORMAT = False

name = "q5"
points = 4

@test_case(points=None, hidden=False)
def test_importances(importances, top5, X_train):
    import numpy as np
    import pandas as pd
    assert isinstance(importances, pd.Series), 'importances must be a pandas Series'
    assert len(importances) == X_train.shape[1]
    assert (importances.diff().dropna() <= 1e-12).all(), 'importances must be sorted descending'
    assert np.isclose(importances.sum(), 1.0), 'impurity importances sum to 1'
    assert list(top5) == list(importances.index[:5]), 'top5 must be the 5 most important feature names, most important first'

@test_case(points=None, hidden=False)
def test_forest_beats_tree(rf_train_acc, rf_test_acc, best_tree_test_acc):
    import numpy as np
    assert np.isclose(rf_train_acc, 1.0, atol=1e-06)
    assert rf_test_acc > best_tree_test_acc, 'the forest should beat the best tree'
    assert rf_test_acc > 0.95

