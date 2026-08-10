from otter.test_files import test_case

OK_FORMAT = False

name = "q2"
points = 3

@test_case(points=None, hidden=False)
def test_best_split_matches_sklearn(best_thr, best_score, X_train, y_train, FEATURE, RANDOM_STATE):
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    assert best_thr is not None, 'best_split_for_feature returned no threshold'
    stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)
    stump.fit(X_train[[FEATURE]], y_train)
    sk_thr = stump.tree_.threshold[0]
    n_nodes = stump.tree_.n_node_samples
    impurity = stump.tree_.impurity
    sk_score = (n_nodes[1] * impurity[1] + n_nodes[2] * impurity[2]) / n_nodes[0]
    assert np.isclose(best_thr, sk_thr, atol=1e-06), f"your threshold {best_thr} != sklearn's {sk_thr}"
    assert np.isclose(best_score, sk_score, atol=1e-09), f"your weighted Gini {best_score} != sklearn's {sk_score}"

@test_case(points=None, hidden=False)
def test_best_split_values(best_thr, best_score, root_gini):
    import numpy as np
    assert np.isclose(best_thr, 113.75, atol=1e-06), f'expected a threshold of 113.75, got {best_thr}'
    assert np.isclose(best_score, 0.146339, atol=1e-05), f'expected a weighted child Gini of ~0.146339, got {best_score}'
    assert best_score < root_gini, 'a real split must lower impurity'

