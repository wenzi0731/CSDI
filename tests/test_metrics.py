import numpy as np
import pytest

from baseline5.metrics import interval_score, joint_scores, summarize


def test_interval_penalties():
    # alpha=.05 gives multiplier40; width2 with left/right misses of 1/2.
    np.testing.assert_allclose(interval_score([-1, 1, 4], [0, 0, 0], [2, 2, 2]), [42, 2, 82])
    with pytest.raises(ValueError):
        interval_score([1], [2], [0])


def test_joint_scores_hand_calculated_and_scaling():
    # Two scenarios (0,0),(2,2); truth(1,1). ES=sqrt(2)/2; VS=0.
    samples = np.array([[[[0., 0.]], [[2., 2.]]]])
    target = np.array([[[1., 1.]]])
    es, vs = joint_scores(samples, target, [0], [1])
    np.testing.assert_allclose(es, [np.sqrt(2) / 2])
    np.testing.assert_allclose(vs, [0])
    # Single deterministic scenario: ES reduces to Euclidean error.
    es, vs = joint_scores(np.array([[[[0., 0.]]]]), np.array([[[0., 4.]]]), [0], [1])
    np.testing.assert_allclose(es, [4])
    np.testing.assert_allclose(vs, [4])
    original = joint_scores(samples, target, [0], [1])
    transformed = joint_scores(samples * 7 + 5, target * 7 + 5, [5], [7])
    np.testing.assert_allclose(original, transformed)


def test_r2_not_pearson_squared_and_normalized_is():
    truth = np.array([[[0., 1.]], [[2., 3.]]])
    samples = np.repeat((truth + 1)[:, None], 2, axis=1)
    scores = summarize(samples, truth, ["X"], np.array([0]), np.array([2]))
    assert scores["X_R2"] == pytest.approx(0.2)  # Pearson^2 would be 1!
    assert scores["X_IS"] == pytest.approx(40.)
    assert scores["X_IS_Z"] == pytest.approx(20.)
    assert scores["X_RMSE_Z"] == pytest.approx(0.5)
    perfect = summarize(np.repeat(truth[:, None], 2, axis=1), truth, ["X"], [0], [2])
    assert perfect["X_R2"] == 1 and perfect["X_IS"] == 0
    constant = summarize(np.ones((2, 2, 1, 2)), np.ones((2, 1, 2)), ["X"], [0], [1])
    assert np.isnan(constant["X_R2"])


def test_joint_scores_match_bruteforce_and_detect_dependence():
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(3, 5, 4, 24)), rng.normal(size=(3, 4, 24))
    es, vs = joint_scores(x, y, np.zeros(4), np.ones(4))
    for n in range(3):
        a, b = x[n].reshape(5, -1), y[n].reshape(-1)
        expected_es = np.linalg.norm(a - b, axis=1).mean() - 0.5 * np.linalg.norm(a[:, None] - a[None], axis=-1).mean()
        i, j = np.triu_indices(96, 1)
        expected_vs = np.mean([(abs(b[k]-b[l])**0.5 - np.mean(abs(a[:, k]-a[:, l])**0.5))**2 for k, l in zip(i, j)])
        assert es[n] == pytest.approx(expected_es)
        assert vs[n] == pytest.approx(expected_vs)
    with pytest.raises(ValueError, match="positive"):
        joint_scores(x, y, np.zeros(4), np.zeros(4))
