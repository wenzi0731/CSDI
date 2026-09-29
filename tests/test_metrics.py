import numpy as np
import pytest

from baseline5.metrics import interval_score, joint_scores, summarize, save_additional_scores


def test_interval_penalties():
    # alpha=.05 gives multiplier40; width2 with left/right misses of 1/2.
    np.testing.assert_allclose(interval_score([-1, 1, 4], [0, 0, 0], [2, 2, 2]), [42, 2, 82])
    with pytest.raises(ValueError):
        interval_score([1], [2], [0])


def test_joint_scores_hand_calculated_and_scaling():
    # Two scenarios (0,0),(2,2); truth(1,1). ES=sqrt(2)/2; VS=0.
    samples = np.array([[[[0.], [0.]], [[2.], [2.]]]])
    target = np.array([[[1.], [1.]]])
    es, vs = joint_scores(samples, target, [0, 0], [1, 1])
    np.testing.assert_allclose(es, [np.sqrt(2) / 2])
    np.testing.assert_allclose(vs, [0])
    # Single deterministic scenario: ES reduces to Euclidean error.
    es, vs = joint_scores(np.zeros((1, 1, 2, 1)), np.array([[[0.], [4.]]]), [0, 0], [1, 1])
    np.testing.assert_allclose(es, [4])
    np.testing.assert_allclose(vs, [4])
    original = joint_scores(samples, target, [0, 0], [1, 1])
    transformed = joint_scores(samples * 7 + 5, target * 7 + 5, [5, 5], [7, 7])
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
        # Independent paper implementation: six channel pairs x all 24x24 hours.
        expected_vs = 0.0
        for c1 in range(4):
            for c2 in range(c1 + 1, 4):
                for h1 in range(24):
                    for h2 in range(24):
                        observed = abs(y[n, c1, h1] - y[n, c2, h2]) ** 0.5
                        predicted = np.mean(abs(x[n, :, c1, h1] - x[n, :, c2, h2]) ** 0.5)
                        expected_vs += (observed - predicted) ** 2
        assert es[n] == pytest.approx(expected_es)
        assert vs[n] == pytest.approx(expected_vs)
    with pytest.raises(ValueError, match="positive"):
        joint_scores(x, y, np.zeros(4), np.zeros(4))


def test_vs_excludes_within_channel_and_sums_all_cross_hours():
    samples = np.array([[[[0., 4.], [1., 9.]]]])
    target = np.zeros((1, 2, 2))
    _, vs = joint_scores(samples, target, [0, 0], [1, 1])
    # Deterministic forecast, p=.5: squared discrepancies are 1,9,3,5.
    # Same-hour pairs contribute 6; different-hour pairs contribute 12.
    # Within-channel discrepancies 4 and 8 must not contribute.
    np.testing.assert_allclose(vs, [18.])
    # A single channel has no positive-weight pairs, even with temporal errors.
    _, within_only = joint_scores(samples[:, :, :1], target[:, :1], [0], [1])
    np.testing.assert_array_equal(within_only, [0.])


def test_vs_daily_mean_and_definition_exports(tmp_path):
    import csv
    import json

    first = np.broadcast_to(np.arange(4.)[:, None], (4, 24))
    samples = np.stack([first, first * 4])[:, None]
    targets = np.zeros((2, 4, 24))
    labels = ["Electricity", "Heat", "Cooling", "PV"]
    result = save_additional_scores(samples, targets, np.zeros(4), np.ones(4),
                                    ["2022-01-01", "2022-01-02"], labels, tmp_path)
    # sum_{c1<c2} |c1-c2| = 10, with 24*24 pairs per channel pair.
    expected_daily = np.array([5760., 23040.])
    assert result["VS"] == pytest.approx(expected_daily.mean())
    assert result["VS_Z"] == result["VS"]
    assert result["VS_pair_count"] == 3456
    assert result["VS_definition"] == "cross_channel_unit_weights_v1"
    with (tmp_path / "daily_scores.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    np.testing.assert_allclose([float(row["VS_Z"]) for row in rows], expected_daily)
    definitions = json.loads((tmp_path / "metric_definitions.json").read_text())
    assert definitions["VS_pair_count"] == 3456
    assert definitions["VS_definition"] == result["VS_definition"]
