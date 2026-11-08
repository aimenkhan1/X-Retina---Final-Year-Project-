import numpy as np
from src.evaluation.metrics import compute_metrics


def test_perfect_predictions():
    y = [0, 1, 2, 3, 4, 0, 2]
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0 and abs(m["qwk"] - 1.0) < 1e-9
    assert m["per_class_recall"] == [1.0] * 5
    assert np.trace(np.array(m["confusion_matrix"])) == len(y)


def test_qwk_punishes_big_mistakes_more():
    y = [0, 0, 1, 2, 3, 4, 4, 2]
    small = [1, 0, 1, 2, 3, 4, 4, 2]   # one grade 0 called grade 1
    big = [4, 0, 1, 2, 3, 4, 4, 2]     # one grade 0 called grade 4
    assert compute_metrics(y, small)["accuracy"] == compute_metrics(y, big)["accuracy"]
    assert compute_metrics(y, small)["qwk"] > compute_metrics(y, big)["qwk"]


def test_missing_grade_and_single_class_do_not_crash():
    m = compute_metrics([0, 0, 2], [0, 0, 0])
    assert len(m["per_class_recall"]) == 5 and m["per_class_recall"][1] == 0.0
    assert np.array(m["confusion_matrix"]).shape == (5, 5)
    assert compute_metrics([0, 0], [0, 0])["qwk"] == 0.0
