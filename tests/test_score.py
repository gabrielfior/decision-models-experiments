"""Tests for the frozen scorer in prepare.py.

Conventions under test (see program.md):
- probs: float array [N, K_max], rows padded with zeros beyond n_options[r].
- Brier is the multi-class sum form, sum_i (p_i - y_i)^2, in [0, 2].
- ECE uses 10 equal-width confidence bins on max-prob.
- selection = mean over question types of 0.5 * ((acc - chance)/(1 - chance) + (1 - brier)).
- question types use JevBench's names: noul (yes/no), choice, score.
"""
import numpy as np
import pytest

import prepare


def _rows(probs, labels, qtypes, n_opts):
    return np.asarray(probs, dtype=np.float64), np.asarray(labels), list(qtypes), np.asarray(n_opts)


def test_perfect_one_hot_predictions_score_perfectly():
    probs, labels, qtypes, n = _rows([[1, 0, 0], [0, 1, 0]], [0, 1], ["choice", "choice"], [3, 3])
    m = prepare.score(probs, labels, qtypes, n)
    assert m["accuracy"] == 1.0
    assert m["brier"] == 0.0
    assert m["ece"] == pytest.approx(0.0)


def test_brier_is_multiclass_sum_form():
    probs, labels, qtypes, n = _rows([[0.7, 0.3]], [0], ["noul"], [2])
    m = prepare.score(probs, labels, qtypes, n)
    assert m["brier"] == pytest.approx(0.3**2 + 0.3**2)


def test_chance_uses_per_row_option_count():
    # one 2-option row and one 4-option row, both answered correctly
    probs, labels, qtypes, n = _rows([[1, 0, 0, 0], [0, 0, 1, 0]], [0, 2], ["noul", "choice"], [2, 4])
    m = prepare.score(probs, labels, qtypes, n)
    assert m["per_type"]["noul"]["chance"] == pytest.approx(0.5)
    assert m["per_type"]["choice"]["chance"] == pytest.approx(0.25)
    # chance-corrected, as JevBench's intelligence axis: (acc - chance) / (1 - chance)
    assert m["per_type"]["noul"]["acc_above_chance"] == pytest.approx(1.0)
    assert m["per_type"]["choice"]["acc_above_chance"] == pytest.approx(1.0)


def test_selection_is_mean_over_types_of_acc_above_chance_and_one_minus_brier():
    probs, labels, qtypes, n = _rows(
        [[1, 0, 0, 0], [0.5, 0.5, 0, 0]], [0, 1], ["choice", "noul"], [4, 2]
    )
    m = prepare.score(probs, labels, qtypes, n)
    # choice row: acc 1, chance .25 -> corrected 1; brier 0 -> 0.5*(1 + 1) = 1.0
    # noul row: argmax ties -> index 0 is wrong, acc 0, chance .5 -> corrected -1; brier .5 -> 0.5*(-1 + 0.5) = -0.25
    assert m["selection"] == pytest.approx((1.0 - 0.25) / 2)


def test_selection_se_is_zero_for_identical_rows_and_positive_otherwise():
    probs, labels, qtypes, n = _rows([[1, 0], [1, 0]], [0, 0], ["noul", "noul"], [2, 2])
    assert prepare.score(probs, labels, qtypes, n)["selection_se"] == pytest.approx(0.0)
    probs, labels, qtypes, n = _rows([[1, 0], [1, 0]], [0, 1], ["noul", "noul"], [2, 2])
    assert prepare.score(probs, labels, qtypes, n)["selection_se"] > 0.0


def test_ece_zero_when_calibrated_and_large_when_overconfident():
    N = 1000
    conf = np.full((N, 2), [0.9, 0.1])
    # calibrated: exactly 90% of rows correct at confidence 0.9
    labels_cal = np.where(np.arange(N) % 10 == 0, 1, 0)
    m = prepare.score(conf, labels_cal, ["noul"] * N, np.full(N, 2))
    assert m["ece"] == pytest.approx(0.0, abs=1e-9)
    # overconfident: exactly 50% correct at confidence 0.9
    labels_bad = np.arange(N) % 2
    m = prepare.score(conf, labels_bad, ["noul"] * N, np.full(N, 2))
    assert m["ece"] == pytest.approx(0.4, abs=1e-9)


def test_rejects_probabilities_that_do_not_sum_to_one_over_valid_options():
    probs = np.array([[0.6, 0.6, 0.0]])
    with pytest.raises(ValueError):
        prepare.score(probs, np.array([0]), ["choice"], np.array([2]))


def test_keep_rule_requires_more_than_two_standard_errors():
    base = [0.50, 0.51, 0.49]
    se = 0.01
    assert prepare.keep(base, [0.53, 0.52, 0.53], se) is True   # +0.027 > 0.02
    assert prepare.keep(base, [0.515, 0.51, 0.52], se) is False  # +0.015 < 0.02
    assert prepare.keep(base, [0.50, 0.51, 0.49], se) is False   # no change


def test_keep_rule_noise_floor_is_the_larger_of_dev_se_and_seed_se():
    # seeds disagree wildly: seed std 0.03 -> seed SE 0.0173 > dev SE 0.01, so the bar is 2 * 0.0173
    base = [0.26, 0.29, 0.32]
    assert prepare.keep(base, [0.31, 0.31, 0.31], 0.01) is False    # +0.02 clears 2*dev_se but not 2*seed_se
    assert prepare.keep(base, [0.33, 0.33, 0.33], 0.01) is True     # +0.04 clears both
    assert prepare.noise_floor(base, [0.31, 0.31, 0.31], 0.01) == pytest.approx(0.03 / 3**0.5, rel=1e-6)
