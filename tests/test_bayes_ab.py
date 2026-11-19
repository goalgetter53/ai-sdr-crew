from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ai_sdr_crew.experiments.bayes_ab import (
    Arm,
    decide,
    expected_loss,
    monte_carlo_prob_greater,
    prob_greater,
)
from ai_sdr_crew.experiments.simulate import aa_test, ab_test, calibration

params = st.integers(min_value=1, max_value=200)


@pytest.mark.parametrize(
    ("a1", "b1", "a2", "b2"), [(1, 1, 1, 1), (11, 91, 21, 81), (5, 300, 7, 298), (40, 960, 52, 948)]
)
def test_exact_matches_monte_carlo(a1: int, b1: int, a2: int, b2: int) -> None:
    exact = prob_greater(a1, b1, a2, b2)
    assert exact == pytest.approx(monte_carlo_prob_greater(a1, b1, a2, b2), abs=0.005)


def test_identical_posteriors_give_one_half() -> None:
    assert prob_greater(30, 700, 30, 700) == pytest.approx(0.5, abs=1e-9)


@settings(max_examples=60)
@given(params, params, params, params)
def test_probabilities_are_complementary(a1: int, b1: int, a2: int, b2: int) -> None:
    p = prob_greater(a1, b1, a2, b2)
    assert 0.0 <= p <= 1.0
    assert p + prob_greater(a2, b2, a1, b1) == pytest.approx(1.0, abs=1e-9)


@settings(max_examples=40)
@given(params, params, params, params)
def test_extra_success_never_lowers_win_probability(a1: int, b1: int, a2: int, b2: int) -> None:
    assert prob_greater(a1, b1, a2 + 1, b2) >= prob_greater(a1, b1, a2, b2) - 1e-12


def test_expected_loss_matches_monte_carlo() -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    a, b = rng.beta(21, 81, 400_000), rng.beta(11, 91, 400_000)
    mc = float(np.maximum(a - b, 0).mean())
    assert expected_loss((21, 81), (11, 91)) == pytest.approx(mc, rel=0.01)
    assert expected_loss((11, 91), (21, 81)) >= 0


def test_non_integer_alpha_rejected() -> None:
    with pytest.raises(ValueError, match="integer"):
        prob_greater(1, 1, 1.5, 1)


def test_arm_validation() -> None:
    with pytest.raises(ValueError, match="replies"):
        Arm("x", 10, 11)


def test_decision_waits_for_min_sends_then_stops_on_clear_winner() -> None:
    early = decide(Arm("control", 100, 4), Arm("variant", 100, 9))
    assert not early.stop and "fewer than" in early.reason
    late = decide(Arm("control", 3000, 90), Arm("variant", 3000, 180))
    assert late.stop and late.winner == "variant" and late.prob_variant_better > 0.999


def test_decision_keeps_testing_when_unclear() -> None:
    d = decide(Arm("control", 400, 16), Arm("variant", 400, 18))
    assert not d.stop and math.isfinite(d.loss_ship_variant)


def test_simulated_probability_is_calibrated() -> None:
    cal = calibration(n_sims=3000, seed=3)
    assert cal.max_abs_gap < 0.08
    assert cal.brier < 0.07


def test_aa_false_positive_rates() -> None:
    aa = aa_test(n_sims=400, max_sends=1500, peek_every=150, seed=5)
    assert 0.01 <= aa.fixed_horizon_fpr <= 0.10
    assert aa.peeking_fpr > aa.fixed_horizon_fpr


def test_loss_rule_picks_the_better_arm_most_of_the_time() -> None:
    ab = ab_test(n_sims=200, control_rate=0.03, variant_rate=0.06, epsilon=0.001, seed=9)
    assert ab.correct_rate_when_stopped > 0.9
    assert ab.mean_regret_points < 0.003
