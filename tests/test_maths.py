"""
tests/test_maths.py
Unit tests verifying mathematical primitives and zero-leakage properties.
"""
import pytest
import numpy as np
from core.maths import (
    compute_ewma_series,
    compute_acwr,
    compute_training_monotony,
    compute_training_strain,
    calculate_match_leverage_weight,
    evaluate_decision_action,
)


def test_ewma_first_element():
    """First EWMA value must equal the first raw load."""
    loads = [500.0, 600.0, 700.0]
    ewma = compute_ewma_series(loads, time_decay_days=7)
    assert ewma[0] == pytest.approx(500.0)


def test_ewma_decay_behavior():
    """Higher subsequent load increases EWMA smoothly."""
    loads = [100.0, 200.0, 200.0, 200.0]
    ewma = compute_ewma_series(loads, time_decay_days=7)
    # lambda = 2 / (7 + 1) = 0.25
    # day 1: 200 * 0.25 + 100 * 0.75 = 50 + 75 = 125
    assert ewma[1] == pytest.approx(125.0)


def test_acwr_calculation():
    """ACWR equals acute / chronic."""
    assert compute_acwr(1200.0, 1000.0) == pytest.approx(1.2)
    assert compute_acwr(0.0, 0.0) == pytest.approx(1.0)
    assert compute_acwr(1000.0, 0.0) == pytest.approx(2.5)


def test_training_monotony_uniform_load():
    """Identical loads across 7 days produce high monotony (zero variance)."""
    uniform_loads = [500.0] * 7
    monotony = compute_training_monotony(uniform_loads)
    assert monotony >= 10.0


def test_training_monotony_varied_load():
    """Realistic periodized week (rest day + hard days) has moderate monotony."""
    periodized = [0.0, 700.0, 800.0, 300.0, 900.0, 200.0, 0.0]
    monotony = compute_training_monotony(periodized)
    assert 0.5 < monotony < 2.0


def test_match_leverage_tier_1_vs_tier_2():
    """Tier 1 must yield 2.0 leverage weight, while Tier 2 yields 1.0."""
    assert calculate_match_leverage_weight("tier_1_must_win") == 2.0
    assert calculate_match_leverage_weight("tier_2_standard") == 1.0
    assert calculate_match_leverage_weight("tier_3_rotation") == 0.6


def test_decision_action_championship_final():
    """
    In a high-stakes final with moderate physical risk (0.20),
    decision should be PLAY_WITH_MINUTE_CAP instead of outright benching.
    """
    decision = evaluate_decision_action(
        calibrated_physical_risk=0.20,
        match_leverage_multiplier=2.0,
        alert_budget_cutoff=0.15
    )
    assert decision["decision"] == "PLAY_WITH_MINUTE_CAP"
    assert decision["max_minutes"] == 60


def test_decision_action_low_stakes_rotation():
    """
    In a routine or rotation game with the same physical risk (0.20),
    decision should ROTATE and prescribe a micro-dose session.
    """
    decision = evaluate_decision_action(
        calibrated_physical_risk=0.20,
        match_leverage_multiplier=1.0,
        alert_budget_cutoff=0.15
    )
    assert decision["decision"] == "ROTATE_AND_PRESCRIBE_MICRODOSE"
    assert decision["max_minutes"] == 0
    assert decision["prescribe_microdose"] is True
