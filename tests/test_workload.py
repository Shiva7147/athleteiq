"""Unit tests for Sports Science Workload Analytics Engine."""

from datetime import date, timedelta
import pytest

from athleteiq.analytics.workload import (
    calculate_acwr_ewma,
    calculate_acwr_uncoupled,
    calculate_acute_workload,
    calculate_chronic_workload,
    calculate_monotony,
    calculate_strain,
    compute_workload_summary,
)
from athleteiq.core.exceptions import InsufficientDataError
from athleteiq.core.models import DailyTelemetry


def test_calculate_acute_workload() -> None:
    """Tests 7-day rolling acute workload calculation."""
    loads = [300.0] * 10
    # 7-day average of 300.0 is 300.0
    assert calculate_acute_workload(loads) == 300.0


def test_calculate_chronic_workload() -> None:
    """Tests 28-day rolling chronic workload calculation."""
    loads = [400.0] * 35
    assert calculate_chronic_workload(loads) == 400.0


def test_calculate_acwr_uncoupled() -> None:
    """Tests ACWR uncoupled ratio calculation."""
    assert calculate_acwr_uncoupled(600.0, 500.0) == 1.2
    assert calculate_acwr_uncoupled(600.0, 0.0) == 0.0


def test_calculate_acwr_ewma() -> None:
    """Tests EWMA ACWR ratio for constant and spike load patterns."""
    constant_loads = [400.0] * 28
    # Constant load should yield EWMA ACWR ~ 1.0
    assert abs(calculate_acwr_ewma(constant_loads) - 1.0) < 0.05

    # Acute spike load: 21 days of 200, then 7 days of 600
    spiked_loads = ([200.0] * 21) + ([600.0] * 7)
    ewma_spike = calculate_acwr_ewma(spiked_loads)
    # Spike should drive EWMA ACWR significantly above 1.0
    assert ewma_spike > 1.25


def test_calculate_monotony_and_strain() -> None:
    """Tests Foster monotony and strain index calculation."""
    # Varied loads over 7 days
    varied_loads = [200.0, 400.0, 300.0, 500.0, 100.0, 450.0, 250.0]
    monotony = calculate_monotony(varied_loads)
    assert monotony > 0.0

    strain = calculate_strain(varied_loads)
    assert strain == round(sum(varied_loads) * monotony, 2)


def test_compute_workload_summary_insufficient_data_raises_error() -> None:
    """Verifies InsufficientDataError is raised when history < 28 records."""
    short_history = [
        DailyTelemetry(
            athlete_id="ATH-001",
            recorded_date=date(2026, 8, 1) + timedelta(days=i),
            hr_rest_bpm=55,
            hrv_rmssd_ms=60.0,
            sleep_hours=8.0,
            rpe_score=5.0,
            session_duration_minutes=60.0,
        )
        for i in range(20)
    ]
    with pytest.raises(InsufficientDataError):
        compute_workload_summary(short_history)


def test_compute_workload_summary_success() -> None:
    """Tests end-to-end workload summary calculation with 30 days of valid history."""
    history = [
        DailyTelemetry(
            athlete_id="ATH-001",
            recorded_date=date(2026, 7, 1) + timedelta(days=i),
            hr_rest_bpm=55,
            hrv_rmssd_ms=65.0,
            sleep_hours=8.0,
            rpe_score=6.0,
            session_duration_minutes=60.0,
        )
        for i in range(30)
    ]
    summary = compute_workload_summary(history)
    assert summary.athlete_id == "ATH-001"
    assert summary.acute_workload_7d == 360.0  # 6.0 * 60.0
    assert summary.chronic_workload_28d == 360.0
    assert summary.acwr_uncoupled == 1.0
