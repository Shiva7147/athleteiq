"""Unit tests for Deterministic Workload Analytics Engine."""

from datetime import date, timedelta
import pytest

from src.core.exceptions import InsufficientDataError
from src.domain.schemas import TelemetryRead
from src.services.analytics import compute_workload_summary


def test_compute_workload_summary_success() -> None:
    """Verifies deterministic workload calculation for 30 days of telemetry history."""
    history = [
        TelemetryRead(
            id=f"TEL-{i}",
            athlete_id="ATH-100",
            recorded_date=date(2026, 7, 1) + timedelta(days=i),
            hr_rest_bpm=52,
            hrv_rmssd_ms=65.0,
            sleep_hours=8.0,
            rpe_score=6.0,
            session_duration_minutes=60.0,
            total_distance_meters=8000.0,
            high_speed_running_meters=600.0,
            injuries_reported=None,
            session_load=360.0,  # 6 * 60
        )
        for i in range(30)
    ]
    summary = compute_workload_summary(history)
    assert summary.athlete_id == "ATH-100"
    assert summary.acute_workload_7d == 360.0
    assert summary.chronic_workload_28d == 360.0
    assert summary.acwr_uncoupled == 1.0
    assert summary.risk_zone == "SWEET_SPOT"


def test_compute_workload_summary_insufficient_data() -> None:
    """Verifies InsufficientDataError when history < 28 records."""
    short_history = [
        TelemetryRead(
            id=f"TEL-{i}",
            athlete_id="ATH-100",
            recorded_date=date(2026, 8, 1) + timedelta(days=i),
            hr_rest_bpm=52,
            hrv_rmssd_ms=65.0,
            sleep_hours=8.0,
            rpe_score=5.0,
            session_duration_minutes=60.0,
            session_load=300.0,
        )
        for i in range(15)
    ]
    with pytest.raises(InsufficientDataError):
        compute_workload_summary(short_history)
