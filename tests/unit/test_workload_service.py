"""Unit tests for WorkloadService orchestrator."""

from datetime import date, timedelta
import pytest

from src.models import TelemetryRead
from src.services.workload_service import WorkloadService
from src.utils.exceptions import InsufficientDataError


def test_workload_service_compute_summary_success() -> None:
    """Verifies WorkloadService generating WorkloadSummaryRead DTO."""
    service = WorkloadService()
    history = [
        TelemetryRead(
            id=f"TEL-{i}",
            athlete_id="ATH-001",
            recorded_date=date(2026, 7, 1) + timedelta(days=i),
            hr_rest_bpm=52,
            hrv_rmssd_ms=65.0,
            sleep_hours=8.0,
            rpe_score=6.0,
            session_duration_minutes=60.0,
            session_load=360.0,
        )
        for i in range(30)
    ]
    summary = service.compute_summary(history)
    assert summary.athlete_id == "ATH-001"
    assert summary.acute_workload_7d == 360.0
    assert summary.chronic_workload_28d == 360.0
    assert summary.acwr_uncoupled == 1.0


def test_workload_service_insufficient_history_raises_error() -> None:
    """Verifies InsufficientDataError when history < 28 days."""
    service = WorkloadService()
    short_history = [
        TelemetryRead(
            id=f"TEL-{i}",
            athlete_id="ATH-001",
            recorded_date=date(2026, 8, 1) + timedelta(days=i),
            hr_rest_bpm=52,
            hrv_rmssd_ms=65.0,
            sleep_hours=8.0,
            rpe_score=6.0,
            session_duration_minutes=60.0,
            session_load=360.0,
        )
        for i in range(10)
    ]
    with pytest.raises(InsufficientDataError):
        service.compute_summary(short_history)
