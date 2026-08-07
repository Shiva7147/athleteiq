"""Integration tests for RiskService domain service."""

from datetime import date, timedelta
from src.models import RiskTier, TelemetryRead
from src.services.risk_service import RiskService


def test_risk_service_low_risk() -> None:
    """Verifies RiskService output for optimal telemetry history."""
    service = RiskService()
    history = [
        TelemetryRead(
            id=f"TEL-{i}",
            athlete_id="ATH-001",
            recorded_date=date(2026, 7, 1) + timedelta(days=i),
            hr_rest_bpm=52,
            hrv_rmssd_ms=65.0,
            sleep_hours=8.0,
            rpe_score=5.0,
            session_duration_minutes=60.0,
            session_load=300.0,
        )
        for i in range(30)
    ]
    assessment = service.evaluate_risk(history, target_sleep_hours=8.0)
    assert assessment.risk_tier == RiskTier.LOW
    assert assessment.risk_score < 0.30


def test_risk_service_critical_risk_spike() -> None:
    """Verifies RiskService output for severe EWMA load spike and suppressed HRV."""
    service = RiskService()
    history = []
    # 25 baseline days
    for i in range(25):
        history.append(
            TelemetryRead(
                id=f"TEL-{i}",
                athlete_id="ATH-002",
                recorded_date=date(2026, 7, 1) + timedelta(days=i),
                hr_rest_bpm=52,
                hrv_rmssd_ms=70.0,
                sleep_hours=8.0,
                rpe_score=5.0,
                session_duration_minutes=60.0,
                session_load=300.0,
            )
        )
    # 5 spiked days
    for i in range(25, 30):
        history.append(
            TelemetryRead(
                id=f"TEL-{i}",
                athlete_id="ATH-002",
                recorded_date=date(2026, 7, 1) + timedelta(days=i),
                hr_rest_bpm=64,  # Elevated RHR
                hrv_rmssd_ms=40.0,  # Suppressed HRV
                sleep_hours=6.0,  # Sleep deficit
                rpe_score=9.0,
                session_duration_minutes=90.0,
                session_load=810.0,
            )
        )

    assessment = service.evaluate_risk(history, target_sleep_hours=8.0)
    assert assessment.risk_tier in [RiskTier.HIGH, RiskTier.CRITICAL]
    assert len(assessment.contributing_factors) >= 2
