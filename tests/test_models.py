"""Unit tests for AthleteIQ Pro Domain Models."""

from datetime import date

from athleteiq.core.models import DailyTelemetry, InjuryRiskAssessment, RiskTier, WorkloadMetrics


def test_daily_telemetry_session_load_calculation() -> None:
    """Tests sRPE session load calculation property on DailyTelemetry."""
    record = DailyTelemetry(
        athlete_id="ATH-001",
        recorded_date=date(2026, 8, 1),
        hr_rest_bpm=55,
        hrv_rmssd_ms=65.0,
        sleep_hours=8.0,
        rpe_score=7.0,
        session_duration_minutes=90.0,
    )
    # 7.0 * 90.0 = 630.0
    assert record.session_load == 630.0


def test_workload_metrics_instantiation() -> None:
    """Tests instantiation and attribute access on WorkloadMetrics DTO."""
    metrics = WorkloadMetrics(
        athlete_id="ATH-001",
        as_of_date=date(2026, 8, 7),
        acute_workload_7d=500.0,
        chronic_workload_28d=450.0,
        acwr_uncoupled=1.111,
        acwr_ewma=1.105,
        monotony=1.2,
        strain=4200.0,
    )
    assert metrics.athlete_id == "ATH-001"
    assert metrics.acwr_uncoupled == 1.111


def test_injury_risk_assessment_tier_enum() -> None:
    """Tests InjuryRiskAssessment DTO with RiskTier enum values."""
    assessment = InjuryRiskAssessment(
        athlete_id="ATH-002",
        assessment_date=date(2026, 8, 7),
        risk_tier=RiskTier.HIGH,
        risk_score=0.82,
        contributing_factors=["High EWMA ACWR (1.65)", "Suppressed HRV (-25%)"],
        recommended_action="Reduce training duration by 40% and prescribe light active recovery.",
    )
    assert assessment.risk_tier == RiskTier.HIGH
    assert len(assessment.contributing_factors) == 2
