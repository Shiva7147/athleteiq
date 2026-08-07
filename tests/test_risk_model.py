"""Unit tests for Soft-Tissue Injury Risk Prediction Engine."""

from datetime import date

from athleteiq.core.models import FeatureVector, RiskTier
from athleteiq.models.injury_risk import SoftTissueInjuryRiskPredictor


def test_predict_risk_low_tier() -> None:
    """Verifies low risk tier prediction for optimal physiological feature vector."""
    predictor = SoftTissueInjuryRiskPredictor()
    optimal_fv = FeatureVector(
        athlete_id="ATH-001",
        feature_date=date(2026, 8, 7),
        hrv_z_score=0.5,
        rhr_z_score=-0.2,
        sleep_deficit_ratio=0.0,
        acwr_ewma=1.05,
        acwr_spike_delta=0.02,
        monotony=1.1,
        strain=1500.0,
        hsr_ratio=0.08,
    )
    assessment = predictor.predict_risk(optimal_fv)
    assert assessment.risk_tier == RiskTier.LOW
    assert assessment.risk_score < 0.30
    assert "LOW RISK" in assessment.recommended_action


def test_predict_risk_critical_tier_and_factors() -> None:
    """Verifies critical risk tier classification and explainability factor extraction."""
    predictor = SoftTissueInjuryRiskPredictor()
    high_stress_fv = FeatureVector(
        athlete_id="ATH-002",
        feature_date=date(2026, 8, 7),
        hrv_z_score=-1.8,  # Severe HRV suppression (+0.25)
        rhr_z_score=1.6,   # Elevated RHR (+0.20)
        sleep_deficit_ratio=0.35,  # High sleep deficit (+0.15)
        acwr_ewma=1.65,    # Critical workload spike (+0.35)
        acwr_spike_delta=0.25,
        monotony=2.4,      # High monotony & strain (+0.15)
        strain=4500.0,
        hsr_ratio=0.18,
    )
    assessment = predictor.predict_risk(high_stress_fv)
    assert assessment.risk_tier == RiskTier.CRITICAL
    assert assessment.risk_score >= 0.80
    assert len(assessment.contributing_factors) >= 4
    assert "IMMEDIATE ACTION REQUIRED" in assessment.recommended_action
