"""Calibrated Soft-Tissue Injury Risk Prediction Engine.

Combines physiological Z-scores, workload spikes, and strain metrics to generate
probabilistic risk scores, explainable factor attributions, and clinical recommendations.
"""

from typing import List, Tuple

from athleteiq.core.models import FeatureVector, InjuryRiskAssessment, RiskTier
from athleteiq.models.base import BaseRiskPredictor


class SoftTissueInjuryRiskPredictor(BaseRiskPredictor):
    """Calibrated predictive engine for soft-tissue injury risk assessment."""

    def predict_risk(self, feature_vector: FeatureVector) -> InjuryRiskAssessment:
        """Evaluates feature vector to predict injury risk tier, probability, and factor attributions.

        Args:
            feature_vector: A populated `FeatureVector` DTO.

        Returns:
            An `InjuryRiskAssessment` DTO.
        """
        raw_score, factors = self._calculate_risk_components(feature_vector)
        calibrated_score = round(max(0.0, min(1.0, raw_score)), 3)
        risk_tier = self._classify_risk_tier(calibrated_score)
        recommended_action = self._generate_recommendation(risk_tier, factors)

        return InjuryRiskAssessment(
            athlete_id=feature_vector.athlete_id,
            assessment_date=feature_vector.feature_date,
            risk_tier=risk_tier,
            risk_score=calibrated_score,
            contributing_factors=factors,
            recommended_action=recommended_action,
        )

    def _calculate_risk_components(self, fv: FeatureVector) -> Tuple[float, List[str]]:
        """Calculates non-linear additive risk contributions and extracts explainability factors."""
        score = 0.15  # Baseline minimal ambient risk
        factors: List[str] = []

        # 1. ACWR EWMA evaluation
        if fv.acwr_ewma > 1.5:
            score += 0.35
            factors.append(f"Critical Workload Spike (EWMA ACWR = {fv.acwr_ewma})")
        elif fv.acwr_ewma > 1.3:
            score += 0.20
            factors.append(f"Elevated Training Load (EWMA ACWR = {fv.acwr_ewma})")
        elif fv.acwr_ewma < 0.8:
            score += 0.10
            factors.append(f"Under-training / Fitness Decay (EWMA ACWR = {fv.acwr_ewma})")

        # 2. HRV Suppression (Z-Score < -1.0)
        if fv.hrv_z_score < -1.5:
            score += 0.25
            factors.append(f"Severe HRV Suppression (Z-score = {fv.hrv_z_score})")
        elif fv.hrv_z_score < -1.0:
            score += 0.15
            factors.append(f"Moderate HRV Suppression (Z-score = {fv.hrv_z_score})")

        # 3. Resting HR Elevation (Z-Score > 1.0)
        if fv.rhr_z_score > 1.5:
            score += 0.20
            factors.append(f"Elevated Resting Heart Rate (Z-score = {fv.rhr_z_score})")
        elif fv.rhr_z_score > 1.0:
            score += 0.10
            factors.append(f"Mild RHR Elevation (Z-score = {fv.rhr_z_score})")

        # 4. Sleep Deficit Ratio
        if fv.sleep_deficit_ratio > 0.25:
            score += 0.15
            factors.append(f"Significant Sleep Deficit ({int(fv.sleep_deficit_ratio * 100)}% below target)")

        # 5. Monotony & Strain Interaction
        if fv.monotony > 2.0 and fv.strain > 3000.0:
            score += 0.15
            factors.append(f"High Training Monotony ({fv.monotony}) & Strain ({fv.strain})")

        return score, factors

    def _classify_risk_tier(self, score: float) -> RiskTier:
        """Maps continuous risk score [0.0, 1.0] to categorical RiskTier."""
        if score >= 0.80:
            return RiskTier.CRITICAL
        if score >= 0.55:
            return RiskTier.HIGH
        if score >= 0.30:
            return RiskTier.MODERATE
        return RiskTier.LOW

    def _generate_recommendation(self, tier: RiskTier, factors: List[str]) -> str:
        """Generates evidence-based clinical decision recommendations."""
        if tier == RiskTier.CRITICAL:
            return (
                "IMMEDIATE ACTION REQUIRED: Mandate 24-48h active recovery. "
                "Deload high-speed running volume by 50% and conduct physio screening for soft-tissue tightness."
            )
        if tier == RiskTier.HIGH:
            return (
                "HIGH RISK DETECTED: Cap high-intensity session duration to 45 mins. "
                "Reduce sprint volume by 35% and prescribe sleep extension protocol."
            )
        if tier == RiskTier.MODERATE:
            return (
                "MODERATE ELEVATION: Monitor athlete during high-speed running drills. "
                "Ensure post-session hydrotherapy or recovery modality."
            )
        return "LOW RISK: Athlete is cleared for full training load."
