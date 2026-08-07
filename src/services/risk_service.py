"""Calibrated Soft-Tissue Injury Risk Service.

Evaluates physiological Z-scores, workload spikes, and strain metrics to output
calibrated risk probabilities, explainable factor attributions, and clinical recommendations.
"""

from typing import List, Optional, Tuple
from src.models import InjuryRiskAssessmentRead, RiskTier, TelemetryRead
from src.services.workload_service import WorkloadService


class RiskService:
    """Domain service for calibrated soft-tissue injury risk assessment."""

    def __init__(self, workload_service: Optional[WorkloadService] = None) -> None:
        self.workload_service = workload_service or WorkloadService()

    def evaluate_risk(
        self,
        telemetry_history: List[TelemetryRead],
        target_sleep_hours: float = 8.0,
    ) -> InjuryRiskAssessmentRead:
        """Evaluates telemetry history and returns an InjuryRiskAssessmentRead DTO.

        Args:
            telemetry_history: Chronological list of `TelemetryRead` entries (min 28 days).
            target_sleep_hours: Clinical sleep target baseline in hours.

        Returns:
            `InjuryRiskAssessmentRead` DTO.
        """
        sorted_history = sorted(telemetry_history, key=lambda t: t.recorded_date)
        latest = sorted_history[-1]
        workload = self.workload_service.compute_summary(sorted_history)

        # 28-day baseline stats for HRV and RHR
        rhrs = [t.hr_rest_bpm for t in sorted_history[-28:]]
        hrvs = [t.hrv_rmssd_ms for t in sorted_history[-28:]]

        mean_rhr = sum(rhrs) / len(rhrs)
        mean_hrv = sum(hrvs) / len(hrvs)

        var_rhr = sum((x - mean_rhr) ** 2 for x in rhrs) / len(rhrs)
        var_hrv = sum((x - mean_hrv) ** 2 for x in hrvs) / len(hrvs)

        std_rhr = (var_rhr ** 0.5) if var_rhr > 1e-6 else 1.0
        std_hrv = (var_hrv ** 0.5) if var_hrv > 1e-6 else 1.0

        rhr_z_score = (latest.hr_rest_bpm - mean_rhr) / std_rhr
        hrv_z_score = (latest.hrv_rmssd_ms - mean_hrv) / std_hrv

        sleep_deficit_ratio = max(0.0, (target_sleep_hours - latest.sleep_hours) / target_sleep_hours)

        score = 0.15
        factors: List[str] = []

        # 1. ACWR EWMA contribution
        if workload.acwr_ewma > 1.5:
            score += 0.35
            factors.append(f"Critical Workload Spike (EWMA ACWR = {workload.acwr_ewma})")
        elif workload.acwr_ewma > 1.3:
            score += 0.20
            factors.append(f"Elevated Workload Ratio (EWMA ACWR = {workload.acwr_ewma})")
        elif workload.acwr_ewma < 0.8:
            score += 0.10
            factors.append(f"Under-training / Fitness Decay (EWMA ACWR = {workload.acwr_ewma})")

        # 2. HRV Suppression
        if hrv_z_score < -1.5:
            score += 0.25
            factors.append(f"Severe HRV Suppression (Z-score = {round(hrv_z_score, 2)})")
        elif hrv_z_score < -1.0:
            score += 0.15
            factors.append(f"Moderate HRV Suppression (Z-score = {round(hrv_z_score, 2)})")

        # 3. Resting HR Elevation
        if rhr_z_score > 1.5:
            score += 0.20
            factors.append(f"Elevated Resting Heart Rate (Z-score = {round(rhr_z_score, 2)})")
        elif rhr_z_score > 1.0:
            score += 0.10
            factors.append(f"Mild RHR Elevation (Z-score = {round(rhr_z_score, 2)})")

        # 4. Sleep Deficit
        if sleep_deficit_ratio > 0.25:
            score += 0.15
            factors.append(f"Significant Sleep Deficit ({int(sleep_deficit_ratio * 100)}% below target)")

        # 5. Monotony & Strain
        if workload.monotony > 2.0 and workload.strain > 3000.0:
            score += 0.15
            factors.append(f"High Monotony ({workload.monotony}) & Strain ({workload.strain})")

        calibrated_score = round(max(0.0, min(1.0, score)), 3)

        if calibrated_score >= 0.80:
            risk_tier = RiskTier.CRITICAL
            recommended_action = (
                "CRITICAL INJURY RISK: Mandate 24-48h active recovery. "
                "Reduce high-speed running duration by 50% and perform physiotherapy tightness screening."
            )
        elif calibrated_score >= 0.55:
            risk_tier = RiskTier.HIGH
            recommended_action = (
                "HIGH RISK: Cap training duration to 45 mins. "
                "Reduce sprint load by 35% and prescribe sleep extension protocol."
            )
        elif calibrated_score >= 0.30:
            risk_tier = RiskTier.MODERATE
            recommended_action = (
                "MODERATE RISK: Monitor during high-intensity drills. "
                "Prescribe post-session active recovery modalities."
            )
        else:
            risk_tier = RiskTier.LOW
            recommended_action = "LOW RISK: Athlete cleared for full planned training load."

        return InjuryRiskAssessmentRead(
            athlete_id=latest.athlete_id,
            assessment_date=latest.recorded_date,
            risk_tier=risk_tier,
            risk_score=calibrated_score,
            contributing_factors=factors,
            recommended_action=recommended_action,
        )
