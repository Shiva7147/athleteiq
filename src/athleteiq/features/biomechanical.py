"""Biomechanical and Physiological Feature Extractor Engine.

Extracts normalized Z-scores, workload spike deltas, sleep deficit ratios,
and high-speed running metrics for machine learning injury risk models.
"""

from typing import List

from athleteiq.analytics.workload import compute_workload_summary
from athleteiq.core.exceptions import FeatureExtractionError, InsufficientDataError
from athleteiq.core.models import AthleteBaseline, DailyTelemetry, FeatureVector
from athleteiq.features.base import BaseFeatureExtractor


class BiomechanicalFeatureExtractor(BaseFeatureExtractor):
    """Transforms raw telemetry and physiological baselines into normalized FeatureVectors."""

    def extract_features(
        self,
        telemetry_history: List[DailyTelemetry],
        baseline: AthleteBaseline,
    ) -> FeatureVector:
        """Transforms daily telemetry history and baseline stats into a FeatureVector.

        Args:
            telemetry_history: Historical list of `DailyTelemetry` records (at least 28 days).
            baseline: `AthleteBaseline` DTO containing mean and std dev metrics.

        Returns:
            A populated `FeatureVector` DTO.

        Raises:
            FeatureExtractionError: If inputs are invalid or historical depth is insufficient.
        """
        if len(telemetry_history) < 28:
            raise FeatureExtractionError(
                f"Feature extraction requires at least 28 days of history. Received {len(telemetry_history)}.",
                details={"record_count": len(telemetry_history)},
            )

        sorted_history = sorted(telemetry_history, key=lambda t: t.recorded_date)
        latest = sorted_history[-1]

        if latest.athlete_id != baseline.athlete_id:
            raise FeatureExtractionError(
                f"Athlete ID mismatch: telemetry '{latest.athlete_id}' vs baseline '{baseline.athlete_id}'.",
                details={"telemetry_id": latest.athlete_id, "baseline_id": baseline.athlete_id},
            )

        # 1. HRV Z-Score calculation
        hrv_z_score = 0.0
        if baseline.std_hrv_rmssd_ms > 1e-6:
            hrv_z_score = (latest.hrv_rmssd_ms - baseline.mean_hrv_rmssd_ms) / baseline.std_hrv_rmssd_ms

        # 2. Resting HR Z-Score calculation
        rhr_z_score = 0.0
        if baseline.std_hr_rest_bpm > 1e-6:
            rhr_z_score = (latest.hr_rest_bpm - baseline.mean_hr_rest_bpm) / baseline.std_hr_rest_bpm

        # 3. Sleep Deficit Ratio calculation
        sleep_deficit_ratio = 0.0
        if baseline.target_sleep_hours > 0.0:
            deficit = baseline.target_sleep_hours - latest.sleep_hours
            sleep_deficit_ratio = max(0.0, min(1.0, deficit / baseline.target_sleep_hours))

        # 4. HSR Ratio calculation
        hsr_ratio = 0.0
        if latest.total_distance_meters > 0.0:
            hsr_ratio = min(1.0, latest.high_speed_running_meters / latest.total_distance_meters)

        # 5. Workload Analytics Summary
        try:
            workload = compute_workload_summary(sorted_history)
        except InsufficientDataError as e:
            raise FeatureExtractionError(str(e.message), details=e.details) from e

        acwr_spike_delta = round(workload.acwr_ewma - workload.acwr_uncoupled, 3)

        return FeatureVector(
            athlete_id=latest.athlete_id,
            feature_date=latest.recorded_date,
            hrv_z_score=round(hrv_z_score, 3),
            rhr_z_score=round(rhr_z_score, 3),
            sleep_deficit_ratio=round(sleep_deficit_ratio, 3),
            acwr_ewma=workload.acwr_ewma,
            acwr_spike_delta=acwr_spike_delta,
            monotony=workload.monotony,
            strain=workload.strain,
            hsr_ratio=round(hsr_ratio, 3),
        )
