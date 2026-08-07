"""Unit tests for Biomechanical Feature Extractor engine."""

from datetime import date, timedelta
import pytest

from athleteiq.core.exceptions import FeatureExtractionError
from athleteiq.core.models import AthleteBaseline, DailyTelemetry
from athleteiq.features.biomechanical import BiomechanicalFeatureExtractor


def test_feature_extractor_success() -> None:
    """Tests successful extraction of normalized FeatureVector from 30 days of history."""
    extractor = BiomechanicalFeatureExtractor()
    history = [
        DailyTelemetry(
            athlete_id="ATH-001",
            recorded_date=date(2026, 7, 1) + timedelta(days=i),
            hr_rest_bpm=52 + (i % 2),
            hrv_rmssd_ms=65.0 - (i % 3),
            sleep_hours=7.5,
            rpe_score=6.0,
            session_duration_minutes=60.0,
            total_distance_meters=8000.0,
            high_speed_running_meters=800.0,
        )
        for i in range(30)
    ]
    baseline = AthleteBaseline(
        athlete_id="ATH-001",
        mean_hr_rest_bpm=52.5,
        std_hr_rest_bpm=2.0,
        mean_hrv_rmssd_ms=64.0,
        std_hrv_rmssd_ms=4.0,
        target_sleep_hours=8.0,
    )

    fv = extractor.extract_features(history, baseline)
    assert fv.athlete_id == "ATH-001"
    assert fv.sleep_deficit_ratio == 0.062  # (8.0 - 7.5)/8.0
    assert fv.hsr_ratio == 0.1  # 800 / 8000
    assert isinstance(fv.hrv_z_score, float)
    assert isinstance(fv.rhr_z_score, float)


def test_feature_extractor_insufficient_history_raises_error() -> None:
    """Verifies that fewer than 28 records raises FeatureExtractionError."""
    extractor = BiomechanicalFeatureExtractor()
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
        for i in range(10)
    ]
    baseline = AthleteBaseline(
        athlete_id="ATH-001",
        mean_hr_rest_bpm=55.0,
        std_hr_rest_bpm=2.0,
        mean_hrv_rmssd_ms=60.0,
        std_hrv_rmssd_ms=3.0,
    )
    with pytest.raises(FeatureExtractionError):
        extractor.extract_features(short_history, baseline)
