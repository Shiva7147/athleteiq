"""
tests/test_features_leakage.py
Rigorous tests verifying point-in-time correctness and absence of future data leakage.
"""
from datetime import date, timedelta
import pytest
import polars as pl
from features.pipeline import extract_features_as_of


def test_point_in_time_feature_leakage_guard():
    """
    CI GATE TEST:
    Verifies that inserting a future high-load session on Day D or D+1 has ZERO effect
    on the feature vector computed as of the morning of Day D.
    """
    player_id = "PLY_001"
    target_date = date(2026, 10, 15)

    # 1. Historical data strictly before Day D (Oct 10 - Oct 14)
    clean_sessions = pl.DataFrame({
        "player_id": [player_id] * 5,
        "session_date": [
            date(2026, 10, 10),
            date(2026, 10, 11),
            date(2026, 10, 12),
            date(2026, 10, 13),
            date(2026, 10, 14),
        ],
        "minutes": [75, 80, 0, 90, 60],
        "total_distance_m": [6000.0, 7000.0, 0.0, 8500.0, 5000.0],
        "high_speed_running_m": [400.0, 500.0, 0.0, 650.0, 300.0],
        "sprint_distance_m": [100.0, 120.0, 0.0, 180.0, 80.0],
        "rpe": [6.0, 7.0, 1.0, 8.5, 6.0],
    })

    # Empty wellness for simplicity
    wellness_df = pl.DataFrame({
        "player_id": pl.Series(dtype=pl.String),
        "record_date": pl.Series(dtype=pl.Date),
        "sleep_hours": pl.Series(dtype=pl.Float64),
        "fatigue_score": pl.Series(dtype=pl.Int64),
    })

    # Compute clean feature baseline
    clean_features = extract_features_as_of(
        clean_sessions, wellness_df, as_of_date=target_date, player_id=player_id
    )

    # 2. Add an extreme future session on target_date (Oct 15) and next day (Oct 16)
    poisoned_future_sessions = pl.DataFrame({
        "player_id": [player_id, player_id],
        "session_date": [date(2026, 10, 15), date(2026, 10, 16)],
        "minutes": [120, 120],
        "total_distance_m": [99999.0, 99999.0],  # Outlier
        "high_speed_running_m": [9999.0, 9999.0],
        "sprint_distance_m": [5000.0, 5000.0],
        "rpe": [10.0, 10.0],
    })

    contaminated_sessions = pl.concat([clean_sessions, poisoned_future_sessions])

    # Compute features from contaminated table
    tested_features = extract_features_as_of(
        contaminated_sessions, wellness_df, as_of_date=target_date, player_id=player_id
    )

    # 3. Assert exact equality: The pipeline MUST ignore future data
    assert clean_features["acute_load_7d_ewma"] == tested_features["acute_load_7d_ewma"], \
        "LEAKAGE FAIL: Future load on target_date leaked into acute EWMA!"
    assert clean_features["chronic_load_28d_ewma"] == tested_features["chronic_load_28d_ewma"], \
        "LEAKAGE FAIL: Future load leaked into chronic EWMA!"
    assert clean_features["acwr_ewma"] == tested_features["acwr_ewma"], \
        "LEAKAGE FAIL: ACWR skewed by future records!"
    assert clean_features["hsr_ratio_7d"] == tested_features["hsr_ratio_7d"], \
        "LEAKAGE FAIL: HSR ratio skewed by future records!"
    assert clean_features["sprint_exposure_7d"] == tested_features["sprint_exposure_7d"], \
        "LEAKAGE FAIL: Sprint exposure skewed by future records!"


def test_missing_data_imputation():
    """Verifies that missing days are imputed gracefully without crashes."""
    player_id = "PLY_002"
    target_date = date(2026, 10, 15)

    empty_sessions = pl.DataFrame({
        "player_id": pl.Series(dtype=pl.String),
        "session_date": pl.Series(dtype=pl.Date),
        "minutes": pl.Series(dtype=pl.Int64),
        "total_distance_m": pl.Series(dtype=pl.Float64),
        "high_speed_running_m": pl.Series(dtype=pl.Float64),
        "sprint_distance_m": pl.Series(dtype=pl.Float64),
        "rpe": pl.Series(dtype=pl.Float64),
    })
    empty_wellness = pl.DataFrame({
        "player_id": pl.Series(dtype=pl.String),
        "record_date": pl.Series(dtype=pl.Date),
        "sleep_hours": pl.Series(dtype=pl.Float64),
        "fatigue_score": pl.Series(dtype=pl.Int64),
    })

    features = extract_features_as_of(empty_sessions, empty_wellness, target_date, player_id)
    assert features["player_id"] == player_id
    assert features["acwr_ewma"] == 1.0
    assert features["mean_sleep_7d"] == 8.0
