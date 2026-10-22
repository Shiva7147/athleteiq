"""
features/pipeline.py
High-performance Polars feature engineering pipeline with point-in-time guarantees.
Calculates acute/chronic EWMA, monotony, strain, exposure ratios, and recovery metrics.
"""
from datetime import date, timedelta
from typing import Optional
import polars as pl
from core.maths import (
    compute_ewma_series,
    compute_acwr,
    compute_training_monotony,
    compute_training_strain,
)


def extract_features_as_of(
    sessions_df: pl.DataFrame,
    wellness_df: pl.DataFrame,
    as_of_date: date,
    player_id: str,
) -> dict:
    """
    Computes a complete feature vector for a player strictly using data prior to as_of_date.
    
    CRITICAL RULE:
    Rows where session_date >= as_of_date are EXCLUDED.
    This guarantees zero lookahead leakage.
    """
    # 1. Enforce Point-in-Time Temporal Boundary
    history_sessions = sessions_df.filter(
        (pl.col("player_id") == player_id) & (pl.col("session_date") < as_of_date)
    ).sort("session_date")

    history_wellness = wellness_df.filter(
        (pl.col("player_id") == player_id) & (pl.col("record_date") < as_of_date)
    ).sort("record_date")

    if history_sessions.is_empty():
        # Baseline cold-start defaults if no history exists
        return {
            "player_id": player_id,
            "as_of_date": as_of_date,
            "acute_load_7d_ewma": 0.0,
            "chronic_load_28d_ewma": 0.0,
            "acwr_ewma": 1.0,
            "training_monotony_7d": 1.0,
            "training_strain_7d": 0.0,
            "hsr_ratio_7d": 0.0,
            "sprint_exposure_7d": 0.0,
            "days_since_last_match": 14,
            "mean_sleep_7d": 8.0,
            "fatigue_index_7d": 3.0,
        }

    # 2. Build Daily Time Series grid for previous 28 days to prevent gaps
    start_date = as_of_date - timedelta(days=28)
    date_grid = pl.date_range(start_date, as_of_date - timedelta(days=1), interval="1d", eager=True).alias("session_date")
    grid_df = pl.DataFrame({"session_date": date_grid})

    player_series = grid_df.join(history_sessions, on="session_date", how="left").with_columns([
        pl.col("total_distance_m").fill_null(0.0),
        pl.col("high_speed_running_m").fill_null(0.0),
        pl.col("sprint_distance_m").fill_null(0.0),
        pl.col("rpe").fill_null(0.0),
        pl.col("minutes").fill_null(0),
    ])

    # Workload proxy = RPE * minutes (or distance if RPE is 0)
    workload = (player_series["rpe"] * player_series["minutes"]).to_numpy()
    if (workload == 0).all():
        workload = (player_series["total_distance_m"] / 100.0).to_numpy()

    # 3. Calculate EWMA Series
    acute_series = compute_ewma_series(workload, time_decay_days=7)
    chronic_series = compute_ewma_series(workload, time_decay_days=28)

    acute_val = float(acute_series[-1]) if len(acute_series) > 0 else 0.0
    chronic_val = float(chronic_series[-1]) if len(chronic_series) > 0 else 0.0
    acwr_val = compute_acwr(acute_val, chronic_val)

    # 4. Monotony & Strain across last 7 days
    last_7d_loads = workload[-7:] if len(workload) >= 7 else list(workload) + [0.0] * (7 - len(workload))
    monotony = compute_training_monotony(last_7d_loads)
    strain = compute_training_strain(last_7d_loads)

    # 5. High-speed running & Sprint exposure
    hsr_7d = float(player_series["high_speed_running_m"][-7:].sum())
    sprint_7d = float(player_series["sprint_distance_m"][-7:].sum())
    total_dist_7d = float(player_series["total_distance_m"][-7:].sum())
    hsr_ratio = (hsr_7d / total_dist_7d) if total_dist_7d > 0 else 0.0

    # 6. Days since last match
    matches = history_sessions.filter(pl.col("minutes") > 45)
    if not matches.is_empty():
        last_match_date = matches["session_date"][-1]
        days_since_match = (as_of_date - last_match_date).days
    else:
        days_since_match = 14

    # 7. Wellness & Autonomic Recovery Aggregation (7d)
    recent_wellness = history_wellness.filter(pl.col("record_date") >= as_of_date - timedelta(days=7))
    if not recent_wellness.is_empty():
        mean_sleep = float(recent_wellness["sleep_hours"].mean())
        fatigue_idx = float(recent_wellness["fatigue_score"].mean())
    else:
        mean_sleep = 8.0
        fatigue_idx = 3.0

    return {
        "player_id": player_id,
        "as_of_date": as_of_date,
        "acute_load_7d_ewma": round(acute_val, 2),
        "chronic_load_28d_ewma": round(chronic_val, 2),
        "acwr_ewma": round(acwr_val, 3),
        "training_monotony_7d": round(monotony, 2),
        "training_strain_7d": round(strain, 2),
        "hsr_ratio_7d": round(hsr_ratio, 4),
        "sprint_exposure_7d": round(sprint_7d, 1),
        "days_since_last_match": days_since_match,
        "mean_sleep_7d": round(mean_sleep, 2),
        "fatigue_index_7d": round(fatigue_idx, 2),
    }


def build_squad_feature_matrix(
    sessions_df: pl.DataFrame,
    wellness_df: pl.DataFrame,
    as_of_date: date,
    squad_ids: list[str],
) -> pl.DataFrame:
    """
    Builds the daily squad feature matrix ready for ML inference and ranking.
    """
    rows = []
    for pid in squad_ids:
        f_vec = extract_features_as_of(sessions_df, wellness_df, as_of_date, pid)
        rows.append(f_vec)
    return pl.DataFrame(rows)
