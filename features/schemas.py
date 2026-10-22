"""
features/schemas.py
Pandera validation schemas for raw ingestion tables and engineered feature tables.
Prevents bad units, impossible physical values, and corrupted data from entering the ML engine.
"""
import pandera.polars as pa
import polars as pl


class RawSessionSchema(pa.DataFrameModel):
    """
    Contract for raw GPS and training session sensor data.
    """
    player_id: str = pa.Field(str_matches=r"^PLY_\d{3}$")
    session_date: pl.Date
    minutes: int = pa.Field(ge=0, le=130)
    total_distance_m: float = pa.Field(ge=0.0, le=20000.0)
    high_speed_running_m: float = pa.Field(ge=0.0, le=3500.0)
    sprint_distance_m: float = pa.Field(ge=0.0, le=1500.0)
    rpe: float = pa.Field(ge=0.0, le=10.0)


class DailyPlayerFeaturesSchema(pa.DataFrameModel):
    """
    Contract for engineered point-in-time features ingested by the ML model ladder.
    """
    player_id: str = pa.Field(str_matches=r"^PLY_\d{3}$")
    as_of_date: pl.Date
    acute_load_7d_ewma: float = pa.Field(ge=0.0)
    chronic_load_28d_ewma: float = pa.Field(ge=0.0)
    acwr_ewma: float = pa.Field(ge=0.0, le=5.0)
    training_monotony_7d: float = pa.Field(ge=0.0, le=10.0)
    training_strain_7d: float = pa.Field(ge=0.0)
    hsr_ratio_7d: float = pa.Field(ge=0.0)
    sprint_exposure_7d: float = pa.Field(ge=0.0)
    days_since_last_match: int = pa.Field(ge=0, le=365)
    mean_sleep_7d: float = pa.Field(ge=0.0, le=16.0)
    fatigue_index_7d: float = pa.Field(ge=1.0, le=5.0)
