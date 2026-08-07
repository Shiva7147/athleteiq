"""Unified Domain Exception Hierarchy for AthleteIQ Pro.

Defines strongly-typed exceptions for data validation, database access,
deterministic analytics calculations, feature engineering, and predictive models.
"""

from typing import Any, Dict, Optional


class AthleteIQError(Exception):
    """Root base exception for all AthleteIQ Pro domain errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Context: {self.details}"
        return self.message


class InvalidTelemetryError(AthleteIQError):
    """Raised when incoming daily wearable telemetry violates physiological bounds."""

    pass


class InsufficientDataError(AthleteIQError):
    """Raised when analytics calculations lack required historical window depth (e.g. < 28 days)."""

    pass


class BaselineValidationError(AthleteIQError):
    """Raised when athlete baseline metrics are statistically invalid."""

    pass


class DatabaseError(AthleteIQError):
    """Raised when database query or transaction failure occurs."""

    pass


class AnalyticsError(AthleteIQError):
    """Raised when deterministic mathematical routines encounter calculation errors."""

    pass


class FeatureExtractionError(AthleteIQError):
    """Raised when feature engineering pipeline fails."""

    pass


class ModelPredictionError(AthleteIQError):
    """Raised when predictive risk engine encounters invalid feature inputs."""

    pass
