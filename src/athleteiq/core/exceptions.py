"""Domain exceptions for AthleteIQ Pro.

Defines a clean, strongly typed exception hierarchy for sports science
telemetry validation, workload mathematical bounds, and model predictions.
"""

from typing import Any, Optional


class AthleteIQError(Exception):
    """Base exception for all domain-specific errors in AthleteIQ Pro.

    Inheriting from standard `Exception` ensures all custom errors caught in upstream
    API handlers or service layers stem from a unified root.
    """

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Context: {self.details}"
        return self.message


class InvalidTelemetryError(AthleteIQError):
    """Raised when incoming wearable or subjective telemetry violates physiological bounds."""

    pass


class InsufficientDataError(AthleteIQError):
    """Raised when workload analytics algorithms lack required historical window depth (e.g. < 28 days for chronic workload)."""

    pass


class BaselineValidationError(AthleteIQError):
    """Raised when athlete baseline metrics are statistically improbable or corrupt."""

    pass
