"""Core Exception Definitions for ACWR Engine.

Defines custom exception hierarchy for workload calculations and input validation.
"""

from typing import Any, Optional


class ACWRError(Exception):
    """Base exception for all errors encountered within the ACWR engine.

    Ensures upstream caller layers can catch all engine failures cleanly.
    """

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Context: {self.details}"
        return self.message


class InsufficientDataError(ACWRError):
    """Raised when calculation demands a historical window depth greater than provided."""

    pass


class InvalidWorkloadValueError(ACWRError):
    """Raised when an RPE or duration value violates physiological boundaries."""

    pass
