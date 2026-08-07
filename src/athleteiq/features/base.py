"""Abstract Base Classes for AthleteIQ Pro Feature Engineering Pipelines.

Enforces SOLID Open/Closed principle allowing modular extension of feature extractors.
"""

from abc import ABC, abstractmethod
from typing import List

from athleteiq.core.models import AthleteBaseline, DailyTelemetry, FeatureVector


class BaseFeatureExtractor(ABC):
    """Abstract Base Class for feature engineering transformers."""

    @abstractmethod
    def extract_features(
        self,
        telemetry_history: List[DailyTelemetry],
        baseline: AthleteBaseline,
    ) -> FeatureVector:
        """Transforms daily telemetry history and baseline stats into a normalized FeatureVector.

        Args:
            telemetry_history: List of daily telemetry records ordered chronologically.
            baseline: The athlete's baseline physiological statistics.

        Returns:
            A populated `FeatureVector` DTO.
        """
        pass
