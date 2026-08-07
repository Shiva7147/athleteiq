"""Abstract Base Classes for AthleteIQ Pro Machine Learning Models.

Implements Strategy pattern enabling seamless model switching and A/B evaluation.
"""

from abc import ABC, abstractmethod

from athleteiq.core.models import FeatureVector, InjuryRiskAssessment


class BaseRiskPredictor(ABC):
    """Abstract Base Class for injury risk prediction models."""

    @abstractmethod
    def predict_risk(self, feature_vector: FeatureVector) -> InjuryRiskAssessment:
        """Predicts soft-tissue injury risk tier and score from a feature vector.

        Args:
            feature_vector: Populated `FeatureVector` DTO.

        Returns:
            An `InjuryRiskAssessment` DTO containing risk tier, score, factors, and action.
        """
        pass
