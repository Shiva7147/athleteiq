"""AthleteIQ Pro Machine Learning Models Package."""

from athleteiq.models.base import BaseRiskPredictor
from athleteiq.models.injury_risk import SoftTissueInjuryRiskPredictor

__all__ = ["BaseRiskPredictor", "SoftTissueInjuryRiskPredictor"]
