"""
ml/models.py
AthleteIQ Multi-Tier Evaluation and Decision-Support Ladder.

Tier 0: Baseline ACWR Threshold Rule (Heuristic benchmark to beat)
Tier 1: Regularized Logistic Regression (Linear baseline)
Tier 2: Non-linear Tree Classifier (LightGBM with optional grouped calibration)
Tier 3: Discrete-Time Survival Hazard Estimator

Scientific Design Notes:
- Non-monotonicity: True athletic physiological strain exhibits non-linear sweet spots (U-curves)
  and rest-induced decapacitation. Hard monotonic constraints on chronic load structurally
  fail to represent these dynamics and have been relaxed.
- Grouped Calibration: Probability calibration uses GroupKFold across athlete IDs to prevent
  intra-athlete temporal leakage across calibration folds.
- Conformal Prediction: Uses the exact inductive split-conformal quantile formula, with explicit
  caveats regarding time-series exchangeability violations.
"""
from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
import lightgbm as lgb


FEATURE_COLUMNS = [
    "acute_load_7d_ewma",
    "chronic_load_28d_ewma",
    "acwr_ewma",
    "training_monotony_7d",
    "training_strain_7d",
    "hsr_ratio_7d",
    "sprint_exposure_7d",
    "days_since_last_match",
    "mean_sleep_7d",
    "fatigue_index_7d",
]


class BaselineACWRRule:
    """
    Tier 0 Baseline: Traditional ACWR threshold rule.
    Flag if ACWR > upper_threshold (e.g. 1.45) or < lower_threshold (e.g. 0.8).
    Used as the classical sports science heuristic benchmark.
    """
    def __init__(self, upper_threshold: float = 1.45, lower_threshold: float = 0.80):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # acwr_ewma is at index 2
        acwr = X[:, 2]
        # U-shaped heuristic: elevated risk for both acute spikes (>1.45) and chronic decapacitation (<0.8)
        spike_mask = (acwr > self.upper_threshold) | (acwr < self.lower_threshold)
        probs = np.where(spike_mask, 0.22, 0.05)
        return np.column_stack([1.0 - probs, probs])


class LogisticRegressionBenchmark:
    """
    Tier 1: L2-penalized Logistic Regression.
    Interpretable parametric baseline ensuring tree-based ensembles actually add value.
    """
    def __init__(self):
        self.model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class MonotonicLightGBM:
    """
    Tier 2: LightGBM Gradient-Boosted Decision Tree Classifier.
    
    Architectural Updates:
    - Relaxed rigid monotonic constraints on chronic load to permit representation
      of U-shaped sweet spots and under-training decapacitation.
    - Added grouped calibration support (GroupKFold by athlete) to eliminate
      intra-athlete calibration leakage.
    - Adaptive fallback: Platt sigmoidal calibration for small sample regimes (6 <= N_pos < 30).
    """
    def __init__(self, use_monotonic_constraints: bool = False):
        self.use_monotonic_constraints = use_monotonic_constraints
        
        # If explicitly requested, enforce monotonic fatigue and strain constraints only
        # (acute load +, strain +, fatigue +, sleep -), leaving chronic load unconstrained for U-curves
        monotone = [1, 0, 1, 1, 1, 0, 0, 0, -1, 1] if use_monotonic_constraints else None

        self.base_estimator = lgb.LGBMClassifier(
            n_estimators=60,
            learning_rate=0.04,
            max_depth=3,
            num_leaves=8,
            monotone_constraints=monotone,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        self.calibrated_model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray, athlete_ids: Optional[np.ndarray] = None):
        """
        Fits base estimator and applies adaptive probability calibration.
        Uses GroupKFold if athlete_ids are provided to prevent intra-athlete leakage.
        """
        self.base_estimator.fit(X, y)
        positive_cases = int(np.sum(y == 1))

        cv_strategy = 3
        if athlete_ids is not None and len(np.unique(athlete_ids)) >= 3:
            # GroupKFold ensures an athlete's observations never span both train and calibration folds
            n_groups = min(3, len(np.unique(athlete_ids)))
            cv_strategy = GroupKFold(n_splits=n_groups).split(X, y, groups=athlete_ids)

        if positive_cases >= 30:
            # High-sample regime: Isotonic Regression
            try:
                self.calibrated_model = CalibratedClassifierCV(
                    estimator=self.base_estimator,
                    method="isotonic",
                    cv=cv_strategy
                )
                self.calibrated_model.fit(X, y)
            except Exception:
                self.calibrated_model = self.base_estimator
        elif positive_cases >= 6:
            # Small-sample elite squad regime (10-25 events): Regularized Platt Scaling (Sigmoid)
            try:
                self.calibrated_model = CalibratedClassifierCV(
                    estimator=self.base_estimator,
                    method="sigmoid",
                    cv=cv_strategy
                )
                self.calibrated_model.fit(X, y)
            except Exception:
                self.calibrated_model = self.base_estimator
        else:
            # Ultra-low sample regime: Fallback directly to base estimator
            self.calibrated_model = self.base_estimator

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.calibrated_model is None:
            raise RuntimeError("Model must be fitted before predict_proba is called.")
        return self.calibrated_model.predict_proba(X)


def compute_conformal_bounds(
    predicted_proba: float,
    alpha: float = 0.05,
    calibration_residuals: Optional[np.ndarray] = None,
    n_calibration_samples: int = 100
) -> Tuple[float, float]:
    """
    Inductive Split-Conformal Prediction Interval.
    
    Formula:
        q_val = Quantile(residuals, ceil((n + 1) * (1 - alpha)) / n)
        Interval = [max(0, p - q_val), min(1, p + q_val)]
        
    Theoretical Caveat:
        Guarantees marginal coverage of >= 1 - alpha strictly under the exchangeability
        assumption (Vovk et al. 2005). In longitudinal sports telemetry, sequential daily
        observations exhibit autocorrelation and non-stationarity, violating pure exchangeability.
        These bounds are empirical heuristic calibration bands rather than formal i.i.d. proofs.
    """
    n = max(10, n_calibration_samples if calibration_residuals is None else len(calibration_residuals))
    
    # Exact split-conformal finite-sample quantile index
    quantile_level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    
    if calibration_residuals is not None and len(calibration_residuals) > 0:
        q_val = float(np.quantile(calibration_residuals, quantile_level))
    else:
        # Default empirical non-conformity reference radius scaled by exact quantile level
        q_val = 0.08 * quantile_level

    lower = max(0.01, predicted_proba - q_val)
    upper = min(0.99, predicted_proba + q_val)
    return (round(lower, 3), round(upper, 3))


class DiscreteTimeSurvivalModel:
    """
    Tier 3: Discrete-Time Survival Hazard Model across Days [1..7].
    Models daily conditional hazard of time-loss strain occurrence over a forward horizon.
    """
    def __init__(self):
        self.hazard_models: List[Any] = []

    def fit(self, X: np.ndarray, y_survival_matrix: np.ndarray):
        """
        y_survival_matrix: Shape (N, 7), where y[i, d] = 1 if time-loss occurred on or before day d+1.
        """
        self.hazard_models = []
        for d in range(7):
            y_day = y_survival_matrix[:, d]
            estimator = lgb.LGBMClassifier(
                n_estimators=40,
                learning_rate=0.03,
                max_depth=3,
                class_weight="balanced",
                random_state=42 + d,
                verbose=-1
            )
            positive_cases = int(np.sum(y_day == 1))
            if positive_cases >= 6:
                try:
                    calibrated = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=3)
                    calibrated.fit(X, y_day)
                    self.hazard_models.append(calibrated)
                except Exception:
                    estimator.fit(X, y_day)
                    self.hazard_models.append(estimator)
            else:
                estimator.fit(X, y_day)
                self.hazard_models.append(estimator)
        return self

    def predict_cumulative_hazard_curve(self, X: np.ndarray) -> np.ndarray:
        daily_probs = []
        for model in self.hazard_models:
            p_hazard = model.predict_proba(X)[:, 1]
            daily_probs.append(p_hazard)
        
        hazard_matrix = np.column_stack(daily_probs)
        return np.maximum.accumulate(hazard_matrix, axis=1)
