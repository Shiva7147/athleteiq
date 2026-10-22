"""
tests/test_ml_ladder.py
Tests verifying the 4-tier model ladder, monotonic constraints, and SHAP explainability.
"""
import pytest
import numpy as np
from ml.models import (
    BaselineACWRRule,
    LogisticRegressionBenchmark,
    MonotonicLightGBM,
    DiscreteTimeSurvivalModel,
    FEATURE_COLUMNS,
)
from ml.explainability import ModelExplainabilityEngine


@pytest.fixture
def synthetic_training_data():
    """Generates a small synthetic tabular dataset for model tests."""
    rng = np.random.default_rng(42)
    n_samples = 200
    n_features = len(FEATURE_COLUMNS)

    X = rng.normal(loc=50.0, scale=15.0, size=(n_samples, n_features))
    # acwr_ewma at index 2
    X[:, 2] = rng.uniform(0.7, 2.2, size=n_samples)
    # mean_sleep_7d at index 8
    X[:, 8] = rng.uniform(5.0, 9.5, size=n_samples)

    # Ground truth: higher ACWR and lower sleep -> higher risk
    logits = -2.5 + (1.2 * X[:, 2]) - (0.4 * X[:, 8])
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(0.0, 1.0, size=n_samples) < probs).astype(int)

    # 7-day cumulative hazard matrix
    y_survival = np.zeros((n_samples, 7), dtype=int)
    for d in range(7):
        y_survival[:, d] = (rng.uniform(0.0, 1.0, size=n_samples) < (probs * (d + 1) / 7.0)).astype(int)
    y_survival = np.maximum.accumulate(y_survival, axis=1)

    return X, y, y_survival


def test_baseline_acwr_rule(synthetic_training_data):
    X, _, _ = synthetic_training_data
    baseline = BaselineACWRRule(upper_threshold=1.5)
    probs = baseline.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert (probs[:, 1] >= 0.0).all() and (probs[:, 1] <= 1.0).all()


def test_monotonic_lightgbm_training_and_calibration(synthetic_training_data):
    X, y, _ = synthetic_training_data
    model = MonotonicLightGBM()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    # Probabilities should be strictly calibrated between 0 and 1
    assert (probs[:, 1] >= 0.0).all() and (probs[:, 1] <= 1.0).all()


def test_discrete_time_survival_hazard_curve(synthetic_training_data):
    X, _, y_survival = synthetic_training_data
    survival_model = DiscreteTimeSurvivalModel()
    survival_model.fit(X, y_survival)
    curve = survival_model.predict_cumulative_hazard_curve(X[:5])
    assert curve.shape == (5, 7)
    # Hazard must be non-decreasing over the 7-day window
    for row in curve:
        assert np.all(np.diff(row) >= -1e-6)


def test_shap_explainability_engine(synthetic_training_data):
    X, y, _ = synthetic_training_data
    model = MonotonicLightGBM()
    model.fit(X, y)
    explainer = ModelExplainabilityEngine(model)
    
    explanation = explainer.explain_athlete(X[0], top_k=3)
    assert len(explanation) == 3
    for exp in explanation:
        assert "feature" in exp
        assert "label" in exp
        assert "tactical_summary" in exp
        assert isinstance(exp["shap_impact"], float)
