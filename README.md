# AthleteIQ Pro 🏃‍♂️⚡

**AthleteIQ Pro** is a production-quality, enterprise-grade AI Decision Support Platform built for sports science, biometric telemetry analysis, feature engineering, and predictive injury risk mitigation.

---

## 🌟 Key Architecture & Features

- **Domain-Driven Core Architecture (`athleteiq.core`)**: Immutable Value Objects and Data Transfer Objects (`DailyTelemetry`, `AthleteBaseline`, `WorkloadMetrics`, `FeatureVector`, `InjuryRiskAssessment`, `RiskTier`).
- **Defensive Telemetry Validation (`athleteiq.data.validation`)**: Physiological telemetry data contracts enforcing valid RPE, RHR, HRV, sleep, and GPS distance ranges.
- **Sports Science Workload Engine (`athleteiq.analytics.workload`)**:
  - sRPE Session Load ($RPE \times Duration$).
  - Uncoupled ACWR (7-day Acute / 28-day Chronic ratio).
  - Exponentially Weighted Moving Average (EWMA) ACWR using time-decay parameter $\lambda = \frac{2}{N+1}$.
  - Foster's Monotony and Strain Index.
- **High-Performance Feature Engineering Pipeline (`athleteiq.features`)**:
  - Biomechanical Z-score normalization ($Z_{HRV}$, $Z_{RHR}$).
  - Sleep Deficit Ratio and High-Speed Running Intensity Ratio.
  - EWMA ACWR Spike Delta ($EWMA\_ACWR - Uncoupled\_ACWR$).
- **Calibrated Injury Risk Predictor & XAI Engine (`athleteiq.models`)**:
  - Bounded probabilistic soft-tissue injury risk model ($[0.0, 1.0]$).
  - Explainable AI (XAI) factor attribution pinpointing exact stress drivers.
  - Actionable clinical recommendations for high-performance coaches and trainers.
- **Athlete Data Repository Layer (`athleteiq.data.repository`)**: Thread-safe in-memory repository for daily telemetry history streams and automatic baseline computation.
- **100% Pytest Coverage**: 21 unit tests passing in 0.11s.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+

### Installation & Test Execution

```bash
# Clone the repository
git clone https://github.com/Shiva7147/athleteiq.git
cd athleteiq

# Run the complete test suite
python -m pytest tests/ -v
```

---

## 🧪 Test Suite Execution

```powershell
pytest tests/ -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.4.2, pluggy-1.5.0
configfile: pyproject.toml
collected 21 items

tests\test_features.py ..                                                [  9%]
tests\test_models.py ...                                                 [ 23%]
tests\test_repository.py ....                                            [ 42%]
tests\test_risk_model.py ..                                              [ 52%]
tests\test_validation.py ...                                             [ 66%]
tests\test_workload.py .......                                           [100%]

============================= 21 passed in 0.11s ==============================
```

---

## 📁 Repository Structure

```
athleteiq/
├── src/
│   └── athleteiq/
│       ├── core/
│       │   ├── exceptions.py       # Domain exception hierarchy
│       │   └── models.py           # Telemetry, Baseline, Workload, FeatureVector & Risk DTOs
│       ├── data/
│       │   ├── validation.py       # Telemetry data validator
│       │   └── repository.py       # Repository interface & thread-safe InMemory implementation
│       ├── features/
│       │   ├── base.py             # Abstract BaseFeatureExtractor interface
│       │   └── biomechanical.py    # Z-score & workload feature extractor
│       ├── models/
│       │   ├── base.py             # Abstract BaseRiskPredictor strategy interface
│       │   └── injury_risk.py      # Calibrated soft-tissue injury risk predictor & XAI
│       └── analytics/
│           └── workload.py         # sRPE, ACWR, EWMA, Monotony, & Strain algorithms
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_repository.py
│   ├── test_risk_model.py
│   ├── test_validation.py
│   └── test_workload.py
├── pyproject.toml
└── README.md
```
