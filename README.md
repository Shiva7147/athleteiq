# AthleteIQ Pro 🏃‍♂️⚡

**AthleteIQ Pro** is a production-quality, enterprise-grade AI Decision Support Platform built for sports science, biometric telemetry analysis, and injury risk mitigation.

---

## 🌟 Architecture & Key Features

- **Domain-Driven Architecture**: Decoupled core domain models, immutable value objects, and standard domain exception hierarchy.
- **Defensive Telemetry Validation**: Enterprise-grade data validation enforcing physiological contracts on wearable inputs (HRV, RHR, RPE, sleep, distance).
- **Sports Science Workload Engine**:
  - **sRPE Session Load**: Session Rate of Perceived Exertion ($RPE \times Duration$).
  - **Uncoupled ACWR**: 7-day acute to 28-day chronic rolling workload ratio.
  - **Exponentially Weighted Moving Average (EWMA) ACWR**: Uses decay constant $\lambda = \frac{2}{N+1}$ to weight recent training load higher.
  - **Monotony & Strain Index**: Foster's method calculating training variation and cumulative strain.
- **100% Test Coverage**: Full suite of unit tests with `pytest`.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+

### Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Shiva7147/athleteiq.git
cd athleteiq

# Run test suite
python -m pytest tests/ -v
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📁 Repository Structure

```
athleteiq/
├── src/
│   └── athleteiq/
│       ├── core/
│       │   ├── exceptions.py       # Domain exception hierarchy
│       │   └── models.py           # Telemetry, Workload, and Risk DTOs
│       ├── analytics/
│       │   └── workload.py         # ACWR, EWMA, Monotony, & Strain calculation engine
│       └── data/
│           └── validation.py       # Physiological telemetry data validator
├── tests/
│   ├── test_models.py
│   ├── test_validation.py
│   └── test_workload.py
├── pyproject.toml                  # Setuptools & pytest packaging configuration
└── README.md                       # Documentation
```
