"""Comprehensive Unit Tests for ACWR Engine (Phase 1)."""

from datetime import date, timedelta
import pytest

from src.core.acwr import (
    calculate_acwr,
    calculate_acute_load,
    calculate_chronic_load,
    calculate_ewma_acwr,
    calculate_ewma_load,
    calculate_uncoupled_acwr,
    classify_risk_zone,
    validate_workload_entry,
)
from src.core.exceptions import InsufficientDataError, InvalidWorkloadValueError
from src.core.models import ACWRResult, ACWRRiskZone, WorkloadEntry


def test_workload_entry_session_load() -> None:
    """Verifies sRPE load calculation on WorkloadEntry."""
    entry = WorkloadEntry(entry_date=date(2026, 8, 1), rpe=7.0, duration_minutes=60.0)
    assert entry.session_load == 420.0


def test_validate_workload_entry_valid() -> None:
    """Verifies that valid WorkloadEntry passes validation."""
    entry = WorkloadEntry(entry_date=date(2026, 8, 1), rpe=5.0, duration_minutes=45.0)
    validate_workload_entry(entry)


def test_validate_workload_entry_invalid_rpe() -> None:
    """Verifies that RPE > 10.0 raises InvalidWorkloadValueError."""
    entry = WorkloadEntry(entry_date=date(2026, 8, 1), rpe=12.0, duration_minutes=45.0)
    with pytest.raises(InvalidWorkloadValueError):
        validate_workload_entry(entry)


def test_validate_workload_entry_negative_duration() -> None:
    """Verifies that duration < 0 raises InvalidWorkloadValueError."""
    entry = WorkloadEntry(entry_date=date(2026, 8, 1), rpe=5.0, duration_minutes=-10.0)
    with pytest.raises(InvalidWorkloadValueError):
        validate_workload_entry(entry)


def test_calculate_acute_load() -> None:
    """Tests 7-day acute workload mean calculation."""
    loads = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0]
    assert calculate_acute_load(loads) == 400.0


def test_calculate_chronic_load() -> None:
    """Tests 28-day chronic workload mean calculation."""
    loads = [100.0] * 28
    assert calculate_chronic_load(loads) == 100.0


def test_calculate_uncoupled_acwr() -> None:
    """Tests uncoupled ACWR ratio calculation."""
    assert calculate_uncoupled_acwr(500.0, 400.0) == 1.25
    assert calculate_uncoupled_acwr(500.0, 0.0) == 0.0


def test_calculate_ewma_acwr_constant_load() -> None:
    """Tests EWMA ACWR for constant training load yields ratio ~ 1.0."""
    constant_loads = [300.0] * 35
    ewma_acwr = calculate_ewma_acwr(constant_loads)
    assert abs(ewma_acwr - 1.0) < 0.05


def test_calculate_ewma_acwr_load_spike() -> None:
    """Tests EWMA ACWR for load spike in acute window yields ratio > 1.3."""
    spiked_loads = ([200.0] * 21) + ([600.0] * 7)
    ewma_acwr = calculate_ewma_acwr(spiked_loads)
    assert ewma_acwr > 1.3


@pytest.mark.parametrize(
    "ratio, expected_zone",
    [
        (0.65, ACWRRiskZone.UNDER_TRAINING),
        (1.10, ACWRRiskZone.SWEET_SPOT),
        (1.40, ACWRRiskZone.ELEVATED_RISK),
        (1.65, ACWRRiskZone.DANGER_ZONE),
    ],
)
def test_classify_risk_zone(ratio: float, expected_zone: ACWRRiskZone) -> None:
    """Tests ACWR risk zone categorization for all thresholds."""
    assert classify_risk_zone(ratio) == expected_zone


def test_calculate_acwr_insufficient_data() -> None:
    """Verifies InsufficientDataError is raised when entries < 28."""
    entries = [
        WorkloadEntry(
            entry_date=date(2026, 8, 1) + timedelta(days=i),
            rpe=5.0,
            duration_minutes=60.0,
        )
        for i in range(20)
    ]
    with pytest.raises(InsufficientDataError):
        calculate_acwr("ATH-101", entries)


def test_calculate_acwr_success() -> None:
    """Tests end-to-end ACWR calculation with 30 valid entries."""
    entries = [
        WorkloadEntry(
            entry_date=date(2026, 7, 1) + timedelta(days=i),
            rpe=6.0,
            duration_minutes=60.0,
        )
        for i in range(30)
    ]
    result = calculate_acwr("ATH-101", entries)
    assert result.athlete_id == "ATH-101"
    assert result.acute_workload == 360.0  # 6 * 60
    assert result.chronic_workload == 360.0
    assert result.acwr_uncoupled == 1.0
    assert result.risk_zone == ACWRRiskZone.SWEET_SPOT
