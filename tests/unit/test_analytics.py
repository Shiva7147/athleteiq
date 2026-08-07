"""Unit tests for deterministic analytics calculations (sRPE, ACWR, EWMA, Strain)."""

import pytest
from src.analytics import (
    calculate_acute_load,
    calculate_chronic_load,
    calculate_ewma_acwr,
    calculate_monotony,
    calculate_srpe_load,
    calculate_strain,
    calculate_uncoupled_acwr,
    classify_risk_zone,
)
from src.models import ACWRRiskZone
from src.utils.exceptions import InvalidTelemetryError


def test_calculate_srpe_load() -> None:
    """Verifies sRPE calculation formula."""
    assert calculate_srpe_load(7.0, 90.0) == 630.0


def test_calculate_srpe_invalid_rpe_raises_error() -> None:
    """Verifies that RPE > 10 raises InvalidTelemetryError."""
    with pytest.raises(InvalidTelemetryError):
        calculate_srpe_load(11.0, 60.0)


def test_calculate_acute_and_chronic_load() -> None:
    """Verifies acute and chronic rolling averages."""
    loads_30d = [400.0] * 30
    assert calculate_acute_load(loads_30d) == 400.0
    assert calculate_chronic_load(loads_30d) == 400.0


def test_calculate_uncoupled_and_ewma_acwr() -> None:
    """Verifies uncoupled and EWMA ACWR calculations."""
    constant_loads = [300.0] * 30
    assert calculate_uncoupled_acwr(300.0, 300.0) == 1.0
    assert abs(calculate_ewma_acwr(constant_loads) - 1.0) < 0.05


@pytest.mark.parametrize(
    "acwr, expected_zone",
    [
        (0.6, ACWRRiskZone.UNDER_TRAINING),
        (1.1, ACWRRiskZone.SWEET_SPOT),
        (1.4, ACWRRiskZone.ELEVATED_RISK),
        (1.6, ACWRRiskZone.DANGER_ZONE),
    ],
)
def test_classify_risk_zone(acwr: float, expected_zone: ACWRRiskZone) -> None:
    """Verifies ACWR risk zone threshold mapping."""
    assert classify_risk_zone(acwr) == expected_zone
