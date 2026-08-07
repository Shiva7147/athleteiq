"""Workload Domain Orchestrator Service.

Coordinates telemetry repository histories with deterministic analytics calculators
to generate comprehensive WorkloadSummaryRead DTOs.
"""

from typing import List
from src.analytics.acwr import (
    calculate_acute_load,
    calculate_chronic_load,
    calculate_ewma_acwr,
    calculate_uncoupled_acwr,
    classify_risk_zone,
)
from src.analytics.strain import calculate_monotony, calculate_strain
from src.models import TelemetryRead, WorkloadSummaryRead
from src.utils.exceptions import InsufficientDataError


class WorkloadService:
    """Domain service for generating deterministic athlete workload analytics summaries."""

    def compute_summary(self, telemetry_history: List[TelemetryRead]) -> WorkloadSummaryRead:
        """Computes deterministic workload summary from telemetry records.

        Args:
            telemetry_history: Chronological list of `TelemetryRead` entries (min 28 records).

        Returns:
            `WorkloadSummaryRead` DTO.

        Raises:
            InsufficientDataError: If history contains fewer than 28 days.
        """
        if len(telemetry_history) < 28:
            raise InsufficientDataError(
                f"Workload analytics require at least 28 days of telemetry history. Received {len(telemetry_history)}.",
                details={"received_records": len(telemetry_history)},
            )

        sorted_history = sorted(telemetry_history, key=lambda t: t.recorded_date)
        daily_loads = [t.session_load for t in sorted_history]
        recent_7d = daily_loads[-7:]

        acute = calculate_acute_load(daily_loads)
        chronic = calculate_chronic_load(daily_loads)
        acwr_uncoupled = calculate_uncoupled_acwr(acute, chronic)
        acwr_ewma = calculate_ewma_acwr(daily_loads)
        monotony = calculate_monotony(recent_7d)
        strain = calculate_strain(recent_7d)
        risk_zone = classify_risk_zone(acwr_ewma)

        latest = sorted_history[-1]

        return WorkloadSummaryRead(
            athlete_id=latest.athlete_id,
            evaluation_date=latest.recorded_date,
            acute_workload_7d=round(acute, 2),
            chronic_workload_28d=round(chronic, 2),
            acwr_uncoupled=acwr_uncoupled,
            acwr_ewma=acwr_ewma,
            monotony=monotony,
            strain=strain,
            risk_zone=risk_zone,
        )
