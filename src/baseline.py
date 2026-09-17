"""Baseline (no-battery) cost and savings versus the optimized schedule."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def baseline_household_cost(
    buy_price: Sequence[float],
    load_kw: Sequence[float],
    dt: float,
) -> np.ndarray:
    """Per-timestep household cost with no battery, in SEK.

    Assumes the full load is imported from the grid:
        cost[t] = buy_price[t] * load_kw[t] * dt
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    buy = np.asarray(buy_price, dtype=float)
    load = np.asarray(load_kw, dtype=float)
    if buy.shape != load.shape:
        raise ValueError("buy_price and load_kw must have the same length")
    return buy * load * dt


def add_cumulative_baseline_and_savings(
    result: pd.DataFrame,
    dt: float,
) -> pd.DataFrame:
    """Add baseline cost, cumulative baseline cost, and cumulative savings.

    Savings are baseline cost minus optimized net cost (including degradation).
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    out = result.copy()
    out["baseline_cost_sek"] = baseline_household_cost(
        out["buy_price"],
        out["load_kw"],
        dt,
    )
    out["cumulative_baseline_cost_sek"] = out["baseline_cost_sek"].cumsum()
    out["cumulative_savings_sek"] = (
        out["cumulative_baseline_cost_sek"] - out["cumulative_net_cost_sek"]
    )
    return out


def summarize_results(result: pd.DataFrame, dt: float) -> dict[str, float]:
    """Summary metrics for one optimized day (or horizon).

    total_battery_discharge_kwh is discharged energy (AC-side discharge * dt),
    matching the quantity used for degradation cost.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if result.empty:
        raise ValueError("result is empty; cannot compute summary metrics")

    if "cumulative_baseline_cost_sek" not in result.columns:
        result = add_cumulative_baseline_and_savings(result, dt)

    return {
        "total_cost_without_battery_sek": float(
            result["cumulative_baseline_cost_sek"].iloc[-1]
        ),
        "total_cost_with_optimized_battery_sek": float(
            result["cumulative_net_cost_sek"].iloc[-1]
        ),
        "total_savings_sek": float(result["cumulative_savings_sek"].iloc[-1]),
        "total_grid_import_kwh": float((result["grid_import_kw"] * dt).sum()),
        "total_grid_export_kwh": float((result["grid_export_kw"] * dt).sum()),
        "total_battery_discharge_kwh": float((result["discharge_kw"] * dt).sum()),
    }
