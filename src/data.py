"""Price and load inputs for the prototype.

Price data is fetched from elprisetjustnu.se, the same source as the notebook.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray

ELPRISET_BASE_URL = "https://www.elprisetjustnu.se/api/v1/prices"
VALID_PRICE_AREAS = ("SE1", "SE2", "SE3", "SE4")


def fetch_se3_prices(
    day: date | str = "2026-09-14",
    area: str = "SE3",
) -> pd.DataFrame:
    """Fetch day-ahead prices and return timestamps plus SEK/kWh.

    The public API path is:
        {base}/{YYYY}/{MM-DD}_{area}.json

    Default day matches the notebook (2026-09-14, SE3).
    area must be one of SE1, SE2, SE3, SE4.
    """
    if area not in VALID_PRICE_AREAS:
        raise ValueError(f"area must be one of {VALID_PRICE_AREAS}, got {area!r}")

    if isinstance(day, str):
        day = datetime.strptime(day, "%Y-%m-%d").date()

    url = f"{ELPRISET_BASE_URL}/{day.year}/{day:%m-%d}_{area}.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.DataFrame(response.json())
    required_columns = {"time_start", "time_end", "SEK_per_kWh"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"API response missing required columns: {sorted(missing)}")

    df["time_start"] = pd.to_datetime(df["time_start"])
    df["time_end"] = pd.to_datetime(df["time_end"])

    return df[["time_start", "time_end", "SEK_per_kWh"]].copy()


def synthetic_household_load(
    n_intervals: int = 96,
    dt: float = 0.25,
) -> NDArray[np.float64]:
    """Synthetic household load in kW (same profile as the notebook).

    Base load plus morning and evening peaks.
    """
    if n_intervals <= 0:
        raise ValueError("n_intervals must be > 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    hours = np.arange(n_intervals) * dt
    return (
        0.6
        + 1.2 * np.exp(-((hours - 7.5) / 1.5) ** 2)
        + 2.0 * np.exp(-((hours - 18.5) / 2.0) ** 2)
    )
