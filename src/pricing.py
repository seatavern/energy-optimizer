"""Retail buy/sell prices from day-ahead spot plus configurable adders."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def build_prices(
    spot_price: Sequence[float],
    supplier_adder: float = 0.0875,
    energy_tax: float = 0.36,
    variable_grid_fee: float = 0.26,
    export_compensation: float = 0.04,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build household buy and sell prices in SEK/kWh.

    buy_price = spot + supplier_adder + energy_tax + variable_grid_fee
    sell_price = spot + export_compensation

    Default adders are illustrative V2 assumptions from the notebook
    prototype. They must remain configurable; pass explicit values for a
    real contract. All adders are SEK/kWh.
    """
    spot = np.asarray(spot_price, dtype=float)
    if spot.size == 0:
        raise ValueError("spot_price must not be empty")

    buy_price = spot + supplier_adder + energy_tax + variable_grid_fee
    sell_price = spot + export_compensation

    return buy_price, sell_price
