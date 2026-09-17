"""Household battery optimization (V2 model).

Minimizes household net cost over a horizon with 15-minute (or other)
timesteps, separate buy/sell prices, and a degradation cost on discharge.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import pulp


def optimize_household_battery(
    buy_price: Sequence[float],
    sell_price: Sequence[float],
    load_kw: Sequence[float],
    capacity: float,
    initial_energy: float,
    min_energy: float,
    max_energy: float,
    max_charge_power: float,
    max_discharge_power: float,
    eta_charge: float,
    eta_discharge: float,
    dt: float,
    degradation_cost: float = 0.0,
) -> pd.DataFrame:
    """Optimize battery charge/discharge against household load and prices.

    Units:
        buy_price, sell_price, degradation_cost: SEK/kWh
        load_kw, charge/discharge/grid power: kW
        capacity, energy: kWh
        dt: hours (0.25 for 15-minute intervals)

    Power balance (each timestep):
        grid_import + battery_discharge
        = household_load + battery_charge + grid_export

    Battery energy:
        E[t+1] = E[t] + eta_charge * charge[t] * dt
                 - discharge[t] * dt / eta_discharge

    Terminal energy is constrained to equal initial_energy.

    Returns a DataFrame with one row per timestep, including costs in SEK.
    """
    T = len(buy_price)

    if not (len(sell_price) == len(load_kw) == T):
        raise ValueError("buy_price, sell_price and load_kw must have same length")

    if capacity <= 0:
        raise ValueError("capacity must be > 0")
    if not (0 < eta_charge <= 1):
        raise ValueError("eta_charge must be in (0, 1]")
    if not (0 < eta_discharge <= 1):
        raise ValueError("eta_discharge must be in (0, 1]")
    if min_energy < 0:
        raise ValueError("min_energy must be >= 0")
    if max_energy > capacity:
        raise ValueError("max_energy must be <= capacity")
    if not (min_energy <= initial_energy <= max_energy):
        raise ValueError("initial_energy must be within [min_energy, max_energy]")

    model = pulp.LpProblem(
        "Household_Battery_Optimizer",
        pulp.LpMinimize,
    )

    charge = pulp.LpVariable.dicts(
        "charge",
        range(T),
        lowBound=0,
        upBound=max_charge_power,
    )

    discharge = pulp.LpVariable.dicts(
        "discharge",
        range(T),
        lowBound=0,
        upBound=max_discharge_power,
    )

    energy = pulp.LpVariable.dicts(
        "energy",
        range(T + 1),
        lowBound=min_energy,
        upBound=max_energy,
    )

    grid_import = pulp.LpVariable.dicts(
        "grid_import",
        range(T),
        lowBound=0,
    )

    grid_export = pulp.LpVariable.dicts(
        "grid_export",
        range(T),
        lowBound=0,
    )

    # Objective: minimize import cost - export revenue + degradation cost
    model += pulp.lpSum(
        (
            buy_price[t] * grid_import[t]
            - sell_price[t] * grid_export[t]
            + degradation_cost * discharge[t]
        )
        * dt
        for t in range(T)
    )

    model += energy[0] == initial_energy

    for t in range(T):
        model += (
            energy[t + 1]
            == energy[t]
            + eta_charge * charge[t] * dt
            - discharge[t] * dt / eta_discharge
        )

        model += (
            grid_import[t] + discharge[t]
            == load_kw[t] + charge[t] + grid_export[t]
        )

    model += energy[T] == initial_energy

    model.solve(pulp.GLPK_CMD(msg=False))

    if pulp.LpStatus[model.status] != "Optimal":
        raise RuntimeError(f"Solver status: {pulp.LpStatus[model.status]}")

    rows = []

    cumulative_grid_cost = 0.0
    cumulative_export_revenue = 0.0
    cumulative_degradation_cost = 0.0
    cumulative_net_cost = 0.0

    for t in range(T):
        charge_t = charge[t].value()
        discharge_t = discharge[t].value()
        energy_t = energy[t].value()
        energy_after_t = energy[t + 1].value()

        grid_import_t = grid_import[t].value()
        grid_export_t = grid_export[t].value()

        grid_cost_t = buy_price[t] * grid_import_t * dt
        export_revenue_t = sell_price[t] * grid_export_t * dt
        degradation_cost_t = degradation_cost * discharge_t * dt

        net_cost_t = grid_cost_t - export_revenue_t + degradation_cost_t

        cumulative_grid_cost += grid_cost_t
        cumulative_export_revenue += export_revenue_t
        cumulative_degradation_cost += degradation_cost_t
        cumulative_net_cost += net_cost_t

        if charge_t > 1e-6:
            action = "charge"
        elif discharge_t > 1e-6:
            action = "discharge"
        else:
            action = "hold"

        rows.append(
            {
                "time": t,
                "buy_price": buy_price[t],
                "sell_price": sell_price[t],
                "load_kw": load_kw[t],
                "grid_import_kw": grid_import_t,
                "grid_export_kw": grid_export_t,
                "charge_kw": charge_t,
                "discharge_kw": discharge_t,
                "energy_kwh": energy_t,
                "energy_after_kwh": energy_after_t,
                "soc": energy_t / capacity,
                "soc_after": energy_after_t / capacity,
                "action": action,
                "grid_cost_sek": grid_cost_t,
                "export_revenue_sek": export_revenue_t,
                "degradation_cost_sek": degradation_cost_t,
                "net_cost_sek": net_cost_t,
                "cumulative_grid_cost_sek": cumulative_grid_cost,
                "cumulative_export_revenue_sek": cumulative_export_revenue,
                "cumulative_degradation_cost_sek": cumulative_degradation_cost,
                "cumulative_net_cost_sek": cumulative_net_cost,
            }
        )

    return pd.DataFrame(rows)
