"""Plotly Dash interface for the residential battery optimizer."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, ctx, dcc, html, no_update
from plotly.subplots import make_subplots

from src.baseline import add_cumulative_baseline_and_savings, summarize_results
from src.data import fetch_se3_prices, synthetic_household_load
from src.optimizer import optimize_household_battery
from src.pricing import build_prices

DT_HOURS = 0.25
PRICE_DAY = "2026-09-14"
FLOW_TOLERANCE = 1e-6

COLORS = {
    "background": "#0f172a",
    "panel": "#1e293b",
    "border": "#334155",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "blue": "#38bdf8",
    "green": "#34d399",
    "amber": "#fbbf24",
    "red": "#fb7185",
    "purple": "#a78bfa",
}

MESSAGE_STYLE = {
    "display": "none",
    "padding": "12px 16px",
    "marginBottom": "18px",
    "borderRadius": "8px",
}
ERROR_MESSAGE_STYLE = {
    **MESSAGE_STYLE,
    "display": "block",
    "backgroundColor": "#451a2b",
    "border": f"1px solid {COLORS['red']}",
    "color": "#fecdd3",
}
WARNING_MESSAGE_STYLE = {
    **MESSAGE_STYLE,
    "display": "block",
    "backgroundColor": "#422006",
    "border": f"1px solid {COLORS['amber']}",
    "color": "#fde68a",
}


def load_model_inputs() -> tuple[pd.DataFrame, Any, Any, Any]:
    """Load the fixed V2 price and household-load inputs once at startup."""
    prices = fetch_se3_prices(PRICE_DAY, area="SE3")
    buy_price, sell_price = build_prices(prices["SEK_per_kWh"].to_numpy())
    load_kw = synthetic_household_load(len(prices), dt=DT_HOURS)
    return prices, buy_price, sell_price, load_kw


try:
    PRICE_DATA, BUY_PRICE, SELL_PRICE, LOAD_KW = load_model_inputs()
    INPUT_ERROR = ""
except Exception as exc:  # Keep the UI available if the price API is unavailable.
    PRICE_DATA, BUY_PRICE, SELL_PRICE, LOAD_KW = pd.DataFrame(), [], [], []
    INPUT_ERROR = f"Could not load SE3 price data: {exc}"


def number_control(
    label: str,
    component_id: str,
    value: float,
    *,
    minimum: float,
    maximum: float,
    step: float,
    unit: str,
) -> html.Div:
    """Create one labelled numeric parameter input."""
    return html.Div(
        [
            html.Label(
                [label, html.Span(f" ({unit})", style={"color": COLORS["muted"]})],
                htmlFor=component_id,
                style={"display": "block", "marginBottom": "6px", "fontSize": "14px"},
            ),
            dcc.Input(
                id=component_id,
                type="number",
                value=value,
                min=minimum,
                max=maximum,
                step=step,
                style={
                    "width": "100%",
                    "boxSizing": "border-box",
                    "padding": "9px 10px",
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "6px",
                    "backgroundColor": COLORS["background"],
                    "color": COLORS["text"],
                },
            ),
        ]
    )


def kpi_card(label: str, component_id: str) -> html.Div:
    """Create one KPI card."""
    return html.Div(
        [
            html.Div(label, style={"color": COLORS["muted"], "fontSize": "13px"}),
            html.Div(
                "—",
                id=component_id,
                style={"fontSize": "25px", "fontWeight": "600", "marginTop": "5px"},
            ),
        ],
        style={
            "backgroundColor": COLORS["panel"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "8px",
            "padding": "16px",
        },
    )


def style_figure(figure: go.Figure, title: str, y_title: str) -> go.Figure:
    """Apply the shared compact dark chart styling."""
    figure.update_layout(
        title={"text": title, "font": {"size": 17}},
        template="plotly_dark",
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font={"color": COLORS["text"]},
        margin={"l": 68, "r": 24, "t": 55, "b": 64},
        height=330,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right"},
        xaxis={
            "title": None,
            "gridcolor": COLORS["border"],
            "tickformat": "%H:%M",
            "hoverformat": "%H:%M",
            "ticklabelstandoff": 16,
            "automargin": True,
        },
        yaxis={
            "title": {"text": y_title, "standoff": 12},
            "gridcolor": COLORS["border"],
            "ticklabelstandoff": 9,
            "automargin": True,
        },
    )
    return figure


def padded_y_range(
    *series: Any,
    include_zero: bool = False,
    top_padding: float = 0.12,
) -> list[float] | None:
    """Return a data-driven y-axis range with reliable visual headroom."""
    arrays = [np.asarray(values, dtype=float).reshape(-1) for values in series]
    if not arrays:
        return None

    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None

    data_min = float(values.min())
    data_max = float(values.max())
    range_min = min(data_min, 0.0) if include_zero else data_min
    range_max = max(data_max, 0.0) if include_zero else data_max
    span = range_max - range_min

    if span == 0:
        span = max(abs(range_min), abs(range_max), 1.0)

    top_headroom = max(top_padding * span, 0.08 * abs(range_max))
    lower = range_min if include_zero and range_min == 0 else range_min - 0.05 * span
    upper = range_max + top_headroom
    return [lower, upper]


def add_y_headroom(
    figure: go.Figure,
    *series: Any,
    include_zero: bool = False,
    top_padding: float = 0.12,
) -> None:
    """Apply a padded, data-driven y-axis range to a figure."""
    y_range = padded_y_range(
        *series,
        include_zero=include_zero,
        top_padding=top_padding,
    )
    if y_range is not None:
        figure.update_yaxes(range=y_range, autorange=False)


def empty_figure(title: str, message: str) -> go.Figure:
    """Create a chart placeholder for validation or data errors."""
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": COLORS["muted"]},
    )
    return style_figure(figure, title, "")


def serialize_result(result: pd.DataFrame) -> dict[str, Any]:
    """Serialize one solved schedule for reuse by visualization callbacks."""
    stored_result = result.copy()
    stored_result["time_start"] = [
        timestamp.isoformat() for timestamp in PRICE_DATA["time_start"]
    ]
    stored_result["time_end"] = [
        timestamp.isoformat() for timestamp in PRICE_DATA["time_end"]
    ]
    return {
        "records": stored_result.to_dict(orient="records"),
        "n_steps": len(stored_result),
    }


def clean_flow_value(value: float) -> float:
    """Remove insignificant solver noise without changing material values."""
    return 0.0 if abs(value) <= FLOW_TOLERANCE else float(value)


TRIANGLE_WIDTH = 540
TRIANGLE_HEIGHT = 520
TRIANGLE_HOUSE = (270, 78)
TRIANGLE_GRID = (100, 412)
TRIANGLE_BATTERY = (440, 412)
TRIANGLE_BASE_Y = round(0.824 * TRIANGLE_HEIGHT) - 8


def _inset_edge(
    start: tuple[float, float],
    end: tuple[float, float],
    start_inset: float,
    end_inset: float,
) -> tuple[float, float, float, float]:
    """Shorten an edge so it meets the node boxes instead of their centers."""
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    ux = (x2 - x1) / length
    uy = (y2 - y1) / length
    return (
        x1 + ux * start_inset,
        y1 + uy * start_inset,
        x2 - ux * end_inset,
        y2 - uy * end_inset,
    )


def _edge_style(
    start: tuple[float, float],
    end: tuple[float, float],
    start_inset: float,
    end_inset: float,
) -> dict[str, str]:
    """Place one triangle edge; geometry is constant across timesteps."""
    x1, y1, x2, y2 = _inset_edge(start, end, start_inset, end_inset)
    length = math.hypot(x2 - x1, y2 - y1)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return {
        "left": f"{x1}px",
        "top": f"{y1}px",
        "width": f"{length}px",
        "--edge-travel": f"{max(length - 8, 0):.2f}px",
        "transform": f"rotate({angle:.2f}deg)",
    }


def _label_style(
    start: tuple[float, float],
    end: tuple[float, float],
    start_inset: float,
    end_inset: float,
    label_side: float,
) -> dict[str, str]:
    """Place the kW label beside a triangle edge."""
    x1, y1, x2, y2 = _inset_edge(start, end, start_inset, end_inset)
    length = math.hypot(x2 - x1, y2 - y1)
    ux = (x2 - x1) / length
    uy = (y2 - y1) / length
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    return {
        "left": f"{mid_x - uy * label_side}px",
        "top": f"{mid_y + ux * label_side}px",
    }


def _edge_class(direction: str, active: bool) -> str:
    """CSS class that encodes flow direction; changing it restarts the pulse."""
    return f"triangle-edge {direction} {'active' if active else 'inactive'}"


def _label_class(active: bool) -> str:
    return "triangle-edge-label " + ("active" if active else "inactive")


def _keep_if_same(new_value: str, old_value: str | None) -> str | Any:
    """Leave a DOM property untouched so CSS animations keep running."""
    if (old_value or "") == (new_value or ""):
        return no_update
    return new_value


def static_triangle_edge(
    component_id: str,
    start: tuple[float, float],
    end: tuple[float, float],
    start_inset: float,
    end_inset: float,
) -> html.Div:
    """Persistent edge; only className updates when direction changes."""
    return html.Div(
        [
            html.Div(className="triangle-edge-track"),
            html.Div(className="triangle-edge-pulse"),
        ],
        id=component_id,
        className="triangle-edge forward inactive",
        style=_edge_style(start, end, start_inset, end_inset),
    )


def static_triangle_label(
    component_id: str,
    start: tuple[float, float],
    end: tuple[float, float],
    start_inset: float,
    end_inset: float,
    label_side: float,
) -> html.Div:
    return html.Div(
        [
            html.Span("\u00a0", id=f"{component_id}-dir", className="triangle-edge-dir"),
            html.Strong("0.00 kW", id=f"{component_id}-kw"),
        ],
        id=component_id,
        className="triangle-edge-label inactive",
        style=_label_style(start, end, start_inset, end_inset, label_side),
    )


def build_triangle_system() -> html.Div:
    """Triangle chrome that stays mounted so charge-line pulses can continue."""
    grid_battery_start = (TRIANGLE_GRID[0], TRIANGLE_BASE_Y)
    grid_battery_end = (TRIANGLE_BATTERY[0], TRIANGLE_BASE_Y)
    return html.Div(
        [
            html.Div("00:00", id="triangle-time", className="triangle-time"),
            html.Div(
                [
                    html.Div(className="triangle-fill"),
                    static_triangle_edge(
                        "triangle-edge-grid-house",
                        TRIANGLE_GRID,
                        TRIANGLE_HOUSE,
                        64,
                        48,
                    ),
                    static_triangle_label(
                        "triangle-label-grid-house",
                        TRIANGLE_GRID,
                        TRIANGLE_HOUSE,
                        64,
                        48,
                        -56,
                    ),
                    static_triangle_edge(
                        "triangle-edge-house-battery",
                        TRIANGLE_HOUSE,
                        TRIANGLE_BATTERY,
                        48,
                        64,
                    ),
                    static_triangle_label(
                        "triangle-label-house-battery",
                        TRIANGLE_HOUSE,
                        TRIANGLE_BATTERY,
                        48,
                        64,
                        -56,
                    ),
                    static_triangle_edge(
                        "triangle-edge-grid-battery",
                        grid_battery_start,
                        grid_battery_end,
                        68,
                        68,
                    ),
                    static_triangle_label(
                        "triangle-label-grid-battery",
                        grid_battery_start,
                        grid_battery_end,
                        68,
                        68,
                        28,
                    ),
                    html.Div(
                        [
                            html.Div("⌂", className="flow-node-icon house-icon"),
                            html.Div("HOUSE", className="flow-node-title"),
                            html.Div(
                                "0.00 kW load",
                                id="triangle-house-load",
                                className="flow-node-primary",
                            ),
                            html.Div(
                                "\u00a0",
                                id="triangle-house-split",
                                className="flow-node-split is-off",
                            ),
                        ],
                        id="triangle-node-house",
                        className="flow-node triangle-node triangle-house",
                    ),
                    html.Div(
                        [
                            html.Div("⚡", className="flow-node-icon"),
                            html.Div("GRID", className="flow-node-title"),
                            html.Div(
                                "Idle",
                                id="triangle-grid-primary",
                                className="flow-node-primary",
                            ),
                            html.Div(
                                "",
                                id="triangle-grid-secondary",
                                className="flow-node-subtitle",
                            ),
                        ],
                        id="triangle-node-grid",
                        className="flow-node triangle-node triangle-grid",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        id="battery-fill",
                                        style={"height": "0%"},
                                        className="battery-fill",
                                    ),
                                ],
                                id="battery-shell",
                                className="battery-shell",
                            ),
                            html.Div("BATTERY", className="flow-node-title"),
                            html.Div("0.0% SOC", id="battery-soc", className="battery-soc"),
                            html.Div(
                                "0.00 kWh",
                                id="battery-energy",
                                className="flow-node-subtitle",
                            ),
                            html.Div("\u00a0", id="battery-mode", className="battery-mode"),
                        ],
                        id="triangle-node-battery",
                        className="flow-node triangle-node triangle-battery battery-node",
                    ),
                ],
                className="triangle-scene",
                style={"width": f"{TRIANGLE_WIDTH}px", "height": f"{TRIANGLE_HEIGHT}px"},
            ),
        ],
        id="triangle-system",
        className="triangle-system",
        style={"height": f"{TRIANGLE_HEIGHT}px"},
    )


def _price_rank_sentence(current: float, prices: np.ndarray, kind: str) -> str | None:
    """Describe where the current price sits in today's solved day."""
    if prices.size == 0:
        return None
    upper_share = float(np.mean(prices >= current - 1e-12))
    lower_share = float(np.mean(prices <= current + 1e-12))
    label = "buy price" if kind == "buy" else "sell price"
    if upper_share <= 0.10:
        return f"Current {label} is in the upper 10% of today's prices."
    if upper_share <= 0.25:
        return f"Current {label} is in the upper 25% of today's prices."
    if lower_share <= 0.10:
        return f"Current {label} is in the lowest 10% of today's prices."
    if lower_share <= 0.25:
        return f"Current {label} is in the lowest 25% of today's prices."
    return None


def current_decision_badge(
    *,
    charge: float,
    battery_to_house: float,
    battery_to_grid: float,
) -> tuple[str, str]:
    """Return the dominant battery action for the current timestep."""
    actions = (
        ("CHARGING", charge),
        ("DISCHARGING TO HOUSE", battery_to_house),
        ("EXPORTING", battery_to_grid),
    )
    label, power = max(actions, key=lambda item: item[1])
    if power <= FLOW_TOLERANCE:
        return "HOLDING", "decision-badge holding"
    class_name = {
        "CHARGING": "decision-badge charging",
        "DISCHARGING TO HOUSE": "decision-badge discharging",
        "EXPORTING": "decision-badge exporting",
    }[label]
    return label, class_name


def describe_current_action(
    *,
    load: float,
    buy_price: float,
    sell_price: float,
    grid_to_battery: float,
    battery_to_grid: float,
    battery_to_house: float,
    charge: float,
    discharge: float,
    soc_percent: float,
    energy_kwh: float,
    buy_prices: np.ndarray,
    sell_prices: np.ndarray,
    later_discharge: float,
    later_export: float,
    later_max_buy: float | None,
) -> str:
    """Return a short deterministic explanation of the current schedule."""
    buy_rank = _price_rank_sentence(buy_price, buy_prices, "buy")
    sell_rank = _price_rank_sentence(sell_price, sell_prices, "sell")
    later_use = later_discharge > FLOW_TOLERANCE or later_export > FLOW_TOLERANCE
    later_higher_buy = (
        later_max_buy is not None and later_max_buy > buy_price + 1e-6
    )
    inexpensive = buy_rank is not None and "lowest" in buy_rank
    if buy_prices.size and buy_price <= float(np.median(buy_prices)):
        inexpensive = True

    badge_label, _badge_class = current_decision_badge(
        charge=charge,
        battery_to_house=battery_to_house,
        battery_to_grid=battery_to_grid,
    )

    if badge_label == "EXPORTING":
        first = (
            f"Battery is exporting {battery_to_grid:.2f} kW to the grid "
            f"at {sell_price:.2f} SEK/kWh."
        )
        extra = sell_rank or (
            "This is a relatively high sell price in the solved day."
            if sell_prices.size and sell_price >= float(np.median(sell_prices))
            else ""
        )
        return f"{first} {extra}".strip()

    if badge_label == "CHARGING":
        if inexpensive:
            first = (
                f"Charging {grid_to_battery:.2f} kW from the grid while "
                f"electricity is relatively inexpensive ({buy_price:.2f} SEK/kWh)."
            )
        else:
            first = (
                f"Charging {grid_to_battery:.2f} kW from the grid at "
                f"{buy_price:.2f} SEK/kWh."
            )
        if later_use or later_higher_buy:
            return first + " The stored energy is scheduled for later, more expensive periods."
        return f"{first} {buy_rank or ''}".strip()

    if badge_label == "DISCHARGING TO HOUSE":
        first = (
            f"Battery is supplying {battery_to_house:.2f} kW of the {load:.2f} kW "
            f"household load, reducing grid purchases while the buy price is "
            f"{buy_price:.2f} SEK/kWh."
        )
        extra = buy_rank or ""
        if battery_to_grid > FLOW_TOLERANCE:
            extra = (
                f"It is also exporting {battery_to_grid:.2f} kW to the grid. {extra}"
            ).strip()
        return f"{first} {extra}".strip()

    if charge <= FLOW_TOLERANCE and discharge <= FLOW_TOLERANCE:
        first = (
            f"Battery is holding at {soc_percent:.0f}% SOC. "
            f"Current buy price is {buy_price:.2f} SEK/kWh."
        )
        if later_use and energy_kwh > FLOW_TOLERANCE:
            return first + " The optimized schedule preserves the stored energy for later use."
        extra = buy_rank or (
            "Charging or discharging is not used at this timestep in the solved schedule."
        )
        return f"{first} {extra}".strip()

    return (
        f"The schedule is transferring {max(charge, discharge):.2f} kW "
        "through the battery at this timestep."
    )


def flow_detail(label: str, value: str) -> html.Div:
    """Create one compact current-timestep detail."""
    return html.Div(
        [
            html.Div(label, className="flow-detail-label"),
            html.Div(value, className="flow-detail-value"),
        ],
        className="flow-detail-card",
    )


def _now_marker(
    figure: go.Figure,
    *,
    timestamp: Any,
    value: float,
    color: str,
    row: int,
    visible: bool = True,
) -> None:
    """Add a current-timestep marker that stands out from the series line."""
    figure.add_scatter(
        x=[timestamp],
        y=[value],
        mode="markers",
        marker={
            "size": 11,
            "color": color,
            "line": {"width": 2, "color": "#f8fafc"},
        },
        showlegend=False,
        hoverinfo="skip",
        opacity=1 if visible else 0,
        cliponaxis=False,
        row=row,
        col=1,
    )


def build_day_context_figure(
    records: list[dict[str, Any]],
    timestep: int,
) -> go.Figure:
    """Build synchronized whole-day panels with a current-time cursor."""
    timestamps = pd.to_datetime([record["time_start"] for record in records])
    buy_price = np.asarray([record["buy_price"] for record in records], dtype=float)
    sell_price = np.asarray([record["sell_price"] for record in records], dtype=float)
    load_kw = np.asarray([record["load_kw"] for record in records], dtype=float)
    charge_kw = np.asarray([record["charge_kw"] for record in records], dtype=float)
    discharge_kw = np.asarray(
        [record["discharge_kw"] for record in records],
        dtype=float,
    )
    grid_import = np.asarray(
        [record["grid_import_kw"] for record in records],
        dtype=float,
    )
    grid_export = np.asarray(
        [record["grid_export_kw"] for record in records],
        dtype=float,
    )
    soc_percent = np.asarray([record["soc"] for record in records], dtype=float) * 100

    current_time = timestamps[timestep]
    charge_now = float(charge_kw[timestep])
    discharge_now = float(discharge_kw[timestep])
    import_now = float(grid_import[timestep])
    export_now = float(grid_export[timestep])

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.30, 0.40, 0.30],
        subplot_titles=(
            "Electricity prices",
            "Power  ·  load/charge +, discharge/export −",
            "Battery state",
        ),
    )

    figure.add_scatter(
        x=timestamps,
        y=buy_price,
        name="Buy price",
        line={"color": COLORS["blue"], "width": 2.2},
        row=1,
        col=1,
    )
    figure.add_scatter(
        x=timestamps,
        y=sell_price,
        name="Sell price",
        line={"color": COLORS["green"], "width": 2.2},
        row=1,
        col=1,
    )
    _now_marker(figure,
        timestamp=current_time,
        value=float(buy_price[timestep]),
        color=COLORS["blue"],
        row=1,
    )
    _now_marker(figure,
        timestamp=current_time,
        value=float(sell_price[timestep]),
        color=COLORS["green"],
        row=1,
    )

    figure.add_scatter(
        x=timestamps,
        y=load_kw,
        name="Household load",
        line={"color": COLORS["amber"], "width": 2.2},
        row=2,
        col=1,
    )
    figure.add_scatter(
        x=timestamps,
        y=charge_kw,
        name="Battery charge",
        line={"color": COLORS["green"], "width": 2.2},
        row=2,
        col=1,
    )
    figure.add_scatter(
        x=timestamps,
        y=-discharge_kw,
        name="Battery discharge",
        line={"color": COLORS["red"], "width": 2.2},
        row=2,
        col=1,
    )
    figure.add_scatter(
        x=timestamps,
        y=grid_import,
        name="Grid import",
        line={"color": "#7dd3fc", "width": 1.4, "dash": "dot"},
        opacity=0.7,
        row=2,
        col=1,
    )
    figure.add_scatter(
        x=timestamps,
        y=-grid_export,
        name="Grid export",
        line={"color": "#cbd5e1", "width": 1.4, "dash": "dot"},
        opacity=0.65,
        row=2,
        col=1,
    )
    _now_marker(figure,
        timestamp=current_time,
        value=float(load_kw[timestep]),
        color=COLORS["amber"],
        row=2,
    )
    _now_marker(figure,
        timestamp=current_time,
        value=charge_now,
        color=COLORS["green"],
        row=2,
        visible=charge_now > FLOW_TOLERANCE,
    )
    _now_marker(figure,
        timestamp=current_time,
        value=-discharge_now,
        color=COLORS["red"],
        row=2,
        visible=discharge_now > FLOW_TOLERANCE,
    )
    _now_marker(figure,
        timestamp=current_time,
        value=import_now,
        color="#7dd3fc",
        row=2,
        visible=import_now > FLOW_TOLERANCE,
    )
    _now_marker(figure,
        timestamp=current_time,
        value=-export_now,
        color="#cbd5e1",
        row=2,
        visible=export_now > FLOW_TOLERANCE,
    )

    figure.add_scatter(
        x=timestamps,
        y=soc_percent,
        name="Battery SOC",
        line={"color": COLORS["purple"], "width": 2.5},
        row=3,
        col=1,
    )
    _now_marker(figure,
        timestamp=current_time,
        value=float(soc_percent[timestep]),
        color=COLORS["purple"],
        row=3,
    )

    figure.add_shape(
        type="line",
        x0=current_time,
        x1=current_time,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"color": "#f8fafc", "width": 2, "dash": "dot"},
    )
    figure.add_annotation(
        x=current_time,
        y=1.02,
        xref="x",
        yref="paper",
        text=current_time.strftime("%H:%M"),
        showarrow=False,
        xanchor=_cursor_xanchor(timestep, len(records)),
        font={"color": "#f8fafc", "size": 13, "family": "Arial"},
        bgcolor=COLORS["background"],
        bordercolor=COLORS["blue"],
        borderpad=4,
    )

    price_range = padded_y_range(buy_price, sell_price, top_padding=0.16)
    power_range = padded_y_range(
        load_kw,
        charge_kw,
        -discharge_kw,
        grid_import,
        -grid_export,
        include_zero=True,
        top_padding=0.16,
    )
    figure.update_yaxes(
        title={"text": "SEK/kWh", "standoff": 12},
        range=price_range,
        gridcolor=COLORS["border"],
        automargin=False,
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title={"text": "kW", "standoff": 12},
        range=power_range,
        gridcolor=COLORS["border"],
        automargin=False,
        row=2,
        col=1,
    )
    figure.update_yaxes(
        title={"text": "SOC %", "standoff": 12},
        range=[-2, 104],
        gridcolor=COLORS["border"],
        automargin=False,
        row=3,
        col=1,
    )

    tick_indices = sorted({0, 24, 48, 72, len(records) - 1})
    tick_values = [timestamps[index] for index in tick_indices if index < len(records)]
    tick_text = [timestamp.strftime("%H:%M") for timestamp in tick_values]
    figure.update_xaxes(
        range=[timestamps[0], timestamps[-1]],
        gridcolor=COLORS["border"],
        showticklabels=False,
        automargin=False,
        row=1,
        col=1,
    )
    figure.update_xaxes(
        range=[timestamps[0], timestamps[-1]],
        gridcolor=COLORS["border"],
        showticklabels=False,
        automargin=False,
        row=2,
        col=1,
    )
    figure.update_xaxes(
        range=[timestamps[0], timestamps[-1]],
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_text,
        gridcolor=COLORS["border"],
        ticklabelstandoff=10,
        automargin=False,
        showticklabels=True,
        row=3,
        col=1,
    )

    figure.update_annotations(font={"color": COLORS["text"], "size": 13})
    figure.layout.annotations[3].font = {
        "color": "#f8fafc",
        "size": 13,
        "family": "Arial",
    }
    figure.update_layout(
        template="plotly_dark",
        height=520,
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font={"color": COLORS["text"]},
        margin={"l": 72, "r": 47, "t": 78, "b": 42},
        autosize=True,
        uirevision="day-context",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.16,
            "xanchor": "left",
            "font": {"size": 10},
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    return figure


def _cursor_xanchor(timestep: int, n_steps: int) -> str:
    """Keep the time label inside the plot near the start and end of the day."""
    if timestep <= 3:
        return "left"
    if timestep >= n_steps - 4:
        return "right"
    return "center"


def patch_day_context_cursor(
    records: list[dict[str, Any]],
    timestep: int,
) -> Patch:
    """Move context markers/cursor without resending the whole-day curves."""
    row = records[timestep]
    current_time = row["time_start"]
    charge = float(row["charge_kw"])
    discharge = float(row["discharge_kw"])
    grid_import = float(row["grid_import_kw"])
    grid_export = float(row["grid_export_kw"])
    marker_values = {
        2: (float(row["buy_price"]), True),
        3: (float(row["sell_price"]), True),
        9: (float(row["load_kw"]), True),
        10: (charge, charge > FLOW_TOLERANCE),
        11: (-discharge, discharge > FLOW_TOLERANCE),
        12: (grid_import, grid_import > FLOW_TOLERANCE),
        13: (-grid_export, grid_export > FLOW_TOLERANCE),
        15: (float(row["soc"]) * 100, True),
    }

    patched_figure = Patch()
    for trace_index, (marker_value, visible) in marker_values.items():
        patched_figure["data"][trace_index]["x"] = [current_time]
        patched_figure["data"][trace_index]["y"] = [marker_value]
        patched_figure["data"][trace_index]["opacity"] = 1 if visible else 0

    patched_figure["layout"]["shapes"][0]["x0"] = current_time
    patched_figure["layout"]["shapes"][0]["x1"] = current_time
    patched_figure["layout"]["annotations"][3]["x"] = current_time
    patched_figure["layout"]["annotations"][3]["text"] = pd.to_datetime(
        current_time
    ).strftime("%H:%M")
    return patched_figure


def build_flow_visualization(
    row: dict[str, Any],
    records: list[dict[str, Any]],
    timestep: int,
) -> dict[str, Any]:
    """Update the schematic from the stored schedule without remounting charge lines."""
    load = clean_flow_value(float(row["load_kw"]))
    grid_import = clean_flow_value(float(row["grid_import_kw"]))
    grid_export = clean_flow_value(float(row["grid_export_kw"]))
    charge = clean_flow_value(float(row["charge_kw"]))
    discharge = clean_flow_value(float(row["discharge_kw"]))

    grid_to_battery = clean_flow_value(min(grid_import, charge))
    battery_to_grid = clean_flow_value(min(grid_export, discharge))
    grid_to_house = clean_flow_value(grid_import - grid_to_battery)
    battery_to_house = clean_flow_value(discharge - battery_to_grid)

    soc_percent = max(0.0, min(100.0, float(row["soc"]) * 100))
    energy_kwh = float(row["energy_kwh"])
    current_time = pd.to_datetime(row["time_start"]).strftime("%H:%M")
    buy_price = float(row["buy_price"])
    sell_price = float(row["sell_price"])
    buy_prices = np.asarray([float(item["buy_price"]) for item in records], dtype=float)
    sell_prices = np.asarray([float(item["sell_price"]) for item in records], dtype=float)
    later = records[timestep + 1 :]
    later_discharge = sum(clean_flow_value(float(item["discharge_kw"])) for item in later)
    later_export = sum(clean_flow_value(float(item["grid_export_kw"])) for item in later)
    later_max_buy = (
        max(float(item["buy_price"]) for item in later) if later else None
    )

    if grid_to_house > FLOW_TOLERANCE:
        grid_house_label = "Grid → House"
        grid_house_value = grid_to_house
        grid_house_direction = "forward"
    else:
        grid_house_label = "Grid ↔ House"
        grid_house_value = 0.0
        grid_house_direction = "forward"

    if battery_to_house > FLOW_TOLERANCE:
        house_battery_label = "Battery → House"
        house_battery_value = battery_to_house
        house_battery_direction = "reverse"
    else:
        house_battery_label = "House ↔ Battery"
        house_battery_value = 0.0
        house_battery_direction = "forward"

    if grid_to_battery > FLOW_TOLERANCE:
        grid_battery_label = "Grid → Battery"
        grid_battery_value = grid_to_battery
        grid_battery_direction = "forward"
    elif battery_to_grid > FLOW_TOLERANCE:
        grid_battery_label = "Battery → Grid"
        grid_battery_value = battery_to_grid
        grid_battery_direction = "reverse"
    else:
        grid_battery_label = "Grid ↔ Battery"
        grid_battery_value = 0.0
        grid_battery_direction = "forward"

    house_split = (
        f"Grid {grid_to_house:.2f}  ·  Battery {battery_to_house:.2f}"
        if battery_to_house > FLOW_TOLERANCE
        else "\u00a0"
    )
    house_split_class = (
        "flow-node-split"
        if battery_to_house > FLOW_TOLERANCE
        else "flow-node-split is-off"
    )

    if grid_export > FLOW_TOLERANCE:
        grid_primary = f"Exporting {grid_export:.2f} kW"
        grid_secondary = (
            f"{grid_import:.2f} kW import" if grid_import > FLOW_TOLERANCE else ""
        )
    elif grid_import > FLOW_TOLERANCE:
        grid_primary = f"Importing {grid_import:.2f} kW"
        grid_secondary = ""
    else:
        grid_primary = "Idle"
        grid_secondary = ""

    battery_mode = ""
    battery_shell_class = "battery-shell"
    if charge > FLOW_TOLERANCE:
        battery_mode = "Charging"
        battery_shell_class += " charging"
    elif discharge > FLOW_TOLERANCE:
        battery_mode = "Discharging"
        battery_shell_class += " discharging"

    details = [
        flow_detail("Current time", current_time),
        flow_detail("Household load", f"{load:.2f} kW"),
        flow_detail("Buy price", f"{buy_price:.3f} SEK/kWh"),
        flow_detail("Sell price", f"{sell_price:.3f} SEK/kWh"),
        flow_detail("Grid import", f"{grid_import:.2f} kW"),
        flow_detail("Grid export", f"{grid_export:.2f} kW"),
        flow_detail("Battery charge", f"{charge:.2f} kW"),
        flow_detail("Battery discharge", f"{discharge:.2f} kW"),
        flow_detail("Battery SOC", f"{soc_percent:.1f}%"),
        flow_detail("Battery energy", f"{energy_kwh:.2f} kWh"),
    ]

    explanation = describe_current_action(
        load=load,
        buy_price=buy_price,
        sell_price=sell_price,
        grid_to_battery=grid_to_battery,
        battery_to_grid=battery_to_grid,
        battery_to_house=battery_to_house,
        charge=charge,
        discharge=discharge,
        soc_percent=soc_percent,
        energy_kwh=energy_kwh,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        later_discharge=later_discharge,
        later_export=later_export,
        later_max_buy=later_max_buy,
    )
    badge_label, badge_class = current_decision_badge(
        charge=charge,
        battery_to_house=battery_to_house,
        battery_to_grid=battery_to_grid,
    )

    aggregate_error = grid_import + discharge - load - charge - grid_export
    decomposition_errors = (
        grid_import - grid_to_house - grid_to_battery,
        discharge - battery_to_house - battery_to_grid,
        load - grid_to_house - battery_to_house,
        charge - grid_to_battery,
        grid_export - battery_to_grid,
    )
    balance_error = max(abs(aggregate_error), *(abs(error) for error in decomposition_errors))
    invalid_negative_flow = min(
        grid_to_house,
        grid_to_battery,
        battery_to_house,
        battery_to_grid,
    ) < -FLOW_TOLERANCE

    if balance_error > 1e-5 or invalid_negative_flow:
        balance_message = (
            f"Balance warning at {current_time}: maximum error {balance_error:.3g} kW."
        )
        balance_class = "flow-balance flow-balance-warning"
    else:
        balance_message = "Balance OK"
        balance_class = "flow-balance flow-balance-ok"

    return {
        "triangle_time": current_time,
        "grid_house_edge_class": _edge_class(
            grid_house_direction, grid_house_value > FLOW_TOLERANCE
        ),
        "house_battery_edge_class": _edge_class(
            house_battery_direction, house_battery_value > FLOW_TOLERANCE
        ),
        "grid_battery_edge_class": _edge_class(
            grid_battery_direction, grid_battery_value > FLOW_TOLERANCE
        ),
        "grid_house_label_class": _label_class(grid_house_value > FLOW_TOLERANCE),
        "house_battery_label_class": _label_class(house_battery_value > FLOW_TOLERANCE),
        "grid_battery_label_class": _label_class(grid_battery_value > FLOW_TOLERANCE),
        "grid_house_dir": grid_house_label,
        "grid_house_kw": f"{grid_house_value:.2f} kW",
        "house_battery_dir": house_battery_label,
        "house_battery_kw": f"{house_battery_value:.2f} kW",
        "grid_battery_dir": grid_battery_label,
        "grid_battery_kw": f"{grid_battery_value:.2f} kW",
        "house_load": f"{load:.2f} kW load",
        "house_split": house_split,
        "house_split_class": house_split_class,
        "grid_primary": grid_primary,
        "grid_secondary": grid_secondary,
        "battery_shell_class": battery_shell_class,
        "soc_percent": round(soc_percent, 2),
        "battery_soc": f"{soc_percent:.1f}% SOC",
        "battery_energy": f"{energy_kwh:.2f} kWh",
        "battery_mode": battery_mode or "\u00a0",
        "details": details,
        "explanation": explanation,
        "badge_label": badge_label,
        "badge_class": badge_class,
        "balance_message": balance_message,
        "balance_class": balance_class,
    }


app = Dash(__name__)
app.title = "Residential Battery Optimizer"

overview_content = html.Div(
    [
        html.Div(
            [
                html.H2("Battery parameters", style={"fontSize": "18px", "marginTop": 0}),
                html.Div(
                    [
                        number_control(
                            "Battery capacity", "capacity", 4.0,
                            minimum=0.1, maximum=50, step=0.1, unit="kWh",
                        ),
                        number_control(
                            "Initial SOC", "initial-soc", 50,
                            minimum=0, maximum=100, step=1, unit="%",
                        ),
                        number_control(
                            "Minimum SOC", "minimum-soc", 0,
                            minimum=0, maximum=100, step=1, unit="%",
                        ),
                        number_control(
                            "Maximum SOC", "maximum-soc", 100,
                            minimum=0, maximum=100, step=1, unit="%",
                        ),
                        number_control(
                            "Maximum charge power", "charge-power", 2.0,
                            minimum=0, maximum=20, step=0.1, unit="kW",
                        ),
                        number_control(
                            "Maximum discharge power", "discharge-power", 2.0,
                            minimum=0, maximum=20, step=0.1, unit="kW",
                        ),
                        number_control(
                            "Charge efficiency", "charge-efficiency", 0.95,
                            minimum=0.01, maximum=1, step=0.01, unit="0–1",
                        ),
                        number_control(
                            "Discharge efficiency", "discharge-efficiency", 0.95,
                            minimum=0.01, maximum=1, step=0.01, unit="0–1",
                        ),
                        number_control(
                            "Degradation cost", "degradation-cost", 0.20,
                            minimum=0, maximum=5, step=0.01, unit="SEK/kWh",
                        ),
                    ],
                    className="parameter-grid",
                ),
            ],
            style={
                "backgroundColor": COLORS["panel"],
                "border": f"1px solid {COLORS['border']}",
                "borderRadius": "8px",
                "padding": "18px",
                "marginBottom": "18px",
            },
        ),
        html.Div(
            id="validation-message",
            style=MESSAGE_STYLE,
        ),
        html.Div(
            [
                kpi_card("Cost without battery", "kpi-baseline-cost"),
                kpi_card("Cost with optimized battery", "kpi-optimized-cost"),
                kpi_card("Savings", "kpi-savings"),
                kpi_card("Grid import", "kpi-grid-import"),
                kpi_card("Grid export", "kpi-grid-export"),
                kpi_card("Battery discharged energy", "kpi-battery-discharge"),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(175px, 1fr))",
                "gap": "12px",
                "marginBottom": "18px",
            },
        ),
        html.Div(
            [
                dcc.Graph(id="price-chart", config={"displaylogo": False}),
                dcc.Graph(id="load-chart", config={"displaylogo": False}),
                dcc.Graph(id="soc-chart", config={"displaylogo": False}),
                dcc.Graph(id="power-chart", config={"displaylogo": False}),
                dcc.Graph(id="cost-chart", config={"displaylogo": False}),
                dcc.Graph(id="savings-chart", config={"displaylogo": False}),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(440px, 1fr))",
                "gap": "14px",
            },
        ),
    ],
    className="overview-content",
)

timeline_marks = {
    0: "00:00",
    24: "06:00",
    48: "12:00",
    72: "18:00",
    95: "23:45",
}

energy_flow_content = html.Div(
    [
        html.Div(
            [
                html.H2("Energy Flow", style={"margin": 0, "fontSize": "22px"}),
                html.P(
                    "Day context on the left, energy system on the right. Both show the same playback timestep.",
                    style={"color": COLORS["muted"], "margin": "6px 0 0"},
                ),
            ],
            className="flow-header",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Button("▶ Play", id="flow-play", className="flow-button"),
                                html.Button("❚❚ Pause", id="flow-pause", className="flow-button"),
                                html.Span(
                                    "Paused",
                                    id="playback-status",
                                    className="playback-status",
                                ),
                                html.Span(
                                    "00:00",
                                    id="playback-clock",
                                    className="playback-clock",
                                ),
                            ],
                            className="playback-controls",
                        ),
                        html.Div("Timeline", className="timeline-label"),
                        dcc.Slider(
                            id="flow-timestep",
                            min=0,
                            max=95,
                            step=1,
                            value=0,
                            marks=timeline_marks,
                            tooltip=None,
                        ),
                    ],
                    className="timeline-panel",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("What is happening?", className="flow-explanation-title"),
                                html.Div(
                                    "HOLDING",
                                    id="flow-decision-badge",
                                    className="decision-badge holding",
                                ),
                            ],
                            className="flow-explanation-header",
                        ),
                        html.Div(id="flow-explanation", className="flow-explanation-text"),
                    ],
                    className="flow-explanation",
                ),
                html.Div(
                    [
                        html.Div("Day context", className="section-heading"),
                        dcc.Graph(
                            id="day-context-chart",
                            config={"displaylogo": False, "responsive": True},
                            className="day-context-chart",
                            style={"height": "520px"},
                        ),
                    ],
                    className="flow-stage-chart",
                ),
                html.Div(
                    [
                        html.Div("Energy system", className="section-heading"),
                        html.Div(
                            [
                                html.Div(
                                    "No optimized schedule is available.",
                                    id="flow-empty-state",
                                    className="flow-empty-state",
                                ),
                                build_triangle_system(),
                            ],
                            id="flow-visualization",
                        ),
                    ],
                    className="flow-stage-system",
                ),
            ],
            className="flow-layout",
        ),
        html.Div(
            "Waiting for an optimized schedule.",
            id="flow-balance-message",
            className="flow-balance",
        ),
        html.H3("Current timestep", className="flow-details-heading"),
        html.Div(id="flow-details", className="flow-details-grid"),
    ],
    className="energy-flow-content",
)

tab_style = {
    "backgroundColor": COLORS["panel"],
    "color": COLORS["muted"],
    "border": f"1px solid {COLORS['border']}",
    "padding": "12px",
}
selected_tab_style = {
    **tab_style,
    "backgroundColor": "#243247",
    "color": COLORS["text"],
    "borderTop": f"3px solid {COLORS['blue']}",
    "fontWeight": "600",
}

app.layout = html.Div(
    [
        dcc.Store(id="optimization-store"),
        dcc.Store(id="flow-frame-store"),
        dcc.Store(id="flow-vis-applied"),
        dcc.Interval(id="flow-interval", interval=400, disabled=True),
        html.Div(
            [
                html.H1(
                    "Residential Battery Optimizer",
                    style={"margin": "0", "fontSize": "30px"},
                ),
                html.P(
                    "SE3 day-ahead prices · synthetic household load · 15-minute intervals",
                    style={"color": COLORS["muted"], "margin": "8px 0 0"},
                ),
            ],
            style={"marginBottom": "22px"},
        ),
        dcc.Tabs(
            id="dashboard-tabs",
            value="overview",
            children=[
                dcc.Tab(
                    label="Overview",
                    value="overview",
                    children=overview_content,
                    style=tab_style,
                    selected_style=selected_tab_style,
                ),
                dcc.Tab(
                    label="Energy Flow",
                    value="energy-flow",
                    children=energy_flow_content,
                    style=tab_style,
                    selected_style=selected_tab_style,
                ),
            ],
            colors={
                "border": COLORS["border"],
                "primary": COLORS["blue"],
                "background": COLORS["panel"],
            },
            style={"marginBottom": "18px"},
        ),
    ],
    style={
        "minHeight": "100vh",
        "backgroundColor": COLORS["background"],
        "color": COLORS["text"],
        "fontFamily": "Arial, sans-serif",
        "padding": "28px",
    },
)


@app.callback(
    Output("kpi-baseline-cost", "children"),
    Output("kpi-optimized-cost", "children"),
    Output("kpi-savings", "children"),
    Output("kpi-grid-import", "children"),
    Output("kpi-grid-export", "children"),
    Output("kpi-battery-discharge", "children"),
    Output("price-chart", "figure"),
    Output("load-chart", "figure"),
    Output("soc-chart", "figure"),
    Output("power-chart", "figure"),
    Output("cost-chart", "figure"),
    Output("savings-chart", "figure"),
    Output("optimization-store", "data"),
    Output("validation-message", "children"),
    Output("validation-message", "style"),
    Input("capacity", "value"),
    Input("initial-soc", "value"),
    Input("minimum-soc", "value"),
    Input("maximum-soc", "value"),
    Input("charge-power", "value"),
    Input("discharge-power", "value"),
    Input("charge-efficiency", "value"),
    Input("discharge-efficiency", "value"),
    Input("degradation-cost", "value"),
)
def update_dashboard(
    capacity: float | None,
    initial_soc: float | None,
    minimum_soc: float | None,
    maximum_soc: float | None,
    charge_power: float | None,
    discharge_power: float | None,
    charge_efficiency: float | None,
    discharge_efficiency: float | None,
    degradation_cost: float | None,
) -> tuple[Any, ...]:
    """Run the existing optimization pipeline and update the dashboard."""
    values = (
        capacity,
        initial_soc,
        minimum_soc,
        maximum_soc,
        charge_power,
        discharge_power,
        charge_efficiency,
        discharge_efficiency,
        degradation_cost,
    )

    error = INPUT_ERROR
    if not error and any(value is None for value in values):
        error = "All battery parameters must have a value."
    elif not error and capacity <= 0:
        error = "Battery capacity must be greater than zero."
    elif not error and not (0 <= minimum_soc <= 100 and 0 <= maximum_soc <= 100):
        error = "Minimum and maximum SOC must be between 0% and 100%."
    elif not error and minimum_soc > maximum_soc:
        error = "Minimum SOC cannot be greater than maximum SOC."
    elif not error and not (minimum_soc <= initial_soc <= maximum_soc):
        error = "Initial SOC must be between minimum and maximum SOC."
    elif not error and not (0 < charge_efficiency <= 1):
        error = "Charge efficiency must be greater than 0 and at most 1."
    elif not error and not (0 < discharge_efficiency <= 1):
        error = "Discharge efficiency must be greater than 0 and at most 1."
    elif not error and (charge_power < 0 or discharge_power < 0):
        error = "Charge and discharge power cannot be negative."
    elif not error and degradation_cost < 0:
        error = "Degradation cost cannot be negative."

    chart_titles = (
        "Buy and sell electricity prices",
        "Household load",
        "Battery SOC",
        "Battery charge / discharge power",
        "Cumulative household cost",
        "Cumulative savings",
    )
    if error:
        figures = [empty_figure(title, "Fix the input error to run the model.") for title in chart_titles]
        return ("—",) * 6 + tuple(figures) + (None, error, ERROR_MESSAGE_STYLE)

    try:
        initial_energy = capacity * initial_soc / 100
        min_energy = capacity * minimum_soc / 100
        max_energy = capacity * maximum_soc / 100

        result = optimize_household_battery(
            buy_price=BUY_PRICE,
            sell_price=SELL_PRICE,
            load_kw=LOAD_KW,
            capacity=capacity,
            initial_energy=initial_energy,
            min_energy=min_energy,
            max_energy=max_energy,
            max_charge_power=charge_power,
            max_discharge_power=discharge_power,
            eta_charge=charge_efficiency,
            eta_discharge=discharge_efficiency,
            dt=DT_HOURS,
            degradation_cost=degradation_cost,
        )
        result = add_cumulative_baseline_and_savings(result, dt=DT_HOURS)
        metrics = summarize_results(result, dt=DT_HOURS)
    except Exception as exc:
        figures = [empty_figure(title, "The optimization could not be completed.") for title in chart_titles]
        return (
            ("—",) * 6
            + tuple(figures)
            + (None, f"Optimization error: {exc}", ERROR_MESSAGE_STYLE)
        )

    timestamps = PRICE_DATA["time_start"]
    sanity_warnings = []
    simultaneous_battery = (
        (result["charge_kw"] > 1e-6) & (result["discharge_kw"] > 1e-6)
    )
    simultaneous_grid = (
        (result["grid_import_kw"] > 1e-6) & (result["grid_export_kw"] > 1e-6)
    )
    if simultaneous_battery.any():
        sanity_warnings.append(
            "charge and discharge are both active in "
            f"{int(simultaneous_battery.sum())} interval(s)"
        )
    if simultaneous_grid.any():
        sanity_warnings.append(
            "grid import and export are both active in "
            f"{int(simultaneous_grid.sum())} interval(s)"
        )

    price_figure = go.Figure()
    price_figure.add_scatter(
        x=timestamps,
        y=result["buy_price"],
        name="Buy",
        line={"color": COLORS["blue"], "width": 2.5},
        cliponaxis=False,
    )
    price_figure.add_scatter(
        x=timestamps,
        y=result["sell_price"],
        name="Sell",
        line={"color": COLORS["green"], "width": 2},
        cliponaxis=False,
    )
    style_figure(price_figure, chart_titles[0], "SEK/kWh")
    add_y_headroom(
        price_figure,
        result["buy_price"],
        result["sell_price"],
        top_padding=0.18,
    )

    load_figure = go.Figure()
    load_figure.add_scatter(
        x=timestamps,
        y=result["load_kw"],
        name="Load",
        fill="tozeroy",
        line={"color": COLORS["amber"]},
    )
    style_figure(load_figure, chart_titles[1], "kW")
    add_y_headroom(load_figure, result["load_kw"], include_zero=True)

    soc_timestamps = pd.concat(
        [
            PRICE_DATA["time_start"].iloc[[0]],
            PRICE_DATA["time_end"],
        ],
        ignore_index=True,
    )
    soc_values = pd.concat(
        [
            pd.Series([result["soc"].iloc[0] * 100]),
            result["soc_after"].reset_index(drop=True) * 100,
        ],
        ignore_index=True,
    )
    soc_figure = go.Figure()
    soc_figure.add_scatter(
        x=soc_timestamps,
        y=soc_values,
        name="SOC",
        line={"color": COLORS["purple"]},
    )
    soc_figure.add_hline(y=minimum_soc, line_dash="dot", line_color=COLORS["muted"])
    soc_figure.add_hline(y=maximum_soc, line_dash="dot", line_color=COLORS["muted"])
    style_figure(soc_figure, chart_titles[2], "%")
    soc_figure.update_yaxes(range=[0, 100])

    power_figure = go.Figure()
    power_figure.add_bar(
        x=timestamps,
        y=result["charge_kw"],
        name="Charge",
        marker_color=COLORS["green"],
    )
    power_figure.add_bar(
        x=timestamps,
        y=-result["discharge_kw"],
        name="Discharge",
        marker_color=COLORS["red"],
    )
    style_figure(power_figure, chart_titles[3], "kW")
    power_figure.update_layout(barmode="relative")
    add_y_headroom(
        power_figure,
        result["charge_kw"],
        -result["discharge_kw"],
        include_zero=True,
    )

    cost_figure = go.Figure()
    cost_figure.add_scatter(
        x=timestamps,
        y=result["cumulative_net_cost_sek"],
        name="Optimized",
        line={"color": COLORS["blue"], "width": 2},
        cliponaxis=False,
        legendrank=2,
    )
    cost_figure.add_scatter(
        x=timestamps,
        y=result["cumulative_baseline_cost_sek"],
        name="Without battery",
        line={"color": COLORS["muted"], "width": 2.5, "dash": "dash"},
        cliponaxis=False,
        legendrank=1,
    )
    style_figure(cost_figure, chart_titles[4], "SEK")
    add_y_headroom(
        cost_figure,
        result["cumulative_baseline_cost_sek"],
        result["cumulative_net_cost_sek"],
        include_zero=True,
        top_padding=0.18,
    )

    savings_figure = go.Figure()
    savings_figure.add_scatter(
        x=timestamps,
        y=result["cumulative_savings_sek"],
        name="Savings",
        fill="tozeroy",
        line={"color": COLORS["green"]},
    )
    style_figure(savings_figure, chart_titles[5], "SEK")
    add_y_headroom(
        savings_figure,
        result["cumulative_savings_sek"],
        include_zero=True,
    )

    warning_message = ""
    message_style = MESSAGE_STYLE
    if sanity_warnings:
        warning_message = "Sanity warning: " + "; ".join(sanity_warnings) + "."
        message_style = WARNING_MESSAGE_STYLE

    stored_result = serialize_result(result)

    return (
        f"{metrics['total_cost_without_battery_sek']:.2f} SEK",
        f"{metrics['total_cost_with_optimized_battery_sek']:.2f} SEK",
        f"{metrics['total_savings_sek']:.2f} SEK",
        f"{metrics['total_grid_import_kwh']:.2f} kWh",
        f"{metrics['total_grid_export_kwh']:.2f} kWh",
        f"{metrics['total_battery_discharge_kwh']:.2f} kWh",
        price_figure,
        load_figure,
        soc_figure,
        power_figure,
        cost_figure,
        savings_figure,
        stored_result,
        warning_message,
        message_style,
    )


@app.callback(
    Output("flow-interval", "disabled"),
    Output("playback-status", "children"),
    Input("flow-play", "n_clicks"),
    Input("flow-pause", "n_clicks"),
)
def toggle_playback(
    play_clicks: int | None,
    pause_clicks: int | None,
) -> tuple[bool, str]:
    """Start or pause timeline playback without recomputing the model."""
    del play_clicks, pause_clicks
    if ctx.triggered_id == "flow-play":
        return False, "Playing"
    return True, "Paused"


@app.callback(
    Output("flow-timestep", "value"),
    Input("flow-interval", "n_intervals"),
    State("flow-timestep", "value"),
    State("optimization-store", "data"),
    prevent_initial_call=True,
)
def advance_timestep(
    n_intervals: int,
    current_timestep: int | None,
    stored_result: dict[str, Any] | None,
) -> int | Any:
    """Advance through the stored result and loop after the final interval."""
    del n_intervals
    if not stored_result or not stored_result.get("records"):
        return no_update

    n_steps = int(stored_result["n_steps"])
    current = int(current_timestep or 0)
    return (current + 1) % n_steps


@app.callback(
    Output("day-context-chart", "figure"),
    Input("optimization-store", "data"),
    State("flow-timestep", "value"),
)
def refresh_day_context(
    stored_result: dict[str, Any] | None,
    timestep: int | None,
) -> go.Figure:
    """Rebuild whole-day context only when model inputs produce a new result."""
    if not stored_result or not stored_result.get("records"):
        return empty_figure("Day context", "No optimized schedule is available.")

    records = stored_result["records"]
    selected_timestep = min(max(int(timestep or 0), 0), len(records) - 1)
    return build_day_context_figure(records, selected_timestep)


@app.callback(
    Output("day-context-chart", "figure", allow_duplicate=True),
    Input("flow-timestep", "value"),
    State("optimization-store", "data"),
    prevent_initial_call=True,
)
def update_day_context_cursor(
    timestep: int | None,
    stored_result: dict[str, Any] | None,
) -> Patch | Any:
    """Patch only the current cursor and markers during playback."""
    if not stored_result or not stored_result.get("records"):
        return no_update

    records = stored_result["records"]
    selected_timestep = min(max(int(timestep or 0), 0), len(records) - 1)
    return patch_day_context_cursor(records, selected_timestep)


@app.callback(
    Output("flow-visualization", "className"),
    Output("flow-explanation", "children"),
    Output("flow-decision-badge", "children"),
    Output("flow-decision-badge", "className"),
    Output("playback-clock", "children"),
    Output("flow-details", "children"),
    Output("flow-balance-message", "children"),
    Output("flow-balance-message", "className"),
    Output("flow-frame-store", "data"),
    Input("flow-timestep", "value"),
    Input("optimization-store", "data"),
    State("flow-visualization", "className"),
)
def update_energy_flow(
    timestep: int | None,
    stored_result: dict[str, Any] | None,
    visualization_class: str | None,
) -> tuple[Any, ...]:
    """Update copy and a triangle frame; the illustration DOM is patched in place."""
    if not stored_result or not stored_result.get("records"):
        return (
            _keep_if_same("is-empty", visualization_class),
            "No current action is available.",
            "HOLDING",
            "decision-badge holding",
            "—",
            [],
            "",
            "flow-balance flow-balance-ok",
            None,
        )

    records = stored_result["records"]
    selected_timestep = min(max(int(timestep or 0), 0), len(records) - 1)
    current_time = pd.to_datetime(
        records[selected_timestep]["time_start"]
    ).strftime("%H:%M")
    frame = build_flow_visualization(
        records[selected_timestep],
        records,
        selected_timestep,
    )
    triangle_frame = {
        key: frame[key]
        for key in (
            "triangle_time",
            "grid_house_edge_class",
            "house_battery_edge_class",
            "grid_battery_edge_class",
            "grid_house_label_class",
            "house_battery_label_class",
            "grid_battery_label_class",
            "grid_house_dir",
            "grid_house_kw",
            "house_battery_dir",
            "house_battery_kw",
            "grid_battery_dir",
            "grid_battery_kw",
            "house_load",
            "house_split",
            "house_split_class",
            "grid_primary",
            "grid_secondary",
            "battery_shell_class",
            "soc_percent",
            "battery_soc",
            "battery_energy",
            "battery_mode",
        )
    }
    return (
        _keep_if_same("", visualization_class),
        frame["explanation"],
        frame["badge_label"],
        frame["badge_class"],
        current_time,
        frame["details"],
        frame["balance_message"],
        frame["balance_class"],
        triangle_frame,
    )


app.clientside_callback(
    """
    function(frame) {
        if (!frame) {
            return window.dash_clientside.no_update;
        }
        const setText = (id, value) => {
            const el = document.getElementById(id);
            if (!el) {
                return;
            }
            const next = value == null ? "" : String(value);
            if (el.textContent !== next) {
                el.textContent = next;
            }
        };
        const setClass = (id, value) => {
            const el = document.getElementById(id);
            if (!el || !value || el.className === value) {
                return;
            }
            el.className = value;
        };
        const setHeight = (id, pct) => {
            const el = document.getElementById(id);
            if (!el) {
                return;
            }
            const next = Number(pct).toFixed(2) + "%";
            if (el.style.height !== next) {
                el.style.height = next;
            }
        };
        setText("triangle-time", frame.triangle_time);
        setClass("triangle-edge-grid-house", frame.grid_house_edge_class);
        setClass("triangle-edge-house-battery", frame.house_battery_edge_class);
        setClass("triangle-edge-grid-battery", frame.grid_battery_edge_class);
        setClass("triangle-label-grid-house", frame.grid_house_label_class);
        setClass("triangle-label-house-battery", frame.house_battery_label_class);
        setClass("triangle-label-grid-battery", frame.grid_battery_label_class);
        setText("triangle-label-grid-house-dir", frame.grid_house_dir);
        setText("triangle-label-grid-house-kw", frame.grid_house_kw);
        setText("triangle-label-house-battery-dir", frame.house_battery_dir);
        setText("triangle-label-house-battery-kw", frame.house_battery_kw);
        setText("triangle-label-grid-battery-dir", frame.grid_battery_dir);
        setText("triangle-label-grid-battery-kw", frame.grid_battery_kw);
        setText("triangle-house-load", frame.house_load);
        setText("triangle-house-split", frame.house_split);
        setClass("triangle-house-split", frame.house_split_class);
        setText("triangle-grid-primary", frame.grid_primary);
        setText("triangle-grid-secondary", frame.grid_secondary);
        setClass("battery-shell", frame.battery_shell_class);
        setHeight("battery-fill", frame.soc_percent);
        setText("battery-soc", frame.battery_soc);
        setText("battery-energy", frame.battery_energy);
        setText("battery-mode", frame.battery_mode);
        return window.dash_clientside.no_update;
    }
    """,
    Output("flow-vis-applied", "data"),
    Input("flow-frame-store", "data"),
)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
