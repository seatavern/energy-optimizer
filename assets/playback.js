window.dash_clientside = window.dash_clientside || {};

window.dash_clientside.playback = {
    toggle_playback: function (playClicks, pauseClicks) {
        const triggered = window.dash_clientside.callback_context.triggered_id;
        if (triggered === "flow-play") {
            return [false, "Playing"];
        }
        return [true, "Paused"];
    },

    advance_timestep: function (nIntervals, current, playback) {
        if (!playback || !playback.n_steps) {
            return window.dash_clientside.no_update;
        }
        const currentStep = Number(current) || 0;
        return (currentStep + 1) % playback.n_steps;
    },

    apply_timestep: function (timestep, playback, tab, vizClass) {
        const noUpdate = window.dash_clientside.no_update;
        const nextClass = !playback || !playback.frames || !playback.frames.length ? "is-empty" : "";
        const classOut = (vizClass || "") === nextClass ? noUpdate : nextClass;
        if (!playback || !playback.frames || !playback.frames.length) {
            return [
                classOut,
                "No current action is available.",
                "HOLDING",
                "decision-badge holding",
                "—",
                "",
                "flow-balance flow-balance-ok",
                noUpdate,
            ];
        }

        const last = playback.frames.length - 1;
        const index = Math.min(Math.max(Number(timestep) || 0, 0), last);
        const frame = playback.frames[index];
        applyPlaybackDom(frame);
        applyPlaybackCursor(frame);
        return [
            classOut,
            frame.explanation,
            frame.badge,
            frame.badge_cls,
            frame.clock,
            frame.bal_msg,
            frame.bal_cls,
            index,
        ];
    },
};

function setText(id, value) {
    const el = document.getElementById(id);
    if (!el) {
        return;
    }
    const next = value == null ? "" : String(value);
    if (el.textContent !== next) {
        el.textContent = next;
    }
}

function setClass(id, value) {
    const el = document.getElementById(id);
    if (!el || !value || el.className === value) {
        return;
    }
    el.className = value;
}

function setHeight(id, pct) {
    const el = document.getElementById(id);
    if (!el) {
        return;
    }
    const next = Number(pct).toFixed(2) + "%";
    if (el.style.height !== next) {
        el.style.height = next;
    }
}

function applyPlaybackDom(frame) {
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
    setText("detail-time", frame.clock);
    setText("detail-load", frame.m_load);
    setText("detail-buy", frame.m_buy);
    setText("detail-sell", frame.m_sell);
    setText("detail-import", frame.m_imp);
    setText("detail-export", frame.m_exp);
    setText("detail-charge", frame.m_chg);
    setText("detail-discharge", frame.m_dis);
    setText("detail-soc", frame.m_soc);
    setText("detail-energy", frame.m_kwh);
}

function applyPlaybackCursor(frame) {
    const tryApply = (attempt) => {
        const graph = document.querySelector("#day-context-chart .js-plotly-plot");
        if (!graph || !window.Plotly || !graph.data || graph.data.length < 16) {
            if (attempt < 12) {
                window.setTimeout(() => tryApply(attempt + 1), 50);
            }
            return;
        }
        const t = frame.t;
        const visible = (value) => (Math.abs(value) > 1e-6 ? 1 : 0);
        Plotly.restyle(
            graph,
            {
                x: [[t], [t], [t], [t], [t], [t], [t], [t]],
                y: [
                    [frame.buy],
                    [frame.sell],
                    [frame.load],
                    [frame.chg],
                    [-frame.dis],
                    [frame.imp],
                    [-frame.exp],
                    [frame.soc],
                ],
                opacity: [
                    1,
                    1,
                    1,
                    visible(frame.chg),
                    visible(frame.dis),
                    visible(frame.imp),
                    visible(frame.exp),
                    1,
                ],
            },
            [2, 3, 9, 10, 11, 12, 13, 15]
        );
        Plotly.relayout(graph, {
            "shapes[0].x0": t,
            "shapes[0].x1": t,
            "annotations[3].x": t,
            "annotations[3].text": frame.clock,
        });
    };
    tryApply(0);
}
