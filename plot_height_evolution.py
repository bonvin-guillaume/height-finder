#!/usr/bin/env python3
"""Interactive time series of mean and median auroral height.

Reads a WISC batch results CSV and writes an HTML figure with four
toggleable traces: greyscale mean, greyscale median, RGB-product mean,
and RGB-product median. Click a legend entry to hide or show that
series; double-click isolates one series.

A second panel plots series A minus series B. Two dropdowns list every
series. The default is greyscale mean minus RGB-product mean.

LYR and NYA mean and median traces start hidden in the legend. A checkbox
shows their standard deviation or standard error as error bars.

    python plot_height_evolution.py
    python plot_height_evolution.py "out/BACC_image_pairs_triangulation - 0901/WISC_batch_results.csv"
"""

from __future__ import annotations

import csv
import json
import sys
import webbrowser
from datetime import datetime
from html import escape
from pathlib import Path

import plotly.graph_objects as go

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = (
    SCRIPT_DIR
    / "out"
    / "BACC_image_pairs_triangulation - 0901"
    / "WISC_batch_results.csv"
)

SERIES = (
    {
        "sample": "greyscale",
        "column": "lyr_mean_km",
        "std_column": "lyr_std_km",
        "sem_column": "lyr_sem_km",
        "name": "Greyscale LYR mean",
        "group": "LYR",
        "color": "#9b2226",
        "dash": "solid",
        "statistic": "mean",
        "visible": "legendonly",
    },
    {
        "sample": "greyscale",
        "column": "lyr_median_km",
        "std_column": "lyr_std_km",
        "sem_column": "lyr_sem_km",
        "name": "Greyscale LYR median",
        "group": "LYR",
        "color": "#9b2226",
        "dash": "dash",
        "statistic": "median",
        "visible": "legendonly",
    },
    {
        "sample": "RGBproduct",
        "column": "lyr_mean_km",
        "std_column": "lyr_std_km",
        "sem_column": "lyr_sem_km",
        "name": "RGB product LYR mean",
        "group": "LYR",
        "color": "#e09f3e",
        "dash": "solid",
        "statistic": "mean",
        "visible": "legendonly",
    },
    {
        "sample": "RGBproduct",
        "column": "lyr_median_km",
        "std_column": "lyr_std_km",
        "sem_column": "lyr_sem_km",
        "name": "RGB product LYR median",
        "group": "LYR",
        "color": "#e09f3e",
        "dash": "dash",
        "statistic": "median",
        "visible": "legendonly",
    },
    {
        "sample": "greyscale",
        "column": "nya_mean_km",
        "std_column": "nya_std_km",
        "sem_column": "nya_sem_km",
        "name": "Greyscale NYA mean",
        "group": "NYA",
        "color": "#1b4332",
        "dash": "solid",
        "statistic": "mean",
        "visible": "legendonly",
    },
    {
        "sample": "greyscale",
        "column": "nya_median_km",
        "std_column": "nya_std_km",
        "sem_column": "nya_sem_km",
        "name": "Greyscale NYA median",
        "group": "NYA",
        "color": "#1b4332",
        "dash": "dash",
        "statistic": "median",
        "visible": "legendonly",
    },
    {
        "sample": "RGBproduct",
        "column": "nya_mean_km",
        "std_column": "nya_std_km",
        "sem_column": "nya_sem_km",
        "name": "RGB product NYA mean",
        "group": "NYA",
        "color": "#40916c",
        "dash": "solid",
        "statistic": "mean",
        "visible": "legendonly",
    },
    {
        "sample": "RGBproduct",
        "column": "nya_median_km",
        "std_column": "nya_std_km",
        "sem_column": "nya_sem_km",
        "name": "RGB product NYA median",
        "group": "NYA",
        "color": "#40916c",
        "dash": "dash",
        "statistic": "median",
        "visible": "legendonly",
    },
    {
        "sample": "greyscale",
        "column": "both_mean_km",
        "name": "Greyscale mean",
        "group": "Both stations",
        "color": "#4a4a4a",
        "dash": "solid",
        "statistic": "mean",
    },
    {
        "sample": "greyscale",
        "column": "both_median_km",
        "name": "Greyscale median",
        "group": "Both stations",
        "color": "#4a4a4a",
        "dash": "dash",
        "statistic": "median",
    },
    {
        "sample": "RGBproduct",
        "column": "both_mean_km",
        "name": "RGB product mean",
        "group": "Both stations",
        "color": "#1f77b4",
        "dash": "solid",
        "statistic": "mean",
    },
    {
        "sample": "RGBproduct",
        "column": "both_median_km",
        "name": "RGB product median",
        "group": "Both stations",
        "color": "#1f77b4",
        "dash": "dash",
        "statistic": "median",
    },
)

DEFAULT_A = "Greyscale mean"
DEFAULT_B = "RGB product mean"
DIFF_TRACE_NAME = "Difference"
PLOT_DIV_ID = "height-evolution"
DIFF_DIV_ID = "height-difference"


def _parse_time(text: str) -> datetime | None:
    text = text.strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _parse_km(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_series(
    csv_path: Path,
) -> dict[str, tuple[list[datetime], list[float], list[float | None] | None, list[float | None] | None, list[int | None]]]:
    points: dict[str, list[tuple[datetime, float, float | None, float | None, int | None]]] = {
        spec["name"]: [] for spec in SERIES
    }
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample = (row.get("sample") or "").strip()
            when = _parse_time(row.get("event_utc") or "")
            if when is None:
                continue
            for spec in SERIES:
                if spec["sample"] != sample:
                    continue
                height = _parse_km(row.get(spec["column"]) or "")
                if height is None:
                    continue
                std = sem = None
                if spec.get("std_column"):
                    std = _parse_km(row.get(spec["std_column"]) or "")
                    sem = _parse_km(row.get(spec["sem_column"]) or "")
                n_lines = _parse_km(row.get("n_lines") or "")
                points[spec["name"]].append(
                    (when, height, std, sem, None if n_lines is None else int(n_lines))
                )

    series: dict[
        str,
        tuple[
            list[datetime],
            list[float],
            list[float | None] | None,
            list[float | None] | None,
            list[int | None],
        ],
    ] = {}
    for spec in SERIES:
        ordered = sorted(points[spec["name"]], key=lambda item: item[0])
        times = [item[0] for item in ordered]
        heights = [item[1] for item in ordered]
        n_lines = [item[4] for item in ordered]
        if spec.get("std_column"):
            series[spec["name"]] = (
                times,
                heights,
                [item[2] for item in ordered],
                [item[3] for item in ordered],
                n_lines,
            )
        else:
            series[spec["name"]] = (times, heights, None, None, n_lines)
    return series


def _iso(when: datetime) -> str:
    return when.strftime("%Y-%m-%dT%H:%M:%S")


def pairwise_differences(
    series: dict[str, tuple],
) -> dict[str, dict[str, dict[str, list]]]:
    names = [spec["name"] for spec in SERIES]
    diffs: dict[str, dict[str, dict[str, list]]] = {}
    for name_a in names:
        diffs[name_a] = {}
        times_a, heights_a = series[name_a][0], series[name_a][1]
        for name_b in names:
            other = dict(zip(series[name_b][0], series[name_b][1]))
            xs: list[str] = []
            ys: list[float] = []
            for when, height in zip(times_a, heights_a):
                subtracted = other.get(when)
                if subtracted is None:
                    continue
                xs.append(_iso(when))
                ys.append(height - subtracted)
            diffs[name_a][name_b] = {"x": xs, "y": ys}
    return diffs


def _height_trace(spec: dict, series: dict[str, tuple]) -> go.Scatter:
    times, heights, stds, _sems, n_lines = series[spec["name"]]
    error_y = None
    if stds is not None:
        error_y = {
            "type": "data",
            "array": stds,
            "visible": False,
            "symmetric": True,
            "thickness": 1,
            "width": 3,
        }
    return go.Scatter(
        x=times,
        y=heights,
        mode="lines+markers",
        name=spec["name"],
        legendgroup=spec["group"],
        legendgrouptitle_text=spec["group"],
        visible=spec.get("visible", True),
        line={"color": spec["color"], "dash": spec["dash"], "width": 2},
        marker={"size": 6, "color": spec["color"]},
        error_y=error_y,
        customdata=[
            [count, f"{spec['sample']}/{when.strftime('%Y%m%dT%H%M%SZ')}_combined.png"]
            for when, count in zip(times, n_lines)
        ],
        hovertemplate=(
            "Time: %{x|%Y-%m-%d %H:%M:%S} UTC<br>"
            f"Sampling: {spec['sample']}<br>"
            f"Statistic: {spec['statistic']}<br>"
            "Field lines: %{customdata[0]}<br>"
            "Height: %{y:.1f} km"
            "<extra></extra>"
        ),
    )


def build_height_figure(series: dict[str, tuple]) -> go.Figure:
    fig = go.Figure(data=[_height_trace(spec, series) for spec in SERIES])
    fig.update_layout(
        title="Auroral height over time",
        xaxis_title="UTC time",
        yaxis_title="Height (km)",
        legend={
            "title_text": "Series",
            "tracegroupgap": 16,
            "groupclick": "toggleitem",
        },
        hovermode="closest",
        template="plotly_white",
        height=560,
    )
    return fig


def build_difference_figure(diffs: dict[str, dict[str, dict[str, list]]]) -> go.Figure:
    fig = go.Figure()
    default = diffs[DEFAULT_A][DEFAULT_B]
    fig.add_trace(
        go.Scatter(
            x=default["x"],
            y=default["y"],
            mode="lines+markers",
            name=DIFF_TRACE_NAME,
            line={"color": "#c45c26", "width": 2},
            marker={"size": 6, "color": "#c45c26"},
            hovertemplate=(
                "Time: %{x|%Y-%m-%d %H:%M:%S} UTC<br>"
                "Difference: %{y:.1f} km"
                "<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#888888", line_width=1)
    fig.update_layout(
        xaxis_title="UTC time",
        yaxis_title=f"{DEFAULT_A} − {DEFAULT_B}",
        showlegend=False,
        hovermode="closest",
        template="plotly_white",
        height=280,
        margin={"t": 24},
    )
    return fig


def _controls_html() -> str:
    groups: dict[str, list[str]] = {}
    for spec in SERIES:
        groups.setdefault(spec["group"], []).append(
            f'<option value="{escape(spec["name"])}">{escape(spec["name"])}</option>'
        )
    option_html = "\n".join(
        f'<optgroup label="{escape(group)}">{"".join(items)}</optgroup>'
        for group, items in groups.items()
    )
    selected_a = option_html.replace(
        f'value="{DEFAULT_A}"',
        f'value="{DEFAULT_A}" selected',
        1,
    )
    selected_b = option_html.replace(
        f'value="{DEFAULT_B}"',
        f'value="{DEFAULT_B}" selected',
        1,
    )
    return f"""
<div id="diff-controls" style="font-family: sans-serif; margin: 12px 16px 0; display: flex; gap: 16px; align-items: center; flex-wrap: wrap;">
  <label><input id="show-difference" type="checkbox" checked> Show difference</label>
  <label>Series A
    <select id="series-a" style="margin-left: 6px;">{selected_a}</select>
  </label>
  <span>−</span>
  <label>Series B
    <select id="series-b" style="margin-left: 6px;">{selected_b}</select>
  </label>
  <label><input id="show-error" type="checkbox"> Show error</label>
  <label>Error
    <select id="error-kind" style="margin-left: 6px;">
      <option value="std" selected>std</option>
      <option value="sem">sem</option>
    </select>
  </label>
</div>
"""


def _station_errors(series: dict[str, tuple]) -> dict[str, dict[str, list]]:
    errors: dict[str, dict[str, list]] = {}
    for spec in SERIES:
        if not spec.get("std_column"):
            continue
        _times, _heights, stds, sems, _n_lines = series[spec["name"]]
        errors[spec["name"]] = {"std": stds, "sem": sems}
    return errors


def _page_script(
    diffs: dict[str, dict[str, dict[str, list]]],
    errors: dict[str, dict[str, list]],
) -> str:
    diff_payload = json.dumps(diffs)
    error_payload = json.dumps(errors)
    return f"""
<script>
const HEIGHT_DIFFS = {diff_payload};
const STATION_ERRORS = {error_payload};
function applyHeightDifference() {{
  const plot = document.getElementById("{DIFF_DIV_ID}");
  const a = document.getElementById("series-a").value;
  const b = document.getElementById("series-b").value;
  const diff = HEIGHT_DIFFS[a][b];
  Plotly.restyle(plot, {{x: [diff.x], y: [diff.y]}}, [0]);
  Plotly.relayout(plot, {{"yaxis.title.text": a + " − " + b}}).then(alignDifferenceAxis);
}}
function applyStationError() {{
  const plot = document.getElementById("{PLOT_DIV_ID}");
  const show = document.getElementById("show-error").checked;
  const kind = document.getElementById("error-kind").value;
  const indices = [];
  const arrays = [];
  for (const name of Object.keys(STATION_ERRORS)) {{
    const idx = plot.data.findIndex((trace) => trace.name === name);
    if (idx < 0) continue;
    indices.push(idx);
    arrays.push(STATION_ERRORS[name][kind]);
  }}
  Plotly.restyle(plot, {{
    "error_y.visible": show,
    "error_y.type": "data",
    "error_y.symmetric": true,
    "error_y.array": arrays
  }}, indices);
}}
function applyDifferencePanel() {{
  const show = document.getElementById("show-difference").checked;
  document.getElementById("difference-panel").style.display = show ? "" : "none";
}}
function axisRange(event) {{
  if (event["xaxis.autorange"]) return null;
  if (event["xaxis.range"]) return event["xaxis.range"];
  if (event["xaxis.range[0]"] !== undefined) {{
    return [event["xaxis.range[0]"], event["xaxis.range[1]"]];
  }}
  return undefined;
}}
let syncingX = false;
function syncX(sourceId, targetId) {{
  document.getElementById(sourceId).on("plotly_relayout", (event) => {{
    if (syncingX || !document.getElementById("show-difference").checked) return;
    const range = axisRange(event);
    if (range === undefined) return;
    syncingX = true;
    const layout = range === null ? {{"xaxis.autorange": true}} : {{"xaxis.range": range}};
    Plotly.relayout(document.getElementById(targetId), layout).then(() => {{
      syncingX = false;
    }});
  }});
}}
syncX("{PLOT_DIV_ID}", "{DIFF_DIV_ID}");
syncX("{DIFF_DIV_ID}", "{PLOT_DIV_ID}");
let aligningAxes = false;
function alignDifferenceAxis() {{
  if (aligningAxes) return;
  const top = document.getElementById("{PLOT_DIV_ID}");
  const bottom = document.getElementById("{DIFF_DIV_ID}");
  const topAxis = top._fullLayout && top._fullLayout.xaxis;
  const bottomLayout = bottom._fullLayout;
  const bottomAxis = bottomLayout && bottomLayout.xaxis;
  if (!topAxis || !bottomAxis || !topAxis._length || !bottomAxis._length) return;
  const margin = bottomLayout.margin;
  const newL = margin.l + (topAxis._offset - bottomAxis._offset);
  const newR = bottomLayout.width - newL - topAxis._length;
  if (newL < 0 || newR < 0) return;
  if (Math.abs(newL - margin.l) < 2 && Math.abs(newR - margin.r) < 2) return;
  aligningAxes = true;
  Plotly.relayout(bottom, {{"margin.l": newL, "margin.r": newR}}).then(() => {{
    aligningAxes = false;
  }});
}}
document.getElementById("{PLOT_DIV_ID}").on("plotly_afterplot", alignDifferenceAxis);
alignDifferenceAxis();
let combinedScale = 1;
let combinedPanX = 0;
let combinedPanY = 0;
function applyCombinedTransform() {{
  const image = document.getElementById("combined-image");
  image.style.transformOrigin = "0 0";
  image.style.transform = "translate(" + combinedPanX + "px, " + combinedPanY + "px) scale(" + combinedScale + ")";
}}
function resetCombinedZoom() {{
  combinedScale = 1;
  combinedPanX = 0;
  combinedPanY = 0;
  applyCombinedTransform();
}}
function zoomCombined(factor, clientX, clientY) {{
  const viewport = document.getElementById("combined-viewport");
  const rect = viewport.getBoundingClientRect();
  const originX = clientX - rect.left;
  const originY = clientY - rect.top;
  const next = Math.min(8, Math.max(1, combinedScale * factor));
  const ratio = next / combinedScale;
  combinedPanX = originX - ratio * (originX - combinedPanX);
  combinedPanY = originY - ratio * (originY - combinedPanY);
  combinedScale = next;
  if (combinedScale === 1) {{
    combinedPanX = 0;
    combinedPanY = 0;
  }}
  applyCombinedTransform();
}}
function closeCombinedPlot() {{
  document.getElementById("combined-overlay").style.display = "none";
  document.getElementById("combined-image").removeAttribute("src");
  resetCombinedZoom();
}}
let combinedPlainPath = "";
function currentCombinedPath() {{
  if (!document.getElementById("combined-coloured").checked) return combinedPlainPath;
  return combinedPlainPath.replace(/_combined\.png$/, "_combined_coloured.png");
}}
function showCombinedImage(path) {{
  const image = document.getElementById("combined-image");
  const missing = document.getElementById("combined-missing");
  missing.style.display = "none";
  image.style.display = "block";
  image.onerror = () => {{
    image.style.display = "none";
    missing.style.display = "block";
  }};
  image.src = path;
}}
function timeLabelFromPath(path) {{
  const stamp = String(path).match(/(\d{{4}})(\d{{2}})(\d{{2}})T(\d{{2}})(\d{{2}})(\d{{2}})Z/);
  if (!stamp) return path;
  return stamp[1] + "-" + stamp[2] + "-" + stamp[3] + " " + stamp[4] + ":" + stamp[5] + ":" + stamp[6];
}}
function combinedPathsFor(sample) {{
  const paths = new Set();
  const plot = document.getElementById("{PLOT_DIV_ID}");
  for (const trace of plot.data) {{
    for (const row of trace.customdata || []) {{
      const path = row && row[1];
      if (path && String(path).startsWith(sample + "/")) paths.add(path);
    }}
  }}
  return Array.from(paths).sort();
}}
function switchCombinedSample() {{
  const parts = String(combinedPlainPath).split("/");
  const sample = parts[0];
  const other = sample === "greyscale" ? "RGBproduct" : "greyscale";
  const path = other + "/" + parts.slice(1).join("/");
  if (!combinedPathsFor(other).includes(path)) return;
  openCombinedPlot(path, other, timeLabelFromPath(path));
}}
function stepCombinedPlot(delta) {{
  const sample = String(combinedPlainPath).split("/")[0];
  const paths = combinedPathsFor(sample);
  const index = paths.indexOf(combinedPlainPath);
  const next = index + delta;
  if (index < 0 || next < 0 || next >= paths.length) return;
  openCombinedPlot(paths[next], sample, timeLabelFromPath(paths[next]));
}}
function openCombinedPlot(path, sample, timeLabel) {{
  const overlay = document.getElementById("combined-overlay");
  const image = document.getElementById("combined-image");
  combinedPlainPath = path;
  document.getElementById("combined-caption").textContent = sample + "  " + timeLabel;
  image.alt = sample + " combined plot " + timeLabel;
  resetCombinedZoom();
  showCombinedImage(currentCombinedPath());
  overlay.style.display = "flex";
}}
document.getElementById("{PLOT_DIV_ID}").on("plotly_click", (event) => {{
  const point = event.points && event.points[0];
  if (!point || !point.customdata) return;
  const path = point.customdata[1];
  const sample = String(path).split("/")[0];
  const timeLabel = point.x;
  openCombinedPlot(path, sample, timeLabel);
}});
document.getElementById("combined-close").addEventListener("click", (event) => {{
  event.stopPropagation();
  closeCombinedPlot();
}});
document.getElementById("combined-overlay").addEventListener("click", closeCombinedPlot);
document.getElementById("combined-panel").addEventListener("click", (event) => {{
  event.stopPropagation();
}});
document.getElementById("combined-zoom-in").addEventListener("click", () => {{
  const rect = document.getElementById("combined-viewport").getBoundingClientRect();
  zoomCombined(1.25, rect.left + rect.width / 2, rect.top + rect.height / 2);
}});
document.getElementById("combined-zoom-out").addEventListener("click", () => {{
  const rect = document.getElementById("combined-viewport").getBoundingClientRect();
  zoomCombined(0.8, rect.left + rect.width / 2, rect.top + rect.height / 2);
}});
document.getElementById("combined-zoom-reset").addEventListener("click", resetCombinedZoom);
document.getElementById("combined-coloured").addEventListener("change", () => {{
  showCombinedImage(currentCombinedPath());
}});
document.getElementById("combined-viewport").addEventListener("wheel", (event) => {{
  event.preventDefault();
  zoomCombined(event.deltaY < 0 ? 1.15 : 1 / 1.15, event.clientX, event.clientY);
}}, {{passive: false}});
let combinedDragging = false;
let combinedLastX = 0;
let combinedLastY = 0;
const combinedViewport = document.getElementById("combined-viewport");
combinedViewport.addEventListener("pointerdown", (event) => {{
  if (combinedScale <= 1) return;
  combinedDragging = true;
  combinedLastX = event.clientX;
  combinedLastY = event.clientY;
  combinedViewport.setPointerCapture(event.pointerId);
}});
combinedViewport.addEventListener("pointermove", (event) => {{
  if (!combinedDragging) return;
  combinedPanX += event.clientX - combinedLastX;
  combinedPanY += event.clientY - combinedLastY;
  combinedLastX = event.clientX;
  combinedLastY = event.clientY;
  applyCombinedTransform();
}});
combinedViewport.addEventListener("pointerup", () => {{
  combinedDragging = false;
}});
combinedViewport.addEventListener("pointercancel", () => {{
  combinedDragging = false;
}});
document.addEventListener("keydown", (event) => {{
  const open = document.getElementById("combined-overlay").style.display !== "none";
  if (!open) return;
  if (event.key === "Escape") closeCombinedPlot();
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {{
    event.preventDefault();
    stepCombinedPlot(event.key === "ArrowLeft" ? -1 : 1);
  }}
  if (event.key === "ArrowUp" || event.key === "ArrowDown") {{
    event.preventDefault();
    switchCombinedSample();
  }}
}});
document.getElementById("{PLOT_DIV_ID}").on("plotly_restyle", (data) => {{
  const update = (data && data[0]) || {{}};
  if (!Object.prototype.hasOwnProperty.call(update, "visible")) return;
  const shown = document.getElementById("{PLOT_DIV_ID}").data.filter(
    (trace) => trace.visible === true
  );
  if (shown.length !== 2) return;
  document.getElementById("series-a").value = shown[0].name;
  document.getElementById("series-b").value = shown[1].name;
  applyHeightDifference();
}});
document.getElementById("series-a").addEventListener("change", applyHeightDifference);
document.getElementById("series-b").addEventListener("change", applyHeightDifference);
document.getElementById("show-difference").addEventListener("change", applyDifferencePanel);
document.getElementById("show-error").addEventListener("change", applyStationError);
document.getElementById("error-kind").addEventListener("change", applyStationError);
</script>
"""


def render_page(
    height_fig: go.Figure,
    diff_fig: go.Figure,
    diffs: dict[str, dict[str, dict[str, list]]],
    errors: dict[str, dict[str, list]],
) -> str:
    height_div = height_fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        div_id=PLOT_DIV_ID,
    )
    diff_div = diff_fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=DIFF_DIV_ID,
    )
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body>
{_controls_html()}
<div id="combined-overlay" style="display:none; position:fixed; inset:0; z-index:30; background:rgba(0,0,0,0.72); align-items:center; justify-content:center; padding:24px;">
  <div id="combined-panel" style="background:#fff; max-width:95vw; max-height:95vh; overflow:auto; padding:16px 16px 20px; border-radius:8px; font-family:sans-serif;">
    <div style="display:flex; justify-content:space-between; align-items:center; gap:16px; margin-bottom:8px;">
      <div id="combined-caption"></div>
      <div style="display:flex; gap:8px; align-items:center;">
        <label><input id="combined-coloured" type="checkbox"> Coloured</label>
        <button id="combined-zoom-out" type="button">Zoom out</button>
        <button id="combined-zoom-reset" type="button">Reset</button>
        <button id="combined-zoom-in" type="button">Zoom in</button>
        <button id="combined-close" type="button">Close</button>
      </div>
    </div>
    <p style="margin:0 0 8px; color:#555; font-size:13px;">Scroll to zoom. Drag to move when zoomed in. Left and right arrows show the previous and next time. Up and down switch between greyscale and RGB product.</p>
    <div id="combined-viewport" style="overflow:hidden; max-width:90vw; max-height:78vh; cursor:grab;">
      <img id="combined-image" alt="Combined plot" style="max-width:90vw; max-height:78vh; display:block;">
    </div>
    <p id="combined-missing" style="display:none;">Combined plot not found.</p>
  </div>
</div>
{height_div}
<div id="difference-panel">{diff_div}</div>
{_page_script(diffs, errors)}
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    csv_path = Path(args[0]) if args else DEFAULT_CSV
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    series = load_series(csv_path)
    if not any(times for times, _heights, _stds, _sems, _n_lines in series.values()):
        print(f"No height rows in {csv_path}", file=sys.stderr)
        return 1

    diffs = pairwise_differences(series)
    html_path = csv_path.with_name("height_evolution.html")
    errors = _station_errors(series)
    html_path.write_text(
        render_page(build_height_figure(series), build_difference_figure(diffs), diffs, errors),
        encoding="utf-8",
    )
    print(f"Wrote {html_path}")
    webbrowser.open(html_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
