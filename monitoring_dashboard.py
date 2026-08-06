"""
====================================================================
MONITORING DASHBOARD — standalone module
Aarogya Aarohan / TANUH Oral Cancer Project

Owns everything shown under the top-level "Monitoring Dashboard" view
(the Overall / Image-Based Screening / AI-Enabled Screening tabs),
separate from the Research Dashboard (see research_dashboard.py).

Contains:
  - Responsive layout helpers (screen-width breakpoints, dynamic
    chart/card sizing) used by every chart/card/tab below.
  - Analytics helpers (column detection, status masks, monthly stats).
  - Chart builders (Plotly figures) and UI component helpers (metric
    cards, map cards, etc).
  - The three tab renderers (Overall / Image-Based Screening /
    AI-Enabled Screening) plus the tab-navigation bar and footer.

Public entry point: render(screen_width, df_all, df_p1, df_p2,
df_all_map), called by app.py's main() whenever the user is on the
"Monitoring Dashboard" top-level view.
====================================================================
"""

from __future__ import annotations
import html
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_log = logging.getLogger(__name__)

try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path.cwd()

# ── Data source (same source dir as app.py's data loading; only this
#    module's own local files — the map JSON + India GeoJSON — live
#    here, since only Monitoring Dashboard charts consume them) ──────
LOCAL_DATA_DIR = Path(
    os.environ.get(
        "OCP_DATA_DIR",
        r"/mnt/d/OneDrive/IISC/TANUH/OralCancer_Project/Raw_Data/Dashboard",
    )
)
GEOJSON_DATA_DIR = BASE / "static"

AMBER_HIGH = "#E0631A"
AMBER_LOW  = "#F7C548"

# Count-up animation duration (ms) shared by every animated metric card.
# The original count-up finished in 900ms — fast enough that the number
# barely looked like it was moving. Bumped up so it's clearly a visible
# scroll rather than a near-instant snap into place.
COUNT_ANIM_MS = 2200

# ════════════════════════════════════════════════════════════════════
# 1.  RESPONSIVE LAYOUT — screen-width breakpoints & dynamic sizing
# ════════════════════════════════════════════════════════════════════

BP_PHONE  = 640   # < 640px  → phone
BP_TABLET = 1024  # 640–1024 → tablet, 1024–1440 → laptop, > 1440 → desktop
BP_LAPTOP = 1440  # 13"/14" laptops typically report ~1280–1440 CSS px here
_SIDEBAR_PX = 260

# Module-level layout state, (re)computed at the top of every render()
# call from the screen_width app.py detects — every chart/card/tab
# function below reads these as plain module globals, exactly as they
# were read as script globals in the original single-file app.
IS_PHONE = IS_TABLET = IS_LAPTOP = IS_DESKTOP = False
CONTENT_WIDTH = 0
CHART_WIDTH = 0


def _set_layout(screen_width: int) -> None:
    """(Re)derive every responsive layout global from the current
    screen_width. Called once at the top of render() on every rerun."""
    global IS_PHONE, IS_TABLET, IS_LAPTOP, IS_DESKTOP, CONTENT_WIDTH, CHART_WIDTH

    IS_PHONE   = screen_width < BP_PHONE
    IS_TABLET  = BP_PHONE <= screen_width < BP_TABLET
    IS_LAPTOP  = BP_TABLET <= screen_width < BP_LAPTOP
    IS_DESKTOP = screen_width >= BP_LAPTOP

    CONTENT_WIDTH = screen_width if IS_PHONE else max(screen_width - _SIDEBAR_PX, 320)
    # Registrations / suspicious-status charts and the phase-2 sankey always
    # render as one of two side-by-side columns on tablet/laptop/desktop, and
    # full-width on phone — used to size their legends/heights off their
    # actual rendered width instead of the full window width.
    CHART_WIDTH = CONTENT_WIDTH if IS_PHONE else _row_width(2)


# A 14" laptop window (~1280–1440px) used to fall into the same catch-all
# "desktop" bucket as a 1800–2500px external monitor, so charts/cards were
# sized as if they had a big monitor's worth of room and then clipped when
# they didn't get it. IS_LAPTOP splits that bucket out; CONTENT_WIDTH and
# _interp() below replace the old hardcoded per-bucket values with sizing
# that scales continuously with the space actually available, so every
# width in between — not just the three original buckets — gets a
# proportionate result instead of jumping between fixed sizes.

# Rough width Streamlit's expanded sidebar + page padding subtract from the
# window, so width-driven sizing is based on the space actually left for
# content rather than the raw (sidebar-inclusive) window width reported by
# the browser. The sidebar collapses to an overlay on phone, so it doesn't
# eat into content width there.


def _interp(x: float, points: list[tuple[float, float]]) -> float:
    """Piecewise-linear interpolation of `points` (a list of (x, value)
    pairs) at `x`, clamped to the first/last value outside the given
    range. This is what makes chart/card sizing "dynamic" rather than
    bucketed — any in-between width (e.g. a 1366px laptop sitting between
    the old tablet/desktop cutoffs) gets a proportionate size instead of
    being rounded up or down to whichever fixed bucket it technically
    fell into.
    """
    pts = sorted(points)
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, v0), (x1, v1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return v0 + t * (v1 - v0)
    return pts[-1][1]


def _row_width(n_cols: int) -> int:
    """Estimated pixel width of a single card/chart when CONTENT_WIDTH is
    split into `n_cols` side-by-side st.columns (roughly accounts for the
    gap Streamlit inserts between columns). Sizing cards/legends off this
    — instead of off the raw window width — is what keeps a 4-across card
    row correctly sized whether the window is a 1366px laptop or a
    2560px monitor, rather than only checking the window width and
    assuming every column got an equal share of a much wider screen.
    """
    n_cols = max(n_cols, 1)
    gap = 16 * (n_cols - 1)
    return max((CONTENT_WIDTH - gap) // n_cols, 160)


def _weighted_col_width(weights: list[float], idx: int) -> int:
    """Like `_row_width`, but for a row of st.columns given unequal
    relative widths (as passed to `st.columns([...])`) instead of an
    equal n-way split. Used when one card in a row needs more horizontal
    room than its neighbors (e.g. a longer label) — the other columns in
    the row shrink proportionally to make room for it, rather than every
    column claiming a fixed equal share regardless of content.
    """
    n_cols = max(len(weights), 1)
    gap = 16 * (n_cols - 1)
    avail = max(CONTENT_WIDTH - gap, 0)
    total = sum(weights) or 1.0
    return max(round(avail * weights[idx] / total), 160)


# Registrations / suspicious-status charts and the phase-2 sankey always
# render as one of two side-by-side columns on tablet/laptop/desktop, and
# full-width on phone — used to size their legends/heights off their
# actual rendered width instead of the full window width.


def _cols(n_desktop: int, n_tablet: int | None = None, n_phone: int = 1):
    """Return the column count to use for the current screen size.

    n_desktop applies >= BP_TABLET (laptop and desktop alike — the row is
    still split into this many columns; _row_width()/_interp() are what
    adjust the per-column *sizing* between a laptop and a large monitor),
    n_tablet applies on tablet widths (defaults to n_desktop if not
    given, capped at 2-3 cols), n_phone applies on phones (defaults to a
    single stacked column).
    """
    if IS_PHONE:
        return n_phone
    if IS_TABLET:
        return n_tablet if n_tablet is not None else min(n_desktop, 3)
    return n_desktop


# ════════════════════════════════════════════════════════════════════
# 2.  ANALYTICS HELPERS
# ════════════════════════════════════════════════════════════════════

_BLANK_LIKE = {"", "nan", "none", "-", ".", "na", "n/a", "null"}

_STATE_ALIAS: dict[str, str] = {
    # Delhi variants — shapefile uses "Delhi"
    "new delhi":              "delhi",
    "nct of delhi":           "delhi",
    "national capital territory of delhi": "delhi",
    "nct delhi":              "delhi",
    # Other common aliases
    "uttaranchal":            "uttarakhand",
    "orissa":                 "odisha",
    "pondicherry":            "puducherry",
    "andaman & nicobar":      "andaman and nicobar",
    "andaman & nicobar islands": "andaman and nicobar",
    "jammu & kashmir":        "jammu and kashmir",
    "j&k":                    "jammu and kashmir",
    "dadra & nagar haveli":   "dadra and nagar haveli",
    "daman & diu":            "daman and diu",
}

def _norm_state(s: str) -> str:
    k = s.strip().lower()
    return _STATE_ALIAS.get(k, k)


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def _blank_mask(s: pd.Series) -> pd.Series:
    return s.isna() | _norm(s).isin(_BLANK_LIKE)


def _present_mask(s: pd.Series) -> pd.Series:
    return ~_blank_mask(s)


def _id_col(df: pd.DataFrame) -> str:
    for c in ("case_id", "Unique_case_ID"):
        if c in df.columns:
            return c
    return df.columns[0]


def _ai_result_col(df: pd.DataFrame) -> str | None:
    for c in ("ai_result", "AI_Result", "AI Result"):
        if c in df.columns:
            return c
    return None


def _reviewed_mask_phase1(df: pd.DataFrame) -> pd.Series:
    cols = df.columns
    prov = df["provisional_diagnosis"] if "provisional_diagnosis" in cols else pd.Series(False, index=df.index)
    susp = df["suspicion"]             if "suspicion" in cols             else pd.Series(False, index=df.index)
    risk = df["risk"]                  if "risk" in cols                  else pd.Series(False, index=df.index)
    return _present_mask(prov) & _present_mask(susp) & _present_mask(risk)


def _reviewed_mask_phase2(df: pd.DataFrame) -> pd.Series:
    ai_col = _ai_result_col(df)
    if ai_col is None:
        return pd.Series(False, index=df.index)
    return _present_mask(df[ai_col])


def _is_high_risk(series: pd.Series) -> pd.Series:
    return _norm(series).eq("high risk")


def _is_low_risk(series: pd.Series) -> pd.Series:
    return _norm(series).eq("low risk")


def _months_back(anchor: pd.Timestamp, months: int) -> pd.Timestamp:
    return (anchor - pd.DateOffset(months=months)).normalize()


def _choose_status_masks(
    df: pd.DataFrame,
    phase: str,
) -> tuple[pd.Series, pd.Series | None, str, str]:
    if phase == "phase1":
        reviewed_mask = _reviewed_mask_phase1(df)
        if "suspicion" in df.columns:
            suspicious_mask = reviewed_mask & _norm(df["suspicion"]).eq("suspicious")
        else:
            suspicious_mask = pd.Series(False, index=df.index)
        return reviewed_mask, suspicious_mask, "Suspicion", "Suspicious"

    reviewed_mask = _reviewed_mask_phase2(df)
    ai_col = _ai_result_col(df)
    if ai_col is not None:
        suspicious_mask = reviewed_mask & _norm(df[ai_col]).eq("suspicious")
    else:
        suspicious_mask = pd.Series(False, index=df.index)
    return reviewed_mask, suspicious_mask, "AI Result", "Suspicious"


@st.cache_data(max_entries=8, ttl=3600, show_spinner=False)
def monthly_stats(
    df: pd.DataFrame,
    phase: str = "phase1",
) -> pd.DataFrame:
    """
    Returns one row per calendar month.

    Image-Based Screening columns:
      total, reviewed_total, suspicious, high_risk, low_risk,
      cumulative, susp_pct, non_susp_pct, month_lbl

    AI-Enabled Screening columns:
      total, ai_suspicious, cumulative, month_lbl
    """
    if df.empty or "date_of_case_registered" not in df.columns:
        return pd.DataFrame()

    d = df.dropna(subset=["date_of_case_registered"]).copy()
    if d.empty:
        return pd.DataFrame()

    id_col = _id_col(d)
    d["ym"] = d["date_of_case_registered"].dt.to_period("M")

    total = d.groupby("ym")[id_col].nunique().reset_index(name="total")

    # ── Image-Based Screening ──────────────────────────────────────────────────────
    if phase == "phase1":
        reviewed_mask, suspicious_mask, _, _ = _choose_status_masks(d, phase)
        d["_reviewed"] = reviewed_mask
        d["_susp"]     = suspicious_mask

        reviewed = d[d["_reviewed"]].copy()
        if reviewed.empty:
            mon = total.copy()
            mon["reviewed_total"] = 0
            mon["suspicious"]     = 0
            mon["high_risk"]      = 0
            mon["low_risk"]       = 0
        else:
            reviewed_total = reviewed.groupby("ym")[id_col].nunique().reset_index(name="reviewed_total")
            susp = reviewed.groupby("ym")["_susp"].sum().reset_index(name="suspicious")
            mon  = total.merge(reviewed_total, on="ym", how="left").merge(susp, on="ym", how="left")

            if "risk" in d.columns:
                d["_high"] = d["_susp"] & _is_high_risk(d["risk"])
                d["_low"]  = d["_susp"] & _is_low_risk(d["risk"])
            else:
                d["_high"] = False
                d["_low"]  = False

            high = d.groupby("ym")["_high"].sum().reset_index(name="high_risk")
            low  = d.groupby("ym")["_low"].sum().reset_index(name="low_risk")
            mon  = mon.merge(high, on="ym", how="left").merge(low, on="ym", how="left")

        mon = mon.sort_values("ym").reset_index(drop=True)
        for c in ["reviewed_total", "suspicious", "high_risk", "low_risk"]:
            if c not in mon.columns:
                mon[c] = 0
            mon[c] = mon[c].fillna(0).astype(int)

        mon["cumulative"]   = mon["total"].cumsum()
        denom               = mon["reviewed_total"].replace(0, np.nan)
        mon["susp_pct"]     = (mon["suspicious"] / denom * 100).round(1).fillna(0)
        # When nobody has been reviewed yet this month (reviewed_total == 0,
        # e.g. provisional_diagnosis is still "-"/blank for all cases), there
        # is no "non-suspicious" data either — it must read 0%, not 100%.
        # Without this guard, 100 - susp_pct(=0) incorrectly evaluates to 100.
        mon["non_susp_pct"] = np.where(
            mon["reviewed_total"] > 0,
            (100 - mon["susp_pct"]).round(1),
            0.0,
        )
        mon["month_lbl"]    = mon["ym"].dt.start_time.dt.strftime("%b %Y")
        return mon

    # ── AI-Enabled Screening ──────────────────────────────────────────────────────
    ai_col = _ai_result_col(d)

    if ai_col is not None:
        d["_ai_susp"] = _present_mask(d[ai_col]) & _norm(d[ai_col]).eq("suspicious")
    else:
        d["_ai_susp"] = False

    ai_susp_grp = d.groupby("ym")["_ai_susp"].sum().reset_index(name="ai_suspicious")
    mon = total.merge(ai_susp_grp, on="ym", how="left")
    mon = mon.sort_values("ym").reset_index(drop=True)

    if "ai_suspicious" not in mon.columns:
        mon["ai_suspicious"] = 0
    mon["ai_suspicious"] = mon["ai_suspicious"].fillna(0).astype(int)

    mon["cumulative"] = mon["total"].cumsum()
    mon["month_lbl"]  = mon["ym"].dt.start_time.dt.strftime("%b %Y")
    return mon


# ════════════════════════════════════════════════════════════════════
# 3.  MAP & GEOJSON DATA LOADING
# ════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, max_entries=1, show_spinner=False)
def _load_map_data() -> dict:
    """Load map_data.json from the local filesystem."""
    map_path = LOCAL_DATA_DIR / "map_data.json"
    if not map_path.exists():
        _log.warning("map_data.json not found at %s — returning empty dict", map_path)
        return {}
    with open(map_path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading Map...")
def _load_india_geojson():
    """Return (geojson_dict, state_property_name) or (None, None).

    Source: GEOJSON_DATA_DIR/india_states.geojson (the app's static/
    folder). No shapefile fallback; if the GeoJSON is missing or
    invalid, the map is skipped and a status message is shown instead.
    """
    _PROP_CANDS = ("State_Name", "NAME_1", "ST_NM", "name", "NAME", "GEO_ID", "STNAME", "state")

    def _detect_prop(gj: dict) -> str:
        props = (gj.get("features") or [{}])[0].get("properties", {})
        for p in _PROP_CANDS:
            if p in props:
                return p
        return next(iter(props), "NAME_1")

    gjp = GEOJSON_DATA_DIR / "india_states.geojson"
    if not gjp.exists():
        _log.error("india_states.geojson not found at %s. Map will not render.", gjp)
        return None, None
    try:
        with open(gjp, encoding="utf-8") as f:
            gj = json.load(f)
        return gj, _detect_prop(gj)
    except Exception as exc:
        _log.error("Failed to parse india_states.geojson: %s", exc)
        return None, None

# ════════════════════════════════════════════════════════════════════
# 4.  CHART BUILDERS
# ════════════════════════════════════════════════════════════════════


def map_height_from_width(width: int) -> int:
    # Continuous instead of bucketed, so a 14" laptop's content width
    # (typically ~1000-1200px after the sidebar) gets a proportionate
    # height instead of being rounded up to whatever the nearest bucket
    # used to be.
    return round(_interp(width, [
        (640, 500), (1000, 750), (1200, 800), (1400, 850),
        (1800, 1000), (2400, 1150),
    ]))


def plot_height_from_width(width: int) -> int:
    # NOTE: `width` here is expected to be CHART_WIDTH — the actual
    # rendered width of one half of a 2-column chart row, not the full
    # window/content width. A 1920px laptop window nets a ~820px chart
    # column; a 2560px monitor nets a ~1140px column. The anchors below
    # are calibrated to that (roughly 300–1800px) range so the height
    # keeps growing meaningfully all the way up to large monitors,
    # instead of the old anchors (tuned for full window width, topping
    # out at 2200px — a figure a half-column practically never reaches)
    # which made every screen from a 14" laptop up to a 32" 4K monitor
    # land in the same flat, undersized range.
    return round(_interp(width, [
        (300, 360), (500, 410), (700, 460), (822, 500),
        (1000, 580), (1142, 660), (1400, 740), (1800, 820),
    ]))


# Standard browser rendering is ~96 CSS px/inch. Plotly's exported-image
# "scale" multiplies the on-screen pixel dimensions, so scale = target_dpi
# / 96 gives an export at roughly that effective print DPI. 400 / 96 ≈
# 4.17; export width/height are set generously large on top of that so
# the download is a big, print-quality image rather than a small chart
# blown up past its native resolution.
_HIRES_SCALE = round(400 / 96, 2)


def _hires_plot_config(filename: str, width: int = 1800, height: int = 1100, extra: dict | None = None) -> dict:
    """Plotly config that shows just the camera/download modebar button and
    exports a large, 400-DPI-equivalent PNG when clicked."""
    cfg = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "zoom2d", "pan2d", "select2d", "lasso2d",
            "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
        ],
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "width": width,
            "height": height,
            "scale": _HIRES_SCALE,
        },
    }
    if extra:
        cfg.update(extra)
    return cfg


_BASE_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=650,
    margin=dict(t=10, b=70, l=56, r=64),
    legend=dict(orientation="h", y=1.07, x=0, font=dict(size=14)),
    bargap=0.30,
    hovermode="x unified",
    xaxis=dict(
        tickangle=-40,
        showgrid=False,
        tickfont=dict(size=11),
        showline=True,
        linecolor="#ddd",
        linewidth=1,
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#f0f0f0",
        zeroline=False,
        tickfont=dict(size=11),
    ),
)


def _fig_registrations(mon: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=mon["month_lbl"],
        y=mon["total"],
        name="Monthly new",
        marker_color="rgba(180,245,180,.55)",
        marker_line_color="rgba(34,139,34,.40)",
        marker_line_width=1,
        yaxis="y2",
        zorder=1,
        hovertemplate="<b>%{x}</b><br>Monthly: <b>%{y:,}</b><extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=mon["month_lbl"],
        y=mon["cumulative"],
        name="Cumulative",
        mode="lines",
        line=dict(color="#6036f8", width=3),
        yaxis="y1",
        zorder=10,
        hovertemplate="<b>%{x}</b><br>Cumulative: <b>%{y:,}</b><extra></extra>",
    ))

    if not mon.empty:
        last = mon.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last["month_lbl"]],
            y=[last["cumulative"]],
            mode="markers",
            name="Latest point",
            marker=dict(size=24, color="rgba(96, 54, 248, 0.2)", line=dict(width=0)),
            hoverinfo="skip",
            showlegend=False,
            yaxis="y1",
            zorder=11,
        ))
        fig.add_trace(go.Scatter(
            x=[last["month_lbl"]],
            y=[last["cumulative"]],
            mode="markers",
            name="Latest point",
            marker=dict(size=10, color="#6036f8", line=dict(width=1, color="#ffffff")),
            hoverinfo="skip",
            showlegend=False,
            yaxis="y1",
            zorder=12,
        ))

    reg_layout = {
        **_BASE_LAYOUT,
        "height": 650,
        "xaxis": dict(
            tickangle=45,
            tickfont=dict(size=14, color="black"),
        ),
        "yaxis": dict(
            range=[
                0,
                mon["cumulative"].max() + (5000 if mon["cumulative"].max() > 1000 else 50),
            ],
            title=dict(text="Cumulative  Screening", font=dict(size=14, color="#6036f8")),
            showgrid=True,
            gridcolor="#f0f0f0",
            zeroline=False,
            tickfont=dict(size=14, color="#6036f8"),
        ),
        "yaxis2": dict(
            range=[
                0,
                mon["total"].max() + (500 if mon["total"].max() > 1000 else 50),
            ],
            title=dict(text="Monthly  Screening", font=dict(size=14, color="#3f9d3f")),
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=14, color="#3f9d3f"),
        ),
    }
    fig.update_layout(**reg_layout)
    fig.update_layout(height=plot_height_from_width(CHART_WIDTH))
    return fig


def _fig_status(mon: pd.DataFrame, status_label: str, suspicious_label: str) -> go.Figure:
    m = mon[mon["total"] > 0].copy()

    susp_denom = m["suspicious"].replace(0, np.nan)
    m["high_pct_of_susp"] = (m["high_risk"] / susp_denom * 100).round(1).fillna(0)
    m["low_pct_of_susp"]  = (m["low_risk"]  / susp_denom * 100).round(1).fillna(0)

    rev_denom = m["reviewed_total"].replace(0, np.nan)
    m["high_bar"] = (m["high_risk"] / rev_denom * 100).round(1).fillna(0)
    m["low_bar"]  = (m["low_risk"]  / rev_denom * 100).round(1).fillna(0)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=m["month_lbl"],
        y=m["non_susp_pct"],
        name="Non-suspicious",
        marker_color="#6B6B6B",
        customdata=np.stack([
            m["reviewed_total"] - m["suspicious"],
            m["non_susp_pct"],
        ], axis=-1),
        text=m["non_susp_pct"].apply(lambda v: f"{int(round(v))}%"),
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white", size=14, family="Arial Black"),
        hoverinfo="skip",
    ))

    fig.add_trace(go.Bar(
        x=m["month_lbl"],
        y=m["low_bar"],
        name=f"{status_label} · Low risk",
        marker_color=AMBER_LOW,
        customdata=np.stack([m["low_risk"], m["low_bar"]], axis=-1),
        text=m["low_bar"].apply(lambda v: f"{int(round(v))}%" if v >= 6 else ""),
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="#3a2a00", size=13, family="Arial Black"),
        hoverinfo="skip",
    ))

    fig.add_trace(go.Bar(
        x=m["month_lbl"],
        y=m["high_bar"],
        name=f"{status_label} · High risk",
        marker_color=AMBER_HIGH,
        customdata=np.stack([m["high_risk"], m["high_bar"]], axis=-1),
        text=m["high_bar"].apply(lambda v: f"{int(round(v))}%" if v >= 6 else ""),
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white", size=13, family="Arial Black"),
        hoverinfo="skip",
    ))

    # Hover-only traces (invisible bars, drive unified hover)
    fig.add_trace(go.Bar(
        x=m["month_lbl"], y=[0] * len(m),
        name=f"{status_label} · High risk hover",
        marker=dict(color=AMBER_HIGH, line=dict(color=AMBER_HIGH, width=0)),
        showlegend=False,
        customdata=np.stack([m["high_risk"], m["high_bar"]], axis=-1),
        hovertemplate=(
            "&nbsp;&nbsp;High risk: <b>%{customdata[0]:,}</b> "
            "(%{customdata[1]:.1f}% of screened)<extra></extra>"
        ),
    ))
    fig.add_trace(go.Bar(
        x=m["month_lbl"], y=[0] * len(m),
        name=f"{status_label} · Low risk hover",
        marker=dict(color=AMBER_LOW, line=dict(color=AMBER_LOW, width=0)),
        showlegend=False,
        customdata=np.stack([m["low_risk"], m["low_bar"]], axis=-1),
        hovertemplate=(
            "&nbsp;&nbsp;Low risk: <b>%{customdata[0]:,}</b> "
            "(%{customdata[1]:.1f}% of screened)<extra></extra>"
        ),
    ))
    fig.add_trace(go.Bar(
        x=m["month_lbl"], y=[0] * len(m),
        name=f"{status_label} · Suspicious hover",
        marker=dict(color="#FFFFFF", line=dict(color="#FFFFFF", width=0)),
        showlegend=False,
        customdata=np.stack([m["suspicious"], m["susp_pct"]], axis=-1),
        hovertemplate=(
            "Suspicious: <b>%{customdata[0]:,}</b> "
            "(%{customdata[1]:.1f}%)<extra></extra>"
        ),
    ))
    fig.add_trace(go.Bar(
        x=m["month_lbl"], y=[0] * len(m),
        name="Non-suspicious hover",
        marker=dict(color="#484848", line=dict(color="#484848", width=0)),
        showlegend=False,
        customdata=np.stack([
            m["reviewed_total"] - m["suspicious"],
            m["non_susp_pct"],
        ], axis=-1),
        hovertemplate=(
            "Non-suspicious: <b>%{customdata[0]:,}</b> "
            "(%{customdata[1]:.1f}%)<extra></extra>"
        ),
    ))

    # Legend tends to wrap to 2 lines on narrow screens (3 entries don't
    # fit on one line), so give it a smaller font and more headroom there
    # to avoid colliding with the bars. Sized continuously off CHART_WIDTH
    # (this chart's actual rendered width — half of CONTENT_WIDTH on
    # tablet/laptop/desktop, full width on phone) rather than three fixed
    # buckets, so a 14" laptop's narrower half-column doesn't get the same
    # font/margin as a large monitor's and wrap/clip against the bars.
    legend_font = round(_interp(CHART_WIDTH, [
        (300, 10), (450, 11), (600, 12), (800, 13), (1000, 14),
    ]))
    top_margin = round(_interp(CHART_WIDTH, [
        (300, 90), (450, 80), (600, 72), (800, 66), (1000, 64),
    ]))
    legend_y = _interp(CHART_WIDTH, [
        (300, 1.24), (450, 1.20), (600, 1.18), (800, 1.16), (1000, 1.16),
    ])

    susp_layout = {
        **_BASE_LAYOUT,
        "height": 650,
        "barmode": "stack",
        "margin": dict(t=top_margin, b=70, l=56, r=20),
        "legend": dict(
            orientation="h", y=legend_y, x=0,
            font=dict(size=legend_font),
            xanchor="left",
        ),
        "yaxis": dict(
            range=[0, 100],
            ticksuffix="%",
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=14, color="black"),
        ),
        "xaxis": dict(
            tickangle=45,
            tickfont=dict(size=14, color="black"),
        ),
    }
    fig.update_layout(**susp_layout)
    fig.update_layout(height=plot_height_from_width(CHART_WIDTH))
    return fig

@st.cache_data(max_entries=4, ttl=3600, show_spinner=False)
def _fig_sankey_phase2(df: pd.DataFrame, sw: int = 1200) -> go.Figure:
    """
    Sankey — AI-Enabled Screening pathway.
    """
    ai_col = _ai_result_col(df)
    if ai_col is None or "provisional_diagnosis" not in df.columns:
        return go.Figure()

    ai_s_m  = _present_mask(df[ai_col]) & _norm(df[ai_col]).eq("suspicious")
    ai_ns_m = _present_mask(df[ai_col]) & _norm(df[ai_col]).eq("non suspicious")
    tr_m    = _present_mask(df["provisional_diagnosis"])
    sc_m    = (
        _norm(df["suspicion"]).eq("suspicious")
        if "suspicion" in df.columns
        else pd.Series(False, index=df.index)
    )
    if "risk" in df.columns:
        hi_m = sc_m & _is_high_risk(df["risk"])
        lo_m = sc_m & _is_low_risk(df["risk"])
    else:
        hi_m = pd.Series(False, index=df.index)
        lo_m = pd.Series(False, index=df.index)

    N = lambda m: int(m.sum())

    ai_s  = N(ai_s_m);  ai_ns = N(ai_ns_m)
    total = ai_s + ai_ns

    tr_s  = N(ai_s_m  &  tr_m)
    pend  = N(ai_s_m  & ~tr_m)
    tr_ns = N(ai_ns_m &  tr_m)
    norev = N(ai_ns_m & ~tr_m)

    tele_rev   = tr_s + tr_ns
    tele_susp  = N(tr_m &  sc_m)
    tele_nsusp = N(tr_m & ~sc_m)

    susp_from_ai_s   = N(ai_s_m  & tr_m &  sc_m)
    susp_from_ai_ns  = N(ai_ns_m & tr_m &  sc_m)
    nsusp_from_ai_s  = N(ai_s_m  & tr_m & ~sc_m)
    nsusp_from_ai_ns = N(ai_ns_m & tr_m & ~sc_m)

    high = N(hi_m);  low = N(lo_m)

    high_ai_s  = N(ai_s_m  & hi_m)
    low_ai_s   = N(ai_s_m  & lo_m)
    high_ai_ns = N(ai_ns_m & hi_m)
    low_ai_ns  = N(ai_ns_m & lo_m)

    def pf(v, b): return f"{v / b * 100:.1f}%" if b else "-%"

    labels = [
        f"AI Suspicious\n{ai_s} ({pf(ai_s, total)})",
        f"AI Non-Suspicious\n{ai_ns} ({pf(ai_ns, total)})",
        f"Pending TSD\n{pend} ({pf(pend, total)})",
        "",
        f"Total TSD\n{tele_rev} ({pf(tele_rev, total)})",
        "",
        f"TSD Non-Suspicious\n{tele_nsusp} ({pf(tele_nsusp, total)})",
    ]

    customdata_nodes = [
        (f"AI Suspicious: {ai_s} ({pf(ai_s, total)} of total screened)<br>"
         f"Tele-Diagnosis High Risk: {high_ai_s} | Tele-Diagnosis Low Risk: {low_ai_s}"),
        (f"AI Non-Suspicious: {ai_ns} ({pf(ai_ns, total)} of total screened)<br>"
         f"Tele-Diagnosis High Risk: {high_ai_ns} | Tele-Diagnosis Low Risk: {low_ai_ns}"),
        f"Pending Tele-Diagnosis: {pend} ({pf(pend, total)} of total screened)",
        f"Not Reviewed: {norev} ({pf(norev, total)} of total screened)",
        (f"Total Tele-Diagnosis: {tele_rev} ({pf(tele_rev, total)} of total screened)<br>"
         f"From AI-Suspicious: {tr_s} ({pf(tr_s, ai_s)}) | From AI-Non-Suspicious: {tr_ns} ({pf(tr_ns, ai_ns)})"),
        (f"Tele-Diagnosis Suspicious: {tele_susp} ({pf(tele_susp, total)} of total screened)<br>"
         f"High Risk: {high} ({pf(high, total)}) | Low Risk: {low} ({pf(low, total)})<br>"
         f"Tele-Diagnosis from AI Suspicious: {susp_from_ai_s} ({pf(susp_from_ai_s, ai_s)}) | Tele-Diagnosis from AI Non-Suspicious: {susp_from_ai_ns} ({pf(susp_from_ai_ns, ai_ns)})<br>"
         f"From AI-Suspicious path: High {high_ai_s}, Low {low_ai_s}<br>"
         f"From AI-Non-Suspicious path: High {high_ai_ns}, Low {low_ai_ns}"),
        (f"Tele-Diagnosis Non-Suspicious: {tele_nsusp} ({pf(tele_nsusp, total)} of total screened)<br>"
         f"Tele-Diagnosis from AI Suspicious: {nsusp_from_ai_s} ({pf(nsusp_from_ai_s, ai_s)}) | Tele-Diagnosis from AI Non-Suspicious: {nsusp_from_ai_ns} ({pf(nsusp_from_ai_ns, ai_ns)})"),
    ]

    node_colors = [
        AMBER_HIGH, "#484848", "#c8c8c8", "#e0e0e0",
        "#c87d18", AMBER_HIGH, "#484848",
    ]

    # -------- Positions: AI Non‑Suspicious moved up for tighter gap --------
    node_x = [0.06, 0.06, 0.45, 0.45, 0.58, 0.80, 0.80]
    node_y = [0.12, 0.70, 0.06, 0.94, 0.50, 0.22, 0.78]

    links = []
    link_labels = []
    def _add_link(src: int, tgt: int, val: int, color: str, label: str = "") -> None:
        if val > 0:
            pct_total = (val / total * 100) if total else 0.0
            links.append((src, tgt, val, color, pct_total))
            link_labels.append(label)

    _add_link(0, 2, pend,       "rgba(100,180,240,0.30)")
    _add_link(0, 4, tr_s,       "rgba(224,99,26,0.30)")
    _add_link(1, 3, norev,      "rgba(100,180,240,0.30)")
    _add_link(1, 4, tr_ns,      "rgba(100,180,240,0.30)")
    _add_link(4, 5, tele_susp,  "rgba(224,99,26,0.50)")
    _add_link(4, 6, tele_nsusp, "rgba(100,180,240,0.30)")

    risk_start = len(labels)
    if high > 0 and low > 0:
        labels.extend([
            f"High Risk\n{high} ({pf(high, total)})",
            f"Low Risk\n{low} ({pf(low, total)})",
        ])
        customdata_nodes.extend([
            (f"High Risk: {high} ({pf(high, total)} of total screened)<br>"
             f"From AI-Suspicious path: {high_ai_s} ({pf(high_ai_s, ai_s)}) | From AI-Non-Suspicious path: {high_ai_ns} ({pf(high_ai_ns, ai_ns)})"),
            (f"Low Risk: {low} ({pf(low, total)} of total screened)<br>"
             f"From AI-Suspicious path: {low_ai_s} ({pf(low_ai_s, ai_s)}) | From AI-Non-Suspicious path: {low_ai_ns} ({pf(low_ai_ns, ai_ns)})"),
        ])
        node_colors.extend(["#D94040", AMBER_LOW])
        node_x.extend([1.00, 1.00])
        node_y.extend([-0.10, 0.16])
        _add_link(5, risk_start,     high, "rgba(217,64,64,1)")
        _add_link(5, risk_start + 1, low,  "rgba(247,197,72,0.40)")
    elif high > 0:
        labels.append(f"High Risk\n{high} ({pf(high, total)})")
        customdata_nodes.append(
            f"High Risk: {high} ({pf(high, total)} of total screened)<br>"
            f"From AI-Suspicious path: {high_ai_s} ({pf(high_ai_s, ai_s)}) | From AI-Non-Suspicious path: {high_ai_ns} ({pf(high_ai_ns, ai_ns)})"
        )
        node_colors.append("#D94040")
        node_x.append(1.00)
        node_y.append(0.10)
        _add_link(5, risk_start, high, "rgba(217,64,64,0.32)")
    elif low > 0:
        labels.append(f"Low Risk\n{low} ({pf(low, total)})")
        customdata_nodes.append(
            f"Low Risk: {low} ({pf(low, total)} of total screened)<br>"
            f"From AI-Suspicious path: {low_ai_s} ({pf(low_ai_s, ai_s)}) | From AI-Non-Suspicious path: {low_ai_ns} ({pf(low_ai_ns, ai_ns)})"
        )
        node_colors.append(AMBER_LOW)
        node_x.append(1.00)
        node_y.append(0.10)
        _add_link(5, risk_start, low, "rgba(247,197,72,0.40)")

    # -------- Squeeze vertically: narrower safe band --------
    if node_y:
        y_min, y_max = min(node_y), max(node_y)
        safe_low, safe_high = 0.08, 0.92   # tighter than before
        if y_max > y_min:
            span   = y_max - y_min
            node_y = [safe_low + ((y - y_min) / span) * (safe_high - safe_low) for y in node_y]
        else:
            node_y = [min(max(y, safe_low), safe_high) for y in node_y]

    if not links:
        return go.Figure()
    srcs, tgts, vals, clrs, pcts = zip(*links)

    used_nodes = sorted(set(srcs) | set(tgts))
    remap      = {old: new for new, old in enumerate(used_nodes)}

    labels_f           = [labels[i] for i in used_nodes]
    customdata_nodes_f = [customdata_nodes[i] for i in used_nodes]
    node_colors_f      = [node_colors[i] for i in used_nodes]
    node_x_f           = [node_x[i] for i in used_nodes]
    node_y_f           = [node_y[i] for i in used_nodes]

    srcs_f = [remap[s] for s in srcs]
    tgts_f = [remap[t] for t in tgts]
    pcts_f = list(pcts)

    vals_f = [max(v, total * 0.025) if total else v for v in vals]
    link_customdata = np.stack([list(vals), pcts_f], axis=-1)

    # -------- Height & margins: slightly shorter, balanced bottom --------
    # Height is capped off a 1920×1200-laptop-equivalent column width
    # (~822px) rather than the actual (possibly much wider) `sw`. Past
    # that point a bigger monitor gives the sankey more *horizontal*
    # room (via width="stretch" in the caller) but the figure stops
    # growing *taller* — otherwise a 2560×1440 screen's ~1142px column
    # pushed the height anchors well past what looked right at
    # 1920×1200, making the chart look oversized/sparse rather than
    # just bigger.
    _height_sw = min(sw, 822)
    fig_height = plot_height_from_width(_height_sw) + 140
    top_mg     = max(40, int(fig_height * 0.05))
    bot_mg     = max(120, int(fig_height * 0.08)) 
    right_mg   = max(80, int(sw * 0.12))

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
        node=dict(
            pad=22,
            thickness=20,
            line=dict(color="rgba(0,0,0,0.25)", width=1),
            label=labels_f,
            customdata=customdata_nodes_f,
            color=node_colors_f,
            x=node_x_f,
            y=node_y_f,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        link=dict(
            source=srcs_f,
            target=tgts_f,
            value=vals_f,
            customdata=link_customdata,
            color=list(clrs),
            label=link_labels,
            hovertemplate=(
                "From: %{source.customdata}<br>"
                "→ To: %{target.customdata}<br>"
                "Count: %{customdata[0]:,} (%{customdata[1]:.1f}% of total screened)<extra></extra>"
            ),
        ),
    ))

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=fig_height,
        margin=dict(t=top_mg, b=bot_mg, l=10, r=right_mg),
        font=dict(size=10, family="Arial, sans-serif", color="#666666"),
    )
    fig.update_traces(
        textfont=dict(color="#666666", size=10, family="Arial, sans-serif"),
        selector=dict(type="sankey"),
    )
    return fig


# ════════════════════════════════════════════════════════════════════
# 5.  UI COMPONENT HELPERS
# ════════════════════════════════════════════════════════════════════
def _duration_text(df: pd.DataFrame) -> str:
    """Return a formatted duration string like '(Jun 2023 – Dec 2024 · 18 months)'."""
    if "date_of_case_registered" not in df.columns:
        return ""
    dates = df["date_of_case_registered"].dropna()
    if dates.empty:
        return ""
    start, end = dates.min(), dates.max()
    months = (end.year - start.year) * 12 + (end.month - start.month) + 1
    month_label = "month" if months == 1 else "months"
    return f"({start.strftime('%b %Y')} – {end.strftime('%b %Y')} · {months} {month_label})"

def _card_font_sizes(card_w: int, ticker: bool = False) -> tuple[int, int, str, int]:
    """Continuous font/padding/height sizing for the HTML metric cards,
    driven by the card's *actual* rendered width (CONTENT_WIDTH divided by
    however many st.columns it shares a row with — see _row_width())
    rather than the raw window width. This is what a fixed 3-bucket
    (phone/tablet/desktop) scheme got wrong: a 4-across card row on a 14"
    laptop has a much narrower per-card width than a 4-across row on a
    big external monitor, even though both windows fall in the same
    "desktop" bucket — the old fixed desktop font size overflowed the
    laptop's narrower card and, combined with the card's overflow:hidden,
    clipped the bottom scrolling state-name ticker line.
    """
    big = round(_interp(card_w, [
        (160, 18), (220, 22), (300, 27), (380, 31), (460, 34), (600, 36),
    ]))
    sub = round(_interp(card_w, [
        (160, 10), (220, 12), (300, 13), (380, 15), (460, 16), (600, 17),
    ]))
    pad_h = round(_interp(card_w, [(160, 12), (460, 24)]))
    pad_t = round(_interp(card_w, [(160, 7), (460, 10)]))
    pad_b = round(_interp(card_w, [(160, 5), (460, 8)]))
    height = round(_interp(card_w, [
        (160, 130), (220, 140), (300, 152), (380, 160), (460, 168), (600, 172),
    ]))
    if ticker:
        # The scrolling state-name line sits below the label and needs a
        # bit of extra headroom on top of the base card height, or its
        # last few pixels get clipped by the card's overflow:hidden.
        height += 12
    pad = f"{pad_t}px {pad_h}px {pad_b}px"
    return big, sub, pad, height


def _animated_metric_card(
    value: int,
    suffix: str = "",
    sub_text: str = "",
    duration_text: str = "",
    big_color: str = "#228B22",
    border_color: str = "#228B22",
    animate: bool = False,
    n_cols: int = 2,
    height_override: int | None = None,
    card_width: int | None = None,
    pct_value: float | None = None,
) -> None:
    safe_sub = sub_text          # allow HTML
    safe_dur_text = duration_text  # allow HTML
    safe_suffix = json.dumps(suffix)
    pct_static = f" ({pct_value}%)" if pct_value is not None else ""
    safe_static = html.escape(f"{value:,}{suffix}{pct_static}")
    dur_html = (
        f'<div style="font-size:12px;color:{big_color};font-style:italic;'
        f'margin-top:1px;margin-bottom:3px;">{safe_dur_text}</div>'
        if duration_text else ""
    )

    big_font, sub_font, pad, card_height = _card_font_sizes(
        card_width if card_width is not None else _row_width(n_cols)
    )
    if height_override is not None:
        card_height = height_override

    if animate:
        pct_js = (
            f"+' ('+(ease(p)*{pct_value}).toFixed(1)+'%)'" if pct_value is not None else ""
        )
        script   = f"""<script>
(function(){{
var el=document.getElementById('cnt');
var target={value};var sfx={safe_suffix};var start=null;
function ease(t){{return t<.5?2*t*t:-1+(4-2*t)*t;}}
function step(ts){{
  if(!start)start=ts;
  var p=Math.min((ts-start)/{COUNT_ANIM_MS},1);
  el.textContent=Math.round(ease(p)*target).toLocaleString('en-US')+sfx{pct_js};
  if(p<1)requestAnimationFrame(step);
}}
requestAnimationFrame(step);
}})();
</script>"""
        big_html = '<div class="bignum" id="cnt">0</div>'
    else:
        script   = ""
        big_html = f'<div class="bignum">{safe_static}</div>'

    _html = f"""<!DOCTYPE html><html><head><style>
                html,body{{margin:0;padding:0;overflow:hidden;}}
                body{{background:#fff;font-family:'Segoe UI',Arial,sans-serif;}}
                .card{{background:#fff;border-radius:14px;padding:{pad};
                    box-shadow:0 2px 16px rgba(0,0,0,.07);border-left:6px solid {border_color};
                    box-sizing:border-box;height:{card_height - 16}px;margin:8px;
                    display:flex;flex-direction:column;justify-content:center;overflow:hidden;}}
                .bignum{{font-size:{big_font}px;font-weight:800;color:{big_color};line-height:1.1;margin:2px 0;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
                .subtext{{font-size:{sub_font}px;font-weight:700;color:#000;margin-top:4px;
                    overflow:hidden;text-overflow:ellipsis;display:-webkit-box;
                    -webkit-line-clamp:2;-webkit-box-orient:vertical;}}
                </style></head><body>
                <div class="card">
                {big_html}
                {dur_html}
                <div class="subtext">{safe_sub}</div>
                </div>
                {script}
                </body></html>"""
    st.iframe(_html, height=card_height)


def _map_stat_card(
    value: int,
    label: str,
    items: list[str],
    big_color: str,
    border_color: str,
    n_cols: int = 4,
    height_override: int | None = None,
    card_width: int | None = None,
) -> None:
    """Animated card with count-up and scrolling state-name ticker.
    `n_cols` is how many st.columns share the row this card is placed in
    — used to size fonts/height off the card's actual rendered width
    (see _card_font_sizes) so the ticker line has enough headroom instead
    of getting clipped by the card's fixed overflow:hidden height.
    `height_override` forces a specific total card height — pass this
    alongside plain `_animated_metric_card`s in the same row so every card
    in the row ends up the same height instead of just the ticker cards
    being taller. `card_width` overrides the width used for font sizing —
    pass this when the row's columns are not equal width (see
    `_weighted_col_width`) instead of the equal n-way split
    `_row_width(n_cols)` would assume."""
    # Escape content sourced from external JSON to prevent HTML injection.
    safe_label = html.escape(label)
    safe_items = [html.escape(s) for s in items]
    items_str  = " · ".join(safe_items) if safe_items else ""

    card_w = card_width if card_width is not None else _row_width(n_cols)
    big_font, sub_font, pad, card_height = _card_font_sizes(card_w, ticker=True)
    if height_override is not None:
        card_height = height_override
    ticker_font = round(_interp(card_w, [(160, 9), (300, 10), (460, 11)]))

    scroll_html = ""
    if items_str:
        scroll_html = (
            '<div style="overflow:hidden;white-space:nowrap;margin-top:5px;">'
            f'<span style="display:inline-block;font-size:{ticker_font}px;color:#888;'
            f'animation:ov-scroll {max(8, len(items)*2)}s linear infinite;">'
            f'{items_str}&nbsp;&nbsp;&nbsp;</span></div>'
            '<style>@keyframes ov-scroll{'
            '0%{transform:translateX(100%)}100%{transform:translateX(-100%)}}'
            '</style>'
        )
    _html = f"""<!DOCTYPE html><html><head><style>
                html,body{{margin:0;padding:0;overflow:hidden;}}
                body{{background:#fff;font-family:'Segoe UI',Arial,sans-serif;}}
                .card{{background:#fff;border-radius:14px;padding:{pad};
                    box-shadow:0 2px 16px rgba(0,0,0,.07);border-left:6px solid {border_color};
                    box-sizing:border-box;height:{card_height - 16}px;margin:8px;
                    display:flex;flex-direction:column;justify-content:center;overflow:hidden;}}
                .bignum{{font-size:{big_font}px;font-weight:800;color:{big_color};line-height:1.1;margin:2px 0;}}
                .subtext{{font-size:{sub_font}px;font-weight:700;color:#000;margin-top:4px;
                    overflow:hidden;text-overflow:ellipsis;display:-webkit-box;
                    -webkit-line-clamp:2;-webkit-box-orient:vertical;}}
                @keyframes ov-scroll{{0%{{transform:translateX(100%)}}100%{{transform:translateX(-100%)}}}}
                </style></head><body>
                <div class="card">
                <div class="bignum" id="cnt">0</div>
                <div class="subtext">{safe_label}</div>
                {scroll_html}
                </div>
                <script>
                (function(){{
                var el=document.getElementById('cnt');
                var target={value};var start=null;
                function ease(t){{return t<.5?2*t*t:-1+(4-2*t)*t;}}
                function step(ts){{
                    if(!start)start=ts;
                    var p=Math.min((ts-start)/{COUNT_ANIM_MS},1);
                    el.textContent=Math.round(ease(p)*target).toLocaleString('en-US');
                    if(p<1)requestAnimationFrame(step);
                }}
                requestAnimationFrame(step);
                }})();
                </script>
                </body></html>"""
    st.iframe(_html, height=card_height)


def _animated_dualstat_card(
    left_label: str,
    left_value: int,
    left_pct: float,
    right_label: str,
    right_value: int,
    right_pct: float,
    border_color: str = "#F4A900",
    left_color: str = "#F4A900",
    right_color: str = "#D94040",
    height: int | None = None,
    font_size: int = 35,
) -> None:
    """Two-stat card (e.g. "Suspicious: N (P%) | High risk: N (P%)"),
    styled to match `_animated_metric_card`'s rounded/shadowed/left-accented
    look. Both counts AND their percentages count up together over
    COUNT_ANIM_MS instead of the old plain st.markdown version, which
    rendered every number — including the percentages — as static text
    that never animated at all."""
    safe_left_label  = html.escape(left_label)
    safe_right_label = html.escape(right_label)
    card_height = height if height is not None else 160

    _html = f"""<!DOCTYPE html><html><head><style>
                html,body{{margin:0;padding:0;overflow:hidden;}}
                body{{background:#fff;font-family:'Segoe UI',Arial,sans-serif;}}
                .card{{background:#fff;border-radius:14px;padding:10px 24px 8px;
                    box-shadow:0 2px 16px rgba(0,0,0,.07);border-left:6px solid {border_color};
                    box-sizing:border-box;height:{card_height - 16}px;margin:8px;
                    display:flex;flex-direction:column;justify-content:center;overflow:hidden;}}
                .row{{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 10px;
                    font-weight:800;font-size:{font_size}px;line-height:1.2;white-space:normal;
                    color:{left_color};}}
                .sep{{color:#ccc;font-weight:300;}}
                .high{{color:{right_color};}}
                </style></head><body>
                <div class="card">
                <div class="row">
                <span id="l">{safe_left_label}: 0 (0.0%)</span>
                <span class="sep">|</span>
                <span class="high" id="r">{safe_right_label}: 0 (0.0%)</span>
                </div>
                </div>
                <script>
                (function(){{
                var lEl=document.getElementById('l');
                var rEl=document.getElementById('r');
                var lTarget={left_value};var lPct={left_pct};
                var rTarget={right_value};var rPct={right_pct};
                var start=null;
                function ease(t){{return t<.5?2*t*t:-1+(4-2*t)*t;}}
                function step(ts){{
                    if(!start)start=ts;
                    var p=Math.min((ts-start)/{COUNT_ANIM_MS},1);
                    var e=ease(p);
                    lEl.textContent="{safe_left_label}: "+Math.round(e*lTarget).toLocaleString('en-US')+" ("+(e*lPct).toFixed(1)+"%)";
                    rEl.textContent="{safe_right_label}: "+Math.round(e*rTarget).toLocaleString('en-US')+" ("+(e*rPct).toFixed(1)+"%)";
                    if(p<1)requestAnimationFrame(step);
                }}
                requestAnimationFrame(step);
                }})();
                </script>
                </body></html>"""
    st.iframe(_html, height=card_height)


def _phone_deployment_plot_card(
    deployment_data: "list[dict] | dict",
    phones_total: int = 0,
    n_cols: int = 4,
    height: "int | None" = None,
    border_color: str = "#0771eb",
    card_width: int | None = None,
) -> None:
    """Card showing the total phones deployed plus their breakdown by
    period as a horizontal bar plot, styled as the same
    rounded/shadowed/left-accented HTML card as `_animated_metric_card` /
    `_map_stat_card` (rather than a Streamlit-native container + Plotly
    chart, which render with different corners/shadow/background and
    don't visually match the other cards in the row). `n_cols`/`height`
    follow the same sizing convention as the other cards so the whole row
    lines up. `card_width` overrides the width used for font sizing —
    pass this when the row's columns are not equal width (see
    `_weighted_col_width`).

    The `phones_total` figure count-up animates on load (same ease-in-out
    timing as `_animated_metric_card`/`_map_stat_card`) rather than
    rendering as static text.

    Labels wrap onto a second line rather than truncating with an
    ellipsis on the first available line — truncation (via line-clamp)
    is only a last-resort fallback if a label still doesn't fit after
    wrapping, which combined with the extra column width the row gives
    this card (see `_tab_overall`) should only bite at the very narrowest
    screen widths."""
    card_w = card_width if card_width is not None else _row_width(n_cols)
    big_font, sub_font, pad, card_height = _card_font_sizes(card_w)
    if height is not None:
        card_height = height

    d = deployment_data[0] if isinstance(deployment_data, list) else deployment_data
    labels: list[str] = []
    values: list[int] = []
    if d:
        try:
            labels = [str(k) for k in d.keys()]
            values = [int(v) for v in d.values()]
        except (TypeError, ValueError):
            labels, values = [], []

    bar_font = round(_interp(card_w, [(160, 10), (300, 12), (460, 13)]))
    label_font = round(_interp(card_w, [(160, 10), (300, 10), (460, 14)]))
    max_v = max(values) if values else 1

    rows_html = ""
    for lbl, val in zip(labels, values):
        pct = round(val / max_v * 100, 1) if max_v else 0
        rows_html += (
            '<div class="bar-row">'
            f'<div class="bar-label">{html.escape(lbl)}</div>'
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width:{pct}%;"></div>'
            '</div>'
            f'<div class="bar-value">{val:,}</div>'
            '</div>'
        )
    if not rows_html:
        rows_html = (
            '<div style="font-size:{0}px;color:#999;">No deployment '
            'breakdown available.</div>'
        ).format(sub_font)

    safe_label = html.escape("🚩 Deployment Sites: ")

    _html = f"""<!DOCTYPE html><html><head><style>
                html,body{{margin:0;padding:0;overflow:hidden;}}
                body{{background:#fff;font-family:'Segoe UI',Arial,sans-serif;}}
                .card{{background:#fff;border-radius:14px;padding:{pad};
                    box-shadow:0 2px 16px rgba(0,0,0,.07);border-left:6px solid {border_color};
                    box-sizing:border-box;height:{card_height - 16}px;margin:8px;
                    display:flex;flex-direction:column;justify-content:center;overflow:hidden;}}
                .cardtitle{{font-size:{sub_font}px;font-weight:700;color:#000;margin-bottom:6px;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:flex;
                    align-items:baseline;flex-shrink:0;}}
                .cardnumber{{font-size:{big_font}px;font-weight:800;color:#E0730B;margin-left:2px;}}
                .bar-row{{display:flex;align-items:center;margin:5px 0;flex-shrink:0;}}
                .bar-label{{flex:0 0 44%;max-width:44%;font-size:{label_font}px;color:#000;
                    font-weight:700;white-space:normal;overflow-wrap:break-word;line-height:1.2;
                    display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;
                    overflow:hidden;padding-right:6px;box-sizing:border-box;}}
                .bar-track{{flex:1;min-width:32px;height:16px;border-radius:3px;overflow:hidden;
                    align-self:center;}}
                .bar-fill{{background:{border_color};height:100%;border-radius:3px;}}
                .bar-value{{margin-left:8px;font-size:{bar_font}px;font-weight:700;
                    color:{border_color};white-space:nowrap;min-width:24px;text-align:right;
                    align-self:center;}}
                </style></head><body>
                <div class="card">
                <div class="cardtitle">{safe_label}<span class="cardnumber" id="cnt">0</span></div>
                {rows_html}
                </div>
                <script>
                (function(){{
                var el=document.getElementById('cnt');
                var target={phones_total};var start=null;
                function ease(t){{return t<.5?2*t*t:-1+(4-2*t)*t;}}
                function step(ts){{
                    if(!start)start=ts;
                    var p=Math.min((ts-start)/{COUNT_ANIM_MS},1);
                    el.textContent=Math.round(ease(p)*target).toLocaleString('en-US');
                    if(p<1)requestAnimationFrame(step);
                }}
                requestAnimationFrame(step);
                }})();
                </script>
                </body></html>"""
    st.iframe(_html, height=card_height)


@st.cache_data(max_entries=12, ttl=3600, show_spinner=False)
def _india_map_state_counts(df: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    state_col   = next((c for c in ("states", "state", "State") if c in df.columns), None)
    id_col_name = _id_col(df)
    state_total: dict[str, int] = {}
    state_curr:  dict[str, int] = {}

    if state_col and not df.empty:
        d = df.dropna(subset=[state_col]).copy()
        d["_sk"] = d[state_col].astype(str).apply(_norm_state)
        state_total = d.groupby("_sk")[id_col_name].count().to_dict()

        if "date_of_case_registered" in d.columns:
            max_dt = d["date_of_case_registered"].dropna().max()
            if pd.notna(max_dt):
                cm = d[
                    (d["date_of_case_registered"].dt.year  == max_dt.year) &
                    (d["date_of_case_registered"].dt.month == max_dt.month)
                ]
                state_curr = cm.groupby("_sk")[id_col_name].count().to_dict()

    return state_total, state_curr


def _fig_india_map(df: pd.DataFrame, map_data: dict, width: int = 1200) -> "go.Figure | None":
    geojson, state_prop = _load_india_geojson()
    if geojson is None:
        return None

    ongoing_raw = map_data.get("ongoing_states", [])
    future_raw  = map_data.get("upcoming_states", [])
    ongoing_set = {_norm_state(s) for s in ongoing_raw}
    future_set  = {_norm_state(s) for s in future_raw}
    state_total, state_curr = _india_map_state_counts(df)

    rows = []
    for feat in geojson.get("features", []):
        raw_name = feat.get("properties", {}).get(state_prop, "")
        key      = _norm_state(raw_name)
        if key in ongoing_set:
            status, z_val = "Ongoing", 2
        elif key in future_set:
            status, z_val = "Upcoming", 1
        else:
            status, z_val = "Not Covered", 0
        rows.append({
            "state":  raw_name,
            "z":      z_val,
            "status": status,
            "total":  state_total.get(key, 0),
            "curr":   state_curr.get(key, 0),
        })

    dmap = pd.DataFrame(rows)

    colorscale = [
        [0.00, "#dfe0ea"], [0.33, "#dfe0ea"],
        [0.34, "#F4CA67"], [0.66, "#F4CA67"],
        [0.67, "#64E64A"], [1.00, "#64E64A"],
    ]

    fig = go.Figure(go.Choropleth(
        geojson=geojson,
        locations=dmap["state"],
        z=dmap["z"],
        featureidkey=f"properties.{state_prop}",
        colorscale=colorscale,
        zmin=0, zmax=2,
        showscale=False,
        customdata=np.column_stack([dmap["status"], dmap["total"], dmap["curr"]]),
        hovertemplate=(
            "<b>%{location}</b><br>"
            "Status: %{customdata[0]}<br>"
            "Total Screened: <b>%{customdata[1]:,}</b><br>"
            "Current Month: <b>%{customdata[2]:,}</b>"
            "<extra></extra>"
        ),
        marker_line_color="#ffffff",
        marker_line_width=0.8,
    ))

    # Site-name labels next to pins, and the legend block below, are both
    # sized off `width` (the map's actual rendered width, passed in by the
    # caller as CONTENT_WIDTH) continuously rather than a single fixed
    # size — a 14" laptop's ~1000-1200px content width previously got the
    # same fixed sizes as an 1800px+ external monitor and overflowed.
    _pin_font_size = round(_interp(width, [
        (640, 11), (1000, 13), (1400, 15), (1800, 16),
    ]))

    # ── Site pins from map_data ──────────────────────────────────────
    map_sites = map_data.get("map_sites", [])

    _PIN_LABELS = {
        "district_sentinel": "District Level Deployment and Sentinel Site",
        "sentinel":          "Sentinel Site",
    }

    # Per-site text position overrides to avoid label collisions
    _TEXT_POS = {
        "West Bengal":           "top center",
        "Kolkata":               "middle right",
        "Mathura":               "middle right",
        "Kohima":                "middle right",
        "Goa":                   "middle right",
        "Thanjavur":             "middle right",
        "Delhi":                 "top center",
        "Varanasi":              "middle right",
        "Silchar":               "middle right",
        "Guwahati":              "top center",
        "KLE":                   "top center",
        "MSMF":                  "middle right",
        "Anekal":                "middle left",
    }

    # Subsite font colors by status — green for ongoing, yellow for
    # upcoming.
    _SUBSITE_COLORS = {
        "ongoing":  "#1CBD1C",
        "upcoming": "#C08909",
    }

    def _shown(d: dict) -> bool:
        """True if this site/subsite's map_show flag is 'yes' (case-
        insensitive). Only status='...' == 'no' items are excluded by
        this flag - `status` itself (ongoing/upcoming) does not affect
        whether something is plotted, only its color once it's on the map."""
        return str(d.get("map_show", "")).strip().lower() == "yes"

    for site_type, label in _PIN_LABELS.items():
        sites = [
            s for s in map_sites
            if s.get("type") == site_type and _shown(s)
        ]
        if not sites:
            continue
        base_color = "#b61ff7" if site_type == "district_sentinel" else "#2980B9"

        # Add each site as a separate trace so textposition can vary per pin
        for i, s in enumerate(sites):
            name = s.get("name", "")
            tpos = _TEXT_POS.get(name, "middle right")
            color = base_color

            # Hover text: site name, then only the subsites with
            # map_show == "yes", listed indented on their own line below
            # it, font-colored by status (green = ongoing, yellow =
            # upcoming).
            shown_subs = [sub for sub in s.get("subsites", []) if _shown(sub)]
            hover_lines = [f"<b>{html.escape(name)}</b>"]
            for sub in shown_subs:
                sub_name   = sub.get("name", "")
                sub_status = sub.get("status", "")
                sub_color  = _SUBSITE_COLORS.get(sub_status, "#484848")
                hover_lines.append(
                    f'&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:{sub_color};">'
                    f"{html.escape(sub_name)}</span>"
                )
            hover_text = "<br>".join(hover_lines)

            fig.add_trace(go.Scattergeo(
                lat=[s["coordinates"][0]],
                lon=[s["coordinates"][1]],
                mode="markers+text",
                marker=dict(
                    symbol="circle",
                    size=12,
                    color=color,
                    line=dict(color="white", width=1.5),
                ),
                text=[name],
                textposition=tpos,
                textfont=dict(size=_pin_font_size, color=color),
                name=label if i == 0 else "",
                customdata=[hover_text],
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=(i == 0),
            ))

    # ── Legend annotation ────────────────────────────────────────────
    # On phone there's no room for a 220px right-side legend column, so
    # it moves below the map instead, letting the map itself center and
    # use the full figure width.
    is_phone_map = width < 640

    if is_phone_map:
        fig.add_annotation(
            x=0.5,
            y=-0.06,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="top",
            showarrow=False,
            text=(
                '<span style="color:#64E64A;font-size:15px;">■</span>'
                '<span style="font-size:11px;"> Ongoing &nbsp;</span>'
                '<span style="color:#F4CA67;font-size:15px;">■</span>'
                '<span style="font-size:11px;"> Upcoming</span>'
                '<br>'
                '<span style="color:#b61ff7;font-size:13px;">⬤</span>'
                '<span style="font-size:10px;"> District/Sentinel Site &nbsp;</span>'
                '<span style="color:#2980B9;font-size:13px;">⬤</span>'
                '<span style="font-size:10px;"> Sentinel Site</span>'
            ),
            font=dict(size=11),
            align="center",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        )
    else:
        # Continuous legend sizing: was a single fixed 22px-square /
        # 16px-text block regardless of width, which is what overflowed
        # the (also-fixed) 220px right margin on a 14" laptop's narrower
        # content width. Both the glyph/text sizes and the x position now
        # scale down together with `width` so the legend still fits
        # inside whatever margin `margin_r` below actually reserves for it.
        _leg_text = round(_interp(width, [
            (640, 11), (900, 12), (1100, 13), (1400, 14), (1800, 15), (2400, 16),
        ]))
        _leg_box = _leg_text + 6
        _leg_dot = _leg_text + 4
        _leg_x = _interp(width, [(640, 0.66), (1100, 0.72), (1800, 0.75)])
        fig.add_annotation(
            x=_leg_x,
            y=0.25,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="middle",
            showarrow=False,
            text=(
                f'<span style="color:#64E64A;font-size:{_leg_box}px;">■</span>'
                f'<span style="font-size:{_leg_text}px;"> Ongoing</span>'
                '<br>'
                f'<span style="color:#F4CA67;font-size:{_leg_box}px;">■</span>'
                f'<span style="font-size:{_leg_text}px;"> Upcoming</span>'
                '<br><br>'
                f'<span style="color:#b61ff7;font-size:{_leg_dot}px;">⬤</span>'
                f'<span style="font-size:{_leg_text}px;"> District Level Deployment &amp; Sentinel Site</span>'
                '<br>'
                f'<span style="color:#2980B9;font-size:{_leg_dot}px;">⬤</span>'
                f'<span style="font-size:{_leg_text}px;"> Sentinel Site</span>'
            ),
            font=dict(size=_leg_text),
            align="left",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        )

    # Use explicit lat/lon bounds that fully cover India including J&K, PoK,
    # Ladakh and Andaman & Nicobar — prevents the map from clipping the north.
    fig.update_geos(
        visible=False,
        # Bounding box: lon 68°E–98°E, lat 6°N–38°N covers mainland + A&N + full J&K/PoK
        lonaxis_range=[68, 98],
        lataxis_range=[6, 38],
        projection_type="mercator",
    )
    if is_phone_map:
        # No right-side legend column on phone, so the map can center and
        # use the full figure width; extra bottom margin makes room for
        # the legend stacked underneath instead. A touch more top margin
        # keeps the always-visible zoom mode bar clear of the map outline.
        margin = dict(t=26, b=70, l=4, r=4)
    else:
        # Right margin reserved for the legend column — scales with width
        # so it always has (roughly) enough room for the legend text
        # above at that width, instead of a fixed 220px that was sized
        # for a large monitor and left too little room at laptop widths.
        _margin_r = round(_interp(width, [
            (640, 130), (900, 155), (1100, 175), (1400, 195), (1800, 215), (2400, 230),
        ]))
        margin = dict(t=26, b=30, l=10, r=_margin_r)  # tight top margin keeps stat cards close to the map

    fig.update_layout(
        height=map_height_from_width(width),
        hoverlabel_font_size=14,
        margin=margin,
        plot_bgcolor="white",
        paper_bgcolor="white",
        geo=dict(bgcolor="white"),
        dragmode=False,  # no click-drag pan/zoom — zoom is via the +/- buttons only
        showlegend=False,
    )
    # Scroll-wheel zoom stays off (it fights page scrolling); dedicated
    # +/- buttons are exposed via a trimmed mode bar in the config below.
    return fig


# ════════════════════════════════════════════════════════════════════
# 6.  TAB RENDERERS
# ════════════════════════════════════════════════════════════════════

@st.fragment
def _tab_overall(df: pd.DataFrame, df_map: "pd.DataFrame | None" = None) -> None:
    """Overall tab — shows combined Image-Based Screening + AI-Enabled Screening statistics."""
    if df.empty:
        st.info("No data matches the current filters.")
        return

    mon = monthly_stats(df, phase="phase1")
    if mon.empty:
        st.info("No monthly data available for the current selection.")
        return

    total_cum      = int(mon["cumulative"].iloc[-1])
    last_row       = mon.iloc[-1]
    last_n         = int(last_row["total"])
    last_lbl       = last_row["month_lbl"]
    reviewed_total = int(mon["reviewed_total"].sum())
    total_susp     = int(mon["suspicious"].sum())
    high_total     = int(mon["high_risk"].sum())

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

    ov_n = _cols(2, 2, 1)
    ov_cards = st.columns(ov_n, gap="large")
    col_l = ov_cards[0]
    col_r = ov_cards[1] if len(ov_cards) > 1 else ov_cards[0]

    with col_l:
        dur_text = _duration_text(df)
        _animated_metric_card(
            value=total_cum,
            suffix=" screened",
            sub_text=f"{last_n:,} screened in {last_lbl}",
            duration_text=dur_text,
            big_color="#228B22",
            border_color="#228B22",
            animate=True,
            n_cols=ov_n,
        )
        st.plotly_chart(
            _fig_registrations(mon),
            width="stretch",
            config=_hires_plot_config("overall_registrations"),
            key="ov_reg",
        )

    with col_r:
        # % is of total screened (total_cum) — matches the Site-wise
        # Summary table below, not the reviewed-only subset.
        susp_rate = round(total_susp / total_cum * 100, 1) if total_cum else 0
        high_pct  = round(high_total / total_cum * 100, 1) if total_cum else 0

        # Force this card to occupy the same total vertical footprint as
        # the Total Screened card in col_l (an _animated_metric_card
        # iframe of height card_height) — pass the same computed height
        # straight through, exactly as the Total Screened card's own
        # iframe does, instead of the old margin-bottom arithmetic that
        # was only needed to compensate for st.markdown's container
        # quirks (this card is now an iframe too, so that quirk no
        # longer applies).
        _, _, _, _dualstat_h = _card_font_sizes(_row_width(ov_n))
        _animated_dualstat_card(
            left_label="Suspicious",
            left_value=total_susp,
            left_pct=susp_rate,
            right_label="High risk",
            right_value=high_total,
            right_pct=high_pct,
            height=_dualstat_h,
        )
        st.plotly_chart(
            _fig_status(mon, "Suspicious", "Suspicious"),
            width="stretch",
            config=_hires_plot_config("overall_suspicious_status"),
            key="ov_susp",
        )

    # ── India State Map Section ──────────────────────────────────────
    map_data = _load_map_data()
    if map_data:
        st.markdown(
            "<hr style='border:none;border-top:1px solid #eee;margin:8px 0 4px;'>",
            unsafe_allow_html=True,
        )
        # NOTE: map_data.json's actual key is "deployment_phone_sites" —
        # "phones_deployed" is kept as a fallback only in case an older
        # JSON export still uses that name, so a stale file doesn't
        # silently show 0.
        phones          = int(map_data.get("deployment_phone_sites", map_data.get("phones_deployed", 0)))
        fhw             = int(map_data.get("fhw_trained",        0))
        ongoing         = map_data.get("ongoing_states", [])
        future          = map_data.get("upcoming_states",  [])
        deployment_data = map_data.get("deployment_data", [])

        mc_n = _cols(4, 3, 1)

        # The deployment card's period labels ("April 2026 onwards", etc.)
        # run longer than the other cards' short titles. Rather than a
        # fixed guess at how much extra room to give it, size the boost
        # off the longest actual label and give that column extra
        # relative width in the row — st.columns() then shrinks the other
        # columns proportionally to make room, instead of every column
        # claiming a fixed equal share regardless of what it needs to
        # display. `_phone_deployment_plot_card`'s own label wrapping (see
        # its docstring) is the last-resort fallback if a label still
        # doesn't fit after this redistribution.
        _dd = deployment_data[0] if isinstance(deployment_data, list) else deployment_data
        _max_label_len = max((len(str(k)) for k in _dd.keys()), default=0) if _dd else 0
        _deploy_extra = min(max(_max_label_len - 6, 0) * 0.045, 1.0)

        _deploy_card_idx = 2  # position of the deployment card in mc_cards below
        _col_weights = [1.0] * mc_n
        _col_weights[_deploy_card_idx % mc_n] += _deploy_extra
        mc_cols = st.columns(_col_weights, gap="small")

        def _mc_card_width(card_idx: int) -> int:
            return _weighted_col_width(_col_weights, card_idx % mc_n)

        # Two of these cards (_map_stat_card) carry a scrolling state-name
        # ticker line and need a bit more height than the plain
        # _animated_metric_cards — left to size themselves independently,
        # that height difference showed up as the row's boxes not lining
        # up. Compute one shared height (off the narrowest column in the
        # row — the worst case) up front and force every card in the row
        # to that height so they stay visually uniform.
        mc_card_h = _card_font_sizes(
            min(_weighted_col_width(_col_weights, i) for i in range(mc_n)),
            ticker=True,
        )[3]
        mc_cards = [
            (lambda: _map_stat_card(len(ongoing), "📍Ongoing States/UTs", ongoing, "#237213", "#237113", height_override=mc_card_h, card_width=_mc_card_width(0))),
            (lambda: _animated_metric_card(fhw,        "", "👩‍⚕️ Frontline Health Worker Trained", "", "#0771eb", "#0771eb", animate=True, height_override=mc_card_h, card_width=_mc_card_width(1))),
            (lambda: _phone_deployment_plot_card(deployment_data, phones_total=phones, height=mc_card_h, card_width=_mc_card_width(2))),
            (lambda: _map_stat_card(len(future),  "📍 Upcoming States", future,  "#F4CA67", "#F4CA67", height_override=mc_card_h, card_width=_mc_card_width(3))),
        ]
        for i, card_fn in enumerate(mc_cards):
            with mc_cols[i % mc_n]:
                card_fn()

        gap_offset = -4 if IS_PHONE else -12
        st.markdown(
            f"<div style='margin-top:{gap_offset}px;'></div>",
            unsafe_allow_html=True,
        )  # tighten gap above the map

        df_for_map = df_map if df_map is not None else df
        with st.spinner("Loading Map..."):
            fig_map = _fig_india_map(df_for_map, map_data, width=CONTENT_WIDTH)
        if fig_map is not None:
            st.plotly_chart(
                fig_map,
                width='stretch',
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    # Geo traces only expose zoomInGeo / zoomOutGeo / resetGeo /
                    # hoverClosestGeo on the mode bar — keep +/- zoom, reset,
                    # and the download button; drop just the hover-toggle icon.
                    "modeBarButtonsToRemove": ["hoverClosestGeo"],
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "india_coverage_map",
                        "width": 1650,
                        "height": 950,
                        "scale": _HIRES_SCALE,
                    },
                    "scrollZoom": False,
                    "responsive": True,
                    "doubleClick": "reset",
                },
                key="ov_india_map",
            )
        else:
            st.warning(
                "🗺️ Map unavailable — `india_states.geojson` could not be found or "
                "read. Check that the file is present in the configured data source."
            )

    # ── Site-wise Summary Table ───────────────────────────────────────
    # Site | Screened | Suspicious | High risk — % taken row-wise against
    # that site's total screened (same colours as the summary cards
    # above: green for screened, amber for suspicious, red for high risk).
    if "site_full_id" in df.columns:
        id_col = _id_col(df)
        reviewed_mask, suspicious_mask, _, _ = _choose_status_masks(df, phase="phase1")
        high_mask = (
            suspicious_mask & _is_high_risk(df["risk"])
            if "risk" in df.columns
            else pd.Series(False, index=df.index)
        )

        site_key = df["site_full_id"].astype(str)
        screened_by_site   = df.groupby(site_key)[id_col].nunique()
        suspicious_by_site = suspicious_mask.groupby(site_key).sum()
        high_by_site        = high_mask.groupby(site_key).sum()

        if "study_site_id" in df.columns:
            site_type_by_site = df.groupby(site_key)["study_site_id"].agg(
                lambda s: next((v for v in s if pd.notna(v) and str(v).strip()), "")
            )
        else:
            site_type_by_site = pd.Series(dtype=object)

        if "date_of_case_registered" in df.columns:
            min_date_by_site = df.groupby(site_key)["date_of_case_registered"].min()
        else:
            min_date_by_site = pd.Series(dtype="datetime64[ns]")

        # Exact map-JSON name per site (written by merge_data.py from
        # config.MAP_JSON_NAME_OVERRIDE) — lets stopped-status lookup use
        # an exact match instead of guessing via substring matching
        # against the display name, which silently misses cases like
        # "Goa Dental College & Hospital, Goa" (table) vs. "Goa Dental
        # College & Hospital (Sentinel Site)" (map JSON).
        if "map_site_name" in df.columns:
            map_name_by_site = df.groupby(site_key)["map_site_name"].agg(
                lambda s: next((v for v in s if pd.notna(v) and str(v).strip()), "")
            )
        else:
            map_name_by_site = pd.Series(dtype=object)

        # Group by site type — District rows first, then Hospital rows
        # (anything else/unlabeled sorts last) — and within each group,
        # highest Screened count first.
        def _site_sort_key(site: str) -> tuple[int, int]:
            st_type = str(site_type_by_site.get(site, "") or "").strip().lower()
            group = 0 if st_type == "district" else (1 if st_type == "hospital" else 2)
            return (group, -int(screened_by_site.get(site, 0)))

        sites_present = sorted(screened_by_site.index, key=_site_sort_key)

        if sites_present:
            # ---- Build stopped sites set from map_data ----
            # Walks both top-level sites and their subsites — a site can be
            # "stopped" at either level (e.g. Krishnagiri is stopped at the
            # top level, while Goa Dental College & Hospital is stopped only
            # as a subsite under an otherwise-ongoing "Goa" parent).
            stopped_names = set()
            for msite in map_data.get("map_sites", []):
                if msite.get("status") == "stopped":
                    stopped_names.add(msite["name"].strip().lower())
                for sub in msite.get("subsites", []):
                    if sub.get("status") == "stopped":
                        stopped_names.add(sub["name"].strip().lower())

            def is_stopped(site_name: str) -> bool:
                # Prefer the exact map_site_name bridge column; fall back
                # to the old fuzzy substring match only for parquet files
                # produced before this column existed.
                map_name = str(map_name_by_site.get(site_name, "") or "").strip().lower()
                if map_name:
                    return map_name in stopped_names
                site_lower = site_name.strip().lower()
                return any(stopped_name in site_lower for stopped_name in stopped_names)

            rows_html = []
            for site in sites_present:
                screened     = int(screened_by_site.get(site, 0))
                susp     = int(suspicious_by_site.get(site, 0))
                high     = int(high_by_site.get(site, 0))
                susp_pct = round(susp / screened * 100, 1) if screened else 0.0
                high_pct = round(high / screened * 100, 1) if screened else 0.0

                site_type_str = str(site_type_by_site.get(site, "") or "—")
                _site_type_lower = site_type_str.strip().lower()
                if _site_type_lower == "district":
                    site_type_color = "#0BB8E8"
                elif _site_type_lower == "hospital":
                    site_type_color = "#7A2DE4"
                else:
                    site_type_color = "#555"

                min_dt = min_date_by_site.get(site)
                start_date_str = min_dt.strftime('%d-%b-%Y') if pd.notna(min_dt) else "—"

                rows_html.append(
                    "<tr>"
                    f"<td style='padding:5px 10px;text-align:left;font-weight:600;"
                    f"color:#333;border-top:1px solid #eee;'>{html.escape(site)}</td>"
                    f"<td style='padding:5px 10px;text-align:center;font-weight:700;"
                    f"color:{site_type_color};border-top:1px solid #eee;white-space:nowrap;'>{html.escape(site_type_str)}</td>"
                    f"<td style='padding:5px 10px;text-align:center;font-weight:600;"
                    f"color:#555;border-top:1px solid #eee;white-space:nowrap;'>{html.escape(start_date_str)}</td>"
                    f"<td style='padding:5px 10px;text-align:center;font-weight:700;"
                    f"color:#228B22;border-top:1px solid #eee;'>{screened:,}</td>"
                    f"<td style='padding:5px 10px;text-align:center;font-weight:700;"
                    f"color:#F4A900;border-top:1px solid #eee;'>{susp:,} ({susp_pct}%)</td>"
                    f"<td style='padding:5px 10px;text-align:center;font-weight:700;"
                    f"color:#D94040;border-top:1px solid #eee;'>{high:,} ({high_pct}%)</td>"
                    "</tr>"
                )

            table_html = (
                "<div style='overflow-x:auto;'>"
                "<table style='width:100%;border-collapse:collapse;background:#fff;"
                "border-radius:12px;overflow:hidden;box-shadow:0 2px 16px rgba(0,0,0,.07);'>"
                "<thead><tr style='background:#fafafa;'>"
                "<th style='padding:10px 14px;text-align:left;font-size:14px;"
                "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
                "color:#555;'>Sites</th>"
                "<th style='padding:10px 14px;text-align:center;font-size:14px;"
                "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
                "color:#555;'>Site Type</th>"
                "<th style='padding:10px 14px;text-align:center;font-size:14px;"
                "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
                "color:#555;'>Start Date</th>"
                "<th style='padding:10px 14px;text-align:center;font-size:14px;"
                "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
                "color:#228B22;'>Total Screened</th>"
                "<th style='padding:10px 14px;text-align:center;font-size:14px;"
                "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
                "color:#F4A900;'>Suspicious</th>"
                "<th style='padding:10px 14px;text-align:center;font-size:14px;"
                "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
                "color:#D94040;'>High Risk</th>"
                "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table></div>"
            )

            st.markdown(
                "<hr style='border:none;border-top:1px solid #eee;margin:18px 0 10px;'>"
                "<div style='font-weight:700;font-size:25px;color:#333;"
                "margin-bottom:8px;'>🏥 Summary</div>",
                unsafe_allow_html=True,
            )
            st.markdown(table_html, unsafe_allow_html=True)

            footnote_text = str(map_data.get("footnote", "") or "").strip()
            if footnote_text:
                # footnote lines are '*'-prefixed and separated by literal
                # \n in the JSON; render each on its own line under the table.
                footnote_lines = [
                    line.strip() for line in footnote_text.split("\n") if line.strip()
                ]
                footnote_html = "<br>".join(html.escape(line) for line in footnote_lines)
                st.markdown(
                    f"<div style='margin-top:6px;font-size:12px;color:#888;"
                    f"font-style:italic;line-height:1.5;'>{footnote_html}</div>",
                    unsafe_allow_html=True,
                )


@st.fragment
def _tab_phase1(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No data matches the current filters.")
        return

    mon = monthly_stats(df, phase="phase1")
    if mon.empty:
        st.info("No monthly data available for the current selection.")
        return

    total_cum      = int(mon["cumulative"].iloc[-1])
    last_row       = mon.iloc[-1]
    last_n         = int(last_row["total"])
    last_lbl       = last_row["month_lbl"]
    reviewed_total = int(mon["reviewed_total"].sum())
    total_susp     = int(mon["suspicious"].sum())
    high_total     = int(mon["high_risk"].sum())

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

    p1_n = _cols(2, 2, 1)
    p1_cards = st.columns(p1_n, gap="large")
    col_l = p1_cards[0]
    col_r = p1_cards[1] if len(p1_cards) > 1 else p1_cards[0]

    with col_l:
        dur_text = _duration_text(df)
        _animated_metric_card(
            value=total_cum,
            suffix=" screened",
            sub_text=f"{last_n:,} screened in {last_lbl}",
            duration_text=dur_text,
            big_color="#228B22",
            border_color="#228B22",
            animate=True,
            n_cols=p1_n,
        )
        st.plotly_chart(
            _fig_registrations(mon),
            width="stretch",
            config=_hires_plot_config("phase1_registrations"),
            key="p1_reg",
        )

    with col_r:
        # % is of total screened (total_cum) — matches the Overall tab's
        # Site-wise Summary table, not the reviewed-only subset.
        susp_rate = round(total_susp / total_cum * 100, 1) if total_cum else 0
        high_pct  = round(high_total / total_cum * 100, 1) if total_cum else 0

        # See _tab_overall for why the height is passed straight through
        # to the iframe rather than the old margin-bottom arithmetic.
        _, _, _, _dualstat_h = _card_font_sizes(_row_width(p1_n))
        _animated_dualstat_card(
            left_label="Suspicious",
            left_value=total_susp,
            left_pct=susp_rate,
            right_label="High risk",
            right_value=high_total,
            right_pct=high_pct,
            height=_dualstat_h,
        )
        st.plotly_chart(
            _fig_status(mon, "Suspicious", "Suspicious"),
            width="stretch",
            config=_hires_plot_config("phase1_suspicious_status"),
            key="p1_susp",
        )


@st.fragment
def _tab_phase2(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No data matches the current filters.")
        return

    mon = monthly_stats(df, phase="phase2")
    if mon.empty:
        st.info("No monthly data available for the current selection.")
        return

    total_cum     = int(mon["cumulative"].iloc[-1])
    last_row      = mon.iloc[-1]
    last_n        = int(last_row["total"])
    last_lbl      = last_row["month_lbl"]
    total_screened = int(mon["total"].sum())
    ai_susp_total  = int(mon["ai_suspicious"].sum())
    ai_susp_rate   = round(ai_susp_total / total_screened * 100, 1) if total_screened else 0

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

    p2_n = _cols(2, 2, 1)
    p2_cards = st.columns(p2_n, gap="large")
    col_l = p2_cards[0]
    col_r = p2_cards[1] if len(p2_cards) > 1 else p2_cards[0]

    with col_l:
        dur_text = _duration_text(df)
        _animated_metric_card(
            value=total_cum,
            suffix=" screened",
            sub_text=f"{last_n:,} screened in {last_lbl}",
            duration_text=dur_text,
            big_color="#228B22",
            border_color="#228B22",
            animate=True,
            n_cols=p2_n,
        )
        st.plotly_chart(
            _fig_registrations(mon),
            width="stretch",
            config=_hires_plot_config("phase2_registrations"),
            key="p2_reg",
        )

    # Compute AI suspicious directly from df — mirrors sankey logic
    # (only rows with a valid AI result: suspicious or non-suspicious)
    ai_col        = _ai_result_col(df)
    if ai_col is not None:
        ai_s_m    = _present_mask(df[ai_col]) & _norm(df[ai_col]).eq("suspicious")
        ai_ns_m   = _present_mask(df[ai_col]) & _norm(df[ai_col]).eq("non suspicious")
        ai_s_tot  = int(ai_s_m.sum())
        ai_total  = int(ai_s_m.sum()) + int(ai_ns_m.sum())
    else:
        ai_s_tot  = 0
        ai_total  = 0
    # % is of total screened (total_screened) — matches the Overall tab's
    # Site-wise Summary table, not the AI-reviewed-only subset (ai_total).
    ai_susp_rate_r = round(ai_s_tot / total_screened * 100, 1) if total_screened else 0

    # Current month suspicious from mon (already computed per month)
    last_ai_susp    = int(mon.iloc[-1]["ai_suspicious"]) if "ai_suspicious" in mon.columns else 0
    last_n_screened = int(mon.iloc[-1]["total"])
    last_susp_rate  = round(last_ai_susp / last_n_screened * 100, 1) if last_n_screened else 0

    
    with col_r:
        _animated_metric_card(
            value=ai_s_tot,
            suffix=" suspicious",
            pct_value=ai_susp_rate_r,
            sub_text=(
                f"{last_ai_susp:,} suspicious in {last_lbl}"
                '<br><span style="font-size:11px; font-weight:500; color:#888888; font-style:italic;">'
                '(TSD = Tele-Specialist Diagnosis)'
                '</span>'
            ),
            duration_text="",
            big_color="#F4A900",
            border_color="#F4A900",
            animate=True,
            n_cols=p2_n,
        )

        # Download size matches the figure's own width→height formula
        # (plot_height_from_width(sw) + 80) at a large export width,
        # instead of a fixed 1500x850 that could mismatch the margins
        # baked into the figure at build time and clip content.
        _sankey_dl_w = 1800
        _sankey_dl_h = plot_height_from_width(_sankey_dl_w) + 80
        st.plotly_chart(
            _fig_sankey_phase2(df, CHART_WIDTH),
            width="stretch",
            config=_hires_plot_config(
                "phase2_ai_pathway_sankey", width=_sankey_dl_w, height=_sankey_dl_h
            ),
            key="p2_sankey",
        )


# ════════════════════════════════════════════════════════════════════
# 7.  PUBLIC ENTRY POINT — called by app.py's main()
# ════════════════════════════════════════════════════════════════════

def render(
    screen_width: int,
    df_all: pd.DataFrame,
    df_p1: pd.DataFrame,
    df_p2: pd.DataFrame,
    df_all_map: "pd.DataFrame | None" = None,
) -> None:
    """Render the full Monitoring Dashboard view: tab navigation bar
    (Overall / Image-Based Screening / AI-Enabled Screening), the
    active tab's content, and the footer.

    Parameters
    ----------
    screen_width : raw window width (px) detected by app.py, used to
                   (re)derive every responsive layout global below.
    df_all       : all phases combined, already filtered by the
                   caller's sidebar filters (date range, gender, study
                   setting, site) — feeds the Overall tab.
    df_p1        : phase == '1' (Image-Based Screening) data, same
                   filters applied — feeds the Image-Based Screening tab.
    df_p2        : phase == '2' (AI-Enabled Screening) data, same
                   filters applied — feeds the AI-Enabled Screening tab.
    df_all_map   : Overall-tab map snapshot (date + gender filters only,
                   no site filter), passed straight through to
                   _tab_overall.
    """
    _set_layout(screen_width)

    # ════════════════════════════════════════════════════════════════
    # Tab Navigation  —  Overall · Image-Based Screening · AI-Enabled Screening
    # ════════════════════════════════════════════════════════════════

    if "tab" not in st.session_state:
        st.session_state.tab = 0

    if IS_PHONE:
        tc1, tc2, tc3 = st.columns(3, gap="small")
    else:
        tc1, tc2, tc3, _sp = st.columns([1, 1, 1, 3.5])

    with tc1:
        if st.button(
            "Overall",
            key="btn_ov",
            type="primary" if st.session_state.tab == 0 else "secondary",
            width="stretch",
        ):
            st.session_state.tab = 0
            st.rerun()

    with tc2:
        if st.button(
            "Image-Based Screening",
            key="btn_p1",
            type="primary" if st.session_state.tab == 1 else "secondary",
            width="stretch",
        ):
            st.session_state.tab = 1
            st.rerun()

    with tc3:
        if st.button(
            "AI-Enabled Screening",
            key="btn_p2",
            type="primary" if st.session_state.tab == 2 else "secondary",
            width="stretch",
        ):
            st.session_state.tab = 2
            st.rerun()

    st.markdown(
        "<hr style='border:none;border-top:1.5px solid #ddd;margin:10px 0 18px;'>",
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════════
    # Render Active Tab
    # ════════════════════════════════════════════════════════════════

    if st.session_state.tab == 0:
        _tab_overall(df_all, df_all_map)
    elif st.session_state.tab == 1:
        _tab_phase1(df_p1)
    elif st.session_state.tab == 2:
        _tab_phase2(df_p2)


    # Footer
    if st.session_state.tab == 2:
        # Custom footer for tab 2: hr and text with negative margin to pull up
        st.markdown(
            "<hr style='border:none;border-top:1px solid #ddd;margin:-30px 0 5px;'>"
            '<div style="text-align:center;padding:2px 0;margin-top:20px;font-size:12px;color:#737373;">'
            '<b style="color:#0771eb;">Aarogya Aarohan</b>&nbsp;·&nbsp;'
            'TANUH Oral Cancer Screening Project<br>'
            'Email: <a href="mailto:oralcancerscreening@tanuh.ai" '
            'style="color:#0771eb;text-decoration:none;">oralcancerscreening@tanuh.ai</a>'
            '&nbsp;·&nbsp;© 2026'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # Default footer for other tabs
        st.markdown("---")
        st.markdown(
            '<div style="text-align:center;padding:2px 0;margin-top:10px;font-size:12px;color:#737373;">'
            '<b style="color:#0771eb;">Aarogya Aarohan</b>&nbsp;·&nbsp;'
            'TANUH Oral Cancer Screening Project<br>'
            'Email: <a href="mailto:oralcancerscreening@tanuh.ai" '
            'style="color:#0771eb;text-decoration:none;">oralcancerscreening@tanuh.ai</a>'
            '&nbsp;·&nbsp;© 2026'
            '</div>',
            unsafe_allow_html=True,
    )