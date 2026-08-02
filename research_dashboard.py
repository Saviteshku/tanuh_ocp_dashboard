"""
====================================================================
RESEARCH DASHBOARD — standalone module
Aarogya Aarohan / TANUH Oral Cancer Project

Two tabs:
  - "Descriptive": site‑wise summary table with the "Screened in
    <month>" column.
  - "Leaderboard": top‑10 FLW (field health worker) leaderboards —
    cases screened (stacked by Non‑suspicious / Suspicious·Low risk /
    Suspicious·High risk) and % of AI-Enabled Screening cases where
    the AI result was overridden.

Uses the same tab‑button style as the Monitoring Dashboard.
====================================================================
"""

from __future__ import annotations
import html
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

AMBER_HIGH = "#E0631A"
AMBER_LOW  = "#F7C548"


# ════════════════════════════════════════════════════════════════════
# Password gate (research_dash_password, read from map_data.json)
# ════════════════════════════════════════════════════════════════════
LOCAL_DATA_DIR = Path(
    os.environ.get(
        "OCP_DATA_DIR",
        r"/mnt/d/OneDrive/IISC/TANUH/OralCancer_Project/Raw_Data/Dashboard",
    )
)
MAP_DATA_PATH = LOCAL_DATA_DIR / "map_data.json"

SESSION_TIMEOUT_SECONDS = 5 * 60


@st.cache_data(ttl=3600, show_spinner=False)
def _load_map_data(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _research_dashboard_password() -> str | None:
    pw = _load_map_data(str(MAP_DATA_PATH)).get("research_dash_password")
    return str(pw) if pw else None


def _check_password() -> bool:
    authed_at = st.session_state.get("research_dash_authed_at")
    if authed_at is not None:
        if time.time() - authed_at <= SESSION_TIMEOUT_SECONDS:
            st.session_state.research_dash_authed_at = time.time()
            return True
        del st.session_state["research_dash_authed_at"]
        st.info("Session timed out after 5 minutes of inactivity — please re-enter the password.")

    correct_pw = _research_dashboard_password()
    if not correct_pw:
        st.error(
            "Research Dashboard password isn't configured — add "
            '"research_dash_password" to map_data.json.'
        )
        return False

    with st.form("research_dash_password_form"):
        entered = st.text_input("🔒 Password", type="password")
        submitted = st.form_submit_button("Unlock")

    if submitted:
        if entered == correct_pw:
            st.session_state.research_dash_authed_at = time.time()
            st.rerun()
        else:
            st.error("Incorrect password.")

    return False


# ════════════════════════════════════════════════════════════════════
# Helper functions (copied from monitoring_dashboard for independence)
# ════════════════════════════════════════════════════════════════════

_BLANK_LIKE = {"", "nan", "none", "-", ".", "na", "n/a", "null"}

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

def _is_high_risk(series: pd.Series) -> pd.Series:
    return _norm(series).eq("high risk")

def _is_low_risk(series: pd.Series) -> pd.Series:
    return _norm(series).eq("low risk")

def _ai_result_col(df: pd.DataFrame) -> str | None:
    for c in ("ai_result", "AI_Result", "AI Result"):
        if c in df.columns:
            return c
    return None

def _ai_override_col(df: pd.DataFrame) -> str | None:
    for c in ("ai_override", "AI_Override", "AI Override"):
        if c in df.columns:
            return c
    return None

# Values in the ai_override column that mean "no override happened" —
# anything else present (e.g. "Yes", an overridden diagnosis label) counts
# as an override. Blank-like values (see _BLANK_LIKE) are never an override.
_NOT_OVERRIDE_LIKE = {"no", "false", "0", "not overridden", "no override", "non override"}

def _override_mask(df: pd.DataFrame) -> pd.Series:
    col = _ai_override_col(df)
    if col is None:
        return pd.Series(False, index=df.index)
    return _present_mask(df[col]) & ~_norm(df[col]).isin(_NOT_OVERRIDE_LIKE)

def _reviewed_mask_phase1(df: pd.DataFrame) -> pd.Series:
    cols = df.columns
    prov = df["provisional_diagnosis"] if "provisional_diagnosis" in cols else pd.Series(False, index=df.index)
    susp = df["suspicion"]             if "suspicion" in cols             else pd.Series(False, index=df.index)
    risk = df["risk"]                  if "risk" in cols                  else pd.Series(False, index=df.index)
    return _present_mask(prov) & _present_mask(susp) & _present_mask(risk)

def _choose_status_masks_phase1(df: pd.DataFrame):
    reviewed_mask = _reviewed_mask_phase1(df)
    if "suspicion" in df.columns:
        suspicious_mask = reviewed_mask & _norm(df["suspicion"]).eq("suspicious")
    else:
        suspicious_mask = pd.Series(False, index=df.index)
    return reviewed_mask, suspicious_mask


def _valid_flw(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (rows with a non-blank flw_username, that column as a
    cleaned string Series aligned to those rows)."""
    if df.empty or "flw_username" not in df.columns:
        return df.iloc[0:0], pd.Series(dtype=object)
    flw = df["flw_username"].astype(str).str.strip()
    valid = flw.ne("") & ~flw.str.lower().isin(_BLANK_LIKE)
    return df.loc[valid], flw.loc[valid]


def _current_month_df(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return (rows from the most recent calendar month present in
    date_of_case_registered, that month's label e.g. 'Jul 2026'). Empty
    df / missing dates yield an empty frame and an empty label."""
    if df.empty or "date_of_case_registered" not in df.columns:
        return df.iloc[0:0], ""
    dates = df["date_of_case_registered"].dropna()
    if dates.empty:
        return df.iloc[0:0], ""
    max_dt = dates.max()
    mask = (
        (df["date_of_case_registered"].dt.year == max_dt.year) &
        (df["date_of_case_registered"].dt.month == max_dt.month)
    )
    return df.loc[mask], max_dt.strftime("%b %Y")


# ════════════════════════════════════════════════════════════════════
# Leaderboard — per-FLW stats & charts
# ════════════════════════════════════════════════════════════════════

def _leaderboard_flw_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-FLW totals: screened count, split into Non-suspicious /
    Suspicious·Low risk / Suspicious·High risk (high/low only come out of
    the suspicious subset — the non-suspicious bucket is never split)."""
    d, flw = _valid_flw(df)
    if d.empty:
        return pd.DataFrame()

    id_col = _id_col(d)
    reviewed_mask, suspicious_mask = _choose_status_masks_phase1(d)
    if "risk" in d.columns:
        high_mask = suspicious_mask & _is_high_risk(d["risk"])
        low_mask  = suspicious_mask & _is_low_risk(d["risk"])
    else:
        high_mask = pd.Series(False, index=d.index)
        low_mask  = pd.Series(False, index=d.index)
    non_susp_mask = reviewed_mask & ~suspicious_mask

    total    = d.groupby(flw)[id_col].nunique()
    high     = high_mask.groupby(flw).sum()
    low      = low_mask.groupby(flw).sum()
    non_susp = non_susp_mask.groupby(flw).sum()

    if "site_full_id" in d.columns:
        site = d.groupby(flw)["site_full_id"].agg(
            lambda s: next((v for v in s if pd.notna(v) and str(v).strip()), "")
        )
    else:
        site = pd.Series("", index=total.index, dtype=object)

    out = pd.DataFrame({"flw_username": total.index, "total": total.values}).set_index("flw_username")
    out["high"]     = high
    out["low"]      = low
    out["non_susp"] = non_susp
    out["site_full_id"] = site.reindex(out.index).fillna("")
    out = out.fillna(0)
    for c in ("total", "high", "low", "non_susp"):
        out[c] = out[c].astype(int)
    return out.reset_index()


def _nice_dtick(range_max: float, target_ticks: int = 5) -> float:
    """A round tick-step (1/2/2.5/5/10 x a power of ten) that yields
    roughly `target_ticks` ticks across [0, range_max]. Used so the
    axis always has a labeled tick sitting past the highest bar rather
    than the range simply ending in blank padding."""
    if range_max <= 0:
        return 1
    raw_step = range_max / target_ticks
    magnitude = 10 ** np.floor(np.log10(raw_step))
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * magnitude
        if step >= raw_step:
            return step
    return magnitude * 10


def _fig_leaderboard_flw_counts(df: pd.DataFrame, month_label: str = "") -> go.Figure:
    """Top-10 FLWs by total screened, horizontal stacked bar (highest at
    top, lowest at bottom). Hover shows the FLW's site_full_id plus
    Non-suspicious / Suspicious·Low risk / Suspicious·High risk counts
    together."""
    stats = _leaderboard_flw_counts(df)
    if stats.empty:
        return go.Figure()

    top = stats.sort_values("total", ascending=False).head(10)
    # Ascending order so the highest bar ends up plotted at the top of a
    # horizontal bar chart (Plotly stacks categories bottom-to-top).
    plot_df = top.sort_values("total", ascending=True)

    fig = go.Figure()
    # Invisible zero-width bars purely to surface the FLW's site and total
    # screened count in the unified hover tooltip alongside the visible
    # segments below.
    fig.add_trace(go.Bar(
        y=plot_df["flw_username"], x=[0] * len(plot_df), orientation="h",
        name="Site", marker=dict(color="rgba(0,0,0,0)"), showlegend=False,
        customdata=plot_df["site_full_id"],
        hovertemplate="Site: %{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=plot_df["flw_username"], x=[0] * len(plot_df), orientation="h",
        name="Total screened", marker=dict(color="rgba(0,0,0,0)"), showlegend=False,
        customdata=plot_df["total"],
        hovertemplate="Total screened: <b>%{customdata:,}</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=plot_df["flw_username"], x=plot_df["non_susp"], orientation="h",
        name="Non-suspicious", marker_color="#6B6B6B",
        hovertemplate="Non-suspicious: <b>%{x:,}</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=plot_df["flw_username"], x=plot_df["low"], orientation="h",
        name="Suspicious · Low risk", marker_color=AMBER_LOW,
        hovertemplate="Suspicious · Low risk: <b>%{x:,}</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=plot_df["flw_username"], x=plot_df["high"], orientation="h",
        name="Suspicious · High risk", marker_color=AMBER_HIGH,
        hovertemplate="Suspicious · High risk: <b>%{x:,}</b><extra></extra>",
    ))

    max_total  = float(plot_df["total"].max())
    range_max  = max_total + 20
    x_title = f"Cases screened{f' — {month_label}' if month_label else ''}"
    fig.update_layout(
        barmode="stack",
        height=max(420, 46 * len(plot_df) + 100),
        margin=dict(t=10, b=40, l=10, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="y unified",
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=12, color="black")),
        xaxis=dict(
            title=dict(text=x_title, font=dict(color="black")),
            tickfont=dict(color="black"),
            range=[0, range_max],
            tick0=0,
            dtick=_nice_dtick(range_max),
            showgrid=True, gridcolor="#f0f0f0", zeroline=False,
        ),
        yaxis=dict(
            automargin=True,
            tickfont=dict(size=12, color="black"),
        ),
    )
    return fig


def _leaderboard_ai_override_stats(df_p2: pd.DataFrame) -> pd.DataFrame:
    """Per-FLW % of AI-Enabled Screening cases where the AI result was
    overridden, plus the site each FLW is associated with (for hover)."""
    d, flw = _valid_flw(df_p2)
    if d.empty:
        return pd.DataFrame()

    id_col = _id_col(d)
    override_mask = _override_mask(d)

    total      = d.groupby(flw)[id_col].nunique()
    overridden = override_mask.groupby(flw).sum()

    if "site_full_id" in d.columns:
        site = d.groupby(flw)["site_full_id"].agg(
            lambda s: next((v for v in s if pd.notna(v) and str(v).strip()), "")
        )
    else:
        site = pd.Series("", index=total.index, dtype=object)

    out = pd.DataFrame({"flw_username": total.index, "total": total.values}).set_index("flw_username")
    out["overridden"]   = overridden.fillna(0).astype(int)
    out["site_full_id"] = site.reindex(out.index).fillna("")
    out = out[out["total"] > 0]
    out["pct"] = (out["overridden"] / out["total"] * 100).round(1)
    return out.reset_index()


def _fig_leaderboard_ai_override(df_p2: pd.DataFrame) -> go.Figure:
    """Top-10 FLWs by % of AI-Enabled Screening cases overridden (x-axis
    is a percentage, not a count — AI override only exists in phase 2).
    Hover shows the FLW's site_full_id."""
    stats = _leaderboard_ai_override_stats(df_p2)
    stats = stats[stats["pct"] > 0]
    if stats.empty:
        return go.Figure()

    top = stats.sort_values("pct", ascending=False).head(10)
    plot_df = top.sort_values("pct", ascending=True)

    fig = go.Figure(go.Bar(
        y=plot_df["flw_username"], x=plot_df["pct"], orientation="h",
        marker_color="#7A2DE4",
        customdata=np.stack(
            [plot_df["overridden"], plot_df["total"], plot_df["site_full_id"]], axis=-1
        ),
        text=plot_df["pct"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>Site: %{customdata[2]}<br>"
            "AI Override: <b>%{customdata[0]:,}</b> / %{customdata[1]:,} "
            "(<b>%{x:.1f}%</b>)<extra></extra>"
        ),
    ))
    max_pct   = float(plot_df["pct"].max())
    range_max = max_pct + 2
    fig.update_layout(
        height=max(420, 46 * len(plot_df) + 100),
        margin=dict(t=10, b=40, l=10, r=120),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        xaxis=dict(
            title=dict(text="% AI override (of total screened)", font=dict(color="black")),
            tickfont=dict(color="black"),
            ticksuffix="%",
            range=[0, range_max],
            tick0=0,
            dtick=_nice_dtick(range_max),
            showgrid=True, gridcolor="#f0f0f0", zeroline=False,
        ),
        yaxis=dict(
            automargin=True,
            tickfont=dict(size=12, color="black"),
        ),
    )
    return fig


def _render_leaderboard(df_all: pd.DataFrame, df_p2: pd.DataFrame) -> None:
    """Leaderboard tab: current-month FLW screening-volume leaderboard
    next to the overall FLW AI-override leaderboard."""
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        cur_df, month_lbl = _current_month_df(df_all)
        title_suffix = f" — {month_lbl}" if month_lbl else ""
        st.markdown(
            "<div style='font-weight:700;font-size:15px;color:#333;margin-bottom:4px;'>"
            f"🏆 Top 10 FLWs by cases screened{title_suffix}</div>",
            unsafe_allow_html=True,
        )
        fig_counts = _fig_leaderboard_flw_counts(cur_df, month_label=month_lbl)
        if fig_counts.data:
            st.plotly_chart(fig_counts, width="stretch", key="lb_flw_counts")
        else:
            st.info("No FLW-level data available for the current month.")

    with col_r:
        st.markdown(
            "<div style='font-weight:700;font-size:15px;color:#333;margin-bottom:4px;'>"
            "🤖 Top 10 FLWs by % AI override "
            "<span style='font-size:11px;font-weight:500;color:#888;font-style:italic;'>"
            "(AI-Enabled Screening only)</span></div>",
            unsafe_allow_html=True,
        )
        fig_override = _fig_leaderboard_ai_override(df_p2)
        if fig_override.data:
            st.plotly_chart(fig_override, width="stretch", key="lb_ai_override")
        else:
            st.info("No AI-override data available for the current filters.")


# ════════════════════════════════════════════════════════════════════
# Main table renderer – includes "Screened in <month>" column
# ════════════════════════════════════════════════════════════════════

def _render_site_table(df: pd.DataFrame) -> None:
    """Display the site-wise summary table including the 'Screened in <month>' column."""
    if df.empty:
        st.info("No data matches the current filters.")
        return

    # 1. Current month label and mask
    if "date_of_case_registered" not in df.columns:
        st.info("Date column missing.")
        return
    dates = df["date_of_case_registered"].dropna()
    if dates.empty:
        st.info("No valid dates available.")
        return
    max_dt = dates.max()
    last_lbl = max_dt.strftime("%b %Y")
    current_month_mask = (
        (df["date_of_case_registered"].dt.year == max_dt.year) &
        (df["date_of_case_registered"].dt.month == max_dt.month)
    )

    # 2. Site-wise aggregates
    id_col = _id_col(df)
    reviewed_mask, suspicious_mask = _choose_status_masks_phase1(df)
    high_mask = (
        suspicious_mask & _is_high_risk(df["risk"])
        if "risk" in df.columns
        else pd.Series(False, index=df.index)
    )

    if "site_full_id" not in df.columns:
        st.info("No site information available.")
        return

    site_key = df["site_full_id"].astype(str)

    screened_by_site = df.groupby(site_key)[id_col].nunique()
    suspicious_by_site = suspicious_mask.groupby(site_key).sum()
    high_by_site = high_mask.groupby(site_key).sum()

    cur_month_by_site = (
        df.loc[current_month_mask]
        .groupby(site_key[current_month_mask])[id_col]
        .nunique()
    )

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

    # map_site_name for stopped status (optional)
    if "map_site_name" in df.columns:
        map_name_by_site = df.groupby(site_key)["map_site_name"].agg(
            lambda s: next((v for v in s if pd.notna(v) and str(v).strip()), "")
        )
    else:
        map_name_by_site = pd.Series(dtype=object)

    # Load map_data for stopped sites
    map_data = _load_map_data(str(MAP_DATA_PATH))
    stopped_names = set()
    for msite in map_data.get("map_sites", []):
        if msite.get("status") == "stopped":
            stopped_names.add(msite["name"].strip().lower())
        for sub in msite.get("subsites", []):
            if sub.get("status") == "stopped":
                stopped_names.add(sub["name"].strip().lower())

    def is_stopped(site_name: str) -> bool:
        map_name = str(map_name_by_site.get(site_name, "") or "").strip().lower()
        if map_name:
            return map_name in stopped_names
        site_lower = site_name.strip().lower()
        return any(stopped_name in site_lower for stopped_name in stopped_names)

    # Group by site type: District first, then Hospital, then others
    def _site_sort_key(site: str) -> tuple[int, int]:
        st_type = str(site_type_by_site.get(site, "") or "").strip().lower()
        group = 0 if st_type == "district" else (1 if st_type == "hospital" else 2)
        return (group, -int(screened_by_site.get(site, 0)))

    sites_present = sorted(screened_by_site.index, key=_site_sort_key)

    if not sites_present:
        st.info("No sites found.")
        return

    rows_html = []
    for site in sites_present:
        screened = int(screened_by_site.get(site, 0))
        susp = int(suspicious_by_site.get(site, 0))
        high = int(high_by_site.get(site, 0))
        susp_pct = round(susp / screened * 100, 1) if screened else 0.0
        high_pct = round(high / screened * 100, 1) if screened else 0.0

        if is_stopped(site):
            cur_screened_display = "–"
        else:
            cur_screened = int(cur_month_by_site.get(site, 0))
            cur_screened_display = f"{cur_screened:,}"

        site_type_str = str(site_type_by_site.get(site, "") or "—")
        _st_lower = site_type_str.strip().lower()
        if _st_lower == "district":
            site_type_color = "#0BB8E8"
        elif _st_lower == "hospital":
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
            f"color:#4CA64C;border-top:1px solid #eee;'>{cur_screened_display}</td>"
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
        f"<th style='padding:10px 14px;text-align:center;font-size:14px;"
        "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
        f"color:#4CA64C;'>Screened in<br>{html.escape(last_lbl)}</th>"
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

    st.markdown(table_html, unsafe_allow_html=True)

    footnote_text = str(map_data.get("footnote", "") or "").strip()
    if footnote_text:
        footnote_lines = [line.strip() for line in footnote_text.split("\n") if line.strip()]
        footnote_html = "<br>".join(html.escape(line) for line in footnote_lines)
        st.markdown(
            f"<div style='margin-top:6px;font-size:12px;color:#888;"
            f"font-style:italic;line-height:1.5;'>{footnote_html}</div>",
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════
# Public entry point — called by app.py's main()
# ════════════════════════════════════════════════════════════════════

def render(df_all: pd.DataFrame, df_p1: pd.DataFrame, df_p2: pd.DataFrame) -> None:
    """
    Render the Research Dashboard – "Descriptive" and "Leaderboard" tabs.

    Parameters
    ----------
    df_all : combined dataset (all phases), already filtered by sidebar.
             Feeds the Descriptive site table and the Leaderboard's
             cases-screened chart (Non-suspicious/Suspicious status comes
             from phase-1 fields, present across the combined dataset).
    df_p1  : phase-1 data (not used, kept for signature compatibility).
    df_p2  : phase-2 (AI-Enabled Screening) data — feeds the Leaderboard's
             % AI-override chart, since AI results/overrides only exist
             in phase 2.
    """
    # Password gate
    if not _check_password():
        return

    # ── Style the tab buttons to match Monitoring Dashboard ──
    st.markdown(
        """
        <style>
        /* Make the research tab buttons look exactly like the monitoring tabs */
        .st-key-btn_research_descriptive button,
        .st-key-btn_research_leaderboard button {
            padding: 10px 12px !important;
            font-size: 22px !important;
            font-weight: 800 !important;
            min-height: 0 !important;
            line-height: 1.3 !important;
        }
        .st-key-btn_research_descriptive button *,
        .st-key-btn_research_leaderboard button * {
            font-weight: 800 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "research_tab" not in st.session_state:
        st.session_state.research_tab = 0

    # ── Tab button row — Descriptive · Leaderboard ──
    col1, col2, _sp = st.columns([1, 1, 4])
    with col1:
        if st.button(
            "📊 Descriptive",
            key="btn_research_descriptive",
            type="primary" if st.session_state.research_tab == 0 else "secondary",
            use_container_width=True,
        ):
            st.session_state.research_tab = 0
            st.rerun()
    with col2:
        if st.button(
            "🏆 Leaderboard",
            key="btn_research_leaderboard",
            type="primary" if st.session_state.research_tab == 1 else "secondary",
            use_container_width=True,
        ):
            st.session_state.research_tab = 1
            st.rerun()

    st.markdown(
        "<hr style='border:none;border-top:1.5px solid #ddd;margin:10px 0 18px;'>",
        unsafe_allow_html=True,
    )

    # ── Render the active tab ──
    if st.session_state.research_tab == 0:
        if df_all.empty:
            st.info("No data available for the current filters.")
        else:
            _render_site_table(df_all)
    else:
        if df_all.empty and df_p2.empty:
            st.info("No data available for the current filters.")
        else:
            _render_leaderboard(df_all, df_p2)

    # ── Footer with email (matching monitoring dashboard style) ──
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

    # ── Stop execution to prevent app.py from adding its own footer ──
    st.stop()