"""
====================================================================
RESEARCH DASHBOARD — standalone module
Aarogya Aarohan / TANUH Oral Cancer Project

Single tab ("Descriptive") showing site‑wise summary table with
the "Screened in <month>" column.

Uses the same tab‑button style as the Monitoring Dashboard.
====================================================================
"""

from __future__ import annotations
import html
import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st


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
    Render the Research Dashboard – single "Descriptive" tab with site table.

    Parameters
    ----------
    df_all : combined dataset (all phases), already filtered by sidebar.
    df_p1  : phase-1 data (not used, kept for signature compatibility).
    df_p2  : phase-2 data (not used, kept for signature compatibility).
    """
    # Password gate
    if not _check_password():
        return

    # ── Style the single tab button to match Monitoring Dashboard ──
    st.markdown(
        """
        <style>
        /* Make the research tab button look exactly like the monitoring tabs */
        .st-key-btn_research_descriptive button {
            padding: 10px 12px !important;
            font-size: 22px !important;
            font-weight: 800 !important;
            min-height: 0 !important;
            line-height: 1.3 !important;
        }
        .st-key-btn_research_descriptive button * {
            font-weight: 800 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Tab button row (mimics the layout of monitoring's 3 tabs) ──
    col1, _ = st.columns([1, 5])
    with col1:
        st.button(
            "📊 Descriptive",
            key="btn_research_descriptive",
            type="primary",
            disabled=True,          # only one tab, always active
            use_container_width=True,
        )

    st.markdown(
        "<hr style='border:none;border-top:1.5px solid #ddd;margin:10px 0 18px;'>",
        unsafe_allow_html=True,
    )

    # ── Render the site table ──
    if df_all.empty:
        st.info("No data available for the current filters.")
    else:
        _render_site_table(df_all)

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