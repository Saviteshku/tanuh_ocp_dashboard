"""
====================================================================
RESEARCH DASHBOARD — standalone module
Aarogya Aarohan / TANUH Oral Cancer Project

Owns everything shown under the top-level "Research Dashboard" view
(separate from the Monitoring Dashboard's Overall / Image-Based
Screening / AI-Enabled Screening tabs).

Currently contains:
  - Site Screening Targets table (Phase 2 / AI-Enabled Screening data)

To add more Research Dashboard content later, add new render_* /
build_* functions here and call them from render(df_p2) below —
main app code (code.py) only needs to call research_dashboard.render().
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
# map_data.json lives alongside the combined parquet in LOCAL_DATA_DIR
# (same OCP_DATA_DIR resolution app.py uses), and already carries the
# deployment-stats/map data the Research Dashboard's map view reads —
# the password lives in that same file under "research_dash_password".

LOCAL_DATA_DIR = Path(
    os.environ.get(
        "OCP_DATA_DIR",
        r"/mnt/d/OneDrive/IISC/TANUH/OralCancer_Project/Raw_Data/Dashboard",
    )
)
MAP_DATA_PATH = LOCAL_DATA_DIR / "map_data.json"

# How long an unlocked Research Dashboard stays unlocked without any
# interaction before it re-locks itself and asks for the password
# again. A full page reload also re-locks it: session_state (where the
# "authed" flag below lives) is per browser-tab WebSocket session and
# is wiped from scratch on reload, so there's nothing extra to do for
# that case — it re-locks automatically.
SESSION_TIMEOUT_SECONDS = 5 * 60


@st.cache_data(ttl=3600, show_spinner=False)
def _load_map_data(path_str: str) -> dict:
    """Read map_data.json from disk. Returns {} if missing/unreadable
    (e.g. not deployed yet), rather than raising, so a missing file
    fails safe into "password not configured" below."""
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
    """Gate the Research Dashboard behind research_dash_password.

    Returns True once the correct password has been entered for this
    browser session AND that unlock is still within the 5-minute
    inactivity window, else renders the password prompt/error and
    returns False. Every call that returns True also refreshes the
    "last active" timestamp, so the 5 minutes measures time since the
    last dashboard interaction/rerun, not just time since login. A
    full page reload always re-locks, since session_state itself is
    wiped on reload.
    """
    authed_at = st.session_state.get("research_dash_authed_at")
    if authed_at is not None:
        if time.time() - authed_at <= SESSION_TIMEOUT_SECONDS:
            st.session_state.research_dash_authed_at = time.time()
            return True
        # Timed out — clear and fall through to the password prompt.
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
# Site Screening Targets (Phase 2 / AI-Enabled Screening only)
# ════════════════════════════════════════════════════════════════════
# Every site gets a flat 2,500 screenings/month rate. For each site:
#   Months Initial   = months from that site's own Start Date (earliest
#                       date_of_case_registered in the data) to the
#                       fixed study completion date, 31-Mar-2027.
#   Target           = 2,500 x Months Initial
#   Months Left      = months from TODAY (system clock) to 31-Mar-2027
#   New Target/Month = (Target - Screened) / Months Left
#
# Only sites that actually appear in the phase-2 data are shown. Sites
# with no phase-2 records yet are intentionally left out of the table
# (see commented-out list below) rather than shown as empty placeholder
# rows.

SITE_TARGET_RATE_PER_MONTH = 2500
STUDY_END_DATE = pd.Timestamp("2027-03-31")

# Sites expected in the overall rollout that may not yet have phase-2
# data. Left here for reference only -- NOT rendered until they show up
# in df_p2 (site_full_id). Uncomment / extend as sites go live if you
# want a static population reference alongside them later.
# PLANNED_SITES_NOT_YET_LIVE = [
#     "Tamil Nadu",
#     "Nagaland",
#     "Mathura",
#     "West Bengal",
#     "Guwahati",
#     "Silchar",
#     "Varanasi",
#     "Bangalore",
#     "Delhi",
# ]


def _id_col(df: pd.DataFrame) -> str:
    for c in ("case_id", "Unique_case_ID"):
        if c in df.columns:
            return c
    return df.columns[0]


def _normalize_site_name(name: str) -> str:
    """Lowercase + collapse whitespace, purely for de-duplicating minor
    formatting variants of the same site name (e.g. extra spaces)."""
    return " ".join(str(name).strip().lower().split())


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Fractional months between two timestamps (30.44-day month)."""
    return (end - start).days / 30.44


def build_site_target_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the Site / Start Date / Target / Screened / Months Left /
    New Target-per-Month table, using ONLY phase-2 data (caller must
    already have filtered df to phase == '2').

    All dates/targets are derived directly from the data:
    - Start Date  = earliest date_of_case_registered per site, in data.
    - Target      = 2,500 x months from that Start Date to 31-Mar-2027.
    - Months Left = months from system "now" to 31-Mar-2027.
    - New Target/Month = (Target - Screened) / Months Left.
    Sites with zero phase-2 rows are not included.
    """
    if df is None or df.empty or "site_full_id" not in df.columns:
        return pd.DataFrame()

    id_col = _id_col(df)
    site_key_norm = df["site_full_id"].astype(str).map(_normalize_site_name)

    screened_by_site = df.groupby(site_key_norm)[id_col].nunique()

    if "date_of_case_registered" in df.columns:
        start_by_site = df.groupby(site_key_norm)["date_of_case_registered"].min()
    else:
        start_by_site = pd.Series(dtype="datetime64[ns]")

    # Keep the original (as-typed-in-data) display label for each
    # normalized key, so the table shows real site names from the data.
    display_name_by_norm = (
        df.groupby(site_key_norm)["site_full_id"]
        .agg(lambda s: next((v for v in s if pd.notna(v) and str(v).strip()), ""))
    )

    now = pd.Timestamp.now()
    months_left = round(max(_months_between(now, STUDY_END_DATE), 0), 2)

    rows = []
    for key, screened in screened_by_site.items():
        screened = int(screened)
        start_dt = start_by_site.get(key)
        has_start = pd.notna(start_dt)

        if has_start:
            months_initial = _months_between(start_dt, STUDY_END_DATE)
            target = round(months_initial * SITE_TARGET_RATE_PER_MONTH)
        else:
            target = None

        if target is not None and months_left > 0:
            new_target_per_month = round((target - screened) / months_left, 1)
        else:
            new_target_per_month = None

        display_name = display_name_by_norm.get(key) or key.title()

        rows.append({
            "Site": display_name,
            "Start Date": start_dt.strftime("%d-%b-%Y") if has_start else "Not started",
            "Target": target,
            "Screened": screened,
            "Months Left": months_left if has_start else None,
            "New Target per Month": new_target_per_month,
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(by="Screened", ascending=False).reset_index(drop=True)
    return out


def render_site_target_table(df: pd.DataFrame) -> None:
    """Render the Phase 2 site-target tracking table as styled HTML,
    matching the look of the Monitoring Dashboard's Site-wise Summary
    table."""
    table = build_site_target_table(df)
    if table.empty:
        return

    def _fmt_num(v):
        return f"{int(v):,}" if pd.notna(v) else "\u2014"

    rows_html = []
    for _, r in table.iterrows():
        target = r["Target"]
        screened = r["Screened"]
        pct_html = ""
        if pd.notna(target) and target:
            pct = round(screened / target * 100, 1)
            pct_color = "#228B22" if pct >= 100 else ("#F4A900" if pct >= 50 else "#D94040")
            pct_html = f"<span style='color:{pct_color};font-weight:700;'> ({pct}%)</span>"

        new_target = r["New Target per Month"]
        new_target_html = _fmt_num(new_target) if pd.notna(new_target) else "\u2014"

        month_rem = r["Months Left"]
        month_rem_html = f"{month_rem:g}" if pd.notna(month_rem) else "\u2014"

        rows_html.append(
            "<tr>"
            f"<td style='padding:5px 10px;text-align:left;font-weight:600;"
            f"color:#333;border-top:1px solid #eee;'>{html.escape(str(r['Site']))}</td>"
            f"<td style='padding:5px 10px;text-align:center;font-weight:600;"
            f"color:#555;border-top:1px solid #eee;white-space:nowrap;'>{html.escape(str(r['Start Date']))}</td>"
            f"<td style='padding:5px 10px;text-align:center;font-weight:700;"
            f"color:#0771eb;border-top:1px solid #eee;'>{_fmt_num(target)}</td>"
            f"<td style='padding:5px 10px;text-align:center;font-weight:700;"
            f"color:#228B22;border-top:1px solid #eee;'>{_fmt_num(screened)}{pct_html}</td>"
            f"<td style='padding:5px 10px;text-align:center;font-weight:600;"
            f"color:#555;border-top:1px solid #eee;'>{month_rem_html}</td>"
            f"<td style='padding:5px 10px;text-align:center;font-weight:700;"
            f"color:#7A2DE4;border-top:1px solid #eee;'>{new_target_html}</td>"
            "</tr>"
        )

    table_html = (
        "<div style='overflow-x:auto;'>"
        "<table style='width:100%;border-collapse:collapse;background:#fff;"
        "border-radius:12px;overflow:hidden;box-shadow:0 2px 16px rgba(0,0,0,.07);'>"
        "<thead><tr style='background:#fafafa;'>"
        "<th style='padding:10px 14px;text-align:left;font-size:14px;"
        "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
        "color:#555;'>Site</th>"
        "<th style='padding:10px 14px;text-align:center;font-size:14px;"
        "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
        "color:#555;'>Start Date</th>"
        "<th style='padding:10px 14px;text-align:center;font-size:14px;"
        "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
        "color:#0771eb;'>Target</th>"
        "<th style='padding:10px 14px;text-align:center;font-size:14px;"
        "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
        "color:#228B22;'>Screened</th>"
        "<th style='padding:10px 14px;text-align:center;font-size:14px;"
        "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
        "color:#555;'>Months Left</th>"
        "<th style='padding:10px 14px;text-align:center;font-size:14px;"
        "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
        "color:#7A2DE4;'>New Target/Month</th>"
        "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table></div>"
    )

    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown(
        "<div style='margin-top:6px;font-size:12px;color:#888;font-style:italic;'>"
        "Target = 2,500 screenings/month x months from each site's Start Date "
        "(earliest recorded screening in the data) to study completion on "
        "31-Mar-2027. Months Left = months from today to 31-Mar-2027. "
        "New Target/Month is the revised monthly pace needed, given screenings "
        "so far, to still reach target by the completion date."
        "</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
# Public entry point — called by code.py's main() when the user is on
# the "Research Dashboard" top-level view.
# ════════════════════════════════════════════════════════════════════

def render(df_p2: pd.DataFrame) -> None:
    """Render the full Research Dashboard view.

    Parameters
    ----------
    df_p2 : phase-2 (AI-Enabled Screening) data only, already filtered
            by the caller's sidebar filters (date range, gender,
            study setting, site).
    """
    st.markdown(
        '<div style="font-weight:700;font-size:28px;color:#333;'
        'margin-bottom:12px;">🔬 Research Dashboard</div>',
        unsafe_allow_html=True,
    )

    if not _check_password():
        return

    if df_p2 is None or df_p2.empty:
        st.info("No AI-Enabled Screening (Phase 2) data available yet for the current filters.")
        return

    st.markdown(
        "<div style='font-weight:700;font-size:25px;color:#333;"
        "margin-bottom:8px;'>\U0001f3af Site Screening Targets</div>",
        unsafe_allow_html=True,
    )
    render_site_target_table(df_p2)

    # ── Add future Research Dashboard sections below this line ──────
    # e.g.:
    # st.markdown("---")
    # render_some_other_research_section(df_p2)