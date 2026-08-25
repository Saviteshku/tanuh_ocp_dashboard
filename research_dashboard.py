"""
====================================================================
RESEARCH DASHBOARD — standalone module
Aarogya Aarohan / TANUH Oral Cancer Project

Three tabs:
  - "Descriptive": site‑wise summary table with the "Screened in
    <month>" column.
  - "Leaderboard": top‑10 FLW (field health worker) leaderboards —
    cases screened (stacked by Non‑suspicious / Suspicious·Low risk /
    Suspicious·High risk) and % of AI-Enabled Screening cases where
    the AI result was overridden.
  - "AI Inference": phase‑2 confusion matrix (AI Result vs the TSD's
    Suspicion field as ground truth) alongside its derived diagnostic
    metrics — sensitivity, specificity, PPV, NPV, prevalence, +LR, -LR.

Uses the same tab‑button style as the Monitoring Dashboard.
====================================================================
"""

from __future__ import annotations
import hashlib
import hmac
import html
import json
import math
import os
import secrets
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
import streamlit as st

AMBER_HIGH = "#E0631A"
AMBER_LOW  = "#F7C548"


# ════════════════════════════════════════════════════════════════════
# Password gate — salted+hashed password, read from Streamlit secrets
# or environment variables (never plaintext, never from a data file)
# ════════════════════════════════════════════════════════════════════
LOCAL_DATA_DIR = Path(
    os.environ.get(
        "OCP_DATA_DIR",
        r"/mnt/d/OneDrive/IISC/TANUH/OralCancer_Project/Raw_Data/Dashboard",
    )
)
MAP_DATA_PATH = LOCAL_DATA_DIR / "map_data.json"

SESSION_TIMEOUT_SECONDS = 5 * 60

# PBKDF2 iteration count used both when hashing at setup time and when
# verifying a login attempt — must match whatever was used to generate
# the stored hash (see _hash_password_for_setup below).
_PBKDF2_ITERATIONS = 200_000


@st.cache_data(ttl=3600, show_spinner=False)
def _load_map_data(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _hash_password_for_setup(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    """One-time helper to generate the (salt_hex, hash_hex) pair to put in
    st.secrets / env vars. Not called by the app itself — run it once,
    e.g. `python -c "from research_dashboard import _hash_password_for_setup as h; print(h('mypassword'))"`,
    then store the two returned hex strings as RESEARCH_DASH_PASSWORD_SALT
    and RESEARCH_DASH_PASSWORD_HASH (or the st.secrets equivalents)."""
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return salt.hex(), digest.hex()


def _stored_password_credentials() -> tuple[str, str] | None:
    """Reads (salt_hex, hash_hex) — never a plaintext password — from
    Streamlit secrets first, then environment variables."""
    salt_hex = hash_hex = None
    try:
        salt_hex = st.secrets.get("research_dash_password_salt")
        hash_hex = st.secrets.get("research_dash_password_hash")
    except Exception:
        pass
    salt_hex = salt_hex or os.environ.get("RESEARCH_DASH_PASSWORD_SALT")
    hash_hex = hash_hex or os.environ.get("RESEARCH_DASH_PASSWORD_HASH")
    if salt_hex and hash_hex:
        return str(salt_hex), str(hash_hex)
    return None


def _verify_password(entered: str, salt_hex: str, hash_hex: str) -> bool:
    """Constant-time comparison against the stored PBKDF2 hash."""
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", entered.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return hmac.compare_digest(candidate, expected)


def _check_password() -> bool:
    authed_at = st.session_state.get("research_dash_authed_at")
    if authed_at is not None:
        if time.time() - authed_at <= SESSION_TIMEOUT_SECONDS:
            st.session_state.research_dash_authed_at = time.time()
            return True
        del st.session_state["research_dash_authed_at"]
        st.info("Session timed out after 5 minutes of inactivity — please re-enter the password.")

    creds = _stored_password_credentials()
    if creds is None:
        st.error(
            "Research Dashboard password isn't configured — set "
            "research_dash_password_salt / research_dash_password_hash "
            "in st.secrets, or RESEARCH_DASH_PASSWORD_SALT / "
            "RESEARCH_DASH_PASSWORD_HASH as environment variables."
        )
        return False
    salt_hex, hash_hex = creds

    with st.form("research_dash_password_form"):
        entered = st.text_input("🔒 Password", type="password")
        submitted = st.form_submit_button("Unlock")

    if submitted:
        if _verify_password(entered, salt_hex, hash_hex):
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


def _flw_state_series(d: pd.DataFrame) -> pd.Series:
    """Cleaned 'states' Series aligned to `d` — "" where missing/blank.
    Shared by every leaderboard stat function so an FLW is always grouped
    by (flw_username, states, site_full_id), since the same flw_username
    can appear under more than one state/site."""
    state_col = "states" if "states" in d.columns else ("state" if "state" in d.columns else None)
    if state_col is None:
        return pd.Series("", index=d.index, dtype=object, name="states")
    state = d[state_col].astype(str).str.strip()
    state = state.where(_present_mask(d[state_col]), "")
    state.name = "states"
    return state


def _flw_site_series(d: pd.DataFrame) -> pd.Series:
    """Cleaned 'site_full_id' Series aligned to `d` — "" where missing/blank."""
    site_col = "site_full_id" if "site_full_id" in d.columns else None
    if site_col is None:
        return pd.Series("", index=d.index, dtype=object, name="site_full_id")
    site = d[site_col].astype(str).str.strip()
    site = site.where(_present_mask(d[site_col]), "")
    site.name = "site_full_id"
    return site


def _flw_label_col(out: pd.DataFrame) -> pd.Series:
    """Y-axis label: "<flw_username>" then the state on the next line,
    e.g. "flw2" / "Karnataka" — so the same flw_username shows up as
    separate, clearly-labeled bars per state instead of colliding."""
    return out.apply(
        lambda r: f"{r['flw_username']}<br>{r['states']}" if r["states"] else r["flw_username"],
        axis=1,
    )


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


def _window_label_and_weeks(df: pd.DataFrame) -> tuple[str, float]:
    """Label for the actual date range present in `df`'s
    date_of_case_registered column (e.g. '01 May 2026 - 12 Aug 2026',
    or just '12 Aug 2026' if start and end are the same day), and the
    number of weeks that range spans (minimum 1 week). Reflects
    whatever the sidebar's global date filter has already produced on
    `df` — no additional lookback window is applied here, so both
    values update automatically whenever the global filter changes."""
    if df.empty or "date_of_case_registered" not in df.columns:
        return "", 1.0
    dates = df["date_of_case_registered"].dropna()
    if dates.empty:
        return "", 1.0
    min_dt, max_dt = dates.min(), dates.max()
    start_lbl = min_dt.strftime("%d %b %Y")
    end_lbl = max_dt.strftime("%d %b %Y")
    label = end_lbl if start_lbl == end_lbl else f"{start_lbl} - {end_lbl}"
    weeks = max((max_dt - min_dt).days / 7, 1.0)
    return label, weeks


def _filter_window_label_and_weeks(
    filter_start, filter_end
) -> tuple[str, float]:
    """Label + week-count derived from the sidebar's explicit "Date
    range" selection itself (`filter_start`/`filter_end`, the actual
    picked boundary dates) rather than from whichever case dates
    happen to be present in the filtered data. This is what makes
    "Current month" mean exactly the 1st of the month through today,
    and a 7-day range mean exactly 1 week — regardless of whether data
    exists on the very first or very last day of that range.

    Used as the single shared denominator for cases/week everywhere it
    needs to match across charts: the same FLW's cases/week comes out
    identical in the "Top 10 FLWs by cases/week" leaderboard and the
    "FLWs by site" bar chart, since both now divide by this same
    filter-derived week count instead of each computing its own
    (previously data-derived, or per-FLW-span) weeks value.

    Falls back to ("", 1.0) if no filter dates are available (e.g. the
    dataset has no date column at all)."""
    if filter_start is None or filter_end is None:
        return "", 1.0
    start_ts = pd.Timestamp(filter_start)
    end_ts = pd.Timestamp(filter_end)
    start_lbl = start_ts.strftime("%d %b %Y")
    end_lbl = end_ts.strftime("%d %b %Y")
    label = end_lbl if start_lbl == end_lbl else f"{start_lbl} - {end_lbl}"
    days = (end_ts - start_ts).days
    weeks = max(days, 1) / 7
    return label, weeks


# ════════════════════════════════════════════════════════════════════
# Leaderboard — per-FLW stats & charts
# ════════════════════════════════════════════════════════════════════

def _leaderboard_flw_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(FLW, state) totals: screened count, split into Non-suspicious /
    Suspicious·Low risk / Suspicious·High risk (high/low only come out of
    the suspicious subset — the non-suspicious bucket is never split).

    Grouped by (flw_username, states) rather than flw_username alone, since
    the same flw_username can appear under multiple states — each such
    pairing gets its own row (and its own bar)."""
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
    pending_mask  = ~reviewed_mask

    state = _flw_state_series(d)
    site_full_id = _flw_site_series(d)

    grp = [flw, state, site_full_id]
    total    = d.groupby(grp)[id_col].nunique()
    high     = high_mask.groupby(grp).sum()
    low      = low_mask.groupby(grp).sum()
    non_susp = non_susp_mask.groupby(grp).sum()
    pending  = pending_mask.groupby(grp).sum()

    out = total.rename("total").to_frame()
    out["high"]     = high
    out["low"]      = low
    out["non_susp"] = non_susp
    out["pending"]  = pending
    out = out.fillna(0)
    for c in ("total", "high", "low", "non_susp", "pending"):
        out[c] = out[c].astype(int)
    out = out.reset_index()
    out.columns = ["flw_username", "states", "site_full_id", "total", "high", "low", "non_susp", "pending"]
    # Display label used on the y-axis / legend so the same flw_username
    # shows up as separate, clearly-labeled bars per state, e.g.
    # "flw2" on one line, "Karnataka" on the next.
    out["label"] = _flw_label_col(out)
    return out


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


def _fig_leaderboard_flw_counts(df: pd.DataFrame, window_label: str = "", weeks: float = 1.0) -> go.Figure:
    """Top-10 (FLW, state) pairs by cases/week — total screened divided
    by `weeks` (the number of weeks spanned by whatever data is
    currently passed in, i.e. the sidebar's global filter — see
    `_window_label_and_weeks`). The same `weeks` value is used for
    every FLW, so the ranking order is identical to ranking by raw
    total; only the displayed scale changes. Horizontal stacked bar
    (highest at top, lowest at bottom). The same flw_username can
    appear under multiple states, so each pairing gets its own bar,
    labeled "<flw_username>" then state on the next line on the
    y-axis.

    Each stacked segment's bar length is the rate (raw count / weeks),
    so the whole stack sums to the FLW's cases/week — but the hover
    for every segment still reports the underlying RAW count (via
    customdata), same as before; a new "Cases/week" line is added on
    top of that, showing the FLW's overall rate."""
    stats = _leaderboard_flw_counts(df)
    if stats.empty:
        return go.Figure()

    weeks = weeks if weeks else 1.0
    top = stats.sort_values("total", ascending=False).head(10)
    # Ascending order so the highest bar ends up plotted at the top of a
    # horizontal bar chart (Plotly stacks categories bottom-to-top).
    plot_df = top.sort_values("total", ascending=True).copy()
    plot_df["rate"] = plot_df["total"] / weeks

    fig = go.Figure()
    # Invisible zero-width bar to surface the site_full_id in the unified hover
    fig.add_trace(go.Bar(
        y=plot_df["label"], x=[0] * len(plot_df), orientation="h",
        name="Site", marker=dict(color="rgba(0,0,0,0)"), showlegend=False,
        customdata=plot_df["site_full_id"],
        hovertemplate="Site: %{customdata}<extra></extra>",
    ))
    # Invisible zero-width bar to surface total screened in the unified hover
    fig.add_trace(go.Bar(
        y=plot_df["label"], x=[0] * len(plot_df), orientation="h",
        name="Total screened", marker=dict(color="rgba(0,0,0,0)"), showlegend=False,
        customdata=plot_df["total"],
        hovertemplate="Total screened: <b>%{customdata:,}</b><extra></extra>",
    ))
    # Invisible zero-width bar to surface the FLW's cases/week rate in the unified hover
    fig.add_trace(go.Bar(
        y=plot_df["label"], x=[0] * len(plot_df), orientation="h",
        name="Cases/week", marker=dict(color="rgba(0,0,0,0)"), showlegend=False,
        customdata=plot_df["rate"].apply(lambda v: f"{v:.2f}"),
        hovertemplate="Cases/week: <b>%{customdata}</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=plot_df["label"], x=plot_df["pending"] / weeks, orientation="h",
        name="Pending review", marker_color="#C9C9C9",
        customdata=plot_df["pending"],
        hovertemplate="Pending review: <b>%{customdata:,}</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=plot_df["label"], x=plot_df["non_susp"] / weeks, orientation="h",
        name="Non-suspicious", marker_color="#6B6B6B",
        customdata=plot_df["non_susp"],
        hovertemplate="Non-suspicious: <b>%{customdata:,}</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=plot_df["label"], x=plot_df["low"] / weeks, orientation="h",
        name="Suspicious · Low risk", marker_color=AMBER_LOW,
        customdata=plot_df["low"],
        hovertemplate="Suspicious · Low risk: <b>%{customdata:,}</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=plot_df["label"], x=plot_df["high"] / weeks, orientation="h",
        name="Suspicious · High risk", marker_color=AMBER_HIGH,
        customdata=plot_df["high"],
        hovertemplate="Suspicious · High risk: <b>%{customdata:,}</b><extra></extra>",
    ))

    max_rate = float(plot_df["rate"].max())
    # Axis extends a little past the highest bar, same idea as before
    # but scaled to the rate instead of the raw count.
    range_max = max_rate * 1.15 + 0.5
    x_title = f"Cases/week{f' — {window_label}' if window_label else ''}"
    chart_h = max(420, 46 * len(plot_df) + 100)

    fig.update_layout(
        barmode="stack",
        height=chart_h,
        margin=dict(t=0, b=70, l=10, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="y unified",
        # Legend placed *below* the x-axis title — this never conflicts
        # with the modebar (top-right), unlike a top-anchored legend.
        legend=dict(
            orientation="h", y=-0.22, x=0, yanchor="top", xanchor="left",
            font=dict(size=12, color="black"),
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(color="black")),
            tickfont=dict(size=14, color="black"),
            range=[0, range_max],
            tick0=0,
            dtick=_nice_dtick(range_max),
            showgrid=True, gridcolor="#f0f0f0", zeroline=False,
        ),
        yaxis=dict(
            automargin=True,
            tickfont=dict(size=14, color="black"),
        ),
    )
    return fig


def _leaderboard_ai_override_stats(df_p2: pd.DataFrame) -> pd.DataFrame:
    """Per-(FLW, state, site) % of AI-Enabled Screening cases where the AI
    result was overridden. Grouped by (flw_username, states, site_full_id)
    rather than flw_username alone, since the same flw_username can appear
    under multiple states/sites — each such pairing gets its own row."""
    empty = pd.DataFrame(
        columns=["flw_username", "states", "site_full_id", "total", "overridden", "pct", "label"]
    )
    d, flw = _valid_flw(df_p2)
    if d.empty:
        return empty

    id_col = _id_col(d)
    override_mask = _override_mask(d)
    state = _flw_state_series(d)
    site_full_id = _flw_site_series(d)

    grp = [flw, state, site_full_id]
    total      = d.groupby(grp)[id_col].nunique()
    overridden = override_mask.groupby(grp).sum()

    out = total.rename("total").to_frame()
    out["overridden"] = overridden
    out = out.fillna(0)
    out["total"] = out["total"].astype(int)
    out["overridden"] = out["overridden"].astype(int)
    out = out.reset_index()
    out.columns = ["flw_username", "states", "site_full_id", "total", "overridden"]
    out = out[out["total"] > 0]
    out["pct"] = (out["overridden"] / out["total"] * 100).round(1)
    out["label"] = _flw_label_col(out)
    return out


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
        y=plot_df["label"], x=plot_df["pct"], orientation="h",
        marker_color="#7A2DE4",
        customdata=np.stack(
            [plot_df["overridden"], plot_df["total"], plot_df["site_full_id"], plot_df["flw_username"]],
            axis=-1,
        ),
        text=plot_df["pct"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        textfont=dict(size=14, color="black"),
        hovertemplate=(
            "<b>%{customdata[3]}</b><br>Site: %{customdata[2]}<br>"
            "AI Override: <b>%{customdata[0]:,}</b> / %{customdata[1]:,} "
            "(<b>%{x:.1f}%</b>)<extra></extra>"
        ),
    ))
    max_pct = float(plot_df["pct"].max())
    # Range extends to max value + 10 percentage points of headroom, so the
    # outside "xx.x%" label never gets clipped at the right edge. Ticks at
    # a fixed 10%-interval.
    range_max = max_pct + 10
    fig.update_layout(
        height=max(420, 46 * len(plot_df) + 100),
        margin=dict(t=0, b=40, l=10, r=25),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        xaxis=dict(
            title=dict(text="% AI override (of total screened)", font=dict(color="black")),
            tickfont=dict(size=14, color="black"),
            ticksuffix="%",
            range=[0, range_max],
            tick0=0,
            dtick=10,
            showgrid=True, gridcolor="#f0f0f0", zeroline=False,
        ),
        yaxis=dict(
            automargin=True,
            tickfont=dict(size=14, color="black"),
        ),
    )
    return fig


def _last_n_months_df(df: pd.DataFrame, n_months: int = 3) -> tuple[pd.DataFrame, str]:
    """Return (rows from the most recent `n_months` calendar months present
    in date_of_case_registered, inclusive of the latest month; a label for
    that window, e.g. 'May - Jul 2026' or just 'Jul 2026' if it's a single
    month). Mirrors _current_month_df's "latest date in the data" anchoring,
    just widened to a rolling N-month window instead of a single calendar
    month.

    The label reflects the earliest date actually present in the
    resulting rows, not the theoretical N-month lookback boundary. This
    matters once a caller (e.g. a sidebar "Quick date filter") has
    already narrowed `df` upstream: if only August data survives that
    upstream filter, the 6-month lookback here still computes a March
    cutoff, but since no rows exist before August, the label should read
    just "Aug 2026" rather than the misleading "Mar 2026 - Aug 2026"."""
    if df.empty or "date_of_case_registered" not in df.columns:
        return df.iloc[0:0], ""
    dates = df["date_of_case_registered"].dropna()
    if dates.empty:
        return df.iloc[0:0], ""
    max_dt = dates.max()
    start_period = max_dt.to_period("M") - (n_months - 1)
    cutoff = start_period.to_timestamp()
    mask = df["date_of_case_registered"] >= cutoff
    result = df.loc[mask]
    actual_min = result["date_of_case_registered"].dropna().min()
    start_lbl = (actual_min if pd.notna(actual_min) else max_dt).strftime("%b %Y")
    end_lbl = max_dt.strftime("%b %Y")
    label = end_lbl if start_lbl == end_lbl else f"{start_lbl} - {end_lbl}"
    return result, label


def _leaderboard_retake_photo_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Per-(FLW, state, site) % of cases (past 3 months, all phases
    combined) where specialist_recommendation was "Retake photo". Grouped
    by (flw_username, states, site_full_id) rather than flw_username alone,
    since the same flw_username can appear under multiple states/sites —
    each such pairing gets its own row. Also returns the 6-month window
    label."""
    empty = pd.DataFrame(
        columns=["flw_username", "states", "site_full_id", "total", "retake", "pct", "label"]
    )
    recent, window_lbl = _last_n_months_df(df, n_months=6)
    d, flw = _valid_flw(recent)
    if d.empty or "specialist_recommendation" not in d.columns:
        return empty, window_lbl

    id_col = _id_col(d)
    retake_mask = _norm(d["specialist_recommendation"]).eq("retake photo")
    state = _flw_state_series(d)
    site_full_id = _flw_site_series(d)

    grp = [flw, state, site_full_id]
    total  = d.groupby(grp)[id_col].nunique()
    retake = retake_mask.groupby(grp).sum()

    out = total.rename("total").to_frame()
    out["retake"] = retake
    out = out.fillna(0)
    out["total"] = out["total"].astype(int)
    out["retake"] = out["retake"].astype(int)
    out = out.reset_index()
    out.columns = ["flw_username", "states", "site_full_id", "total", "retake"]
    out = out[out["total"] > 0]
    out["pct"] = (out["retake"] / out["total"] * 100).round(1)
    out["label"] = _flw_label_col(out)
    return out, window_lbl


def _fig_leaderboard_retake_photo(df: pd.DataFrame) -> tuple[go.Figure, str]:
    """Top-10 FLWs by % of cases (past 6 months, all phases combined)
    flagged "Retake photo" by the specialist. Hover shows the FLW's
    site_full_id plus the raw retake / total counts. Also returns the
    6-month window label (e.g. 'Mar - Aug 2026') for the chart title."""
    stats, window_lbl = _leaderboard_retake_photo_stats(df)
    stats = stats[stats["pct"] > 0]
    if stats.empty:
        return go.Figure(), window_lbl

    top = stats.sort_values("pct", ascending=False).head(10)
    plot_df = top.sort_values("pct", ascending=True)

    fig = go.Figure(go.Bar(
        y=plot_df["label"], x=plot_df["pct"], orientation="h",
        marker_color="#0F9D8C",
        customdata=np.stack(
            [plot_df["retake"], plot_df["total"], plot_df["site_full_id"], plot_df["flw_username"]],
            axis=-1,
        ),
        text=plot_df["pct"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        textfont=dict(size=14, color="black"),
        hovertemplate=(
            "<b>%{customdata[3]}</b><br>Site: %{customdata[2]}<br>"
            "Retake photo: <b>%{customdata[0]:,}</b> / %{customdata[1]:,} "
            "(<b>%{x:.1f}%</b>)<extra></extra>"
        ),
    ))
    max_pct = float(plot_df["pct"].max())
    # Range extends to max value + 10 percentage points of headroom, so the
    # outside "xx.x%" label never gets clipped at the right edge. Ticks at
    # a fixed 10%-interval.
    range_max = max_pct + 10
    x_title = "% Retake photo (of total screened"
    x_title += f", {window_lbl})" if window_lbl else ")"
    fig.update_layout(
        height=max(420, 46 * len(plot_df) + 100),
        margin=dict(t=0, b=40, l=10, r=25),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        xaxis=dict(
            title=dict(text=x_title, font=dict(color="black")),
            tickfont=dict(size=14, color="black"),
            ticksuffix="%",
            range=[0, range_max],
            tick0=0,
            dtick=10,
            showgrid=True, gridcolor="#f0f0f0", zeroline=False,
        ),
        yaxis=dict(
            automargin=True,
            tickfont=dict(size=14, color="black"),
        ),
    )
    return fig, window_lbl


def _site_flw_activity_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, str, float]:
    """Per-(site_full_id, flw_username) case count, computed directly
    from whatever `df` already contains — i.e. respects the sidebar's
    global date filter (All data / Current month / Last 3 months /
    Last 6 months / custom range) as-is, with no additional lookback
    window applied here. So this updates automatically whenever the
    global filter changes: how many cases each FLW recruited/screened
    at each site, over whatever period is currently selected.

    Also returns a label for the actual date range present in `df`,
    and the number of weeks that range spans (based on the earliest
    and latest case-registration dates actually present, not a fixed
    assumption), used to turn the raw cases/FLW ratio into a
    cases/FLW/week rate.

    The returned DataFrame also carries each FLW's own earliest and
    latest `date_of_case_registered` within the currently filtered
    data (`first_date` / `last_date`), for display in hover text —
    this is per-FLW, not the overall window bounds.

    Note: this only covers FLWs with at least one case in the window.
    An FLW who recruited zero cases in the period leaves no rows in the
    case-level data at all, so they can't be told apart from an FLW who
    was never assigned to that site — there's no FLW roster in this data
    to compare against. The lowest-count active FLWs (sorted first in
    the hover list) are the closest available signal for underperformance."""
    empty = pd.DataFrame(columns=["site_full_id", "flw_username", "total", "first_date", "last_date"])
    if df.empty or "date_of_case_registered" not in df.columns:
        return empty, "", 1.0
    dates = df["date_of_case_registered"].dropna()
    if dates.empty:
        return empty, "", 1.0
    min_dt, max_dt = dates.min(), dates.max()
    start_lbl = min_dt.strftime("%b %Y")
    end_lbl = max_dt.strftime("%b %Y")
    window_lbl = end_lbl if start_lbl == end_lbl else f"{start_lbl} - {end_lbl}"
    weeks = max((max_dt - min_dt).days / 7, 1.0)
    d, flw = _valid_flw(df)
    if d.empty:
        return empty, window_lbl, weeks

    id_col = _id_col(d)
    site = _flw_site_series(d)
    site = site.where(site.ne(""), "Unknown site")

    grouped = d.groupby([site, flw])
    total = grouped[id_col].nunique()
    first_date = grouped["date_of_case_registered"].min()
    last_date = grouped["date_of_case_registered"].max()
    out = total.rename("total").to_frame()
    out["first_date"] = first_date
    out["last_date"] = last_date
    out = out.reset_index()
    out.columns = ["site_full_id", "flw_username", "total", "first_date", "last_date"]
    return out, window_lbl, weeks



_SITE_TILE_PALETTE = [
    "#4C6EF5", "#12B886", "#FA5252", "#F59F00", "#7950F2",
    "#E64980", "#15AABF", "#82C91E", "#FD7E14", "#5C7CFA",
]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _wrap_label(text: str, width: int = 15, max_lines: int = 4) -> str:
    """Word-wrap `text` onto multiple lines (joined with <br>) so long
    site names stay confined inside their own tile instead of
    overflowing into neighboring tiles. Truncates with an ellipsis past
    `max_lines` rather than letting very long names grow the box."""
    lines = textwrap.wrap(str(text), width=width, break_long_words=False)
    if not lines:
        return str(text)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "…"
    return "<br>".join(lines)


def _format_flw_hover_block(sub: pd.DataFrame, max_rows_per_col: int = 15, max_cols: int = 6) -> str:
    """One "name: **count**" line per FLW, sorted ascending (already
    sorted by the caller) — every FLW is listed, with no cap. The count
    is styled bold + blue so it stands out from the FLW name. Once a
    site has more FLWs than fit in a single column (`max_rows_per_col`),
    the list spreads across additional side-by-side columns — 2, 3,
    4... up to `max_cols` — so it grows wider rather than indefinitely
    taller, though a very large site can still end up with a tall,
    multi-column block."""
    entries = []
    for _, r in sub.iterrows():
        flag = " ⚠️" if r["total"] == 0 else ""
        value = f"<span style='color:#1565C0;font-weight:700'>{int(r['total']):,}</span>"
        entries.append(f"{html.escape(str(r['flw_username']))}: {value}{flag}")

    if len(entries) <= max_rows_per_col:
        return "<br>".join(entries)

    n_cols = min(max_cols, math.ceil(len(entries) / max_rows_per_col))
    rows_per_col = math.ceil(len(entries) / n_cols)
    cols = [entries[i * rows_per_col:(i + 1) * rows_per_col] for i in range(n_cols)]
    # No padding needed after the last (rightmost) column.
    col_widths = [max((len(e) for e in col), default=0) + 3 for col in cols[:-1]]

    lines = []
    for row_i in range(rows_per_col):
        parts = []
        for c, col in enumerate(cols):
            if row_i < len(col):
                cell = col[row_i]
                if c < len(cols) - 1:
                    cell = cell.ljust(col_widths[c])
                parts.append(cell)
        lines.append("".join(parts))
    return "<br>".join(lines)


def _fig_single_site_flw_detail(site: str, sub: pd.DataFrame, weeks: float = 1.0) -> go.Figure:
    """Full-detail per-FLW case-count bar chart for a single site — used
    when the sidebar's "Site" filter has narrowed the grid down to
    exactly one site. Every FLW gets its own bar with the count printed
    directly on it, so there's no dependence on hover at all: this
    sidesteps the exact failure mode that made hover unreliable for a
    single large site (e.g. Thanjavur's 41 FLWs) — the numbers are just
    always visible, for however many FLWs the site has, no cap. The
    chart's height scales with the FLW count, so a big site naturally
    renders as a big chart rather than being squeezed into one small
    tile. Sorted descending (lowest-recruiting, most likely
    underperforming FLWs first in the data) — Plotly's default
    category-axis ordering places the first row at the bottom of a
    horizontal bar chart, so this puts the lowest count at the top of
    the chart and the highest count at the bottom.

    Hover also shows each FLW's earliest and latest
    `date_of_case_registered` within the currently filtered data
    (their own recruitment start / last-recruitment dates), plus their
    cases/week rate. That rate — like the site-level cases/FLW/week
    rate in the title — is each FLW's total divided by the same
    `weeks` value (derived from the sidebar's selected Date range, not
    from the data itself), so a given FLW's cases/week always comes
    out identical here and in the "Top 10 FLWs by cases/week"
    leaderboard, which uses the same shared denominator."""
    plot_df = sub.sort_values("total", ascending=False)
    n_flw = len(plot_df)
    total_cases = int(plot_df["total"].sum())
    ratio = (total_cases / n_flw) if n_flw else 0.0
    ratio_per_week = ratio / weeks if weeks else ratio

    def _fmt_date(v) -> str:
        ts = pd.Timestamp(v)
        return ts.strftime("%d %b %Y") if pd.notna(ts) else "—"

    first_dates = plot_df["first_date"].apply(_fmt_date)
    last_dates = plot_df["last_date"].apply(_fmt_date)
    flw_rate_fmt = [
        f"{(t / weeks):.2f}" if weeks else f"{t:.2f}"
        for t in plot_df["total"]
    ]
    customdata = list(zip(first_dates, last_dates, flw_rate_fmt))

    fig = go.Figure(go.Bar(
        y=plot_df["flw_username"], x=plot_df["total"], orientation="h",
        marker_color="#4C6EF5",
        text=plot_df["total"].apply(lambda v: f"{int(v):,}"),
        textposition="outside",
        textfont=dict(size=12, color="black"),
        customdata=customdata,
        hovertemplate=(
            "<b>%{y}</b><br>Cases: <b>%{x:,}</b>"
            "<br>Recruitment start date: <b>%{customdata[0]}</b>"
            "<br>Last Recruitment Date: <b>%{customdata[1]}</b>"
            "<br>Cases/week: <b>%{customdata[2]}</b><extra></extra>"
        ),
    ))
    max_val = float(plot_df["total"].max()) if n_flw else 0.0
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{html.escape(str(site))}</b> — {n_flw} FLW(s) · "
                f"{total_cases:,} cases · <b>{ratio_per_week:.2f}</b> cases/FLW/week"
            ),
            font=dict(size=14, color="#333"),
            x=0,
        ),
        height=max(320, 26 * n_flw + 120),
        margin=dict(t=60, b=30, l=10, r=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        xaxis=dict(
            title=dict(text="Cases", font=dict(color="black")),
            range=[0, max_val * 1.15 + 1],
            tickfont=dict(size=11, color="black"),
            showgrid=True, gridcolor="#f0f0f0", zeroline=False,
        ),
        yaxis=dict(
            automargin=True,
            tickfont=dict(size=11, color="black"),
        ),
    )
    return fig


def _fig_site_flw_grid(stats: pd.DataFrame, weeks: float = 1.0) -> go.Figure:
    """Grid of one colorful box per site — wrapped site name, active-FLW
    count, and cases/FLW/week rate shown directly on the box (no hover
    needed for those). Hovering a box additionally lists every FLW at
    that site and their case count for the window, sorted ascending so
    the lowest-recruiting (most likely underperforming) FLWs appear
    first. Tiles are laid out in descending order of active-FLW count,
    filled row by row, so the top row holds the highest-FLW sites, the
    next row the mid-range sites, and the last row the lowest.

    When the filtered data has exactly one site (e.g. the sidebar's
    "Site" filter is set to a specific site), this returns a full
    per-FLW detail bar chart instead of a 1-tile grid — see
    `_fig_single_site_flw_detail`."""
    if stats.empty:
        return go.Figure()

    site_flw_counts = stats.groupby("site_full_id")["flw_username"].nunique().sort_values(
        ascending=False
    )
    site_order = site_flw_counts.index.tolist()
    n_sites = len(site_order)

    if n_sites == 1:
        site = site_order[0]
        sub = stats.loc[stats["site_full_id"] == site]
        return _fig_single_site_flw_detail(site, sub, weeks)

    n_cols = min(4, n_sites) or 1
    n_rows = math.ceil(n_sites / n_cols)
    max_flw_at_a_site = int(site_flw_counts.max()) if n_sites else 0

    # Annotation font sizes are fixed pixel values in Plotly, so a tile's
    # text doesn't reflow the way HTML/CSS text does when the browser
    # window is resized — only the tile's rendered pixel width changes.
    # The number of grid columns is what actually determines that
    # per-tile width (a fixed-width row split N ways), so we scale font
    # size down as columns increase — the closest practical proxy for
    # "shrink the text so it still fits the box" without a live
    # container-width signal at figure-build time.
    _font_by_cols = {1: (12, 17, 12), 2: (11, 16, 11), 3: (10, 15, 10)}
    site_font, flw_font, ratio_font = _font_by_cols.get(n_cols, (9, 13, 9))

    # Pre-compute each site's cases/FLW/week rate first, so the darkest
    # blue in the grid always maps to the highest rate present (rather
    # than some fixed, possibly-wrong scale) — same "sample the
    # colorscale by each cell's own fraction of the max" approach used
    # for the confusion-matrix grid.
    site_rates = {}
    for site in site_order:
        sub = stats.loc[stats["site_full_id"] == site]
        n_flw = len(sub)
        total_cases = int(sub["total"].sum())
        ratio = (total_cases / n_flw) if n_flw else 0.0
        site_rates[site] = ratio / weeks if weeks else ratio
    rate_max = max(site_rates.values()) if site_rates else 0.0

    pad = 0.06
    shapes, annotations = [], []
    z_grid = [[np.nan] * n_cols for _ in range(n_rows)]
    text_grid = [[""] * n_cols for _ in range(n_rows)]

    for idx, site in enumerate(site_order):
        row, col = divmod(idx, n_cols)
        xi, yi = col, row
        sub = stats.loc[stats["site_full_id"] == site].sort_values("total")
        n_flw = len(sub)
        total_cases = int(sub["total"].sum())
        ratio_per_week = site_rates[site]

        frac = (ratio_per_week / rate_max) if rate_max else 0.0
        fill = pcolors.sample_colorscale("Blues", [frac])[0]
        text_color = "#FFFFFF" if frac > 0.55 else "#222222"
        muted_text_color = "rgba(255,255,255,0.85)" if frac > 0.55 else "#555555"

        header = [
            f"<b>{html.escape(str(site))}</b>",
            f"{n_flw} active FLW(s) · {total_cases:,} cases · {ratio_per_week:.2f} cases/FLW/week",
            "—",
        ]
        hover_text = "<br>".join(header) + "<br>" + _format_flw_hover_block(sub)
        z_grid[row][col] = 1
        text_grid[row][col] = hover_text

        x0, x1 = xi - 0.5 + pad, xi + 0.5 - pad
        y0, y1 = yi - 0.5 + pad, yi + 0.5 - pad
        shapes.append(dict(
            type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=y0, y1=y1,
            line=dict(color="rgba(0,0,0,0.15)", width=1), fillcolor=fill,
        ))
        annotations.append(dict(
            x=xi, y=yi - 0.32, text=_wrap_label(site),
            showarrow=False, font=dict(size=site_font, color=text_color, family="Arial Black"),
            align="center", yanchor="top",
        ))
        annotations.append(dict(
            x=xi, y=yi + 0.12, text=f"<b>{n_flw} FLWs</b>",
            showarrow=False, font=dict(size=flw_font, color=text_color, family="Arial Black"),
            yanchor="middle",
        ))
        annotations.append(dict(
            x=xi, y=yi + 0.42, text=f"<b>{ratio_per_week:.2f} cases/FLW/week</b>",
            showarrow=False, font=dict(size=ratio_font, color=muted_text_color, family="Arial Black"),
            yanchor="bottom",
        ))

    fig = go.Figure()
    # A heatmap trace (fully transparent) instead of fixed-pixel-size
    # scatter markers — its hit area exactly matches each cell's data-space
    # bounding box, so hover fires reliably across the whole tile
    # regardless of container width, rather than only inside a fixed-size
    # marker that may not fully cover a tile at some screen widths.
    #
    # One phantom coordinate is appended past the last column and past
    # the last row (z/text = NaN/"" so it's never drawn). Plotly derives
    # each cell's hover hit-area from the gap between consecutive x/y
    # coordinates; with only a single column and/or single row there's
    # no second coordinate to measure a gap against, and that axis's
    # hit-area can collapse to zero width. The phantom point sits just
    # outside the visible axis range set below, so it's never itself
    # seen or hoverable — it only gives the gap calculation something
    # to work with. (The n_sites == 1 case is now handled by the bar
    # chart above and never reaches this code path, but the padding is
    # kept here too since a 1-row or 1-column *multi-site* grid, e.g. 2
    # or 3 sites total, is still possible.)
    x_coords = list(range(n_cols)) + [n_cols]
    y_coords = list(range(n_rows)) + [n_rows]
    z_grid = [row + [np.nan] for row in z_grid] + [[np.nan] * (n_cols + 1)]
    text_grid = [row + [""] for row in text_grid] + [[""] * (n_cols + 1)]
    fig.add_trace(go.Heatmap(
        z=z_grid, x=x_coords, y=y_coords,
        text=text_grid, hovertemplate="%{text}<extra></extra>",
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
        zmin=0, zmax=1, showscale=False,
        hoverlabel=dict(
            align="left", bgcolor="white",
            font=dict(size=12, family="Courier New, monospace"),
        ),
    ))
    fig.update_layout(
        height=max(210 * n_rows + 30, 130 + 16 * min(max_flw_at_a_site, 30)),
        width=230 * n_cols + 30,
        autosize=False,
        # l > r (rather than equal) shifts the whole plot area — and
        # everything drawn in data coordinates within it (tiles, text) —
        # a bit to the right inside the fixed-width canvas, without
        # resizing anything: increasing the left margin moves the plot
        # area's left edge right, and decreasing the right margin by the
        # same amount moves its right edge right too, so the tiles slide
        # right as a block instead of stretching.
        margin=dict(t=2, b=10, l=25, r=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        # No fixedrange here (matches every other chart in this dashboard,
        # so the full zoom/pan/autoscale toolbar shows up as normal) and
        # an explicit, non-stretching width+height above instead — this
        # grid is shapes/annotations at fixed data coordinates, not a
        # data-driven trace, so letting Streamlit stretch it to a
        # variable container width (e.g. when its native fullscreen
        # toggle resizes the container) was what made the tiles collapse
        # into an overlapping cluster. A fixed pixel size sidesteps that
        # resize path entirely — the plot renders identically no matter
        # what container it's placed in.
        xaxis=dict(range=[-0.5, n_cols - 0.5], visible=False),
        yaxis=dict(
            range=[n_rows - 0.5, -0.5], visible=False,
        ),
        annotations=annotations,
        shapes=shapes,
        # Stashes the exact base pixel sizes used above so the JS
        # font-scaler in _render_leaderboard can multiply from the real
        # Python-computed values instead of trying to read them back out
        # of the DOM (which — depending on exactly when it reads —
        # can race Plotly's own rendering and lock in the wrong number).
        meta=dict(site_font=site_font, flw_font=flw_font, ratio_font=ratio_font),
    )
    return fig


def _render_leaderboard(
    df_all: pd.DataFrame,
    df_p2: pd.DataFrame,
    filter_start=None,
    filter_end=None,
) -> None:
    """Leaderboard tab: FLW cases/week leaderboard (over whatever the
    global filter currently selects) next to the overall FLW
    AI-override leaderboard.

    `filter_start`/`filter_end` are the sidebar's actual selected Date
    range boundaries — used (via `_filter_window_label_and_weeks`) as
    the single shared weeks-denominator for cases/week, so a given
    FLW's rate matches between the "Top 10 FLWs by cases/week" chart
    and the "FLWs by site" bar chart below. Falls back to the data's
    own min/max dates if no explicit filter range was passed in."""
    col_l, col_r = st.columns(2, gap="large")

    counts_window_lbl, counts_weeks = _filter_window_label_and_weeks(
        filter_start, filter_end
    )
    if not counts_window_lbl:
        counts_window_lbl, counts_weeks = _window_label_and_weeks(df_all)

    with col_l:
        title_suffix = f" — {counts_window_lbl}" if counts_window_lbl else ""
        st.markdown(
            "<div style='font-weight:700;font-size:15px;color:#333;margin-bottom:4px;"
            "min-height:40px;'>"
            f"🏆 Top 10 FLWs by cases/week{title_suffix}</div>",
            unsafe_allow_html=True,
        )
        fig_counts = _fig_leaderboard_flw_counts(
            df_all, window_label=counts_window_lbl, weeks=counts_weeks
        )
        if fig_counts.data:
            st.plotly_chart(
                fig_counts, width="stretch", key="lb_flw_counts",
                config={"displaylogo": False},
            )
        else:
            st.info("No FLW-level data available for the current filters.")

        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

        fig_retake, retake_window_lbl = _fig_leaderboard_retake_photo(df_all)
        retake_suffix = f" — {retake_window_lbl}" if retake_window_lbl else ""
        st.markdown(
            "<div style='font-weight:700;font-size:15px;color:#333;margin-bottom:4px;"
            "min-height:40px;'>"
            f"📸 Top 10 FLWs by % Retake Photo{retake_suffix} "
            "<span style='font-size:11px;font-weight:500;color:#888;font-style:italic;'>"
            "(all phases)</span></div>",
            unsafe_allow_html=True,
        )
        if fig_retake.data:
            st.plotly_chart(
                fig_retake, width="stretch", key="lb_retake_photo",
                config={"displaylogo": False},
            )
        else:
            st.info("No 'Retake photo' recommendations available for the last 3 months.")

    with col_r:
        st.markdown(
            "<div style='font-weight:700;font-size:15px;color:#333;margin-bottom:4px;"
            "min-height:40px;'>"
            "🤖 Top 10 FLWs by % AI override "
            "<span style='font-size:11px;font-weight:500;color:#888;font-style:italic;'>"
            "(AI-Enabled Screening only)</span></div>",
            unsafe_allow_html=True,
        )
        fig_override = _fig_leaderboard_ai_override(df_p2)
        if fig_override.data:
            st.plotly_chart(
                fig_override, width="stretch", key="lb_ai_override",
                config={"displaylogo": False},
            )
        else:
            st.info("No AI-override data available for the current filters.")

        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

        site_stats, _site_window_lbl, _site_weeks = _site_flw_activity_stats(df_all)
        # Reuse the same filter-derived label/weeks as the cases/week
        # leaderboard above (not this function's own data-derived
        # values) so a given FLW's cases/week rate matches exactly
        # between the two charts.
        site_suffix = f" — {counts_window_lbl}" if counts_window_lbl else ""
        # Sum of each site's unique-FLW count (i.e. count an FLW once per
        # site they're active at), matching what the tiles below add up
        # to — not a single global nunique(), which would dedupe an FLW
        # who appears at multiple sites down to one and undercount the
        # title relative to the tiles.
        total_flws = int(
            site_stats.groupby("site_full_id")["flw_username"].nunique().sum()
        ) if not site_stats.empty else 0
        st.markdown(
            "<div style='font-weight:700;font-size:15px;color:#333;margin-bottom:4px;"
            "min-height:40px;'>"
            f"🧑‍🤝‍🧑 FLWs by site{site_suffix} ({total_flws} FLWs total) "
            "<span style='font-size:11px;font-weight:500;color:#888;font-style:italic;'>"
            "(hover a box for per-FLW case counts, lowest first)</span></div>"
            "<div style='margin-bottom:-10px;'></div>",
            unsafe_allow_html=True,
        )
        fig_site_grid = _fig_site_flw_grid(site_stats, counts_weeks)
        if fig_site_grid.data:
            # Key includes the grid's shape (rows/cols/site set) rather than
            # a fixed string. This chart is a Heatmap whose grid dimensions,
            # axis ranges, and per-cell hover text all change size when a
            # global filter (site/date/gender) narrows the site list.
            # Streamlit reuses the existing Plotly div and does an in-place
            # Plotly.react() update when the key is unchanged — for a
            # Heatmap that changed shape, the hover hit-testing can end up
            # stale/dead after that in-place update. Varying the key with
            # the data forces Streamlit to fully remount the component on
            # a filter change instead, so hover keeps working.
            grid_sites = tuple(sorted(site_stats["site_full_id"].astype(str).unique()))
            grid_key = "lb_site_flw_grid_" + hashlib.md5(
                "|".join(grid_sites).encode("utf-8")
            ).hexdigest()[:10]
            # The chart is still a fixed-pixel canvas (see the comment on
            # width="content" below — that hasn't changed, and neither
            # has the box/tile size). A plain CSS override on the SVG
            # text's inline style didn't reliably stick, and neither did
            # matching on the text element's rendered font-family string
            # (Plotly doesn't necessarily serialize it back out verbatim),
            # so this selects text a third way instead: Plotly renders
            # each annotation as its own <g class="annotation"
            # data-index="N"> in the exact order fig.layout.annotations
            # was built in Python — 3 per site tile, always in the same
            # site-name / FLW-count / ratio cycle (see the loop in
            # _fig_site_flw_grid) — so `data-index % 3` reliably tells us
            # which role a given annotation is, with no dependency on how
            # Plotly happens to format text/style attributes.
            fig_meta = fig_site_grid.layout.meta or {}
            base_site_font = float(fig_meta.get("site_font", 9))
            base_flw_font = float(fig_meta.get("flw_font", 13))
            base_ratio_font = float(fig_meta.get("ratio_font", 9))

            st.markdown(
                '<style>.st-key-flw_site_grid_wrap { margin-left: 14px; }</style>',
                unsafe_allow_html=True,
            )
            with st.container(key="flw_site_grid_wrap"):
                st.plotly_chart(
                    fig_site_grid, width="content", key=grid_key,
                    # width="content" (not "stretch") plus the fixed
                    # width/height + autosize=False set on the figure itself
                    # (see _fig_site_flw_grid) means this chart is never
                    # asked to resize to fit a variable-width container. That
                    # container-resize path — triggered either by clicking
                    # Plotly's own Autoscale button or by Streamlit's native
                    # per-chart fullscreen toggle — is what was collapsing
                    # the tiles into an overlapping cluster.
                    config={"displaylogo": False},
                )
            st.iframe(
                f"""
                <script>
                (function() {{
                    const doc = window.parent.document;
                    const BASE_SITE = {base_site_font};
                    const BASE_FLW = {base_flw_font};
                    const BASE_RATIO = {base_ratio_font};

                    function scaleFonts() {{
                        const wrap = doc.querySelector('.st-key-flw_site_grid_wrap');
                        if (!wrap) return;
                        const groups = wrap.querySelectorAll('[data-index]');
                        if (!groups.length) return;

                        const w = window.parent.innerWidth;
                        let mult = 1.0;
                        if (w >= 2200) mult = 1.5;
                        else if (w >= 1800) mult = 1.35;
                        else if (w >= 1400) mult = 1.18;

                        groups.forEach(function(g) {{
                            const idx = parseInt(g.getAttribute('data-index'), 10);
                            if (isNaN(idx)) return;
                            const role = idx % 3;
                            const base = role === 0 ? BASE_SITE : (role === 1 ? BASE_FLW : BASE_RATIO);
                            g.querySelectorAll('text').forEach(function(t) {{
                                t.style.setProperty('font-size', (base * mult) + 'px', 'important');
                            }});
                        }});
                    }}

                    // Plotly renders asynchronously after Streamlit inserts
                    // the component, and can re-render on filter changes —
                    // a MutationObserver + resize listener keeps the sizing
                    // correct across both, instead of a one-off call. Since
                    // scaleFonts() always recomputes from the fixed BASE_*
                    // constants (never from the DOM), repeated calls can't
                    // compound or drift.
                    scaleFonts();
                    const observer = new MutationObserver(scaleFonts);
                    observer.observe(doc.body, {{childList: true, subtree: true}});
                    window.parent.addEventListener('resize', scaleFonts);
                }})();
                </script>
                """,
                height=1,
                width=1,
            )
        else:
            st.info("No FLW activity available for the last 3 months.")


# ════════════════════════════════════════════════════════════════════
# AI Inference — phase-2 confusion matrix (AI Result vs TSD Suspicion)
# ════════════════════════════════════════════════════════════════════

def _ai_inference_confusion(df_p2: pd.DataFrame) -> dict:
    """Confusion-matrix counts for phase-2 AI Inference: AI Result
    (prediction) vs the TSD's Suspicion field (ground truth) — "Suspicious"
    is the positive class on both sides. Restricted to rows with a valid
    AI Result ("Suspicious" / "Non suspicious") AND a completed TSD review
    (provisional_diagnosis present) — this matches the "Total TSD" count
    used by the phase-2 sankey in the Monitoring Dashboard. (Not the
    stricter phase-1 "reviewed" definition, which also requires `risk` —
    that field is typically blank for non-suspicious TSD reads and would
    undercount true negatives.)"""
    empty = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    if df_p2.empty:
        return empty
    ai_col = _ai_result_col(df_p2)
    if ai_col is None or "provisional_diagnosis" not in df_p2.columns:
        return empty

    ai_s_m  = _present_mask(df_p2[ai_col]) & _norm(df_p2[ai_col]).eq("suspicious")
    ai_ns_m = _present_mask(df_p2[ai_col]) & _norm(df_p2[ai_col]).eq("non suspicious")
    tsd_reviewed = _present_mask(df_p2["provisional_diagnosis"])
    ground_truth_susp = (
        _norm(df_p2["suspicion"]).eq("suspicious")
        if "suspicion" in df_p2.columns
        else pd.Series(False, index=df_p2.index)
    )

    tp = int((ai_s_m  &  ground_truth_susp & tsd_reviewed).sum())
    fp = int((ai_s_m  & ~ground_truth_susp & tsd_reviewed).sum())
    fn = int((ai_ns_m &  ground_truth_susp & tsd_reviewed).sum())
    tn = int((ai_ns_m & ~ground_truth_susp & tsd_reviewed).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _confusion_matrix_html(conf: dict, n_total: int, accent: str = "#0771eb") -> str:
    """Plain black-and-white HTML table rendering of the confusion
    matrix — same table style as the Descriptive tab's cross-tabs
    (_crosstab_html): no colour fill, just bordered cells, with the
    Total row/column bolded and tinted in the accent colour. AI Result
    (0/1) runs across the columns, TSD Suspicion (0/1) runs down the
    rows — the same labels used elsewhere in the AI Inference tab, with
    the 0/1 codes appended. Counts only, no percentages."""
    tp, fp, fn, tn = conf["tp"], conf["fp"], conf["fn"], conf["tn"]

    # Rows = TSD: 0 = Non-suspicious, 1 = Suspicious
    # Cols = AI:  0 = Non-suspicious, 1 = Suspicious
    core = [[tn, fp], [fn, tp]]
    row_totals = [tn + fp, fn + tp]
    col_totals = [tn + fn, fp + tp]

    col_labels = ["AI: Non-suspicious (0)", "AI: Suspicious (1)", "Total"]
    row_labels = ["TSD: Non-suspicious (0)", "TSD: Suspicious (1)", "Total"]

    header_cells = "".join(
        "<th style='padding:8px 10px;text-align:center;font-size:12px;"
        "font-weight:800;letter-spacing:.5px;text-transform:uppercase;"
        f"color:{accent if c == 'Total' else '#555'};border-left:1px solid #eee;'>"
        f"{html.escape(c)}</th>"
        for c in col_labels
    )

    body_rows = []
    for yi in range(2):
        cells = "".join(
            "<td style='padding:6px 10px;text-align:center;"
            "font-weight:700;color:#333;border-top:1px solid #f0f0f0;"
            "border-left:1px solid #f5f5f5;'>"
            f"{core[yi][xi]:,}</td>"
            for xi in range(2)
        )
        total_cell = (
            f"<td style='padding:6px 10px;text-align:center;font-weight:800;"
            f"color:{accent};border-top:1px solid #f0f0f0;"
            f"border-left:1px solid #f5f5f5;'>{row_totals[yi]:,}</td>"
        )
        body_rows.append(
            "<tr><td style='padding:6px 10px;text-align:left;font-weight:600;"
            "color:#333;border-top:1px solid #f0f0f0;'>"
            f"{html.escape(row_labels[yi])}</td>{cells}{total_cell}</tr>"
        )

    total_cells = "".join(
        f"<td style='padding:6px 10px;text-align:center;font-weight:800;"
        f"color:{accent};border-top:2px solid #ddd;"
        f"border-left:1px solid #f5f5f5;'>{col_totals[xi]:,}</td>"
        for xi in range(2)
    )
    body_rows.append(
        f"<tr><td style='padding:6px 10px;text-align:left;font-weight:800;"
        f"color:{accent};border-top:2px solid #ddd;'>Total</td>{total_cells}"
        f"<td style='padding:6px 10px;text-align:center;font-weight:800;"
        f"color:{accent};border-top:2px solid #ddd;"
        f"border-left:1px solid #f5f5f5;'>{n_total:,}</td></tr>"
    )

    return (
        "<div style='overflow-x:auto;margin-bottom:6px;'>"
        "<table style='border-collapse:collapse;background:#fff;"
        "border-radius:10px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.06);'>"
        "<thead><tr style='background:#fafafa;'>"
        "<th style='padding:8px 10px;text-align:left;font-size:12px;font-weight:800;"
        "letter-spacing:.5px;text-transform:uppercase;color:#555;'>"
        "TSD \\ AI Result</th>"
        f"{header_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def _render_ai_inference(df_p2: pd.DataFrame) -> None:
    """AI Inference tab: phase-2 confusion matrix (AI Result vs TSD
    Suspicion), with row/column totals."""
    conf = _ai_inference_confusion(df_p2)
    n_total = conf["tp"] + conf["fp"] + conf["fn"] + conf["tn"]
    if n_total == 0:
        st.info(
            "No phase-2 cases with both an AI Result and a completed TSD "
            "review are available for the current filters."
        )
        return

    st.markdown(
        "<div style='font-weight:700;font-size:15px;color:#333;margin-bottom:10px;'>"
        "🧩 Confusion Matrix — AI vs TSD</div>",
        unsafe_allow_html=True,
    )
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown(
            _confusion_matrix_html(conf, n_total),
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:13px;color:#888;margin-top:-2px;'>"
            f"n = {n_total:,} phase-2 cases with a completed TSD review</div>",
            unsafe_allow_html=True,
        )


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

    # Phase(s) present for each site — shown in a new "Phase" column
    # before the site name. A site_full_id normally maps to a single
    # phase (phase-varying sites like Goa already get split into
    # separate site_full_id values upstream in merge_data.py), but if
    # more than one phase value is ever seen for the same site_full_id,
    # all of them are shown together (e.g. "Phase 1 & 2") rather than
    # silently picking one.
    if "phase" in df.columns:
        def _phase_label(s: pd.Series) -> str:
            vals = sorted({int(v) for v in s.dropna().unique()})
            return " & ".join(str(v) for v in vals) if vals else "—"

        phase_by_site = df.groupby(site_key)["phase"].agg(_phase_label)
    else:
        phase_by_site = pd.Series(dtype=object)

    if "study_site_id" in df.columns:
        site_type_by_site = df.groupby(site_key)["study_site_id"].agg(
            lambda s: next((v for v in s if pd.notna(v) and str(v).strip()), "")
        )
    else:
        site_type_by_site = pd.Series(dtype=object)

    if "date_of_case_registered" in df.columns:
        min_date_by_site = df.groupby(site_key)["date_of_case_registered"].min()
        max_date_by_site = df.groupby(site_key)["date_of_case_registered"].max()
    else:
        min_date_by_site = pd.Series(dtype="datetime64[ns]")
        max_date_by_site = pd.Series(dtype="datetime64[ns]")

    # Retrieval date (date-only) per site — the most recent
    # "retrieval_date_time" seen for that site, from the cron
    # pipeline's merge step (see merge_data.py).
    if "retrieval_date_time" in df.columns:
        retrieval_dt_series = pd.to_datetime(df["retrieval_date_time"], errors="coerce")
        retrieval_date_by_site = retrieval_dt_series.groupby(site_key).max()
    else:
        retrieval_date_by_site = pd.Series(dtype="datetime64[ns]")

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

        max_dt_site = max_date_by_site.get(site)
        recent_date_str = max_dt_site.strftime('%d-%b-%Y') if pd.notna(max_dt_site) else "—"

        retrieval_dt = retrieval_date_by_site.get(site)
        if pd.notna(retrieval_dt):
            # Date and 24-hour-clock time (hour:minute only) on separate lines.
            retrieval_date_str = (
                f"{html.escape(retrieval_dt.strftime('%d-%b-%Y'))}"
                f"<br>{html.escape(retrieval_dt.strftime('%H:%M'))}"
            )
        else:
            retrieval_date_str = "—"

        phase_str = str(phase_by_site.get(site, "—"))

        rows_html.append(
            "<tr>"
            f"<td style='padding:5px 10px;text-align:center;font-weight:700;"
            f"color:#000;border-top:1px solid #eee;white-space:nowrap;'>{html.escape(phase_str)}</td>"
            f"<td style='padding:5px 10px;text-align:left;font-weight:600;"
            f"color:#333;border-top:1px solid #eee;'>{html.escape(site)}</td>"
            f"<td style='padding:5px 10px;text-align:center;font-weight:700;"
            f"color:{site_type_color};border-top:1px solid #eee;white-space:nowrap;'>{html.escape(site_type_str)}</td>"
            f"<td style='padding:5px 10px;text-align:center;font-weight:600;"
            f"color:#555;border-top:1px solid #eee;white-space:nowrap;line-height:1.4;'>{retrieval_date_str}</td>"
            f"<td style='padding:5px 10px;text-align:center;font-weight:600;"
            f"color:#555;border-top:1px solid #eee;white-space:nowrap;'>{html.escape(start_date_str)}</td>"
            f"<td style='padding:5px 10px;text-align:center;font-weight:600;"
            f"color:#555;border-top:1px solid #eee;white-space:nowrap;'>{html.escape(recent_date_str)}</td>"
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

    # Totals row — summed across every site currently in the table.
    # Current-month total only counts sites that aren't "stopped" (those
    # show "–" per-row and are excluded from the sum, same as their cell).
    total_cur_screened = sum(
        int(cur_month_by_site.get(site, 0))
        for site in sites_present
        if not is_stopped(site)
    )
    total_screened = int(screened_by_site.sum())
    total_susp = int(suspicious_by_site.sum())
    total_high = int(high_by_site.sum())
    total_susp_pct = round(total_susp / total_screened * 100, 1) if total_screened else 0.0
    total_high_pct = round(total_high / total_screened * 100, 1) if total_screened else 0.0

    totals_row_html = (
        "<tr style='background:#fafafa;'>"
        "<td style='padding:6px 10px;text-align:center;font-weight:800;"
        "color:#999;border-top:2px solid #ddd;'>–</td>"
        "<td style='padding:6px 10px;text-align:left;font-weight:800;"
        "color:#333;border-top:2px solid #ddd;'>Total</td>"
        "<td style='padding:6px 10px;text-align:center;font-weight:800;"
        "color:#999;border-top:2px solid #ddd;'>–</td>"
        "<td style='padding:6px 10px;text-align:center;font-weight:800;"
        "color:#999;border-top:2px solid #ddd;'>–</td>"
        "<td style='padding:6px 10px;text-align:center;font-weight:800;"
        "color:#999;border-top:2px solid #ddd;'>–</td>"
        "<td style='padding:6px 10px;text-align:center;font-weight:800;"
        "color:#999;border-top:2px solid #ddd;'>–</td>"
        "<td style='padding:6px 10px;text-align:center;font-weight:800;"
        f"color:#4CA64C;border-top:2px solid #ddd;'>{total_cur_screened:,}</td>"
        "<td style='padding:6px 10px;text-align:center;font-weight:800;"
        f"color:#228B22;border-top:2px solid #ddd;'>{total_screened:,}</td>"
        "<td style='padding:6px 10px;text-align:center;font-weight:800;"
        f"color:#F4A900;border-top:2px solid #ddd;'>{total_susp:,} ({total_susp_pct}%)</td>"
        "<td style='padding:6px 10px;text-align:center;font-weight:800;"
        f"color:#D94040;border-top:2px solid #ddd;'>{total_high:,} ({total_high_pct}%)</td>"
        "</tr>"
    )

    table_html = (
        "<div style='overflow-x:auto;'>"
        "<table style='width:100%;border-collapse:collapse;background:#fff;"
        "border-radius:12px;overflow:hidden;box-shadow:0 2px 16px rgba(0,0,0,.07);'>"
        "<thead><tr style='background:#fafafa;'>"
        "<th style='padding:10px 14px;text-align:center;font-size:14px;"
        "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
        "color:#555;'>Phase</th>"
        "<th style='padding:10px 14px;text-align:left;font-size:14px;"
        "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
        "color:#555;'>Sites</th>"
        "<th style='padding:10px 14px;text-align:center;font-size:14px;"
        "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
        "color:#555;'>Site Type</th>"
        "<th style='padding:10px 14px;text-align:center;font-size:14px;"
        "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
        "color:#555;'>Retrival D&T</th>"
        "<th style='padding:10px 14px;text-align:center;font-size:14px;"
        "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
        "color:#555;'>Start Date</th>"
        "<th style='padding:10px 14px;text-align:center;font-size:14px;"
        "font-weight:800;letter-spacing:.8px;text-transform:uppercase;"
        "color:#555;'>Recent Date</th>"
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
        "</tr></thead><tbody>" + "".join(rows_html) + totals_row_html + "</tbody></table></div>"
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
# Descriptive tab — two cross-tabs below the site table:
# Provisional Diagnosis × Specialist Recommendation, phase-2 only,
# split left/right by AI Result (Suspicious / Non-suspicious).
# ════════════════════════════════════════════════════════════════════

# Fixed display order for provisional_diagnosis rows in the cross-tabs
# below (clinical severity order, not alphabetical). Any value found in
# the data but not in this list is appended after these, still before
# the Total row.
_PROVISIONAL_DIAGNOSIS_ORDER = [
    "Oral cavity normal",
    "Benign",
    "Other",
    "Smokeless tobacco keratosis",
    "Homogenous Oral leukoplakia",
    "Oral submucosal fibrosis",
    "Oral lichen planus",
    "Speckled oral leukoplakia",
    "Erythroplakia",
    "Verrucous oral leukoplakia",
    "Proliferative verrucous oral leukoplakia",
    "Squamous cell carcinoma of oral mucous membrane",
    "Non-diagnostic procedure finding",
]


def _crosstab_with_totals(df: pd.DataFrame, row_col: str, col_col: str, row_order=None):
    """Two-way frequency table of row_col × col_col (rows with a blank
    value in either column are dropped), with a Total row and Total
    column appended. Returns None if either column is missing or no
    rows survive the blank filter.

    row_order : optional list giving the display order for the row
    categories (matched case-insensitively). Categories present in the
    data but not in row_order are appended afterward, in alphabetical
    order, still ahead of the Total row.
    """
    if row_col not in df.columns or col_col not in df.columns:
        return None
    sub = df[[row_col, col_col]].copy()
    mask = _present_mask(sub[row_col]) & _present_mask(sub[col_col])
    sub = sub.loc[mask]
    if sub.empty:
        return None
    sub[row_col] = sub[row_col].astype(str).str.strip()
    sub[col_col] = sub[col_col].astype(str).str.strip()

    ct = pd.crosstab(sub[row_col], sub[col_col])
    ct = ct.reindex(sorted(ct.columns), axis=1)

    if row_order:
        lookup = {str(v).strip().lower(): v for v in ct.index}
        ordered_idx = [lookup[v.strip().lower()] for v in row_order if v.strip().lower() in lookup]
        leftover_idx = sorted(v for v in ct.index if v not in ordered_idx)
        ct = ct.reindex(ordered_idx + leftover_idx, axis=0)
    else:
        ct = ct.reindex(sorted(ct.index), axis=0)

    ct["Total"] = ct.sum(axis=1)
    total_row = ct.sum(axis=0)
    total_row.name = "Total"
    ct = pd.concat([ct, total_row.to_frame().T])
    return ct


def _crosstab_html(ct: pd.DataFrame, heading: str, accent: str, n_total: int) -> str:
    """Render a _crosstab_with_totals() table as a styled HTML table,
    with the Total row/column bolded and tinted in the given accent
    color."""
    col_names = list(ct.columns)
    row_names = list(ct.index)

    header_cells = "".join(
        "<th style='padding:8px 10px;text-align:center;font-size:12px;"
        "font-weight:800;letter-spacing:.5px;text-transform:uppercase;"
        f"color:{accent if c == 'Total' else '#555'};border-left:1px solid #eee;'>"
        f"{html.escape(str(c))}</th>"
        for c in col_names
    )

    body_rows = []
    for r in row_names:
        is_total_row = (r == "Total")
        cells = "".join(
            "<td style='padding:6px 10px;text-align:center;"
            + (f"font-weight:800;color:{accent};" if (is_total_row or c == "Total") else "color:#333;")
            + ("border-top:2px solid #ddd;" if is_total_row else "border-top:1px solid #f0f0f0;")
            + "border-left:1px solid #f5f5f5;'>"
            + f"{int(ct.loc[r, c]):,}</td>"
            for c in col_names
        )
        row_label_style = (
            f"font-weight:800;color:{accent};border-top:2px solid #ddd;"
            if is_total_row else
            "font-weight:600;color:#333;border-top:1px solid #f0f0f0;"
        )
        body_rows.append(
            f"<tr><td style='padding:6px 10px;text-align:left;{row_label_style}'>"
            f"{html.escape(str(r))}</td>{cells}</tr>"
        )

    return (
        "<div style='overflow-x:auto;margin-bottom:18px;'>"
        f"<div style='font-weight:800;font-size:14px;color:{accent};margin-bottom:6px;'>"
        f"{html.escape(heading)} "
        f"<span style='font-weight:600;color:#888;font-size:12px;'>(n = {n_total:,})</span></div>"
        "<table style='width:100%;border-collapse:collapse;background:#fff;"
        "border-radius:10px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.06);'>"
        "<thead><tr style='background:#fafafa;'>"
        "<th style='padding:8px 10px;text-align:left;font-size:12px;font-weight:800;"
        "letter-spacing:.5px;text-transform:uppercase;color:#555;'>"
        "Provisional Dx \\ Specialist Rec.</th>"
        f"{header_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def _row_pct_crosstab(
    df: pd.DataFrame, row_col: str, col_col: str, row_order=None, col_order=None,
    row_canon=None, col_canon=None,
):
    """Two-way frequency + row-percentage table of row_col × col_col —
    mirrors Stata's `tab row_col col_col, row`. Rows with a blank value
    in either column are dropped. Returns (counts_df, row_pct_df), each
    with an appended Total row/column (row percentages are computed so
    every row, including the Total row, sums to 100). Returns (None,
    None) if either column is missing or no rows survive the blank
    filter.

    row_order / col_order : optional lists giving display order for the
    row/column categories (matched case-insensitively). Categories
    present in the data but not listed are appended afterward,
    alphabetically, still ahead of the Total row/column.

    row_canon / col_canon : optional {lowercased_value: canonical_label}
    dicts used to collapse casing variants (e.g. "Non suspicious" and
    "Non Suspicious") into a single category before crosstabbing.
    """
    if row_col not in df.columns or col_col not in df.columns:
        return None, None
    sub = df[[row_col, col_col]].copy()
    mask = _present_mask(sub[row_col]) & _present_mask(sub[col_col])
    sub = sub.loc[mask]
    if sub.empty:
        return None, None
    sub[row_col] = sub[row_col].astype(str).str.strip()
    sub[col_col] = sub[col_col].astype(str).str.strip()
    if row_canon:
        sub[row_col] = sub[row_col].apply(lambda v: row_canon.get(v.lower(), v))
    if col_canon:
        sub[col_col] = sub[col_col].apply(lambda v: col_canon.get(v.lower(), v))

    ct = pd.crosstab(sub[row_col], sub[col_col])

    def _ordered(labels, order):
        if order:
            lookup = {str(v).strip().lower(): v for v in labels}
            picked = [lookup[v.strip().lower()] for v in order if v.strip().lower() in lookup]
            leftover = sorted(v for v in labels if v not in picked)
            return picked + leftover
        return sorted(labels)

    ct = ct.reindex(_ordered(ct.index, row_order), axis=0)
    ct = ct.reindex(_ordered(ct.columns, col_order), axis=1)

    ct["Total"] = ct.sum(axis=1)
    total_row = ct.sum(axis=0)
    total_row.name = "Total"
    ct = pd.concat([ct, total_row.to_frame().T])

    pct = ct.div(ct["Total"], axis=0) * 100
    return ct, pct


def _row_pct_crosstab_html(
    ct: pd.DataFrame, pct: pd.DataFrame, row_label: str, col_label: str,
    heading: str, accent: str, n_total: int,
) -> str:
    """Render a _row_pct_crosstab() pair as a styled HTML table, each
    cell showing frequency on top and row-percentage below it (Stata
    `, row` style), with the Total row/column bolded and tinted."""
    col_names = list(ct.columns)
    row_names = list(ct.index)

    header_cells = "".join(
        "<th style='padding:8px 10px;text-align:center;font-size:12px;"
        "font-weight:800;letter-spacing:.5px;text-transform:uppercase;"
        f"color:{accent if c == 'Total' else '#555'};border-left:1px solid #eee;'>"
        f"{html.escape(str(c))}</th>"
        for c in col_names
    )

    body_rows = []
    for r in row_names:
        is_total_row = (r == "Total")
        cells = []
        for c in col_names:
            emph = is_total_row or (c == "Total")
            freq = int(ct.loc[r, c])
            pctval = float(pct.loc[r, c])
            cells.append(
                "<td style='padding:6px 10px;text-align:center;"
                + ("border-top:2px solid #ddd;" if is_total_row else "border-top:1px solid #f0f0f0;")
                + "border-left:1px solid #f5f5f5;'>"
                + f"<div style='font-weight:{'800' if emph else '700'};"
                + f"color:{accent if emph else '#333'};'>{freq:,}</div>"
                + f"<div style='font-size:11px;font-weight:500;color:#999;'>{pctval:.2f}%</div>"
                + "</td>"
            )
        row_label_style = (
            f"font-weight:800;color:{accent};border-top:2px solid #ddd;"
            if is_total_row else
            "font-weight:600;color:#333;border-top:1px solid #f0f0f0;"
        )
        body_rows.append(
            f"<tr><td style='padding:6px 10px;text-align:left;{row_label_style}'>"
            f"{html.escape(str(r))}</td>{''.join(cells)}</tr>"
        )

    return (
        "<div style='overflow-x:auto;margin-bottom:18px;'>"
        f"<div style='font-weight:800;font-size:14px;color:{accent};margin-bottom:6px;'>"
        f"{html.escape(heading)} "
        f"<span style='font-weight:600;color:#888;font-size:12px;'>(n = {n_total:,})</span></div>"
        "<table style='width:100%;border-collapse:collapse;background:#fff;"
        "border-radius:10px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.06);'>"
        "<thead><tr style='background:#fafafa;'>"
        "<th style='padding:8px 10px;text-align:left;font-size:12px;font-weight:800;"
        "letter-spacing:.5px;text-transform:uppercase;color:#555;'>"
        f"{html.escape(row_label)} \\ {html.escape(col_label)}</th>"
        f"{header_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
        "<div style='font-size:10.5px;color:#aaa;margin-top:4px;'>"
        "Each cell: frequency, row %</div>"
        "</div>"
    )


def _diag_accuracy(
    ct: pd.DataFrame, pos_row: str = "Yes (1)", neg_row: str = "No (0)",
    pos_col: str = "Suspicious (1)", neg_col: str = "Non Suspicious (0)",
) -> dict | None:
    """Sensitivity / Specificity / PPV / NPV / LR+ / LR- from a
    _row_pct_crosstab() counts table, treating pos_row as index-test-
    positive (lesion/patch present) and pos_col as reference-standard-
    positive (Suspicious). Returns None if any of the four cells are
    missing (e.g. that category didn't occur in this subset)."""
    try:
        tp = float(ct.loc[pos_row, pos_col])
        fp = float(ct.loc[pos_row, neg_col])
        fn = float(ct.loc[neg_row, pos_col])
        tn = float(ct.loc[neg_row, neg_col])
    except KeyError:
        return None
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv  = tp / (tp + fp) if (tp + fp) else float("nan")
    npv  = tn / (tn + fn) if (tn + fn) else float("nan")
    lr_pos = (sens / (1 - spec)) if spec < 1 else float("inf")
    lr_neg = ((1 - sens) / spec) if spec > 0 else float("inf")
    return {
        "Sensitivity": sens * 100, "Specificity": spec * 100,
        "PPV": ppv * 100, "NPV": npv * 100,
        "LR+": lr_pos, "LR-": lr_neg,
    }


def _diag_accuracy_html(stats: dict, accent: str) -> str:
    """Compact stat-strip rendering of _diag_accuracy()'s output,
    meant to sit directly beneath a _row_pct_crosstab_html() table."""
    def _fmt(v: float) -> str:
        return "∞" if v == float("inf") else f"{v:.2f}"

    items = [
        ("Sensitivity", f"{stats['Sensitivity']:.1f}%"),
        ("Specificity", f"{stats['Specificity']:.1f}%"),
        ("PPV", f"{stats['PPV']:.1f}%"),
        ("NPV", f"{stats['NPV']:.1f}%"),
        ("LR+", _fmt(stats["LR+"])),
        ("LR-", _fmt(stats["LR-"])),
    ]
    cells = "".join(
        "<div style='text-align:center;flex:1;'>"
        "<div style='font-size:10px;font-weight:700;letter-spacing:.4px;"
        f"text-transform:uppercase;color:#999;'>{label}</div>"
        f"<div style='font-size:14px;font-weight:800;color:{accent};'>{val}</div>"
        "</div>"
        for label, val in items
    )
    return (
        "<div style='display:flex;gap:4px;padding:10px 4px 2px;"
        "border-top:1px dashed #ddd;margin-top:4px;'>"
        f"{cells}</div>"
    )


_SUSPICION_CANON: dict[str, str] = {
    "non suspicious": "Non Suspicious (0)",
    "suspicious": "Suspicious (1)",
}

_LESION_CANON: dict[str, str] = {
    "no": "No (0)",
    "yes": "Yes (1)",
}


def _render_lesion_suspicion_crosstabs(df_all: pd.DataFrame, df_p2: pd.DataFrame) -> None:
    """Descriptive-tab addendum: Suspicion × Lesion/Patch cross-tab with
    row percentages (Stata `tab suspicion lesion_patch, row`) — overall
    on the combined dataset, then split by AI Result (Suspicious /
    Non-suspicious), phase-2 only. Three tables shown side by side.

    Sensitivity/Specificity/PPV/NPV/LR+/LR- are shown only under the
    Overall table. They're deliberately omitted from the two AI-split
    tables: those subsets are conditioned on the AI's own suspicion
    call, which is correlated with both lesion_patch and suspicion, so
    accuracy figures computed inside them would reflect spectrum/
    collider-stratification bias rather than lesion_patch's true
    diagnostic accuracy."""
    if "lesion_patch" not in df_all.columns or "suspicion" not in df_all.columns:
        st.warning(
            "Lesion/Patch × Suspicion table skipped — the Overall data doesn't "
            f"currently have both \"lesion_patch\" and \"suspicion\" columns.\n\n"
            f"\"lesion_patch\" present: {'lesion_patch' in df_all.columns}\n\n"
            f"\"suspicion\" present: {'suspicion' in df_all.columns}"
        )
        return

    st.markdown(
        "<hr style='border:none;border-top:1.5px solid #ddd;margin:26px 0 18px;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-weight:700;font-size:15px;color:#333;margin-bottom:10px;'>"
        "🩺 Lesion/Patch × Suspicion — Overall vs. by AI Result</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        ct, pct = _row_pct_crosstab(
            df_all, "suspicion", "lesion_patch",
            row_canon=_SUSPICION_CANON, col_canon=_LESION_CANON,
        )
        if ct is None:
            st.info("No cases with both Lesion/Patch and Suspicion completed.")
        else:
            st.markdown(
                _row_pct_crosstab_html(
                    ct, pct, "Suspicion", "Lesion/Patch",
                    "📋 Overall", "#0771eb", int(ct.loc["Total", "Total"]),
                ),
                unsafe_allow_html=True,
            )
            # Sensitivity/specificity etc. are defined relative to
            # Lesion/Patch as the index test and Suspicion as the
            # reference standard — that mapping is independent of how
            # the table above is displayed, so compute it off a
            # lesion-as-row/suspicion-as-column crosstab, not off the
            # display-oriented `ct` above.
            ct_stats, _ = _row_pct_crosstab(
                df_all, "lesion_patch", "suspicion",
                row_canon=_LESION_CANON, col_canon=_SUSPICION_CANON,
            )
            stats = _diag_accuracy(ct_stats) if ct_stats is not None else None
            if stats:
                st.markdown(_diag_accuracy_html(stats, "#0771eb"), unsafe_allow_html=True)

    ai_col = _ai_result_col(df_p2) if not df_p2.empty else None
    susp_sub = df_p2.loc[_norm(df_p2[ai_col]).eq("suspicious")] if ai_col else df_p2.iloc[0:0]
    non_sub = df_p2.loc[_norm(df_p2[ai_col]).eq("non suspicious")] if ai_col else df_p2.iloc[0:0]

    with col_b:
        ct, pct = _row_pct_crosstab(
            susp_sub, "suspicion", "lesion_patch",
            row_canon=_SUSPICION_CANON, col_canon=_LESION_CANON,
        )
        if ct is None:
            st.info("No Suspicious phase-2 cases with both fields completed.")
        else:
            st.markdown(
                _row_pct_crosstab_html(
                    ct, pct, "Suspicion", "Lesion/Patch",
                    "🟠 AI: Suspicious", "#F4A900", int(ct.loc["Total", "Total"]),
                ),
                unsafe_allow_html=True,
            )

    with col_c:
        ct, pct = _row_pct_crosstab(
            non_sub, "suspicion", "lesion_patch",
            row_canon=_SUSPICION_CANON, col_canon=_LESION_CANON,
        )
        if ct is None:
            st.info("No Non-suspicious phase-2 cases with both fields completed.")
        else:
            st.markdown(
                _row_pct_crosstab_html(
                    ct, pct, "Suspicion", "Lesion/Patch",
                    "🟢 AI: Non-Suspicious", "#228B22", int(ct.loc["Total", "Total"]),
                ),
                unsafe_allow_html=True,
            )


def _render_ai_result_crosstabs(df_p2: pd.DataFrame) -> None:
    """Descriptive-tab addendum: Provisional Diagnosis × Specialist
    Recommendation cross-tabs, phase-2 data only, shown side by side —
    Suspicious on the left, Non-suspicious on the right (per ai_result)."""
    if df_p2.empty:
        return
    ai_col = _ai_result_col(df_p2)
    if ai_col is None:
        return
    if "provisional_diagnosis" not in df_p2.columns or "specialist_recommendation" not in df_p2.columns:
        return

    susp_mask = _norm(df_p2[ai_col]).eq("suspicious")
    non_mask = _norm(df_p2[ai_col]).eq("non suspicious")

    st.markdown(
        "<hr style='border:none;border-top:1.5px solid #ddd;margin:26px 0 18px;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-weight:700;font-size:15px;color:#333;margin-bottom:10px;'>"
        "🧾 Provisional Diagnosis × Specialist Recommendation — by AI Result "
        "(Phase 2)</div>",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(2)

    with col_left:
        ct = _crosstab_with_totals(
            df_p2.loc[susp_mask], "provisional_diagnosis", "specialist_recommendation",
            row_order=_PROVISIONAL_DIAGNOSIS_ORDER,
        )
        if ct is None:
            st.info("No Suspicious phase-2 cases with both fields completed.")
        else:
            st.markdown(
                _crosstab_html(ct, "🟠 AI: Suspicious", "#F4A900", int(ct.loc["Total", "Total"])),
                unsafe_allow_html=True,
            )

    with col_right:
        ct = _crosstab_with_totals(
            df_p2.loc[non_mask], "provisional_diagnosis", "specialist_recommendation",
            row_order=_PROVISIONAL_DIAGNOSIS_ORDER,
        )
        if ct is None:
            st.info("No Non-suspicious phase-2 cases with both fields completed.")
        else:
            st.markdown(
                _crosstab_html(ct, "🟢 AI: Non-Suspicious", "#228B22", int(ct.loc["Total", "Total"])),
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════
# Public entry point — called by app.py's main()
# ════════════════════════════════════════════════════════════════════

def render(
    df_all: pd.DataFrame,
    df_p1: pd.DataFrame,
    df_p2: pd.DataFrame,
    filter_start=None,
    filter_end=None,
) -> None:
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
    filter_start, filter_end : the sidebar's actual selected "Date
             range" boundaries (date objects). Used as the shared
             cases/week denominator on the Leaderboard tab so a given
             FLW's rate matches between its two cases/week charts —
             see `_render_leaderboard`.
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
        .st-key-btn_research_leaderboard button,
        .st-key-btn_research_ai_inference button {
            padding: 10px 12px !important;
            font-size: 22px !important;
            font-weight: 800 !important;
            min-height: 0 !important;
            line-height: 1.3 !important;
        }
        .st-key-btn_research_descriptive button *,
        .st-key-btn_research_leaderboard button *,
        .st-key-btn_research_ai_inference button * {
            font-weight: 800 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "research_tab" not in st.session_state:
        st.session_state.research_tab = 0

    # ── Tab button row — Descriptive · Leaderboard · AI Inference ──
    col1, col2, col3, _sp = st.columns([1, 1, 1, 3])
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
    with col3:
        if st.button(
            "🧠 AI Inference",
            key="btn_research_ai_inference",
            type="primary" if st.session_state.research_tab == 2 else "secondary",
            use_container_width=True,
        ):
            st.session_state.research_tab = 2
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
        _render_lesion_suspicion_crosstabs(df_all, df_p2)
        _render_ai_result_crosstabs(df_p2)
    elif st.session_state.research_tab == 1:
        if df_all.empty and df_p2.empty:
            st.info("No data available for the current filters.")
        else:
            _render_leaderboard(df_all, df_p2, filter_start, filter_end)
    else:
        if df_p2.empty:
            st.info("No AI-Enabled Screening (phase-2) data available for the current filters.")
        else:
            _render_ai_inference(df_p2)

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