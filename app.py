"""
====================================================================
AAROGYA AAROHAN — REAL-TIME MONITORING DASHBOARD  v3
IISC Oral Cancer Project / TANUH

Combined dataset : OCP_COMB_DATA_OVERALL.parquet
  phase == '1'  → Phase 1 records
  phase == '2'  → Phase 2 records
  Overall tab   → full combined dataset

Run:
    streamlit run app_v4.py
====================================================================
"""

from __future__ import annotations
import base64
import hashlib
import html
import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_js_eval import streamlit_js_eval

_log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# 1.  PAGE CONFIG — must be first Streamlit call
# ════════════════════════════════════════════════════════════════════

try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path.cwd()

logo_path = BASE / "static" / "logo.png"

st.set_page_config(
    page_title="Aarogya Aarohan | TANUH OCP",
    page_icon=str(logo_path.resolve()) if logo_path.exists() else "🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Mobile viewport fix ──────────────────────────────────────────────
# Without a proper viewport meta tag, mobile browsers render the page in a
# desktop-width "layout viewport" (often ~980px) and scale it down, which
# makes window.innerWidth report that wide layout width instead of the
# real device width — silently breaking every phone-width check below.
st.markdown(
    """
    <script>
    (function() {
        var doc = window.parent.document;
        if (!doc.querySelector('meta[name="viewport"]')) {
            var m = doc.createElement('meta');
            m.name = 'viewport';
            m.content = 'width=device-width, initial-scale=1, maximum-scale=1';
            doc.head.appendChild(m);
        }
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

screen_width = streamlit_js_eval(js_expressions="window.innerWidth", key="SCREEN_WIDTH")

if screen_width is None:
    # First paint of a fresh session — the JS round-trip to get the real
    # device width hasn't completed yet. Show a minimal placeholder and
    # stop here instead of rendering the full header/tabs at a guessed
    # (desktop) width, which would otherwise flash briefly before being
    # replaced once the real width resolves and triggers a rerun.
    st.markdown(
        "<div style='display:flex;align-items:center;justify-content:center;"
        "height:60vh;color:#bbb;font-size:14px;font-family:\"Segoe UI\",Arial,sans-serif;'>"
        "Loading…</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Responsive breakpoints (mirrors common phone/tablet/desktop widths) ──
BP_PHONE  = 640   # < 640px  → phone
BP_TABLET = 1024  # 640–1024 → tablet, > 1024 → desktop

IS_PHONE  = screen_width < BP_PHONE
IS_TABLET = BP_PHONE <= screen_width < BP_TABLET
IS_DESKTOP = screen_width >= BP_TABLET


def _cols(n_desktop: int, n_tablet: int | None = None, n_phone: int = 1):
    """Return the column count to use for the current screen size.

    n_desktop applies >= BP_TABLET, n_tablet applies on tablet widths
    (defaults to n_desktop if not given, capped at 2-3 cols), n_phone
    applies on phones (defaults to a single stacked column).
    """
    if IS_PHONE:
        return n_phone
    if IS_TABLET:
        return n_tablet if n_tablet is not None else min(n_desktop, 3)
    return n_desktop


# ════════════════════════════════════════════════════════════════════
# 2.  PATHS & CONSTANTS
# ════════════════════════════════════════════════════════════════════

# ── Data source ──────────────────────────────────────────────────────
# Local filesystem only.
#   • LOCAL_DATA_DIR — where the parquet and map_data.json live. Defaults to
#            the original dev machine path; deployed environments (e.g. the
#            GKE pod, which has an init container stage the data into /data)
#            set OCP_DATA_DIR to override it.
#   • GEOJSON_DATA_DIR — the app's static/ folder, where india_states.geojson
#            lives (this one ships alongside the app, not with the raw data).
LOCAL_DATA_DIR = Path(
    os.environ.get(
        "OCP_DATA_DIR",
        r"/mnt/d/OneDrive/IISC/TANUH/OralCancer_Project/Raw_Data/Dashboard",
    )
)
PARQUET_PATH = LOCAL_DATA_DIR / "OCP_COMB_DATA_OVERALL.parquet"

# Persistent on-disk cache (survives app/server restarts) so the very first
# load of the dashboard can serve instantly from this file instead of
# blocking on a fresh parquet read. Saved alongside app.py.
DATA_CACHE_PATH = BASE / "ocp_data_cache.pkl"

# Fingerprint of this source file. Stored inside the cache alongside the
# data itself; whenever app.py changes (this fingerprint changes), the
# on-disk cache is treated as stale/incompatible and a fresh copy is loaded
# and written out, instead of risking loading data that no longer matches
# the current code's expectations
try:
    _CODE_VERSION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
except NameError:
    _CODE_VERSION = "unknown"

GEOJSON_DATA_DIR = BASE / "static"


SITE_ORDER = [
    "KLE", "AIIMS Delhi", "MSMF", "Krishnagiri", "Thanjavur",
    "MPMMCC", "CCHRC", "Borooah", "Goa",
]

AMBER_HIGH = "#E0631A"
AMBER_LOW  = "#F7C548"

# ════════════════════════════════════════════════════════════════════
# 3.  GLOBAL CSS
# ════════════════════════════════════════════════════════════════════

_CSS = """
<style>
.ocp-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px 24px 15px;
    box-shadow: 0 2px 16px rgba(0,0,0,.07);
    margin-bottom: 10px;
    height: 155px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.ocp-card-grn { border-left: 6px solid #228B22; }
.ocp-card-amb { border-left: 6px solid #F4A900; }

.card-eyebrow {
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .8px;
    color: #bbb;
    margin-bottom: 4px;
}
.card-big-grn {
    font-size: 36px;
    font-weight: 800;
    color: #228B22;
    line-height: 1.1;
    margin: 2px 0 2px;
}
.card-big-amb {
    font-size: 36px;
    font-weight: 800;
    color: #F4A900;
    line-height: 1.1;
    margin: 2px 0 2px;
}

/* Dual-stat card (e.g. "Suspicious | High risk") — shrinks and wraps
   instead of clipping when the column gets narrow. */
.ocp-card-dualstat {
    height: auto !important;
    min-height: 120px;
    overflow: visible !important;
}
.card-dualstat-row {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 6px 10px;
    font-weight: 800;
    font-size: clamp(20px, 4vw, 36px);
    line-height: 1.2;
    white-space: normal;
    color: #F4A900;
}
.card-dualstat-row .dualstat-sep {
    color: #ccc;
    font-weight: 300;
}
.card-dualstat-row .dualstat-high {
    color: #D94040;
}
.card-sub {
    font-size: 17px;
    font-weight: 700;
    color: #000000;
    margin-top: 4px;
}

div[data-testid="metric-container"] {
    background: #fff;
    border: 1px solid #ebebeb;
    border-radius: 10px;
    padding: 14px 18px 10px;
    box-shadow: 0 1px 6px rgba(0,0,0,.05);
}

button[data-testid="baseButton-primary"],
button[kind="primary"],
.stButton button[kind="primary"] {
    background: #16828A !important;
    color: #fff !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
}
button[data-testid="baseButton-secondary"],
button[kind="secondary"],
.stButton button[kind="secondary"] {
    background: #f3f7f3 !important;
    color: #2d3d2d !important;
    border-radius: 8px !important;
}

/* Compact sizing for top-level view tabs (Dashboard / Research Dashboard). */
.st-key-btn_dash_main button,
.st-key-btn_dash_research button {
    padding: 10px 16px !important;
    font-size: 13px !important;
    min-height: 0 !important;
    line-height: 1.3 !important;
}

/* Larger section tabs (Overall / Phase 1 / Phase 2). */
.st-key-btn_ov button,
.st-key-btn_p1 button,
.st-key-btn_p2 button {
    padding: 10px 12px !important;
    font-size: 22px !important;
    font-weight: 800 !important;
    min-height: 0 !important;
    line-height: 1.3 !important;
}

.st-key-btn_ov button *,
.st-key-btn_p1 button *,
.st-key-btn_p2 button * {
    font-weight: 800 !important;
}

.sep  { border: none; border-top: 1px solid #e8e8e8; margin: 8px 0; }
.sep2 { border: none; border-top: 2px solid #e0e0e0; margin: 4px 0 16px; }

.coming-soon {
    text-align: center;
    padding: 110px 0;
    font-size: 1.45rem;
    color: #ccc;
    font-weight: 600;
    letter-spacing: .4px;
}

.admin-name {
    font-weight: 700;
    font-size: 15px;
    padding-top: 8px;
    line-height: 1.35;
}
.admin-note {
    font-size: 11px;
    color: #ccc;
}

.block-container {
    padding-top: 1rem !important;
}

/* ════════════════════════════════════════════════════════════════
   RESPONSIVE — tablet (≤1024px)
   ════════════════════════════════════════════════════════════════ */
@media (max-width: 1024px) {
    .ocp-card {
        height: auto;
        min-height: 130px;
        padding: 16px 18px 12px;
    }
    .card-big-grn, .card-big-amb {
        font-size: clamp(22px, 5vw, 34px) !important;
    }
    .st-key-btn_ov button,
    .st-key-btn_p1 button,
    .st-key-btn_p2 button {
        font-size: 18px !important;
        padding: 8px 10px !important;
    }
}

/* ════════════════════════════════════════════════════════════════
   RESPONSIVE — phone (≤640px)
   ════════════════════════════════════════════════════════════════ */
@media (max-width: 640px) {
    .block-container {
        padding-left: .6rem !important;
        padding-right: .6rem !important;
    }
    .ocp-card {
        min-height: 110px;
        padding: 14px 16px 10px;
        margin-bottom: 8px;
    }
    .card-big-grn, .card-big-amb {
        font-size: clamp(20px, 7vw, 28px) !important;
    }
    .card-dualstat-row {
        font-size: clamp(18px, 6.5vw, 26px) !important;
    }
    .card-sub {
        font-size: 14px;
    }
    .card-eyebrow {
        font-size: 9.5px;
    }
    /* Section tabs (Overall / Phase 1 / Phase 2) shrink to fit 3-across */
    .st-key-btn_ov button,
    .st-key-btn_p1 button,
    .st-key-btn_p2 button {
        font-size: 14px !important;
        padding: 8px 4px !important;
        white-space: nowrap;
    }
    /* Top-level view buttons (Monitoring / Research) */
    .st-key-btn_dash_main button,
    .st-key-btn_dash_research button {
        font-size: 11px !important;
        padding: 8px 8px !important;
    }
    div[data-testid="metric-container"] {
        padding: 10px 12px 8px;
    }
}
</style>
"""

# ════════════════════════════════════════════════════════════════════
# 4.  DATA LOADING
# ════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _img_to_base64(path_str: str) -> str:
    """Read an image file and return a data-URI for inline HTML <img> use."""
    p = Path(path_str)
    suffix = p.suffix.lstrip(".").lower() or "png"
    data = p.read_bytes()
    return f"data:image/{suffix};base64,{base64.b64encode(data).decode()}"


class _DataStore:
    """Shared in-memory holder for the combined dataset with a background
    refresh thread (stale-while-revalidate), backed by a persistent on-disk
    pickle cache.

    On the very first dashboard open after a server (re)start, there is
    nothing in memory yet. Instead of blocking that first request on a
    fresh parquet read, we serve instantly from `cache_path` (the last
    known-good snapshot on disk) if it exists, for every tab. A daemon
    thread then immediately reloads the real parquet source in the
    background, atomically swaps the fresh DataFrame into memory, and
    rewrites the on-disk cache — so subsequent opens (and restarts) keep
    getting faster, always-available data. If no on-disk cache exists yet
    (very first run ever), the first call falls back to a synchronous
    parquet read, same as before, and then seeds the cache file.

    The on-disk cache also embeds a `code_version` fingerprint of app.py.
    If the code has changed since the cache file was written, the old
    cache is treated as stale/incompatible and discarded automatically —
    a fresh copy is loaded from the source and the cache file is
    overwritten with the new version.
    """

    def __init__(
        self,
        path: Path,
        refresh_seconds: int = 3600,
        cache_path: Path | None = None,
        code_version: str = "unknown",
    ):
        self._path = path
        self._refresh_seconds = refresh_seconds
        self._cache_path = cache_path
        self._code_version = code_version
        self._lock = threading.Lock()
        self._df: pd.DataFrame | None = None
        self._loaded_at: float = 0.0
        self._thread_started = False
        self._loaded_from_cache = False

    @staticmethod
    def _read_from_disk(path: Path) -> pd.DataFrame:
        resolved = Path(path)
        if not resolved.exists():
            _log.error("Data file not found: %s", resolved)
            raise FileNotFoundError(str(resolved))

        df = pd.read_parquet(resolved, engine="pyarrow")

        if "date_of_case_registered" in df.columns:
            df["date_of_case_registered"] = pd.to_datetime(
                df["date_of_case_registered"], errors="coerce", dayfirst=True
            )

        _CAT_COLS = (
            "ai_result", "suspicion", "risk",
            "gender", "site_id", "provisional_diagnosis", "phase",
        )
        for _c in _CAT_COLS:
            if _c in df.columns and df[_c].dtype == object:
                df[_c] = df[_c].astype("category")

        return df

    def _read_from_cache(self) -> pd.DataFrame | None:
        if not self._cache_path or not self._cache_path.exists():
            return None
        try:
            payload = pd.read_pickle(self._cache_path)
        except Exception:
            _log.exception("Failed to read on-disk data cache %s; ignoring it", self._cache_path)
            return None

        # Backwards-compat: older cache files stored a bare DataFrame with
        # no version info. Treat those as stale so they get replaced too.
        if not isinstance(payload, dict) or "df" not in payload:
            _log.info("On-disk cache %s has no version info; treating as stale", self._cache_path)
            return None

        if payload.get("code_version") != self._code_version:
            _log.info(
                "On-disk cache %s was written by a different code version; discarding it",
                self._cache_path,
            )
            return None

        return payload["df"]

    def _write_to_cache(self, df: pd.DataFrame) -> None:
        if not self._cache_path:
            return
        try:
            tmp_path = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
            payload = {"code_version": self._code_version, "df": df}
            pd.to_pickle(payload, tmp_path)
            tmp_path.replace(self._cache_path)
        except Exception:
            _log.exception("Failed to write on-disk data cache %s", self._cache_path)

    def _load_fresh(self) -> pd.DataFrame:
        """Read the real source from disk and refresh the on-disk cache."""
        df = self._read_from_disk(self._path)
        self._write_to_cache(df)
        return df

    def _refresh_loop(self, initial_delay: float) -> None:
        first = True
        while True:
            time.sleep(initial_delay if first else self._refresh_seconds)
            first = False
            try:
                new_df = self._load_fresh()
                with self._lock:
                    self._df = new_df
                    self._loaded_at = time.time()
                    self._loaded_from_cache = False
                _log.info("Background refresh: reloaded %d rows", len(new_df))
            except Exception:
                _log.exception("Background data refresh failed; keeping previous data")

    def _ensure_thread(self, initial_delay: float = 0.0) -> None:
        if self._thread_started:
            return
        with self._lock:
            if not self._thread_started:
                threading.Thread(
                    target=self._refresh_loop, args=(initial_delay,), daemon=True
                ).start()
                self._thread_started = True

    def get(self) -> pd.DataFrame:
        """Return a DataFrame instantly wherever possible.

        Order of preference for the very first call in this process:
          1. Already in memory (fast path for every call after the first).
          2. Last known-good snapshot on disk (`cache_path`) — instant,
             served to all tabs, while a background thread kicks off a
             real refresh from the source right away.
          3. Nothing cached anywhere yet — synchronous read from the real
             source (only happens on the very first run ever).
        """
        with self._lock:
            have_data = self._df is not None
        if not have_data:
            cached_df = self._read_from_cache()
            if cached_df is not None:
                with self._lock:
                    if self._df is None:
                        self._df = cached_df
                        self._loaded_at = time.time()
                        self._loaded_from_cache = True
                # Kick off a real refresh from the source right away in the
                # background, instead of waiting a full refresh cycle.
                self._ensure_thread(initial_delay=0.0)
                with self._lock:
                    return self._df
            else:
                df = self._read_from_disk(self._path)
                self._write_to_cache(df)
                with self._lock:
                    if self._df is None:
                        self._df = df
                        self._loaded_at = time.time()
                        self._loaded_from_cache = False
        self._ensure_thread(initial_delay=self._refresh_seconds)
        with self._lock:
            return self._df

    def force_refresh(self) -> pd.DataFrame:
        """Synchronously reload from disk right now (used by the manual
        'Refresh Data' button) and update both the in-memory and on-disk
        cache."""
        df = self._load_fresh()
        with self._lock:
            self._df = df
            self._loaded_at = time.time()
            self._loaded_from_cache = False
        self._ensure_thread(initial_delay=self._refresh_seconds)
        return df


@st.cache_resource(show_spinner=False)
def _get_data_store() -> _DataStore:
    # st.cache_resource makes this a single shared singleton across all
    # sessions in the server process, which is what lets the background
    # refresh thread serve every tab/session from the same in-memory cache.
    return _DataStore(
        PARQUET_PATH,
        refresh_seconds=3600,
        cache_path=DATA_CACHE_PATH,
        code_version=_CODE_VERSION,
    )


def _load_data(path: Path = PARQUET_PATH) -> pd.DataFrame:
    """Return the combined dataset instantly from the shared cache.

    Only the very first call across the whole server process blocks on a
    disk read; after that, every dashboard open (all tabs) is served from
    memory while a background thread keeps the data fresh.
    """
    try:
        return _get_data_store().get()
    except FileNotFoundError as exc:
        st.error(
            f"Data file not found: **{exc}**\n\n"
            "Make sure `OCP_COMB_DATA_OVERALL.parquet` is present in:\n\n"
            f"`{LOCAL_DATA_DIR}`"
        )
        st.stop()


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
# 5.  ANALYTICS HELPERS
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


def _gender_col(df: pd.DataFrame) -> str | None:
    for c in ("gender", "Gender", "Sex"):
        if c in df.columns:
            return c
    return None


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


def _date_window_mask(
    df: pd.DataFrame,
    window: str,
    custom_start: pd.Timestamp | None = None,
    custom_end: pd.Timestamp | None = None,
    anchor: pd.Timestamp | None = None,
) -> pd.Series:
    if "date_of_case_registered" not in df.columns:
        return pd.Series(True, index=df.index)

    dates = df["date_of_case_registered"]
    mask  = dates.notna()
    if not mask.any():
        return pd.Series(False, index=df.index)

    if anchor is None:
        anchor = dates.max()
    if pd.isna(anchor):
        return pd.Series(True, index=df.index)

    if window == "All data":
        return pd.Series(True, index=df.index)

    if window == "Last 6 months":
        start = _months_back(pd.Timestamp(anchor), 6)
        return mask & (dates >= start) & (dates <= anchor)
    if window == "Last 3 months":
        start = _months_back(pd.Timestamp(anchor), 3)
        return mask & (dates >= start) & (dates <= anchor)
    if window == "Custom range":
        if custom_start is None or custom_end is None:
            return pd.Series(True, index=df.index)
        return mask & (dates.dt.date >= custom_start.date()) & (dates.dt.date <= custom_end.date())

    return pd.Series(True, index=df.index)


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

    Phase 1 columns:
      total, reviewed_total, suspicious, high_risk, low_risk,
      cumulative, susp_pct, non_susp_pct, month_lbl

    Phase 2 columns:
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

    # ── Phase 1 ──────────────────────────────────────────────────────
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
        mon["non_susp_pct"] = (100 - mon["susp_pct"]).round(1)
        mon["month_lbl"]    = mon["ym"].dt.start_time.dt.strftime("%b %Y")
        return mon

    # ── Phase 2 ──────────────────────────────────────────────────────
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
# 6.  CHART BUILDERS
# ════════════════════════════════════════════════════════════════════


def map_height_from_width(width:int)->int:
    if width >= 2400:
        return 1100
    elif width >= 1800:
        return 900
    elif width >= 1400:
        return 780
    elif width >= 1000:
        return 650
    elif width >= 640:
        return 500
    # Phone: map + legend share a stacked layout, so neither needs the
    # tall fixed height used on wider screens.
    return 360

def plot_height_from_width(width: int) -> int:
    if width >= 2200:
        return 650
    elif width >= 1700:
        return 550
    elif width >= 1300:
        return 500
    else:
        return 400


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
            title=dict(text="Cumulative", font=dict(size=14, color="#6036f8")),
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
            title=dict(text="Monthly new", font=dict(size=14, color="#3f9d3f")),
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=14, color="#3f9d3f"),
        ),
    }
    fig.update_layout(**reg_layout)
    fig.update_layout(height=plot_height_from_width(screen_width))
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
    # to avoid colliding with the bars.
    if IS_PHONE:
        legend_font, top_margin, legend_y = 10, 86, 1.22
    elif IS_TABLET:
        legend_font, top_margin, legend_y = 12, 76, 1.18
    else:
        legend_font, top_margin, legend_y = 14, 64, 1.16

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
    fig.update_layout(height=plot_height_from_width(screen_width))
    return fig


@st.cache_data(max_entries=4, ttl=3600, show_spinner=False)
def _fig_sankey_phase2(df: pd.DataFrame, sw: int = 1200) -> go.Figure:
    """
    Sankey — Phase 2 pathway:
      Col0: AI result
      Col1: Not-yet-reviewed remainder (per AI path)
      Col2: All Tele Reviewed (merge node)
      Col3: Tele-specialist suspicion verdict
      Col4: Risk outcome (High / Low)
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
        f"Pending TSD\n{pend} ({pf(pend, ai_s)})",
        f"Not Reviewed\n{norev} ({pf(norev, ai_ns)})",
        f"Total TSD\n{tele_rev} ({pf(tele_rev, total)})",
        f"TSD Suspicious\n{tele_susp} ({pf(tele_susp, tele_rev)})",
        f"TSD Non-Suspicious\n{tele_nsusp} ({pf(tele_nsusp, tele_rev)})",
    ]

    customdata_nodes = [
        (f"AI Suspicious: {ai_s} ({pf(ai_s, total)} of total screened)<br>"
         f"TSD High Risk: {high_ai_s} | TSD Low Risk: {low_ai_s}"),
        (f"AI Non-Suspicious: {ai_ns} ({pf(ai_ns, total)} of total screened)<br>"
         f"TSD High Risk: {high_ai_ns} | TSD Low Risk: {low_ai_ns}"),
        f"Pending TSD: {pend} ({pf(pend, total)} of total screened)",
        f"Not Reviewed: {norev} ({pf(norev, total)} of total screened)",
        (f"Total TSD: {tele_rev} ({pf(tele_rev, total)} of total screened)<br>"
         f"From AI-Suspicious: {tr_s} ({pf(tr_s, ai_s)}) | From AI-Non-Suspicious: {tr_ns} ({pf(tr_ns, ai_ns)})"),
        (f"TSD Suspicious: {tele_susp} ({pf(tele_susp, total)} of total screened)<br>"
         f"High Risk: {high} ({pf(high, total)}) | Low Risk: {low} ({pf(low, total)})<br>"
         f"TSD from AI Suspicious: {susp_from_ai_s} ({pf(susp_from_ai_s, ai_s)}) | TSD from AI Non-Suspicious: {susp_from_ai_ns} ({pf(susp_from_ai_ns, ai_ns)})<br>"
         f"From AI-Suspicious path: High {high_ai_s}, Low {low_ai_s}<br>"
         f"From AI-Non-Suspicious path: High {high_ai_ns}, Low {low_ai_ns}"),
        (f"TSD Non-Suspicious: {tele_nsusp} ({pf(tele_nsusp, total)} of total screened)<br>"
         f"TSD from AI Suspicious: {nsusp_from_ai_s} ({pf(nsusp_from_ai_s, ai_s)}) | TSD from AI Non-Suspicious: {nsusp_from_ai_ns} ({pf(nsusp_from_ai_ns, ai_ns)})"),
    ]

    node_colors = [
        AMBER_HIGH, "#484848", "#c8c8c8", "#e0e0e0",
        "#c87d18", AMBER_HIGH, "#484848",
    ]
    node_x = [0.01, 0.01, 0.25, 0.25, 0.52, 0.76, 0.76]
    node_y = [0.18, 0.68, 0.45, 0.80, 0.34, 0.10, 0.64]

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
        node_x.extend([0.985, 0.985])
        node_y.extend([0.04, 0.16])
        _add_link(5, risk_start,     high, "rgba(217,64,64,1)")
        _add_link(5, risk_start + 1, low,  "rgba(247,197,72,0.40)")
    elif high > 0:
        labels.append(f"High Risk\n{high} ({pf(high, total)})")
        customdata_nodes.append(
            f"High Risk: {high} ({pf(high, total)} of total screened)<br>"
            f"From AI-Suspicious path: {high_ai_s} ({pf(high_ai_s, ai_s)}) | From AI-Non-Suspicious path: {high_ai_ns} ({pf(high_ai_ns, ai_ns)})"
        )
        node_colors.append("#D94040")
        node_x.append(0.985)
        node_y.append(0.10)
        _add_link(5, risk_start, high, "rgba(217,64,64,0.32)")
    elif low > 0:
        labels.append(f"Low Risk\n{low} ({pf(low, total)})")
        customdata_nodes.append(
            f"Low Risk: {low} ({pf(low, total)} of total screened)<br>"
            f"From AI-Suspicious path: {low_ai_s} ({pf(low_ai_s, ai_s)}) | From AI-Non-Suspicious path: {low_ai_ns} ({pf(low_ai_ns, ai_ns)})"
        )
        node_colors.append(AMBER_LOW)
        node_x.append(0.985)
        node_y.append(0.10)
        _add_link(5, risk_start, low, "rgba(247,197,72,0.40)")

    # Rescale node y-positions into a safe canvas band.
    if node_y:
        y_min, y_max = min(node_y), max(node_y)
        safe_low, safe_high = 0.14, 0.90
        if y_max > y_min:
            span   = y_max - y_min
            node_y = [safe_low + ((y - y_min) / span) * (safe_high - safe_low) for y in node_y]
        else:
            node_y = [min(max(y, safe_low), safe_high) for y in node_y]

    if not links:
        return go.Figure()
    srcs, tgts, vals, clrs, pcts = zip(*links)

    # Drop linkless nodes & remap indices so positions align correctly.
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

    fig_height = plot_height_from_width(sw)
    top_mg     = 28
    bot_mg     = max(90, int(fig_height * 0.11))

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
        node=dict(
            pad=12,
            thickness=18,
            line=dict(color="rgba(0,0,0,0)", width=0),
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
            value=list(vals),
            customdata=pcts_f,
            color=list(clrs),
            label=link_labels,
            hovertemplate=(
                "From: %{source.customdata}<br>"
                "→ To: %{target.customdata}<br>"
                "Count: %{value:,} (%{customdata:.1f}% of total screened)<extra></extra>"
            ),
        ),
    ))

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=fig_height,
        margin=dict(t=top_mg, b=bot_mg, l=10, r=24),
        font=dict(size=11, family="Arial, sans-serif", color="#666666"),
    )
    fig.update_traces(
        textfont=dict(color="#666666", size=11, family="Arial, sans-serif"),
        selector=dict(type="sankey"),
    )
    return fig

# ════════════════════════════════════════════════════════════════════
# 7.  UI COMPONENT HELPERS
# ════════════════════════════════════════════════════════════════════

def _duration_text(df: pd.DataFrame) -> str:
    """Return a formatted duration string like '(Jun 2023 – Dec 2024 · 18 months)'."""
    if "date_of_case_registered" not in df.columns:
        return ""
    dates = df["date_of_case_registered"].dropna()
    if dates.empty:
        return ""
    start, end = dates.min(), dates.max()
    months = (end.year - start.year) * 12 + (end.month - start.month)
    month_label = "month" if months == 1 else "months"
    return f"({start.strftime('%b %Y')} – {end.strftime('%b %Y')} · {months} {month_label})"


def _animated_metric_card(
    value: int,
    suffix: str = "",
    sub_text: str = "",
    duration_text: str = "",
    big_color: str = "#228B22",
    border_color: str = "#228B22",
    animate: bool = False,
) -> None:
    """Render a metric card; animate count-up only when animate=True."""
    safe_sub      = html.escape(sub_text)
    safe_dur_text = html.escape(duration_text)
    safe_suffix   = json.dumps(suffix)
    safe_static   = html.escape(f"{value:,}{suffix}")
    dur_html = (
        f'<div style="font-size:12px;color:{big_color};font-style:italic;'
        f'margin-top:1px;margin-bottom:3px;">{safe_dur_text}</div>'
        if duration_text else ""
    )

    # Scale type/padding/height down for phone and tablet so nothing clips.
    if IS_PHONE:
        big_font, sub_font, pad, card_height = 26, 13, "8px 16px 6px", 116
    elif IS_TABLET:
        big_font, sub_font, pad, card_height = 31, 15, "9px 20px 7px", 122
    else:
        big_font, sub_font, pad, card_height = 36, 17, "10px 24px 8px", 130

    if animate:
        script   = f"""<script>
(function(){{
var el=document.getElementById('cnt');
var target={value};var sfx={safe_suffix};var start=null;
function ease(t){{return t<.5?2*t*t:-1+(4-2*t)*t;}}
function step(ts){{
  if(!start)start=ts;
  var p=Math.min((ts-start)/900,1);
  el.textContent=Math.round(ease(p)*target).toLocaleString('en-US')+sfx;
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


def _map_stat_card(value: int, label: str, items: list[str], big_color: str, border_color: str) -> None:
    """Animated card with count-up and scrolling state-name ticker."""
    # Escape content sourced from external JSON to prevent HTML injection.
    safe_label = html.escape(label)
    safe_items = [html.escape(s) for s in items]
    items_str  = " · ".join(safe_items) if safe_items else ""

    if IS_PHONE:
        big_font, sub_font, pad, card_height = 24, 13, "8px 16px 6px", 104
    elif IS_TABLET:
        big_font, sub_font, pad, card_height = 29, 15, "9px 20px 7px", 110
    else:
        big_font, sub_font, pad, card_height = 36, 17, "10px 24px 8px", 114

    scroll_html = ""
    if items_str:
        scroll_html = (
            '<div style="overflow:hidden;white-space:nowrap;margin-top:5px;">'
            '<span style="display:inline-block;font-size:11px;color:#888;'
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
                    box-sizing:border-box;height:{card_height}px;margin:8px;
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
                    var p=Math.min((ts-start)/900,1);
                    el.textContent=Math.round(ease(p)*target).toLocaleString('en-US');
                    if(p<1)requestAnimationFrame(step);
                }}
                requestAnimationFrame(step);
                }})();
                </script>
                </body></html>"""
    st.iframe(_html, height=card_height + 16)


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
            status, z_val = "Other", 0
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

    # ── Site pins from map_data ──────────────────────────────────────
    map_sites = map_data.get("map_sites", [])

    _PIN_COLORS = {
        "district_sentinel": {"ongoing": "#C0392B", "future": "#C0392B"},
        "sentinel":          {"ongoing": "#2980B9", "future": "#2980B9"},
    }
    _PIN_LABELS = {
        "district_sentinel": "District Level Deployment and Sentinel Site",
        "sentinel":          "Sentinel Site",
    }

    # Per-site text position overrides to avoid label collisions
    _TEXT_POS = {
        "West Bengal":           "top right",
        "Kolkata Medical College": "bottom right",
        "Mathura":               "middle right",
        "Kohima":                "middle right",
        "Goa":                   "middle right",
        "Thanjavur":             "middle right",
        "AIIMS Delhi":           "top left",
        "Varanasi":              "middle right",
        "Cachar & Guwahati":     "middle right",
        "Bangalore":             "middle right",
    }

    for site_type, label in _PIN_LABELS.items():
        sites = [s for s in map_sites if s.get("type") == site_type]
        if not sites:
            continue
        color = "#C0392B" if site_type == "district_sentinel" else "#2980B9"

        # Add each site as a separate trace so textposition can vary per pin
        for i, s in enumerate(sites):
            name = s.get("name", "")
            tpos = _TEXT_POS.get(name, "middle right")

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
                textfont=dict(size=14, color=color),
                name=label if i == 0 else "",
                hovertemplate="<b>%{text}</b><extra></extra>",
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
                '<span style="font-size:11px;"> Upcoming &nbsp;</span>'
                '<span style="color:#dfe0ea;font-size:15px;">■</span>'
                '<span style="font-size:11px;"> Not Covered</span>'
                '<br>'
                '<span style="color:#C0392B;font-size:13px;">⬤</span>'
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
        fig.add_annotation(
            x=0.75,
            y=0.25,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="middle",
            showarrow=False,
            text=(
                '<span style="color:#64E64A;font-size:22px;">■</span>'
                '<span style="font-size:16px;"> Ongoing</span>'
                '<br>'
                '<span style="color:#F4CA67;font-size:22px;">■</span>'
                '<span style="font-size:16px;"> Upcoming</span>'
                '<br>'
                '<span style="color:#dfe0ea;font-size:22px;">■</span>'
                '<span style="font-size:16px;"> Not Covered</span>'
                '<br><br>'
                '<span style="color:#C0392B;font-size:20px;">⬤</span>'
                '<span style="font-size:16px;"> District Level Deployment &amp; Sentinel Site</span>'
                '<br>'
                '<span style="color:#2980B9;font-size:20px;">⬤</span>'
                '<span style="font-size:16px;"> Sentinel Site</span>'
            ),
            font=dict(size=16),
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
        margin = dict(t=26, b=30, l=10, r=220)  # tight top margin keeps stat cards close to the map

    fig.update_layout(
        height=map_height_from_width(width),
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
# 8.  TAB RENDERERS
# ════════════════════════════════════════════════════════════════════

@st.fragment
def _tab_overall(df: pd.DataFrame, df_map: "pd.DataFrame | None" = None) -> None:
    """Overall tab — shows combined Phase 1 + Phase 2 statistics."""
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

    ov_cards = st.columns(_cols(2, 2, 1), gap="large")
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
        )
        st.plotly_chart(
            _fig_registrations(mon),
            width="stretch",
            config={"displayModeBar": False},
            key="ov_reg",
        )

    with col_r:
        # % is of total screened (total_cum) — matches the Site-wise
        # Summary table below, not the reviewed-only subset.
        susp_rate = round(total_susp / total_cum * 100, 1) if total_cum else 0
        high_pct  = round(high_total / total_cum * 100, 1) if total_cum else 0

        st.markdown(
            f'<div class="ocp-card ocp-card-amb ocp-card-dualstat" style="padding:10px 24px 8px;">'
            f'<div class="card-dualstat-row">'
            f'<span>Suspicious: {total_susp:,} ({susp_rate}%)</span>'
            f'<span class="dualstat-sep">|</span>'
            f'<span class="dualstat-high">High risk: {high_total:,} ({high_pct}%)</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _fig_status(mon, "Suspicious", "Suspicious"),
            width="stretch",
            config={"displayModeBar": False},
            key="ov_susp",
        )

    # ── India State Map Section ──────────────────────────────────────
    map_data = _load_map_data()
    if map_data:
        st.markdown(
            "<hr style='border:none;border-top:1px solid #eee;margin:8px 0 4px;'>",
            unsafe_allow_html=True,
        )
        phones     = int(map_data.get("phones_deployed",    0))
        fhw        = int(map_data.get("fhw_trained",        0))
        facilities = int(map_data.get("facilities_covered", 0))
        ongoing    = map_data.get("ongoing_states", [])
        future     = map_data.get("upcoming_states",  [])

        mc_n = _cols(5, 3, 1)
        mc_cols = st.columns(mc_n, gap="small")
        mc_cards = [
            (lambda: _map_stat_card(len(ongoing), "📍Ongoing States/UTs", ongoing, "#237213", "#237113")),
            (lambda: _animated_metric_card(facilities, "", "🏨 Healthcare Facilities Covered", "", "#0771eb", "#0771eb", animate=True)),
            (lambda: _animated_metric_card(fhw,        "", "👩‍⚕️ Frontline Health Worker Trained", "", "#0771eb", "#0771eb", animate=True)),
            (lambda: _animated_metric_card(phones,     "", "📱 Phones Deployed", "", "#0771eb", "#0771eb", animate=True)),
            (lambda: _map_stat_card(len(future),  "📍 Upcoming States", future,  "#F4CA67", "#F4CA67")),
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
            fig_map = _fig_india_map(df_for_map, map_data, width=screen_width)
        if fig_map is not None:
            st.plotly_chart(
                fig_map,
                width='stretch',
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    # Geo traces only expose zoomInGeo / zoomOutGeo / resetGeo /
                    # hoverClosestGeo on the mode bar — keep just the +/- zoom
                    # and reset, drop the hover-toggle and image-export icons.
                    "modeBarButtonsToRemove": ["hoverClosestGeo", "toImage"],
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
    if "site_id" in df.columns:
        id_col = _id_col(df)
        reviewed_mask, suspicious_mask, _, _ = _choose_status_masks(df, phase="phase1")
        high_mask = (
            suspicious_mask & _is_high_risk(df["risk"])
            if "risk" in df.columns
            else pd.Series(False, index=df.index)
        )

        site_key = df["site_id"].astype(str)
        screened_by_site   = df.groupby(site_key)[id_col].nunique()
        suspicious_by_site = suspicious_mask.groupby(site_key).sum()
        high_by_site        = high_mask.groupby(site_key).sum()

        sites_present = (
            [s for s in SITE_ORDER if s in screened_by_site.index]
            + sorted(set(screened_by_site.index) - set(SITE_ORDER))
        )

        if sites_present:
            rows_html = []
            for site in sites_present:
                screened = int(screened_by_site.get(site, 0))
                susp     = int(suspicious_by_site.get(site, 0))
                high     = int(high_by_site.get(site, 0))
                susp_pct = round(susp / screened * 100, 1) if screened else 0.0
                high_pct = round(high / screened * 100, 1) if screened else 0.0
                rows_html.append(
                    "<tr>"
                    f"<td style='padding:9px 14px;text-align:left;font-weight:600;"
                    f"color:#333;border-top:1px solid #eee;'>{html.escape(site)}</td>"
                    f"<td style='padding:9px 14px;text-align:center;font-weight:700;"
                    f"color:#228B22;border-top:1px solid #eee;'>{screened:,}</td>"
                    f"<td style='padding:9px 14px;text-align:center;font-weight:700;"
                    f"color:#F4A900;border-top:1px solid #eee;'>{susp:,} ({susp_pct}%)</td>"
                    f"<td style='padding:9px 14px;text-align:center;font-weight:700;"
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
                "color:#228B22;'>Screened</th>"
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
                "margin-bottom:8px;'>🏥 Site-wise Summary</div>",
                unsafe_allow_html=True,
            )
            st.markdown(table_html, unsafe_allow_html=True)


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

    p1_cards = st.columns(_cols(2, 2, 1), gap="large")
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
        )
        st.plotly_chart(
            _fig_registrations(mon),
            width="stretch",
            config={"displayModeBar": False},
            key="p1_reg",
        )

    with col_r:
        # % is of total screened (total_cum) — matches the Overall tab's
        # Site-wise Summary table, not the reviewed-only subset.
        susp_rate = round(total_susp / total_cum * 100, 1) if total_cum else 0
        high_pct  = round(high_total / total_cum * 100, 1) if total_cum else 0

        st.markdown(
            f'<div class="ocp-card ocp-card-amb ocp-card-dualstat" style="padding:10px 24px 8px;">'
            f'<div class="card-dualstat-row">'
            f'<span>Suspicious: {total_susp:,} ({susp_rate}%)</span>'
            f'<span class="dualstat-sep">|</span>'
            f'<span class="dualstat-high">High risk: {high_total:,} ({high_pct}%)</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _fig_status(mon, "Suspicious", "Suspicious"),
            width="stretch",
            config={"displayModeBar": False},
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

    p2_cards = st.columns(_cols(2, 2, 1), gap="large")
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
        )
        st.plotly_chart(
            _fig_registrations(mon),
            width="stretch",
            config={"displayModeBar": False},
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
            suffix=f" suspicious ({ai_susp_rate_r}%)",
            sub_text=f"{last_ai_susp:,} suspicious in {last_lbl}",
            duration_text="",
            big_color="#F4A900",
            border_color="#F4A900",
        )
        st.plotly_chart(
            _fig_sankey_phase2(df, screen_width),
            width="stretch",
            config={"displayModeBar": False},
            key="p2_sankey",
        )


# ════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ════════════════════════════════════════════════════════════════════

def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Load combined dataset ────────────────────────────────────────
    df_raw = _load_data()

    if df_raw.empty:
        st.error("No rows were loaded from the Microsoft Graph parquet file.")
        st.stop()

    # ── Split by phase column ────────────────────────────────────────
    has_phase = "phase" in df_raw.columns

    # ── Live sites ───────────────────────────────────────────────────
    live_in_data: set[str] = set()
    if "site_id" in df_raw.columns:
        live_in_data.update(df_raw["site_id"].dropna().astype(str).unique().tolist())
    live_sites = [s for s in SITE_ORDER if s in live_in_data] + sorted(live_in_data - set(SITE_ORDER))

    # ════════════════════════════════════════════════════════════════
    # Page header
    # ════════════════════════════════════════════════════════════════


    if IS_PHONE:
        hdr_l, hdr_r = st.container(), st.container()
    else:
        hdr_l, hdr_r = st.columns([5, 1.5])

    with hdr_l:
        if IS_PHONE:
            c2, c3 = st.container(), None
        else:
            c2, c3 = st.columns([8.5, 8])

        # Logo + title, kept inline together via flexbox so they never
        # drift apart or stack vertically on narrow screens.
        with c2:
            logo = BASE / "static" / "logo.png"
            logo_html = ""
            if logo.exists():
                logo_b64 = _img_to_base64(str(logo))
                logo_px = 48 if IS_PHONE else 70
                logo_html = (
                    f'<img src="{logo_b64}" '
                    f'style="width:{logo_px}px;height:{logo_px}px;object-fit:contain;'
                    f'flex:0 0 auto;display:block;">'
                )

            title_size = "1.3rem" if IS_PHONE else "2rem"
            sub_size   = ".72rem" if IS_PHONE else ".82rem"

            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:10px;flex-wrap:nowrap;">
                    {logo_html}
                    <div style="min-width:0;">
                        <div style="
                            font-size:{title_size};
                            font-weight:800;
                            color:#0771eb;
                            font-family:'Segoe UI',Arial,sans-serif;
                            letter-spacing:-.4px;
                            line-height:1.15;
                            white-space:nowrap;
                            overflow:hidden;
                            text-overflow:ellipsis;">
                            Aarogya Aarohan
                        </div>
                        <div style="
                            font-size:{sub_size};
                            color:#b0b0b0;
                            margin-top:2px;
                            white-space:nowrap;
                            overflow:hidden;
                            text-overflow:ellipsis;">
                            TANUH · IISc Oral Cancer Project · Real-time Dashboard
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Combined logo inline — hidden on phone (not enough room next to
        # the title; it still appears on tablet/desktop as before).
        if c3 is not None:
            with c3:
                logo_comb = BASE / "static" / "logo_comb.png"
                if logo_comb.exists():
                    st.markdown(
                        "<div style='margin-top:-20px;'></div>",
                        unsafe_allow_html=True
                    )
                    comb_width = 220 if IS_TABLET else 300
                    st.image(str(logo_comb), width=comb_width)

    # Right-side buttons — stack full-width below the title on phone,
    # side-by-side column on tablet/desktop.
    with hdr_r:
        if "dashboard_view" not in st.session_state:
            st.session_state.dashboard_view = "main"

        if IS_PHONE:
            btn_cols = st.columns(2, gap="small")
        else:
            btn_cols = [st.container(), st.container()]

        with btn_cols[0]:
            if st.button(
                "📊 Monitoring Dashboard",
                width="stretch",
                type="primary"
                if st.session_state.dashboard_view == "main"
                else "secondary",
                key="btn_dash_main",
            ):
                st.session_state.dashboard_view = "main"
                st.rerun()

        with btn_cols[1]:
            if st.button(
                "🔍 Research Dashboard",
                width="stretch",
                type="primary"
                if st.session_state.dashboard_view == "research"
                else "secondary",
                key="btn_dash_research",
            ):
                st.session_state.dashboard_view = "research"
                st.rerun()

    # ════════════════════════════════════════════════════════════════
    # Sidebar — Global Filters
    # ════════════════════════════════════════════════════════════════

    st.sidebar.markdown("## 🔍 Global Filters")

    if "flt_flip" not in st.session_state:
        st.session_state.flt_flip = False
    if st.sidebar.button("🔄 Reset All Filters"):
        st.session_state.flt_flip = not st.session_state.flt_flip
        st.rerun()
    _f = st.session_state.flt_flip

    quick_window = st.sidebar.selectbox(
        "🗓️ Quick date filter",
        ["All data", "Last 6 months", "Last 3 months"],
        key=f"qw_{_f}",
    )

    all_dates = (
        df_raw["date_of_case_registered"].dropna()
        if "date_of_case_registered" in df_raw.columns
        else pd.Series(dtype="datetime64[ns]")
    )

    quick_mask = _date_window_mask(df_raw, quick_window)
    date_mask  = pd.Series(True, index=df_raw.index)

    if not all_dates.empty:
        min_dt = all_dates.min().date()
        max_dt = all_dates.max().date()
        dr = st.sidebar.date_input(
            "📅 Date range",
            value=(min_dt, max_dt),
            min_value=min_dt,
            max_value=max_dt,
            key=f"dr_{_f}",
        )
        if isinstance(dr, tuple) and len(dr) == 2:
            date_mask = (
                (df_raw["date_of_case_registered"].dt.date >= dr[0]) &
                (df_raw["date_of_case_registered"].dt.date <= dr[1])
            ) if "date_of_case_registered" in df_raw.columns else date_mask

    date_mask = date_mask & quick_mask

    # Gender filter
    gender_col = _gender_col(df_raw)
    gender_values: list[str] = []
    if gender_col:
        gender_values = sorted(
            v for v in df_raw[gender_col].dropna().astype(str).unique().tolist()
            if v.strip().lower() not in ("nan", "none", "", "-", ".")
        )

    gender_sel = "All"
    if gender_values:
        gender_sel = st.sidebar.selectbox("👤 Gender", ["All"] + gender_values, key=f"gx_{_f}")

    # Site filter
    site_sel = "All"
    if live_sites:
        site_sel = st.sidebar.selectbox("🏥 Site", ["All"] + live_sites, key=f"st_{_f}")

    st.sidebar.markdown("---")
    if st.sidebar.button("🔃 Refresh Data", width="stretch", help="Reload data now"):
        _get_data_store().force_refresh()
        st.cache_data.clear()
        st.rerun()

    # ════════════════════════════════════════════════════════════════
    # Apply filters — produce three filtered views
    #   df_all : all phases combined (Overall tab)
    #   df_p1  : phase == '1'        (Phase 1 tab)
    #   df_p2  : phase == '2'        (Phase 2 tab)
    # ════════════════════════════════════════════════════════════════

    df_all = df_raw[date_mask].copy()
    if has_phase:
        df_p1 = df_raw[(df_raw["phase"].astype(str) == "1") & date_mask].copy()
        df_p2 = df_raw[(df_raw["phase"].astype(str) == "2") & date_mask].copy()
    else:
        df_p1 = df_all.copy()
        df_p2 = pd.DataFrame(columns=df_raw.columns)

    # Gender filter
    if gender_sel != "All" and gender_col:
        for _df in (df_all, df_p1, df_p2):
            if gender_col in _df.columns:
                _df.drop(_df.index[_df[gender_col].astype(str) != gender_sel], inplace=True)

    # Snapshot for Overall map — date + gender only, no site filter
    df_all_map = df_all.copy()

    # Site filter
    if site_sel != "All" and "site_id" in df_all.columns:
        for _df in (df_all, df_p1, df_p2):
            if "site_id" in _df.columns:
                _df.drop(_df.index[_df["site_id"].astype(str) != site_sel], inplace=True)

    # ════════════════════════════════════════════════════════════════
    # Research Dashboard — Coming Soon
    # ════════════════════════════════════════════════════════════════

    if st.session_state.get("dashboard_view") == "research":
        st.markdown(
            '<div class="coming-soon">'
            '🔬 Research Dashboard<br>'
            '<span style="font-size:1rem;color:#ddd;font-weight:400;">Coming Soon</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            '<div style="text-align:center;padding:10px 0;font-size:12px;color:#c0c0c0;">'
            '<b style="color:#0771eb;">Aarogya Aarohan</b>&nbsp;·&nbsp;'
            'TANUH Oral Cancer Project&nbsp;·&nbsp;© 2026'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ════════════════════════════════════════════════════════════════
    # Tab Navigation  —  Overall · Phase 1 · Phase 2
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
            "Phase 1",
            key="btn_p1",
            type="primary" if st.session_state.tab == 1 else "secondary",
            width="stretch",
        ):
            st.session_state.tab = 1
            st.rerun()

    with tc3:
        if st.button(
            "Phase 2",
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
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;padding:10px 0;font-size:12px;color:#c0c0c0;">'
        '<b style="color:#0771eb;">Aarogya Aarohan</b>&nbsp;·&nbsp;'
        'TANUH Oral Cancer Project&nbsp;·&nbsp;© 2026'
        '</div>',
        unsafe_allow_html=True,
    )


main()