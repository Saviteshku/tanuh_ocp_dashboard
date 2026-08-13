"""
====================================================================
AAROGYA AAROHAN — REAL-TIME MONITORING DASHBOARD  v3
Oral Cancer Project / TANUH

Combined dataset : OCP_COMB_DATA_OVERALL.parquet
  phase == '1'  → Image-Based Screening records
  phase == '2'  → AI-Enabled Screening records
  Overall tab   → full combined dataset

Owns page config, global CSS/HTML, data loading, sidebar filters, and
top-level view routing. Delegates all actual dashboard content to:
  - monitoring_dashboard.py  (Overall / Image-Based Screening /
                              AI-Enabled Screening tabs)
  - research_dashboard.py    (Research Dashboard view)

This app only LOADS the combined parquet dataset already present on
disk (LOCAL_DATA_DIR / OCP_COMB_DATA_OVERALL.parquet) and renders the
dashboard from it. It no longer fetches or refreshes that data itself
— the fetch step now lives in its own separate script/pipeline (e.g. a
standalone cron job) run independently of the dashboard.

Run:
    streamlit run app.py
====================================================================
"""

from __future__ import annotations
import base64
import hashlib
import logging
import os
import threading
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_js_eval import streamlit_js_eval

import importlib
import sys

_log = logging.getLogger(__name__)


def _render_coming_soon(view_name: str) -> None:
    """Placeholder shown in place of a dashboard view whose module file
    (monitoring_dashboard.py / research_dashboard.py) isn't present
    alongside app.py."""
    st.markdown(
        f"""
        <div class="coming-soon">
            🚧 {view_name} — coming soon
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
# 1.  PAGE CONFIG — must be first Streamlit call
# ════════════════════════════════════════════════════════════════════

try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path.cwd()

# ── Dynamic sibling-module loader ────────────────────────────────────
# Python caches imports in sys.modules, so a plain top-level `import
# monitoring_dashboard` would only ever read the file once per server
# process — deleting monitoring_dashboard.py/research_dashboard.py
# after that would keep "working" (serving the stale cached module)
# until the whole Streamlit server was restarted. To make the
# "coming soon" placeholder react on every rerun (browser refresh /
# any widget interaction), the file's presence on disk is re-checked
# every time this is called, and the module is (re)imported or
# dropped from sys.modules accordingly.
def _load_dashboard_module(module_name: str):
    """Return the module if `<module_name>.py` currently exists next to
    app.py, else None. Re-checks the filesystem on every call."""
    module_path = BASE / f"{module_name}.py"
    if not module_path.exists():
        sys.modules.pop(module_name, None)
        return None
    try:
        if module_name in sys.modules:
            return importlib.reload(sys.modules[module_name])
        return importlib.import_module(module_name)
    except Exception:
        _log.exception("Failed to load %s", module_name)
        return None


logo_path = BASE / "static" / "logo.png"

# ── Force light theme regardless of OS/browser dark-mode setting ────────
# Streamlit's theme is controlled by .streamlit/config.toml, read once at
# server startup — it can't be changed via st.set_page_config or at runtime.
# Write it here (once) so the *next* server start picks up "light" instead
# of following the system preference. Requires an app restart to take
# effect; the CSS override below forces light colors immediately in the
# meantime, independent of that restart.
_config_dir = BASE / ".streamlit"
_config_path = _config_dir / "config.toml"
_theme_block = (
    "[theme]\n"
    'base = "light"\n'
    'primaryColor = "#0771eb"\n'
    'backgroundColor = "#ffffff"\n'
    'secondaryBackgroundColor = "#f5f5f5"\n'
    'textColor = "#000000"\n'
)
try:
    if not _config_path.exists():
        _config_dir.mkdir(parents=True, exist_ok=True)
        _config_path.write_text(_theme_block)
except OSError:
    pass  # e.g. read-only filesystem — config.toml must be added manually

st.set_page_config(
    page_title="Aarogya Aarohan | TANUH OCP",
    page_icon=str(logo_path.resolve()) if logo_path.exists() else "🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Immediate CSS override — forces light colors in the current session even
# before a restart applies config.toml, and even if the OS/browser is set
# to dark mode.
st.markdown(
    """
    <style>
    :root, .stApp {
        color-scheme: light !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
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
BP_TABLET = 1024  # 640–1024 → tablet, 1024–1440 → laptop, > 1440 → desktop
BP_LAPTOP = 1440  # 13"/14" laptops typically report ~1280–1440 CSS px here

IS_PHONE   = screen_width < BP_PHONE
IS_TABLET  = BP_PHONE <= screen_width < BP_TABLET
IS_LAPTOP  = BP_TABLET <= screen_width < BP_LAPTOP
IS_DESKTOP = screen_width >= BP_LAPTOP


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

SITE_ORDER = [
    "Krishnagiri, Tamil Nadu*",
    "Thanjavur, Tamil Nadu",
    "South & North District, Goa",
    "All India Institute of Medical Sciences, Delhi",
    "Anekel, Bangalore, Karnataka",
    "Cachar Cancer Hospital & Research Centre, Silchar, Assam",
    "Dr. Bhubaneswar Borooah Cancer Institute, Guwahati, Assam",
    "Dr. K&T Keditsu Foundation, Kohima, Nagaland",
    "Goa Dental College & Hospital, Goa",
    "KLE Dental College, Bangalore, Karnataka",
    "Mahamana Pandit Madan Mohan Malaviya Cancer Centre, Delhi",
    "Mazumdar Shaw Medical Foundation, Kolkata, West Bengal",
]


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

/* Dual-stat card (e.g. "Suspicious | High risk") — height is set inline
   per-render to match the Total Screened card's box height (see
   _tab_overall / _tab_phase1); min-height is just a floor if that inline
   value ever comes in low, and overflow stays visible so wrapped text
   isn't clipped instead of the box growing. */
.ocp-card-dualstat {
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

/* Larger section tabs (Overall / Image-Based Screening / AI-Enabled Screening). */
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
    padding-top: 0 !important;
    margin-top: -25px !important;
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
    /* Section tabs (Overall / Image-Based Screening / AI-Enabled Screening) shrink to fit 3-across */
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


# Bumps Plotly font sizes on any chart the user expands with Streamlit's
# native chart fullscreen button, and puts them back on exit.
#
# HOW THIS WORKS: st.iframe renders inside a same-origin
# iframe, so `window.parent.document` reaches the real app DOM. We
# listen for the browser Fullscreen API's `fullscreenchange` event
# (which is what Streamlit's chart-expand button triggers) and, when
# the newly-fullscreened element contains a Plotly chart
# (`.js-plotly-plot`, Plotly's own class — stable across Streamlit
# versions), call `Plotly.relayout` to scale up its font/tick/legend
# sizes. Original sizes are cached on the DOM node so they can be
# restored on exit.
#
# CAVEAT: this depends on the Fullscreen API being what Streamlit's
# chart-expand control uses, which has been true across recent
# versions but isn't a documented/guaranteed contract — if a future
# Streamlit release changes how chart fullscreen works, this may need
# updating (e.g. re-check via browser devtools whether expanding a
# chart still sets `document.fullscreenElement`).
_FULLSCREEN_FONT_JS = """
<script>
(function() {
  const SCALE = 1.55;

  function bump(n) { return n ? Math.round(n * SCALE) : n; }
  function unbump(n) { return n ? Math.round(n / SCALE) : n; }

  function isSankey(gd) {
    return !!(gd && gd.data && gd.data.some(function(t) { return t.type === "sankey"; }));
  }

  function applyBigFonts(gd) {
    if (!gd || !gd.layout || gd.dataset.fsFontBumped === "1") return;
    // Sankey labels are drawn per-node next to fixed-position nodes, not
    // on an axis/legend — bumping font.size here (without also growing
    // margins and node spacing) makes label text overflow the node's
    // fixed margin and get clipped in fullscreen. The Sankey's own
    // sizing already adapts to its container, so skip the bump for it.
    if (isSankey(gd)) return;
    const L = gd.layout;
    const update = {};
    if (L.font && L.font.size)                         update["font.size"] = bump(L.font.size);
    if (L.legend && L.legend.font && L.legend.font.size) update["legend.font.size"] = bump(L.legend.font.size);
    if (L.xaxis && L.xaxis.tickfont && L.xaxis.tickfont.size) update["xaxis.tickfont.size"] = bump(L.xaxis.tickfont.size);
    if (L.yaxis && L.yaxis.tickfont && L.yaxis.tickfont.size) update["yaxis.tickfont.size"] = bump(L.yaxis.tickfont.size);
    if (Object.keys(update).length === 0) return;
    gd.dataset.fsFontBumped = "1";
    try { window.parent.Plotly.relayout(gd, update); } catch (e) {}
  }

  function revertFonts(gd) {
    if (!gd || !gd.layout || gd.dataset.fsFontBumped !== "1") return;
    if (isSankey(gd)) return;
    const L = gd.layout;
    const update = {};
    if (L.font && L.font.size)                         update["font.size"] = unbump(L.font.size);
    if (L.legend && L.legend.font && L.legend.font.size) update["legend.font.size"] = unbump(L.legend.font.size);
    if (L.xaxis && L.xaxis.tickfont && L.xaxis.tickfont.size) update["xaxis.tickfont.size"] = unbump(L.xaxis.tickfont.size);
    if (L.yaxis && L.yaxis.tickfont && L.yaxis.tickfont.size) update["yaxis.tickfont.size"] = unbump(L.yaxis.tickfont.size);
    gd.dataset.fsFontBumped = "0";
    if (Object.keys(update).length === 0) return;
    try { window.parent.Plotly.relayout(gd, update); } catch (e) {}
  }

  function handleChange() {
    let doc;
    try { doc = window.parent.document; } catch (e) { return; }
    const fsEl = doc.fullscreenElement;
    doc.querySelectorAll(".js-plotly-plot").forEach(function(gd) {
      if (!fsEl || !fsEl.contains(gd)) revertFonts(gd);
    });
    if (fsEl) {
      const gd = fsEl.querySelector(".js-plotly-plot");
      if (gd) applyBigFonts(gd);
    }
  }

  try {
    window.parent.document.addEventListener("fullscreenchange", handleChange);
  } catch (e) { /* cross-origin: silently do nothing */ }
})();
</script>
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
            "gender", "site_full_id", "provisional_diagnosis", "phase",
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
            # Unique per writer (pid + thread id) so that if two writers
            # ever do end up racing, they never fight over the same .tmp
            # file — each does its own write + atomic replace independently.
            tmp_path = self._cache_path.with_suffix(
                f"{self._cache_path.suffix}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
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
            # Do the entire cold-start load (cache read, or synchronous
            # source read + cache write) while holding the lock, so that
            # if several Streamlit sessions call get() concurrently before
            # any data is loaded, only the first one actually touches
            # disk — the rest just wait and then see self._df already
            # populated. Without this, concurrent cold-start callers could
            # each try to write the on-disk cache at the same time.
            if self._df is None:
                cached_df = self._read_from_cache()
                if cached_df is not None:
                    self._df = cached_df
                    self._loaded_at = time.time()
                    self._loaded_from_cache = True
                else:
                    df = self._read_from_disk(self._path)
                    self._write_to_cache(df)
                    self._df = df
                    self._loaded_at = time.time()
                    self._loaded_from_cache = False
                loaded_from_cache = self._loaded_from_cache
            else:
                loaded_from_cache = None  # not a cold start; no action needed below

        if loaded_from_cache is True:
            # Kick off a real refresh from the source right away in the
            # background, instead of waiting a full refresh cycle.
            self._ensure_thread(initial_delay=0.0)
        elif loaded_from_cache is False:
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


# ════════════════════════════════════════════════════════════════════
# 5.  SIDEBAR-FILTER HELPERS
# ════════════════════════════════════════════════════════════════════
# Column-name detection + date-window masking used only by the sidebar
# filters below in main() (Monitoring Dashboard's tab-internal analytics
# helpers live in monitoring_dashboard.py alongside the code that uses
# them).

def _gender_col(df: pd.DataFrame) -> str | None:
    for c in ("gender", "Gender", "Sex"):
        if c in df.columns:
            return c
    return None


def _study_setting_col(df: pd.DataFrame) -> str | None:
    for c in ("study_setting", "Study_Setting", "Study Setting"):
        if c in df.columns:
            return c
    return None


# ════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ════════════════════════════════════════════════════════════════════

def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    st.iframe(_FULLSCREEN_FONT_JS, height=1)

    # ── Load sibling dashboard modules (re-checked every rerun so a
    #    deleted/restored .py file takes effect immediately, no server
    #    restart needed) ────────────────────────────────────────────
    monitoring_dashboard = _load_dashboard_module("monitoring_dashboard")
    research_dashboard = _load_dashboard_module("research_dashboard")

    # ── Load combined dataset ────────────────────────────────────────
    df_raw = _load_data()

    if df_raw.empty:
        st.error("No rows were loaded from the Microsoft Graph parquet file.")
        st.stop()

    # ── Split by phase column ────────────────────────────────────────
    has_phase = "phase" in df_raw.columns

    # ── Live sites ───────────────────────────────────────────────────
    live_in_data: set[str] = set()
    if "site_full_id" in df_raw.columns:
        live_in_data.update(df_raw["site_full_id"].dropna().astype(str).unique().tolist())
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
            sub_size   = ".82rem" if IS_PHONE else ".92rem"

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
                            color:#737373;
                            margin-top:2px;
                            white-space:nowrap;
                            overflow:hidden;
                            text-overflow:ellipsis;">
                            TANUH · Oral Cancer Screening Project · Dashboard
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Combined logo — inline next to the title on tablet/desktop;
        # own row below the title on phone (not enough horizontal room
        # next to the title at that width).
        logo_comb = BASE / "static" / "logo_comb.png"
        if logo_comb.exists():
            if c3 is not None:
                with c3:
                    st.markdown(
                        "<div style='margin-top:-20px;'></div>",
                        unsafe_allow_html=True
                    )
                    comb_width = 220 if IS_TABLET else 300
                    st.image(str(logo_comb), width=comb_width)
            else:
                st.image(str(logo_comb), width=250)

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

    all_dates = (
        df_raw["date_of_case_registered"].dropna()
        if "date_of_case_registered" in df_raw.columns
        else pd.Series(dtype="datetime64[ns]")
    )

    date_mask = pd.Series(True, index=df_raw.index)
    # The sidebar's actual selected Date range boundaries, passed down
    # to the Research Dashboard (see research_dashboard.render) so its
    # cases/week charts use exactly this calendar span as the
    # denominator — e.g. "current month" = 1st of the month through
    # today — rather than deriving weeks from whichever case dates
    # happen to be present in the filtered data.
    filter_start = filter_end = None

    if not all_dates.empty:
        min_dt = all_dates.min().date()
        data_max_dt = all_dates.max().date()
        # Let users pick an end date up to today, even if no cases have
        # been registered yet for the most recent day(s) — avoids the
        # "outside allowed range" error when someone tries to pick, e.g.,
        # today's date.
        max_dt = max(data_max_dt, date.today())
        # Default the selected end date to one week back from the latest
        # available data, since the most recent week is often incomplete.
        default_end_dt = max(data_max_dt - timedelta(days=7), min_dt)
        dr = st.sidebar.date_input(
            "📅 Date range",
            value=(min_dt, default_end_dt),
            min_value=min_dt,
            max_value=max_dt,
            format="DD/MM/YYYY",
            key=f"dr_{_f}",
        )
        if isinstance(dr, tuple) and len(dr) == 2:
            filter_start, filter_end = dr[0], dr[1]
            date_mask = (
                (df_raw["date_of_case_registered"].dt.date >= dr[0]) &
                (df_raw["date_of_case_registered"].dt.date <= dr[1])
            ) if "date_of_case_registered" in df_raw.columns else date_mask

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

    # Study setting filter
    setting_col = _study_setting_col(df_raw)
    setting_values: list[str] = []
    if setting_col:
        setting_values = sorted(
            v for v in df_raw[setting_col].dropna().astype(str).unique().tolist()
            if v.strip().lower() not in ("nan", "none", "", "-", ".")
        )

    setting_sel = "All"
    if setting_values:
        setting_sel = st.sidebar.selectbox(
            "🏨 Study Setting", ["All"] + setting_values, key=f"ss_{_f}"
        )

    # Site filter
    site_sel = "All"
    if live_sites:
        site_sel = st.sidebar.selectbox("🏥 Site", ["All"] + live_sites, key=f"st_{_f}")

    st.sidebar.markdown("---")
    if st.sidebar.button(
        "🔃 Reload Data",
        width="stretch",
        help="Reload the dashboard from the parquet file already on disk",
    ):
        _get_data_store().force_refresh()
        st.cache_data.clear()
        st.rerun()

    # ════════════════════════════════════════════════════════════════
    # Apply filters — produce three filtered views
    #   df_all : all phases combined (Overall tab)
    #   df_p1  : phase == '1'        (Image-Based Screening tab)
    #   df_p2  : phase == '2'        (AI-Enabled Screening tab)
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

    # Study setting filter
    if setting_sel != "All" and setting_col:
        for _df in (df_all, df_p1, df_p2):
            if setting_col in _df.columns:
                _df.drop(_df.index[_df[setting_col].astype(str) != setting_sel], inplace=True)

    # Snapshot for Overall map — date + gender only, no site filter
    df_all_map = df_all.copy()

    # Site filter
    if site_sel != "All" and "site_full_id" in df_all.columns:
        for _df in (df_all, df_p1, df_p2):
            if "site_full_id" in _df.columns:
                _df.drop(_df.index[_df["site_full_id"].astype(str) != site_sel], inplace=True)

    # ════════════════════════════════════════════════════════════════
    # Research Dashboard — delegated entirely to research_dashboard.py
    # ════════════════════════════════════════════════════════════════

    if st.session_state.get("dashboard_view") == "research":
        if research_dashboard is None:
            _render_coming_soon("Research Dashboard")
        else:
            research_dashboard.render(df_all, df_p1, df_p2, filter_start, filter_end)
        st.markdown("---")
        st.markdown(
            '<div style="text-align:center;padding:10px 0;font-size:12px;color:#c0c0c0;">'
            '<b style="color:#0771eb;">Aarogya Aarohan</b>&nbsp;·&nbsp;'
            'TANUH Oral Cancer Screening Project&nbsp;·&nbsp;© 2026'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ════════════════════════════════════════════════════════════════
    # Monitoring Dashboard — delegated entirely to monitoring_dashboard.py
    # ════════════════════════════════════════════════════════════════

    if monitoring_dashboard is None:
        _render_coming_soon("Monitoring Dashboard")
    else:
        monitoring_dashboard.render(screen_width, df_all, df_p1, df_p2, df_all_map)


main()