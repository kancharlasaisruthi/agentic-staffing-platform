"""
Entry point for the Staffing Prospect Intelligence Platform's Streamlit
app. Run with: streamlit run ui/app.py

This file ONLY sets up navigation between pages — all actual page content
lives in ui/pages_impl/. Using st.navigation(..., position="top") puts the
page switcher as a row of clickable pills along the top of the page
(rather than the sidebar), which is what gives you the "button on top of
header" to jump to Approved / Rejected company lists.

Page split:
- discovery.py — the main run-the-pipeline dashboard (sidebar config,
  Run Discovery button, company cards, approve/reject/edit/re-score).
  This is the original single-file app, extracted as-is.
- approved.py / rejected.py — read-only views backed by
  core/persistence.py's local JSON file (data/decisions.json), NOT by
  st.session_state. This is deliberate: these pages show what you've
  actually decided over time, including decisions made in past sessions
  before a refresh/restart, not just what's in memory right now.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="Staffing Prospect Intelligence", page_icon="🎯", layout="wide")

# ── Global dark-theme stylesheet ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base & background ─────────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main { background-color: #0d0f14 !important; }

    [data-testid="stSidebar"] { background-color: #111318 !important; }
    [data-testid="stSidebar"] * { color: #c9cdd7 !important; }

    /* ── Typography ────────────────────────────────────────────────────── */
    * { color: #e2e6f0; font-family: 'Inter', 'Segoe UI', sans-serif; }

    h1 { font-size: 1.9rem !important; font-weight: 700 !important;
         color: #ffffff !important; letter-spacing: -0.5px; }
    h2 { font-size: 1.35rem !important; font-weight: 600 !important;
         color: #d0d6e8 !important; }
    h3, h4 { color: #c4cad9 !important; }

    /* ── Top nav pills ─────────────────────────────────────────────────── */
    [data-testid="stTopNavigation"] {
        background: #111318 !important;
        border-bottom: 1px solid #1f2330 !important;
        padding: 0 1.5rem !important;
    }
    [data-testid="stTopNavigation"] button {
        border-radius: 6px !important;
        font-weight: 500 !important;
        color: #9ca3b8 !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stTopNavigation"] button:hover { color: #ffffff !important; }
    [data-testid="stTopNavigation"] button[aria-selected="true"] {
        background: #1e3a5f !important;
        color: #60a5fa !important;
        font-weight: 600 !important;
    }

    /* ── Buttons ───────────────────────────────────────────────────────── */
    .stButton > button {
        background: #1a1e2a !important;
        border: 1px solid #2c3347 !important;
        color: #c9cdd7 !important;
        border-radius: 7px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background: #232840 !important;
        border-color: #4a5578 !important;
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 0 12px rgba(37,99,235,0.35) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
        box-shadow: 0 0 18px rgba(59,130,246,0.45) !important;
    }

    /* ── Containers / cards ────────────────────────────────────────────── */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background: #13161f !important;
        border: 1px solid #1f2636 !important;
        border-radius: 10px !important;
        padding: 1rem 1.2rem !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.35) !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div:hover {
        border-color: #2e3a56 !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
        transition: all 0.2s ease !important;
    }

    /* ── Metrics ───────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: #13161f !important;
        border: 1px solid #1f2636 !important;
        border-radius: 8px !important;
        padding: 0.8rem 1rem !important;
    }
    [data-testid="stMetricLabel"] { color: #7b82a0 !important; font-size: 0.78rem !important; }
    [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.6rem !important; font-weight: 700 !important; }

    /* ── Alerts ────────────────────────────────────────────────────────── */
    [data-testid="stInfo"]    { background: #0f1f3d !important; border-left: 3px solid #3b82f6 !important; border-radius: 7px !important; }
    [data-testid="stSuccess"] { background: #0d2818 !important; border-left: 3px solid #22c55e !important; border-radius: 7px !important; }
    [data-testid="stWarning"] { background: #271f0d !important; border-left: 3px solid #f59e0b !important; border-radius: 7px !important; }
    [data-testid="stError"]   { background: #2a0e0e !important; border-left: 3px solid #ef4444 !important; border-radius: 7px !important; }

    /* ── Inputs ────────────────────────────────────────────────────────── */
    input, textarea, select,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #1a1e2a !important;
        border: 1px solid #2c3347 !important;
        border-radius: 6px !important;
        color: #e2e6f0 !important;
    }
    input:focus, textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.18) !important;
    }

    /* ── Expanders ─────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: #111318 !important;
        border: 1px solid #1f2636 !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary { color: #9ca3b8 !important; font-weight: 500 !important; }

    /* ── Dataframe ─────────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] { border-radius: 8px !important; overflow: hidden !important; }

    /* ── Dividers ──────────────────────────────────────────────────────── */
    hr { border-color: #1f2636 !important; opacity: 0.6 !important; }

    /* ── Caption / small text ──────────────────────────────────────────── */
    .stCaption, [data-testid="stCaptionContainer"] { color: #5e6680 !important; }

    /* ── Sidebar dividers ──────────────────────────────────────────────── */
    [data-testid="stSidebar"] hr { border-color: #1f2636 !important; }

    /* ── Scrollbar ─────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0d0f14; }
    ::-webkit-scrollbar-thumb { background: #2c3347; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3d4a6a; }
    </style>
    """,
    unsafe_allow_html=True,
)

discovery_page = st.Page(
    "pages_impl/discovery.py",
    title="Discovery",
    icon="🎯",
    default=True,
)
approved_page = st.Page(
    "pages_impl/approved.py",
    title="Approved Companies",
    icon="✅",
)
rejected_page = st.Page(
    "pages_impl/rejected.py",
    title="Rejected Companies",
    icon="❌",
)

nav = st.navigation([discovery_page, approved_page, rejected_page], position="top")
nav.run()