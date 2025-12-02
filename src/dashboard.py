import os
import sys
import hashlib
import re
import shutil
import time
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import streamlit as st
import polars as pl
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
quality_options_common = ["빠름", "균형", "선명"]

# 결과 폴더 이름은 반드시 '숫자_결과' 패턴을 따라야 합니다.
RESULT_DIR_NAME_PATTERN = re.compile(r"^\d+_결과(?:_\d+)?$")

# Streamlit 페이지 설정: 레이아웃을 와이드로 고정합니다.
# - 이미 페이지 설정이 되어 있거나 이 호출 시점이 맞지 않으면 예외가 발생할 수 있으므로
#   try/except로 안전하게 감쌉니다.
try:
    st.set_page_config(layout="wide")
except Exception:
    # 설정 불가 시 단순히 무시합니다(이미 설정되었거나 호출 시점이 맞지 않을 수 있음).
    try:
        l = globals().get('logger')
        if l and hasattr(l, 'debug'):
            l.debug("st.set_page_config(layout='wide') 호출 실패 또는 이미 설정됨")
    except Exception:
        pass

# (자동 리스캔 호출은 파일 상단이 아닌, UI 렌더링 직전에 수행하도록
# 파일 하단의 적절한 위치에 배치되어 있습니다.)
# -------------------------------------------------------------------------
# 안전 폴백: 모듈의 다른 부분(또는 외부에서)에서 정의되는 전역 심볼들이
# 파일 상단에서 아직 존재하지 않을 때 발생하는 NameError를 방지하기 위한
# 최소한의 기본값을 이 위치에서 설정합니다. 실제 값이 이후에 정의되면
# 덮어쓰지 않으므로 안전합니다.
# 주석은 한국어로 작성했습니다.
# -------------------------------------------------------------------------
try:
    import cv2 as _cv2
except Exception:
    _cv2 = None

if "cv2" not in globals():
    # OpenCV가 없다면 None을 설정합니다. 대체 로직은 _safe_image_open에서 처리됩니다.
    cv2 = _cv2

if "logger" not in globals():
    # 간단한 로거를 설정합니다(실제 로거가 이후 정의되면 덮어씌워집니다).
    import logging
    logger = logging.getLogger("dashboard")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

if "BASE_OUTPUT_DIR" not in globals():
    # 기본 분석 경로는 명시적으로 설정되지 않았다면 빈 문자열으로 둡니다.
    # 기존에는 작업 디렉터리(Path.cwd())가 기본이었는데, 이는 리포지토리
    # 루트가 자동으로 선택되는 원인이 되어 사용자 기대와 달랐습니다.
    BASE_OUTPUT_DIR = ""

if "CLI_BASE_DIR" not in globals():
    CLI_BASE_DIR = BASE_OUTPUT_DIR

if "CLI_OUTPUT_DIR" not in globals():
    CLI_OUTPUT_DIR = BASE_OUTPUT_DIR

if "RESULT_DIRS" not in globals():
    RESULT_DIRS = []

if "SELECTION_ROOT" not in globals():
    SELECTION_ROOT = None

# 사용자가 이전 세션에서 기본값으로 작업 디렉터리를 저장해 둔 경우,
# 앱 시작 시 자동으로 그 값을 기본으로 사용하지 않도록 정리합니다.
try:
    if hasattr(st, 'session_state'):
        cwd_str = str(Path.cwd())
        try:
            if st.session_state.get("result_base_dir") == cwd_str:
                st.session_state["result_base_dir"] = ""
        except Exception:
            pass
        try:
            if st.session_state.get("result_base_input") == cwd_str:
                st.session_state["result_base_input"] = ""
        except Exception:
            pass
        try:
            if st.session_state.get("_cli_base_marker") == cwd_str:
                st.session_state.pop("_cli_base_marker", None)
        except Exception:
            pass
except Exception:
    pass

# ===== 테마 및 CSS 주입 =====
DEFAULT_THEME_KEY = 'Light (기본)'
THEMES = {
    'Light (기본)': {
        'palette': {
            'bg': '#FBFDFF',
            'sidebar_bg': '#FFFFFF',
            'text': '#091223',
            'sidebar_text': '#091223',
            'secondary': '#475569',
            'accent': '#1EA3A1',
            'card_bg': '#FBFDFF',
            'card_border': '#e6eef8',
            'shadow': '0 6px 18px rgba(10,20,40,0.04)'
        },
    },
    'Warm Sepia': {
        'palette': {
            'bg': '#f4efe6',
            'sidebar_bg': '#efe6d9',
            'text': '#2d2a26',
            'sidebar_text': '#2d2a26',
            'secondary': '#6e5a4a',
            'accent': '#1EA3A1',
            'card_bg': '#fbf6ee',
            'card_border': '#e6dccf',
            'shadow': '0 6px 18px rgba(30,20,10,0.08)'
        },
    },
    'Gentle Mint': {
        'palette': {
            'bg': '#f3faf6',
            'sidebar_bg': '#eaf7ef',
            'text': '#082724',
            'sidebar_text': '#082724',
            'secondary': '#4b6b64',
            'accent': '#1EA3A1',
            'card_bg': '#ffffff',
            'card_border': '#e6f0ec',
            'shadow': '0 6px 18px rgba(5,30,25,0.06)'
        },
    },
}

if 'theme' not in st.session_state:
    st.session_state['theme'] = DEFAULT_THEME_KEY


def _get_theme_palette(mode: Optional[str] = None) -> Dict[str, str]:
    """선택된 테마 팔레트를 반환합니다. mode가 없으면 세션 상태를 기본으로 사용합니다."""
    target = mode or st.session_state.get('theme') or DEFAULT_THEME_KEY
    theme = THEMES.get(target, THEMES[DEFAULT_THEME_KEY])
    return theme['palette']


def _inject_theme_css(mode: Optional[str] = None):
    """테마 색상을 기반으로 전역 CSS를 주입합니다."""
    pal = _get_theme_palette(mode)
    sidebar_width = int(st.session_state.get('sidebar_width_px', 350))

    bg = pal.get('bg', '#F7F9FB')
    sidebar_bg = pal.get('sidebar_bg', '#FFFFFF')
    text = pal.get('text', '#0B1726')
    sidebar_text = pal.get('sidebar_text', text)
    secondary_text = pal.get('secondary', '#41515F')
    accent = pal.get('accent', '#1EA3A1')
    card_bg = pal.get('card_bg', '#FFFFFF')
    card_border = pal.get('card_border', '#e6e9ee')
    shadow = pal.get('shadow', 'none')

    css = f"""
    <style>
    .stApp {{ background-color: {bg} !important; color: {text} !important; }}
    [data-testid="stSidebar"] {{ background: linear-gradient(180deg, rgba(30,163,161,0.04), {sidebar_bg}) !important; box-shadow: none !important; color: {sidebar_text} !important; border-left: 6px solid rgba(30,163,161,0.08) !important; }}
    [data-testid="stSidebar"][aria-expanded="true"] {{ width: {sidebar_width}px !important; min-width: {sidebar_width}px !important; }}
    [data-testid="stSidebar"][aria-expanded="false"] {{ width: 0 !important; min-width: 0 !important; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stHeader, [data-testid="stSidebar"] .stMarkdown {{ color: {sidebar_text} !important; opacity: 0.98 !important; }}
    .stBlock, .stCard {{ background-color: {card_bg} !important; border: 1px solid {card_border}; border-radius: 10px; box-shadow: {shadow}; padding: 12px; }}
    .stMetric, .stMetric * {{ color: {text} !important; opacity: 0.98 !important; }}
    .stMetric p, .stMetric span, .stMetric small {{ color: {secondary_text} !important; opacity: 0.95 !important; }}
    input, textarea, select, button {{ color: {text} !important; background-color: transparent !important; border-radius: 8px; }}
    .stApp p, .stApp span, label {{ color: {secondary_text} !important; }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{ color: {sidebar_text} !important; }}
    a, .stButton>button {{ color: {accent} !important; }}
    .stDataFrame table {{ border-collapse: separate; border-spacing: 0 8px; }}
    img {{ border-radius: 8px; box-shadow: 0 8px 24px rgba(2,8,12,0.15); }}
    input[type="text"], .stTextInput>div>div>input {{
        background-color: rgba(255,255,255,0.9) !important;
        border: 1.5px solid {accent} !important;
        box-shadow: 0 4px 10px rgba(30,163,161,0.08) !important;
        padding: 10px 12px !important;
        border-radius: 10px !important;
        font-size: 14px !important;
        color: {text} !important;
    }}
    [data-testid="stSidebar"] input[type="text"] {{
        background-color: rgba(30,163,161,0.03) !important;
        border: 1px solid rgba(30,163,161,0.12) !important;
        color: {text} !important;
        box-shadow: 0 2px 6px rgba(30,163,161,0.04) !important;
    }}
    input::placeholder, textarea::placeholder {{ color: rgba(0,0,0,0.38) !important; font-weight: 500 !important; }}
    hr, .stMarkdown hr {{
        border: none !important;
        height: 4px !important;
        background: linear-gradient(90deg, rgba(30,163,161,0.08), {accent}, rgba(30,163,161,0.08)) !important;
        border-radius: 6px !important;
        margin: 18px 0 !important;
        box-shadow: 0 4px 12px rgba(30,163,161,0.06) inset;
    }}
    [data-testid="stSidebar"] .stSelectbox>div>div {{
        background-color: rgba(255,255,255,0.98) !important;
        border: 1px solid rgba(30,163,161,0.3) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08) !important;
        transition: border-color 0.2s ease !important;
    }}
    [data-testid="stSidebar"] .stSelectbox>div>div:hover {{
        border-color: {accent} !important;
        box-shadow: 0 2px 8px rgba(30,163,161,0.12) !important;
    }}
    [data-testid="stSidebar"] .stSelectbox>div>div>div {{
        color: {text} !important;
        font-weight: 500 !important;
        padding: 10px 12px !important;
        font-size: 14px !important;
    }}
    [data-testid="stSidebar"] .stSelectbox svg {{ color: {accent} !important; opacity: 0.7 !important; }}
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] {{
        background-color: white !important;
        border: 1px solid rgba(30,163,161,0.2) !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        margin-top: 2px !important;
    }}
    [data-testid="stSidebar"] .stSelectbox [role="option"] {{
        color: {text} !important;
        padding: 8px 12px !important;
        margin: 2px 4px !important;
        border-radius: 4px !important;
        font-size: 14px !important;
        transition: background-color 0.15s ease !important;
    }}
    [data-testid="stSidebar"] .stSelectbox [role="option"]:hover {{
        background-color: rgba(30,163,161,0.05) !important;
        color: {accent} !important;
    }}
    [data-testid="stSidebar"] .stSelectbox [aria-selected="true"] {{
        background-color: {accent} !important;
        color: white !important;
        font-weight: 500 !important;
    }}
    [data-testid="stSidebar"] .stSelectbox [role="option"][aria-label^="[재스캔]"] {{
        color: #d62839 !important;
        font-weight: 600 !important;
        background-color: rgba(214,40,57,0.08) !important;
    }}
    [data-testid="stSidebar"] .stSelectbox [role="option"][aria-selected="true"][aria-label^="[재스캔]"] {{
        background-color: rgba(214,40,57,0.14) !important;
        color: #d62839 !important;
    }}
    [data-testid="stSidebar"] .stSelectbox [role="option"][aria-label^="[재스캔]"]::before {{
        content: "⚠ ";
        font-weight: 700;
    }}
    [data-testid="stSidebar"] .stSelectbox>div>div>div[aria-label^="[재스캔]"] {{
        color: #d62839 !important;
        font-weight: 600 !important;
    }}
    [data-testid="stSidebar"] .stSelectbox>div>div>div[aria-label^="[재스캔]"]::before {{
        content: "⚠ ";
        margin-right: 4px;
    }}
    [data-testid="stSidebar"] .stSelectbox [role="option"][aria-selected="true"] {{
        background-color: rgba(0,0,0,0.12) !important;
        color: {sidebar_text} !important;
        font-weight: 700 !important;
        border-radius: 0 0 6px 6px !important;
        position: relative !important;
    }}
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] {{
        position: relative !important;
        overflow: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }}
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] [role="option"][aria-selected="true"],
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] [role="option"][data-selected="true"],
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] [role="option"][aria-current="true"] {{
        background-color: rgba(0,0,0,0.12) !important;
        color: {sidebar_text} !important;
        font-weight: 700 !important;
    }}
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] [role="option"][aria-selected="true"],
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] [role="option"][data-selected="true"] {{
        position: sticky !important;
        top: 0 !important;
        z-index: 10 !important;
        margin-top: 0 !important;
    }}
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] [role="option"][aria-selected="true"]:hover {{
        background-color: rgba(0,0,0,0.12) !important;
        color: {sidebar_text} !important;
    }}
    .baseweb-portal [role="listbox"] [role="option"][aria-selected="true"],
    .baseweb-portal [role="listbox"] [role="option"][data-selected="true"],
    body > [role="listbox"] [role="option"][aria-selected="true"] {{
        background-color: rgba(0,0,0,0.12) !important;
        color: {sidebar_text} !important;
        font-weight: 700 !important;
        position: sticky !important;
        top: 0 !important;
        z-index: 9999 !important;
    }}
    .baseweb-portal [role="listbox"] [role="option"][aria-selected="true"]:hover {{
        background-color: rgba(0,0,0,0.12) !important;
        color: {sidebar_text} !important;
    }}
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] [role="option"][aria-selected="true"] .st-emotion-cache-qiev7j,
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] [role="option"][aria-selected="true"] .etx0m6x1 {{
        background-color: rgba(0,0,0,0.12) !important;
        display: block !important;
        padding: 8px 12px !important;
        margin: -8px -12px !important;
        color: {sidebar_text} !important;
        font-weight: 700 !important;
        border-radius: 4px !important;
    }}
    .baseweb-portal [role="listbox"] [role="option"][aria-selected="true"] .st-emotion-cache-qiev7j,
    .baseweb-portal [role="listbox"] [role="option"][aria-selected="true"] .etx0m6x1,
    body > [role="listbox"] [role="option"][aria-selected="true"] .st-emotion-cache-qiev7j,
    body > [role="listbox"] [role="option"][aria-selected="true"] .etx0m6x1 {{
        background-color: rgba(0,0,0,0.12) !important;
        display: block !important;
        padding: 8px 12px !important;
        margin: -8px -12px !important;
        color: {sidebar_text} !important;
        font-weight: 700 !important;
        border-radius: 4px !important;
    }}
    [data-testid="stSidebar"] .stSelectbox [role="option"][aria-selected="true"]::after {{
        content: "✔";
        position: absolute;
        right: 10px;
        top: 50%;
        transform: translateY(-50%);
        color: {sidebar_text} !important;
        font-weight: 700;
    }}
    [data-testid="stSidebar"] .stSelectbox [role="option"][aria-selected="true"]::before {{
        content: "";
        position: absolute;
        left: 6px;
        top: 8px;
        bottom: 8px;
        width: 4px;
        background: rgba(0,0,0,0.25) !important;
        border-radius: 2px;
    }}
    [data-testid="stSidebar"] .stSelectbox [role="option"] {{
        padding-left: 18px !important;
    }}
    </style>
    """

    extra = f"""
    <style>
    .float-compare {{
        position: sticky;
        top: 78px;
        z-index: 9999;
        background: transparent !important;
        padding: 0 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        margin-bottom: 0 !important;
    }}
    .kpi-row {{ display:flex; gap:18px; align-items:stretch; margin:18px 0 22px; }}
    .kpi-card {{ flex:1; background:{card_bg} !important; border:1px solid {card_border} !important; border-radius:12px; padding:16px; box-shadow:{shadow}; display:flex; flex-direction:column; gap:6px; justify-content:center; min-height:92px; }}
    .kpi-card .kpi-label {{ color:{secondary_text}; font-size:13px; }}
    .kpi-card .kpi-value {{ color:{text}; font-size:22px; font-weight:700; }}
    .kpi-card .kpi-icon {{ font-size:20px; opacity:0.9; }}
    .kpi-card {{ position: relative; }}
    .kpi-card .kpi-delta {{
        position: absolute;
        top: 10px;
        right: 12px;
        font-size:12px;
        padding:4px 8px;
        border-radius:999px;
        background: rgba(34,197,94,0.12);
        color: #16a34a;
        font-weight:700;
        box-shadow: 0 4px 12px rgba(2,8,12,0.06);
        display: inline-block;
    }}
    .kpi-card .kpi-delta.down {{ background: rgba(239,68,68,0.12); color:#ef4444; }}
    .primary-action-btn {{
        background: linear-gradient(180deg, {accent}, #157271) !important;
        color: #fff !important;
        border: none !important;
        padding: 12px 18px !important;
        border-radius: 12px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 28px rgba(30,163,161,0.14) !important;
        cursor: pointer;
    }}
    .group-card {{ background:{card_bg} !important; border:1px solid {card_border} !important; border-radius:14px; padding:14px; box-shadow:{shadow}; margin-bottom:18px; }}
    .group-title {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; font-weight:700; color:{text}; }}
    .thumb-grid {{ display:flex; gap:12px; flex-wrap:wrap; }}
    .thumb-card {{ width:180px; border-radius:10px; overflow:hidden; background:linear-gradient(180deg, rgba(255,255,255,0.98), {card_bg}); border:1px solid rgba(15,23,42,0.04); box-shadow: 0 8px 20px rgba(2,8,12,0.06); padding:8px; position:relative; }}
    .thumb-card img {{ display:block; width:100%; height:140px; object-fit:contain; background: #fff; }}
    .thumb-caption {{ text-align:center; font-size:13px; color:{secondary_text}; margin-top:8px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .thumb-badge {{ position:absolute; top:8px; left:8px; background: rgba(255,255,255,0.95); color:{text}; padding:4px 8px; border-radius:999px; font-weight:600; font-size:12px; box-shadow:0 4px 12px rgba(2,8,12,0.06); }}
    .thumb-check {{ position:absolute; top:8px; right:8px; width:32px; height:32px; border-radius:8px; display:flex; align-items:center; justify-content:center; background: rgba(255,255,255,0.95); box-shadow:0 4px 12px rgba(2,8,12,0.06); }}
    .thumb-card.selected {{ box-shadow: 0 12px 36px rgba(30,163,161,0.12); border:1px solid rgba(30,163,161,0.12); }}
    @media (max-width: 900px) {{ .thumb-card {{ width: calc(50% - 12px); }} .kpi-row {{ flex-direction:column; gap:10px; }} }}
    @media (max-width: 600px) {{ .thumb-card {{ width: calc(100% - 12px); }} }}
    </style>
    """

    try:
        st.markdown(css + extra, unsafe_allow_html=True)
    except Exception:
        # CSS 주입 실패는 UI에만 영향을 주므로 조용히 넘어갑니다.
        pass


# ===== 결과 옵션 라벨 포맷 유틸 (사이드바의 selectbox에서 사용되므로
# 이 정의는 `path_tab` 블록보다 먼저 있어야 합니다.)
def _format_result_option(path_str: str) -> str:
    """결과 폴더 옵션을 사람이 읽기 좋은 라벨로 변환합니다.

    - 입력은 문자열 또는 Path처럼 보이는 값입니다.
    - 가능한 경우 `BASE_OUTPUT_DIR`에 상대 경로로 표시합니다.
    - RESULT_META의 상태에 따라 접두사(예: [재스캔])를 붙입니다.
    이 함수는 사이드바에서 바로 호출되므로 파일 상단(또는 path_tab 앞)에
    정의되어 있어야 합니다.
    """
    try:
        p = Path(path_str)
    except Exception:
        p = Path(str(path_str))

    label = p.name

    # 메타 정보에서 상태를 읽어 접두사 추가
    meta = globals().get('RESULT_META', {}) or {}
    meta_for_path = meta.get(str(p), {})
    prefix = ""
    try:
        if meta_for_path.get("needs_rescan") and not meta_for_path.get("has_report"):
            prefix = "[재스캔] "
    except Exception:
        prefix = ""

    return f"{prefix}{label}" if prefix else label

if "_normalize_base_dir" not in globals():
    # 기본적인 정규화 함수: 전달된 경로를 안전하게 절대경로로 변환합니다.
    def _normalize_base_dir(p: Path, selection_root=None) -> Path:
        try:
            p = Path(p)
            p = p.expanduser()
            return p.resolve()
        except Exception:
            try:
                return Path(str(p))
            except Exception:
                return Path.cwd()

if "_build_result_meta" not in globals():
    # 결과 메타 생성기: 후보 경로를 찾아 report 존재 여부를 확인합니다.
    def _build_result_meta(result_dirs=None) -> Dict[str, Dict[str, bool]]:
        out: Dict[str, Dict[str, bool]] = {}

        def _add_dir(d: Path):
            try:
                target = Path(d).expanduser().resolve()
            except Exception:
                return
            if not target.exists() or not target.is_dir():
                return
            key = str(target)
            if key in out:
                return
            has = _has_result_files(target)
            out[key] = {"has_report": bool(has), "needs_rescan": not bool(has)}

        if isinstance(result_dirs, (list, tuple)) and result_dirs:
            for item in result_dirs:
                if not item:
                    continue
                try:
                    _add_dir(Path(item))
                except Exception:
                    continue

        if not out:
            candidates: List[Path] = []
            try:
                rs = getattr(st, "session_state", {})
                v1 = rs.get("result_base_dir") or rs.get("result_base_input") or rs.get("selected_result_dir")
                if v1:
                    candidates.append(Path(v1))
            except Exception:
                pass

            try:
                env = os.environ
                for k in ("ANSWER_SCAN_BASE_DIR", "ANSWER_SCAN_OUTPUT_DIR", "ANSWER_SCAN_DEFAULT_RESULT"):
                    if env.get(k):
                        candidates.append(Path(env.get(k)))
            except Exception:
                pass

            for name in ("BASE_OUTPUT_DIR", "CLI_BASE_DIR", "CLI_OUTPUT_DIR"):
                try:
                    val = globals().get(name)
                    if val:
                        candidates.append(Path(val))
                except Exception:
                    pass

            try:
                candidates.append(Path.cwd())
                candidates.append(Path.cwd() / "artifacts" / "uploaded_inputs")
            except Exception:
                pass

            for cand in candidates:
                try:
                    candp = cand.expanduser().resolve()
                except Exception:
                    continue
                if not candp.exists():
                    continue
                _add_dir(candp)
                try:
                    for child in candp.iterdir():
                        if child.is_dir():
                            _add_dir(child)
                            try:
                                for grand in child.iterdir():
                                    if grand.is_dir():
                                        _add_dir(grand)
                            except Exception:
                                pass
                except Exception:
                    pass

        try:
            def _sort_key(k):
                v = out.get(k, {})
                has = v.get("has_report", False)
                try:
                    m = Path(k).stat().st_mtime
                except Exception:
                    m = 0
                return (0 if has else 1, -m)

            ordered = sorted(list(out.keys()), key=_sort_key)
        except Exception:
            ordered = list(out.keys())

        try:
            globals()["RESULT_DIRS"] = [Path(p) for p in ordered]
        except Exception:
            try:
                globals()["RESULT_DIRS"] = [Path(p) for p in ordered if p]
            except Exception:
                pass

        return out

if "_register_result_dir" not in globals():
    def _register_result_dir(path: str) -> List[str]:
        """지정한 경로에서 '숫자_결과' 패턴에 맞는 결과 폴더들을 찾아 세션/전역 상태에 등록합니다.

        동작 요약:
        - 전달된 경로 자체가 결과 폴더 패턴에 맞으면 우선 등록합니다.
        - 그렇지 않으면 그 경로의 직계 하위 폴더들에서 패턴에 맞는 폴더들을 수집합니다.
        - 직계에 없을 경우(예: 루트/집합 폴더 구조) 2단계 깊이까지 재귀적으로 검색하여 후보를 수집합니다.
        - 발견된 각 폴더에 대해 `_has_result_files`로 유효성(리포트/요약 존재 여부)를 검사하고
          `st.session_state["available_result_dirs"]`, `RESULT_DIRS`, `RESULT_META`,
          `st.session_state["scan_result_meta"]`를 일관되게 갱신합니다.

        반환값: 등록(또는 발견)된 절대 경로 문자열 목록(순서 보장)
        """
        registered: List[str] = []

        try:
            base_path = Path(path).expanduser().resolve()
        except Exception:
            return registered

        candidates: List[Path] = []

        # 1) 전달된 경로 자체가 결과 폴더 패턴에 맞으면 우선 등록
        try:
            if base_path.is_dir() and RESULT_DIR_NAME_PATTERN.match(base_path.name):
                candidates.append(base_path)
        except Exception:
            pass

        # 2) 직계 하위에서 패턴에 맞는 폴더 수집
        if not candidates and base_path.is_dir():
            try:
                for child in sorted(base_path.iterdir()):
                    try:
                        if child.is_dir() and RESULT_DIR_NAME_PATTERN.match(child.name):
                            candidates.append(child)
                    except Exception:
                        continue
            except Exception:
                pass

        # 3) 직계에도 없을 경우 2단계 깊이까지 검색 (보수적 예비 검색)
        if not candidates and base_path.is_dir():
            try:
                for child in sorted(base_path.iterdir()):
                    try:
                        if not child.is_dir():
                            continue
                        for sub in sorted(child.iterdir()):
                            try:
                                if sub.is_dir() and RESULT_DIR_NAME_PATTERN.match(sub.name):
                                    candidates.append(sub)
                            except Exception:
                                continue
                    except Exception:
                        continue
            except Exception:
                pass

        # 세션 상태 및 메타 준비
        dirs_state = st.session_state.setdefault("available_result_dirs", [])
        meta = globals().get("RESULT_META") or {}

        # 발견된 후보들을 세션/전역에 반영
        for cand in candidates:
            try:
                resolved = str(Path(cand).resolve())
            except Exception:
                resolved = str(cand)

            # 결과 파일 존재 여부 판정
            try:
                has_result = bool(_has_result_files(Path(resolved)))
            except Exception:
                has_result = False

            meta[resolved] = {"has_report": has_result, "needs_rescan": not has_result}

            if resolved not in dirs_state:
                dirs_state.append(resolved)
            registered.append(resolved)

        # 기존에 등록된 경로 중 현재 베이스(루트) 문자열은 직접 등록 대상이 아니면 제거
        try:
            base_str = str(base_path)
            if base_str in dirs_state and base_str not in registered:
                try:
                    dirs_state.remove(base_str)
                except Exception:
                    pass
        except Exception:
            pass

        # 중복 제거(입력 순서 유지) 및 상태 업데이트
        unique_dirs = list(dict.fromkeys(dirs_state))
        st.session_state["available_result_dirs"] = unique_dirs

        try:
            globals()["RESULT_DIRS"] = [Path(p) for p in unique_dirs]
        except Exception:
            globals()["RESULT_DIRS"] = []

        globals()["RESULT_META"] = meta
        st.session_state["scan_result_meta"] = meta

        return registered

if "_has_result_files" not in globals():
    # 결과 폴더 판정 유틸
    def _has_result_files(output_dir: Path) -> bool:
        try:
            p = Path(output_dir)
            if not p.exists() or not p.is_dir():
                return False
            report_parquet = p / "report.parquet"
            report_csv = p / "report.csv"
            summary_csv = p / "images_summary.csv"

            has_report = report_parquet.exists() or report_csv.exists()
            has_imgsum = summary_csv.exists()

            if has_report and has_imgsum:
                return True

            if has_imgsum:
                return True

            if has_report and not has_imgsum:
                art = p / "artifacts"
                if (art / "ordered_paths.txt").exists():
                    return True
                if (art / "thumbnails").exists():
                    return True
                if (p / "grouped").exists() or (p / "ok").exists():
                    return True
                return False

            if (p / "grouped").exists():
                return True
            art = p / "artifacts"
            if (art / "ordered_paths.txt").exists() or (art / "thumbnails").exists():
                return True
            for child in p.iterdir():
                try:
                    if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
                        return True
                except Exception:
                    continue

            return False
        except Exception:
            return False

if "RESAMPLE" not in globals():
    try:
        RESAMPLE = Image.Resampling.LANCZOS
    except Exception:
        try:
            RESAMPLE = Image.LANCZOS
        except Exception:
            RESAMPLE = Image.BICUBIC

try:
    import stat
except Exception:
    stat = None

if "_request_rerun" not in globals():
    def _request_rerun():
        try:
            st.experimental_rerun()
        except Exception:
            try:
                st.rerun()
            except Exception:
                pass


# 사이드바 탭 생성
path_tab, settings_tab, theme_tab = st.sidebar.tabs(["분석 경로", "보기 설정", "테마"])

# ZIP 업로드에 사용되는 세션 상태와 공통 처리 함수들을 전역에서 정의해
# 메인 화면과 사이드바 모두에서 동일한 로직을 재사용합니다.
if "auto_applied_zip_tokens" not in st.session_state:
    # token -> dest_dir 매핑을 저장합니다 (중복 업로드 방지 및 재실행 지원)
    st.session_state["auto_applied_zip_tokens"] = {}

if "available_result_dirs" not in st.session_state:
    # ZIP 업로드로 등록된 결과 폴더 경로 목록을 보관합니다.
    st.session_state["available_result_dirs"] = []


def _start_analysis_uploaded_cb(dest: str, preferred_result_dir: Optional[str] = None) -> None:
    """업로드된 폴더를 분석 대상으로 세션에 적용하고 Streamlit 재실행을 시도합니다.

    dest: 압축이 풀린 디렉터리의 절대 경로
    """
    try:
        # 절대 경로로 정규화하여 세션에 기록
        norm = str(Path(dest).resolve())
        try:
            if preferred_result_dir:
                _register_result_dir(preferred_result_dir)
            else:
                _register_result_dir(norm)
        except Exception:
            pass
        st.session_state["result_base_input"] = norm
        st.session_state["result_base_dir"] = norm
        # 업로드로 풀린 폴더를 '선택된 결과 폴더'로 즉시 설정합니다.
        # 이렇게 하면 사용자가 ZIP을 업로드한 직후 해당 경로가 사이드바의
        # 선택값(selected_result_dir)으로 반영되어 바로 분석 대상이 됩니다.
        # 또한 내부적으로 선택 변경 감지를 위해 _last_selected_dir도 갱신합니다.
        if preferred_result_dir:
            st.session_state["selected_result_dir"] = preferred_result_dir
            st.session_state["_last_selected_dir"] = preferred_result_dir
        else:
            st.session_state.pop("selected_result_dir", None)
            st.session_state.pop("_last_selected_dir", None)
        st.session_state["_cli_base_marker"] = norm
        try:
            # 캐시를 비워 새 경로로의 검색이 반영되게 합니다.
            st.cache_data.clear()
        except Exception:
            pass
    except Exception:
        # 실패해도 UI가 멈추지 않도록 무시
        pass
    # 재실행 시도 (환경에 따라 experimental_rerun/rerun 사용)
    try:
        _request_rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            try:
                st.rerun()
            except Exception:
                pass


def _handle_uploaded_zip(uploaded_zip, source_tag: str = "sidebar") -> None:
    """ZIP 업로드 공통 처리 루틴.

    - 업로드한 ZIP을 안전하게 artifacts/uploaded_inputs 아래에 풀어 놓습니다.
    - 이미 적용된 ZIP은 재사용 버튼만 노출합니다.
    - 성공 시 세션 상태를 갱신해 방금 풀린 경로를 분석 대상으로 지정합니다.
    """
    if uploaded_zip is None:
        return

    try:
        import zipfile
        import time
        import tempfile
    except Exception as exc:
        st.error(f"ZIP 처리를 위한 모듈을 불러오지 못했습니다: {exc}")
        return

    # 안전 상한값: 압축 파일 업로드 크기/압축 해제 용량/파일 개수/단일 파일 크기를 제한합니다.
    # 변경: 전체 업로드 허용치를 10 GiB로 상향합니다.
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024 * 1024        # 10 GiB
    MAX_TOTAL_EXTRACT = 12 * 1024 * 1024 * 1024      # 전체 추출 상한 12 GiB
    MAX_MEMBER_COUNT = 20000                          # 멤버 수 상한 (필요시 더 조정)
    MAX_SINGLE_FILE = 10 * 1024 * 1024 * 1024        # 단일 파일 10 GiB

    # 1) 업로드 스트림을 임시 파일로 복사하며 SHA1 해시를 계산합니다.
    #    메모리에 전체를 올리지 않고 순차적으로 처리해 대용량에서도 안전합니다.
    sha1 = hashlib.sha1()
    total_read = 0
    temp_path = None
    chunk_size = 4 * 1024 * 1024  # 4MB씩 처리

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            temp_path = tmp.name
            try:
                uploaded_zip.seek(0)
            except Exception:
                pass

            while True:
                chunk = uploaded_zip.read(chunk_size)
                if not chunk:
                    break
                total_read += len(chunk)
                if total_read > MAX_UPLOAD_SIZE:
                    raise ValueError("업로드한 ZIP이 허용 용량(10GiB)을 초과했습니다.")
                sha1.update(chunk)
                tmp.write(chunk)

        upload_token = sha1.hexdigest()
    except Exception as exc:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        st.error(f"ZIP 업로드를 처리하는 중 오류가 발생했습니다: {exc}")
        return

    # 2) 기존에 동일 ZIP이 적용된 경우 안내 후 재분석 버튼만 노출합니다.
    if upload_token in st.session_state["auto_applied_zip_tokens"]:
        prev_dest = st.session_state["auto_applied_zip_tokens"].get(upload_token)
        st.warning("이 ZIP은 이미 적용되었습니다.")
        if prev_dest:
            st.markdown(f"- 적용 경로: `{prev_dest}`")
            if st.button("다시 분석", key=f"reapply_zip_{source_tag}_{upload_token}"):
                _start_analysis_uploaded_cb(prev_dest)
        else:
            st.info("다른 파일을 업로드해 주세요.")
        try:
            uploaded_zip.seek(0)
        except Exception:
            pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return

    # 3) ZIP을 추출할 루트를 준비합니다.
    extract_root = os.path.join(str(Path.cwd()), "artifacts", "uploaded_inputs")
    os.makedirs(extract_root, exist_ok=True)

    try:
        orig_name = getattr(uploaded_zip, "name", None) or ""
        stem = Path(orig_name).stem if orig_name else ""
    except Exception:
        stem = ""

    ts = int(time.time())
    if stem:
        candidate = os.path.join(extract_root, stem)
        if os.path.exists(candidate):
            dest_dir = os.path.join(extract_root, f"{stem}_{ts}")
        else:
            dest_dir = candidate
    else:
        dest_dir = os.path.join(extract_root, f"upload_{ts}")

    # 4) ZIP을 검사 및 추출: 경로 탈출, 압축 폭탄, 심볼릭 링크 등을 방지합니다.
    extracted_any = False
    try:
        with zipfile.ZipFile(temp_path, "r") as zf:
            members = zf.infolist()

            if len(members) > MAX_MEMBER_COUNT:
                raise ValueError("ZIP에 포함된 파일 수가 비정상적으로 많습니다.")

            total_uncompressed = 0
            for member in members:
                if member.flag_bits & 0x1:
                    raise ValueError("암호화된 ZIP 파일은 지원하지 않습니다.")
                if member.file_size > MAX_SINGLE_FILE:
                    raise ValueError("ZIP 내부에 허용 크기를 초과하는 파일이 있습니다.")
                total_uncompressed += member.file_size
                if total_uncompressed > MAX_TOTAL_EXTRACT:
                    raise ValueError("압축 해제 예상 용량이 허용치를 초과했습니다.")

            for member in members:
                member_name = member.filename
                normalized = os.path.normpath(member_name)
                if normalized.startswith("..") or os.path.isabs(normalized):
                    logger.warning(f"ZIP 내부 위험 경로 건너뜀: {member_name}")
                    continue

                # 심볼릭 링크/하드 링크는 추출하지 않습니다.
                is_symlink = False
                try:
                    ext_attr = member.external_attr >> 16
                    is_symlink = stat and stat.S_ISLNK(ext_attr)
                except Exception:
                    is_symlink = False
                if is_symlink:
                    logger.warning(f"ZIP 내부 심볼릭 링크 건너뜀: {member_name}")
                    continue

                target_path = os.path.join(dest_dir, *Path(normalized).parts)
                try:
                    dest_root = Path(dest_dir).resolve()
                    target_resolved = Path(target_path).resolve()
                    if not str(target_resolved).startswith(str(dest_root)):
                        logger.warning(f"ZIP 멤버가 허용된 경로 밖에 있어 건너뜀: {member_name}")
                        continue

                    if member.is_dir():
                        target_resolved.mkdir(parents=True, exist_ok=True)
                        continue

                    target_resolved.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(target_resolved, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted_any = True
                except Exception:
                    logger.exception("멤버 추출 중 예외 발생")
    except Exception as exc:
        try:
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
        except Exception:
            pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        st.error(f"압축 해제 실패: {exc}")
        return
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    if not extracted_any:
        try:
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
        except Exception:
            pass
        st.error("ZIP 안에서 추출 가능한 파일을 찾지 못했습니다.")
        return

    # 5) 세션 상태 및 전역 메타를 갱신 후 바로 분석을 재시작합니다.
    try:
        resolved_dest = str(Path(dest_dir).resolve())
    except Exception:
        resolved_dest = dest_dir

    st.session_state["auto_applied_zip_tokens"][upload_token] = resolved_dest
    registered_dirs = _register_result_dir(resolved_dest)

    preferred = registered_dirs[0] if registered_dirs else None

    if not registered_dirs:
        st.warning("ZIP 내부에서 분석 가능한 `_결과` 폴더를 찾지 못했습니다. 추후 직접 경로를 선택해 주세요.")

    st.success(f"압축 해제 완료: {dest_dir}")
    st.info("업로드된 폴더로 바로 분석을 시작합니다.")
    _start_analysis_uploaded_cb(dest_dir, preferred_result_dir=preferred)


def _render_initial_upload_gate() -> None:
    """결과 폴더가 아직 없는 초기 상태에서 대형 업로드 안내 화면을 보여줍니다."""
    st.markdown(
        """
        <style>
        .upload-hero-wrapper { display:flex; align-items:center; justify-content:center; min-height: 70vh; }
        .upload-hero {
            width: min(820px, 100%);
            padding: 52px 48px;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(30,163,161,0.12), rgba(255,255,255,0.92));
            box-shadow: 0 26px 58px rgba(12, 40, 60, 0.08);
            text-align: center;
        }
        .upload-hero h2 { font-size: 32px; font-weight: 800; margin-bottom: 18px; color: #0B1726; }
        .upload-hero p { font-size: 16px; line-height: 1.6; color: #3b4a5d; margin: 0; }
        section.main div[data-testid="stFileUploader"] {
            margin: 28px auto 16px auto;
            max-width: 640px;
            border: 2.5px dashed rgba(30,163,161,0.45);
            border-radius: 18px;
            padding: 32px 28px;
            background: rgba(255,255,255,0.94);
        }
        section.main div[data-testid="stFileUploader"] label { display: none; }
        section.main div[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] {
            background: transparent;
        }
        section.main div[data-testid="stFileUploader"] button[kind="secondary"] {
            width: 100%;
            border-radius: 12px;
        }
        section.main div[data-testid="stFileUploader"] p {
            text-align: center;
            color: #1EA3A1;
            font-weight: 600;
        }
        [data-testid="stSidebar"] div[data-testid="stFileUploader"] {
            border: none;
            padding: 0;
            background: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="upload-hero-wrapper">
            <div class="upload-hero">
                <h2>전체 화면에다가 ZIP 폴더를 넣어주세요</h2>
                <p>폴더를 ZIP으로 압축해 이 영역에 드래그하거나 클릭하여 업로드하세요. 업로드가 끝나면 자동으로 압축을 풀고 분석 경로로 적용합니다.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_zip_main = st.file_uploader(
        "ZIP 업로드",
        type=["zip"],
        key="upload_zip_main",
        label_visibility="collapsed",
        help="ZIP 파일 1개를 업로드하세요.",
    )
    if uploaded_zip_main is not None:
        _handle_uploaded_zip(uploaded_zip_main, source_tag="main")



with path_tab:
    st.markdown("**분석 경로 설정**")
    # result_base_dir 기본값을 초기화하고 업로드된 폴더 경로가 있으면 그대로 보여줍니다.
    if "result_base_dir" not in st.session_state:
        # 기본값으로 현재 작업 디렉터리 대신 빈 문자열을 사용합니다.
        # 사용자가 명시적으로 경로를 입력하거나 ZIP을 업로드할 때까지
        # 분석 경로가 비어있도록 유지합니다.
        default_base = ""
        st.session_state["result_base_dir"] = st.session_state.get("result_base_input", default_base)
    else:
        default_base = ""

    current_base_dir = str(st.session_state.get("result_base_dir", default_base))
    st.session_state["result_base_input"] = current_base_dir

    st.markdown(f"현재 분석 경로: `{current_base_dir}`")
    st.caption("ZIP을 업로드하면 이 경로가 자동으로 업데이트됩니다.")

    def _reset_base_dir_cb():
        # CLI 기본값으로 복원: 마커와 result_base_dir을 갱신하고 표시 문자열도 맞춥니다.
        try:
            st.session_state["result_base_dir"] = str(CLI_BASE_DIR)
            st.session_state.pop("selected_result_dir", None)
            st.session_state["_cli_base_marker"] = str(CLI_BASE_DIR)
            # 텍스트 입력도 같은 값을 반영하도록 설정
            st.session_state["result_base_input"] = str(CLI_BASE_DIR)
        except Exception:
            pass
        # 콜백 내부에서 강제 rerun을 호출하지 않음: 세션 상태 변경으로 자동 재실행됩니다.
        try:
            st.cache_data.clear()
        except Exception:
            pass

    # 사이드바 내 버튼을 좀 더 보기 좋게 확장합니다.
    # - 두 버튼을 동일한 너비로 배치하고
    # - CSS로 최소 너비와 패딩, 글자 크기를 늘려 시각적으로 정돈합니다.
    btn_css = """
    <style>
    /* 사이드바 내부 버튼 스타일 적용 */
    [data-testid="stSidebar"] .stButton>button {
        min-width: 160px !important;
        padding: 10px 22px !important;
        font-size: 16px !important;
        border-radius: 10px !important;
    }
    /* 약간의 간격을 주어 버튼이 붙어 보이지 않게 함 */
    [data-testid="stSidebar"] .stButton {
        margin-bottom: 6px !important;
    }
    </style>
    """
    try:
        st.markdown(btn_css, unsafe_allow_html=True)
    except Exception:
        pass

    st.button("기본 경로 복원", key="reset_base_dir", on_click=_reset_base_dir_cb)
    # (목록 새로고침 버튼 제거됨)

    # ------------------------
    # ZIP(폴더) 업로드: 공통 헬퍼를 호출해 사이드바에서도 동일한 동작을 제공합니다.
    uploaded_zip = st.file_uploader(
        "폴더 업로드(.zip) — 업로드 시 자동으로 분석을 시작합니다",
        type=["zip"],
        help="폴더를 ZIP으로 압축하여 업로드하면 서버에 풀어 분석할 수 있습니다.",
        key="upload_zip_sidebar",
    )
    if uploaded_zip is not None:
        _handle_uploaded_zip(uploaded_zip, source_tag="sidebar")

    # 사이드바에서 탭을 전환할 때 사용할 콜백 함수입니다.
    # 여러 위젯에서 이 함수를 on_change로 참조하므로 파일 상단에서 미리 정의해 NameError를 방지합니다.
    def switch_main_tab(tab_name: str):
        """사이드바 필터 변경 시 해당 탭으로 이동"""
        try:
            st.session_state["main_tab"] = tab_name
        except Exception:
            # 세션 상태 접근이 실패하면 무시합니다.
            pass

    # 세션에서 결과 폴더 목록을 읽어와 셀렉트 박스를 구성합니다.
    result_dirs_val: List[Path] = []
    for item in st.session_state.get("available_result_dirs", []):
        try:
            result_dirs_val.append(Path(item))
        except Exception:
            continue

    selected_from_state = st.session_state.get("selected_result_dir")
    if selected_from_state:
        try:
            resolved_state = Path(selected_from_state).expanduser().resolve()
        except Exception:
            resolved_state = Path(str(selected_from_state))
        already_registered = False
        for existing in result_dirs_val:
            try:
                if Path(existing).resolve() == resolved_state:
                    already_registered = True
                    break
            except Exception:
                if str(existing) == str(resolved_state):
                    already_registered = True
                    break
        if not already_registered:
            result_dirs_val.append(resolved_state)

    try:
        result_options_local = []
        seen = set()
        for p in result_dirs_val:
            s = str(p)
            if s in seen:
                continue
            seen.add(s)
            result_options_local.append(s)
    except Exception:
        result_options_local = []

    try:
        fmt = _format_result_option
    except Exception:
        fmt = lambda x: x

    if result_options_local:
        st.selectbox(
            "분석 결과 폴더",
            options=result_options_local,
            format_func=fmt,
            key="selected_result_dir",
        )
    else:
        st.info("ZIP을 업로드하면 분석 결과 폴더가 여기에 표시됩니다.")

with theme_tab:
    st.markdown("**대시보드 테마**")
    sidebar_width_default = int(st.session_state.get("sidebar_width_px", 350))
    sidebar_slider_args = {
        "label": "사이드바 폭",
        "min_value": 260,
        "max_value": 520,
        "key": "sidebar_width_px",
        "help": "사이드바 영역의 폭을 조정해 긴 라벨이나 컨트롤이 잘려 보이지 않도록 합니다."
    }
    st.slider(value=sidebar_width_default, **sidebar_slider_args)
    theme_keys = list(THEMES.keys())
    default_idx = theme_keys.index(st.session_state.get('theme', theme_keys[0])) if st.session_state.get('theme') in theme_keys else 0
    st.radio('테마 선택', theme_keys, index=default_idx, key='theme', horizontal=True)
    sel = st.session_state.get('theme', theme_keys[0])
    pal = THEMES[sel]['palette']
    swatch_html = '<div style="display:flex;gap:6px;margin-top:8px;align-items:center">'
    for k in ['bg','card_bg','text','accent']:
        if k in pal:
            swatch_html += f"<div style=\"width:36px;height:24px;border-radius:6px;background:{pal[k]};border:1px solid rgba(0,0,0,0.06)\" title=\"{k}\"></div>"
    swatch_html += '</div>'
    st.markdown(swatch_html, unsafe_allow_html=True)
    st.write(THEMES[sel].get('desc',''))

with settings_tab:
    # 보기 설정을 3개의 큰 섹션(재스캔 / 정상·공백 / 전체 보기)으로 정리해 가독성을 높입니다.
    st.markdown("**보기 설정(섹션별로 접어서 보기 가능)**")

    # 1) 재스캔 워크플로
    with st.expander("재스캔 워크플로", expanded=True):
        # df가 아직 정의되지 않았을 수 있으므로 안전하게 조회합니다.
        df_val = globals().get("df")
        if df_val is not None and hasattr(df_val, "columns"):
            try:
                group_list = sorted(list(df_val["그룹ID"].replace('-', pd.NA).dropna().unique())) if "그룹ID" in df_val.columns else []
            except Exception:
                group_list = []
        else:
            group_list = []
        st.selectbox(
            "그룹 선택",
            ["전체"] + group_list,
            key="group_filter",
            help="재스캔 탭의 후보 목록을 특정 그룹으로 한정합니다.",
            on_change=switch_main_tab,
            args=("재스캔 필요",)
        )
        st.radio(
            "보기 방식",
            ["대형 비교(2열)", "그리드(다중 썸네일)"],
            key="group_view_mode",
            horizontal=True,
            help="대형 비교는 앞·뒤면을 크게 보여주고, 그리드는 그룹 내 모든 이미지를 타일로 확인합니다.",
            on_change=switch_main_tab,
            args=("재스캔 필요",)
        )
        rescan_quality_default = st.session_state.get("rescan_quality_profile", "균형")
        rescan_q_idx = quality_options_common.index(rescan_quality_default) if rescan_quality_default in quality_options_common else 1
        st.radio(
            "화질 프로파일",
            quality_options_common,
            index=rescan_q_idx,
            key="rescan_quality_profile",
            horizontal=True,
            help="빠름(512px), 균형(1024px), 선명(1600px) 수준으로 썸네일 품질과 크기를 조정합니다.",
            on_change=switch_main_tab,
            args=("재스캔 필요",)
        )

        # (삭제) 재스캔 워크플로의 파일 삭제 버튼 및 관련 상태 제어는 제거되었습니다.

    # 2) 정상 / 공백 보기
    with st.expander("정상·공백 답안", expanded=True):
        st.radio(
            "보기 옵션",
            ["모두 보기", "정상만", "공백만"],
            key="ok_view_mode",
            horizontal=True,
            help="정상/공백 탭에서 표시할 답안 유형을 빠르게 전환합니다.",
            on_change=switch_main_tab,
            args=("정상/공백 답안",)
        )
        grid_default = int(st.session_state.get("grid_cols", 5))
        grid_slider_args = {
            "label": "그리드 열 개수",
            "min_value": 2,
            "max_value": 10,
            "key": "grid_cols",
            "help": "정상/공백 탭의 썸네일 한 줄 배치를 조정합니다."
        }
        st.slider(value=grid_default, **grid_slider_args)
        ok_quality_default = st.session_state.get("ok_quality_profile", "균형")
        ok_q_idx = quality_options_common.index(ok_quality_default) if ok_quality_default in quality_options_common else 1
        st.radio(
            "화질 프로파일",
            quality_options_common,
            index=ok_q_idx,
            key="ok_quality_profile",
            horizontal=True,
            help="빠름(512px), 균형(1024px), 선명(1600px) 썸네일 품질을 선택합니다.",
            on_change=switch_main_tab,
            args=("정상/공백 답안",)
        )

    # 3) 전체 보기
    with st.expander("전체 보기", expanded=False):
        st.text_input(
            "파일명·경로 검색",
            key="gallery_search",
            placeholder="예: 10002, scan, .png",
            on_change=switch_main_tab,
            args=("전체 보기",)
        )
        ext_options = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        if "gallery_exts" in st.session_state:
            st.multiselect("확장자", ext_options, key="gallery_exts")
        else:
            st.multiselect("확장자", ext_options, default=[], key="gallery_exts")

        st.markdown("**표시 설정**")
        quality_options = quality_options_common
        current_quality = st.session_state.get("gallery_quality_profile")
        q_idx = quality_options.index(current_quality) if current_quality in quality_options else 1
        st.radio(
            "화질 프로파일",
            quality_options,
            index=q_idx,
            horizontal=True,
            key="gallery_quality_profile",
            help="빠름(512px), 균형(1024px), 선명(1600px)",
            on_change=switch_main_tab,
            args=("전체 보기",)
        )
        gallery_grid_default = int(st.session_state.get("gallery_grid_cols", 5))
        gallery_slider_args = {
            "label": "그리드 열 개수",
            "min_value": 2,
            "max_value": 10,
            "key": "gallery_grid_cols",
            "help": "전체 보기 탭에서 한 줄에 배치될 썸네일 개수"
        }
        st.slider(value=gallery_grid_default, **gallery_slider_args)

# 세션에 저장된 결과 폴더 목록을 기반으로 메타 데이터를 초기화합니다.
available_dirs_raw = st.session_state.get("available_result_dirs", [])
filtered_dirs: List[str] = []
for item in available_dirs_raw:
    try:
        p = Path(item).expanduser().resolve()
        if p.is_dir() and RESULT_DIR_NAME_PATTERN.match(p.name):
            filtered_dirs.append(str(p))
    except Exception:
        continue

st.session_state["available_result_dirs"] = filtered_dirs

try:
    RESULT_DIRS = [Path(p) for p in filtered_dirs]
except Exception:
    RESULT_DIRS = []

if "scan_result_meta" in st.session_state:
    RESULT_META = st.session_state.get("scan_result_meta") or {}
else:
    RESULT_META = _build_result_meta(filtered_dirs) if filtered_dirs else {}

# ===== 페이지 설정 =====


result_options = [str(p) for p in RESULT_DIRS]

if "selected_result_dir" in st.session_state and st.session_state["selected_result_dir"] not in result_options:
    st.session_state.pop("selected_result_dir", None)

# 등록된 결과 폴더가 없으면 초기 업로드 게이트를 보여줍니다.
if not result_options:
    st.session_state.pop("selected_result_dir", None)
    _render_initial_upload_gate()
    st.stop()

selected_dir_str = st.session_state.get("selected_result_dir")
if not selected_dir_str and result_options:
    selected_dir_str = result_options[0]
    st.session_state["selected_result_dir"] = selected_dir_str
prev_selected_dir = st.session_state.get("_last_selected_dir")
dir_changed = prev_selected_dir is not None and prev_selected_dir != selected_dir_str
st.session_state["_last_selected_dir"] = selected_dir_str

OUTPUT_DIR = Path(selected_dir_str).resolve()
REPORT_PARQUET = os.path.join(str(OUTPUT_DIR), "report.parquet")
REPORT_CSV = os.path.join(str(OUTPUT_DIR), "report.csv")
IMG_SUMMARY = os.path.join(str(OUTPUT_DIR), "images_summary.csv")
THUMB_DIR = os.path.join(str(OUTPUT_DIR), "artifacts", "thumbnails")

REPORT_BASE_COLUMNS = ["그룹ID", "상태", "파일1", "파일2", "유사도"]

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def _is_back_page(filename: str) -> bool:
    try:
        stem = os.path.splitext(os.path.basename(str(filename)))[0]
    except Exception:
        return False
    return stem.endswith("2")

if OUTPUT_DIR.exists():
    os.makedirs(os.path.join(str(OUTPUT_DIR), "artifacts"), exist_ok=True)
    os.makedirs(THUMB_DIR, exist_ok=True)
    if not _has_result_files(OUTPUT_DIR):
        # UI에서 사용자에게 경고 배너를 띄우지 않도록 변경했습니다.
        # - 결과 파일이 없는 상황은 로그로만 남기고 대시보드 흐름을 막지 않습니다.
        # - 필요하면 후에 st.info/st.warning으로 다시 노출하도록 쉽게 되돌릴 수 있습니다.
        try:
            logger.info(f"결과 파일이 없습니다: {OUTPUT_DIR}")
        except Exception:
            # 로거 호출 실패 시에도 UI가 중단되지 않도록 무시
            pass
else:
    st.warning("선택한 폴더가 존재하지 않습니다. 올바른 경로를 입력하세요.")
# 상단 타이틀(페이지 헤더) 삽입: 페이지 최상단에 보이도록 이동
st.markdown("""
<style>
.app-header { display:flex; align-items:center; justify-content:space-between; padding: 8px 0 18px 0; margin-bottom: 8px; }
.app-title { font-size:20px; font-weight:800; color:#0B1726; display:flex; align-items:center; gap:12px; }
.app-badge { background:linear-gradient(90deg,#eaf9f8,#f0fbfb); color:#1EA3A1; font-weight:700; padding:6px 12px; border-radius:999px; font-size:13px; box-shadow:0 2px 8px rgba(30,163,161,0.06); }
.app-icons { display:flex; gap:10px; align-items:center; }
.app-icon { width:36px; height:36px; border-radius:50%; background:#fff; display:inline-flex; align-items:center; justify-content:center; box-shadow:0 2px 8px rgba(2,8,12,0.06); font-size:16px; }
</style>
<div class="app-header">
    <div class="app-title"><h1>📄 답안지 스캔 검사 대시보드</h1> <span class="app-badge">Handwriting-Optimized</span></div>
    <div class="app-icons">
        <div class="app-icon">🌙</div>
        <div class="app-icon">👤</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 메인 탭 상태 관리 ---
if "main_tab" not in st.session_state:
    st.session_state["main_tab"] = "재스캔 필요"

def switch_main_tab(tab_name: str):
    """사이드바 필터 변경 시 해당 탭으로 이동"""
    st.session_state["main_tab"] = tab_name

# --- Gallery state (전체 보기 탭 전용) ---
if "gallery_limit" not in st.session_state:
    st.session_state.gallery_limit = 120  # 한 번에 보여줄 개수 초기값 (증가)
if "gallery_selected" not in st.session_state:
    st.session_state.gallery_selected = []  # 비교 선택(최대 2장)

# 전체 보기 탭 공유 상태(사이드바 → 전역 적용)
if "gallery_search" not in st.session_state:
    st.session_state.gallery_search = ""
if "gallery_sort" not in st.session_state:
    st.session_state.gallery_sort = "파일명"
if "gallery_render_mode" not in st.session_state:
    st.session_state.gallery_render_mode = "리샘플(권장)"
if "group_filter" not in st.session_state:
    st.session_state.group_filter = "전체"
if "group_view_mode" not in st.session_state:
    st.session_state.group_view_mode = "그리드(다중 썸네일)"
if "ok_view_mode" not in st.session_state:
    st.session_state.ok_view_mode = "모두 보기"

# (삭제) 재스캔 탭의 삭제 워크플로 관련 세션 키 초기화 코드는 제거되었습니다.
# ===== 공용 헬퍼 =====
def _file_mtime(path: str) -> float:
    try: return os.path.getmtime(path)
    except Exception: return 0.0

def _safe_image_open(path: str) -> Image.Image:
    """이미지를 안전하게 열어 PIL.Image를 반환합니다.
    - 우선 PIL.Image.open을 시도합니다.
    - 실패하면 cv2.imread 후 PIL 변환을 시도합니다.
    - 모두 실패하면 UI가 멈추지 않도록 플레이스홀더 이미지를 반환합니다.
    """
    try:
        img = Image.open(path)
        img.load()
    # 하위 처리에서 일관되게 RGB 모드로 정규화
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.debug(f"_safe_image_open: PIL 열기 실패 {path}: {e}")
    # PIL 실패, cv2 대체 방법 시도
    try:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is not None:
            # BGR(A) -> RGB(A) 변환
            if len(arr.shape) == 3:
                if arr.shape[2] == 4:
                    b, g, r, a = cv2.split(arr)
                    arr = cv2.merge((r, g, b, a))
                    pil = Image.fromarray(arr)
                else:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(arr)
            else:
                pil = Image.fromarray(arr)
            return pil
    except Exception as e:
        logger.debug(f"_safe_image_open: cv2 대체 실패 {path}: {e}")

    # 최종 대체: UI가 멈추지 않도록 플레이스홀더 이미지 생성
    try:
        w, h = 640, 480
        ph = Image.new("RGB", (w, h), (220, 220, 220))
        draw = ImageDraw.Draw(ph)
        basename = os.path.basename(path) if path else "unknown"
        txt = f"읽을 수 없음\n{basename}"
        draw.text((8, 8), txt, fill=(80, 80, 80))
        return ph
    except Exception:
        # 플레이스홀더 생성도 실패하면 원래 예외를 재발생시킴
        raise

# ===== 캐싱: 데이터 읽기 =====
@st.cache_data(show_spinner=False)
def load_report(
    report_parquet: str,
    report_csv: str,
    columns: Optional[List[str]] = None,
    cache_token: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    _ = cache_token
    if os.path.exists(report_parquet):
        lf = pl.scan_parquet(report_parquet)
        if columns:
            available_cols = [col for col in lf.columns if col in columns]
            if available_cols:
                lf = lf.select(available_cols)
        df_out = lf.collect(streaming=True).to_pandas(use_pyarrow_extension_array=True)
        if columns:
            ordered = [col for col in columns if col in df_out.columns]
            if ordered:
                df_out = df_out[ordered]
        return df_out
    if os.path.exists(report_csv):
        try:
            if columns:
                df_csv = pd.read_csv(report_csv, usecols=lambda c: c in set(columns))
                ordered = [col for col in columns if col in df_csv.columns]
                if ordered:
                    df_csv = df_csv[ordered]
                return df_csv
            return pd.read_csv(report_csv)
        except Exception as e:
            # 빈 CSV 파일 등으로 인해 pandas가 실패하는 경우를 안전하게 처리합니다.
            # 빈 파일이면 컬럼 정보가 없으므로 빈 DataFrame을 반환해 UI가 멈추지 않도록 합니다.
            try:
                # pandas EmptyDataError를 포함한 모든 읽기 실패는 빈 DataFrame으로 처리
                from pandas.errors import EmptyDataError
                if isinstance(e, EmptyDataError):
                    return pd.DataFrame(columns=columns or [])
            except Exception:
                pass
            # 그 외 예외는 디버그 로깅 후 빈 DataFrame 반환
            logger.debug(f"report.csv 로드 실패 ({report_csv}): {e}")
            return pd.DataFrame(columns=columns or [])
    raise FileNotFoundError("결과 리포트 파일을 찾을 수 없습니다.")

@st.cache_data(show_spinner=False)
def load_img_summary(img_summary_csv: str, cache_buster: Optional[float] = None) -> pd.DataFrame:
    _ = cache_buster
    if os.path.exists(img_summary_csv):
        return pd.read_csv(img_summary_csv)
    return pd.DataFrame(columns=["파일", "밀도", "빈칸여부"])

def _iter_images(root: str):
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(IMAGE_EXTS):
                        yield entry.path
        except PermissionError:
            continue

@st.cache_data(show_spinner=False)
def list_all_images(root: str, cache_buster: float = 0) -> List[str]:
    """
    루트 폴더 아래의 이미지 파일을 재귀적으로 나열합니다.
    cache_buster는 외부에서 캐시를 무효화할 때 사용합니다.
    """
    try:
        # 주의: 전역 IMG_SUMMARY는 현재 선택된 OUTPUT_DIR을 가리키지만,
        # 함수 호출자가 전달한 root를 사용해 이미지 요약 파일 경로를 계산하면
        # 캐시 키가 root에 따라 분리되어 폴더 전환 시 더 안전합니다.
        summary_csv = os.path.join(root, "images_summary.csv")
        summary_df = load_img_summary(summary_csv, cache_buster=cache_buster)
        ordered_txt = os.path.join(root, "artifacts", "ordered_paths.txt")
        ordered_map: Dict[str, List[str]] = {}
        if os.path.isfile(ordered_txt):
            try:
                with open(ordered_txt, "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        p = line.strip().strip('"')
                        if not p:
                            continue
                        path = os.path.normpath(p)
                        if os.path.isfile(path):
                            bn = os.path.basename(path).lower()
                            ordered_map.setdefault(bn, []).append(path)
            except Exception as exc:
                logger.debug(f"ordered_paths 로드 실패: {exc}")

        seen: Dict[str, None] = {}
        if not summary_df.empty and '파일' in summary_df.columns:
            for entry in summary_df['파일']:
                if not isinstance(entry, str) or not entry:
                    continue
                candidates: List[str] = []
                if os.path.isabs(entry):
                    candidates.append(entry)
                else:
                    candidates.append(os.path.join(root, entry))
                bn = os.path.basename(entry).lower()
                candidates.extend(ordered_map.get(bn, []))
                for cand in candidates:
                    path = os.path.normpath(cand)
                    if os.path.isfile(path):
                        seen[path] = None

        if not seen and ordered_map:
            for paths in ordered_map.values():
                for p in paths:
                    if os.path.isfile(p):
                        seen[p] = None

        if seen:
            return sorted(seen.keys())
    except Exception as exc:
        logger.debug(f"images_summary 기반 이미지 목록 활용 실패: {exc}")
    return sorted(_iter_images(root))

# ===== 캐싱: 베이스네임 → 경로 맵 (탐색/해결용) =====
@st.cache_data(show_spinner=False)
def build_basename_map(root: str, cache_buster: float = 0) -> Dict[str, str]:
    """Build basename->path map. If the same filename exists in multiple locations,
    the first discovered path is used. Priority is no longer special-cased for
    grouped/ok/blank_answers because the pipeline no longer creates those folders.
    """
    # cache_buster는 파일 변경 시 호출자가 강제 재계산하도록 허용합니다
    _ = cache_buster
    imgs = list_all_images(root, cache_buster=cache_buster)
    def pri(p: str) -> int:
        # Keep a deterministic pseudo-priority based on path depth and lexicographic order
        low = p.replace("\\", "/").lower()
        # shorter paths (closer to root) get higher priority
        try:
            parts = low.count("/")
        except Exception:
            parts = 0
        return parts
    best: Dict[str, str] = {}
    best_pri: Dict[str, int] = {}
    for p in imgs:
        bn = os.path.basename(p).lower()
        rank = pri(p)
        if (bn not in best) or (rank < best_pri[bn]):
            best[bn] = p
            best_pri[bn] = rank
    return best

SEARCH_CACHE: Dict[str, Optional[str]] = {}


def clear_caches_and_state(session_keys: Optional[List[str]] = None) -> None:
    """안전한 캐시 및 세션 상태 초기화 유틸리티

    - 최신 Streamlit API(`st.cache_data.clear`)를 우선 사용하고, 실패하면
      `st.experimental_memo.clear`로 폴백합니다.
    - 개별 데코레이터 기반 캐시 함수(list_all_images 등)에 대해 `.clear()`가
      가능하면 시도합니다.
    - session_keys에 명시된 세션 키들(예: ['gallery_selected'])을 제거합니다.
    모든 예외는 흘려보내어 대시보드가 중단되지 않도록 설계했습니다.
    """
    try:
        # 최신 Streamlit 전역 캐시 우선 삭제
        if hasattr(st, "cache_data") and callable(getattr(st.cache_data, "clear", None)):
            try:
                st.cache_data.clear()
            except Exception:
                # 내부 구현 차이로 실패할 수 있음; 폴백으로 계속 진행
                pass
    except Exception:
        pass

    try:
        # 구버전 Streamlit 호환
        if hasattr(st, "experimental_memo") and callable(getattr(st.experimental_memo, "clear", None)):
            try:
                st.experimental_memo.clear()
            except Exception:
                pass
    except Exception:
        pass

    # 개별 함수별 clear() 시도 (존재하면 안전하게 호출)
    for _fn in (globals().get('load_img_summary'), globals().get('list_all_images'), globals().get('build_basename_map')):
        try:
            if _fn and hasattr(_fn, 'clear') and callable(getattr(_fn, 'clear')):
                try:
                    _fn.clear()
                except Exception:
                    pass
        except Exception:
            pass

    # SEARCH_CACHE 등 모듈 레벨 캐시 초기화
    try:
        SEARCH_CACHE.clear()
    except Exception:
        pass

    # 세션 상태 키 제거(옵션)
    if session_keys:
        for k in session_keys:
            try:
                if k in st.session_state:
                    del st.session_state[k]
            except Exception:
                pass

    # 일반적인 UI 관련 키 제거(안전하게 시도)
    for k in ("gallery_limit", "gallery_selected"):
        try:
            if k in st.session_state:
                del st.session_state[k]
        except Exception:
            pass

if dir_changed:
    # 선택한 결과 폴더가 바뀐 경우, 안전한 유틸리티를 통해 캐시와 관련 세션 상태를 초기화합니다.
    # 상세 무효화 로직은 `clear_caches_and_state`에 위임됩니다.
    try:
        clear_caches_and_state(session_keys=["gallery_selected", "gallery_limit"])
    except Exception:
        # 극단적 예외가 발생하면 최소한 기본 상태만 초기화
        try:
            st.session_state.gallery_limit = 120
        except Exception:
            pass
        try:
            st.session_state.gallery_selected = []
        except Exception:
            pass
        try:
            SEARCH_CACHE.clear()
        except Exception:
            pass

BASENAME_MAP = build_basename_map(str(OUTPUT_DIR))


def _candidate_roots() -> List[str]:
    """이미지를 탐색할 후보 루트 디렉터리를 우선순위 순으로 반환합니다."""
    raw: List[Optional[Path]] = [
        OUTPUT_DIR,
        BASE_OUTPUT_DIR,
        CLI_BASE_DIR,
        CLI_OUTPUT_DIR,
    ]
    if SELECTION_ROOT is not None:
        raw.append(SELECTION_ROOT)
    try:
        parent = OUTPUT_DIR.parent
        if parent != OUTPUT_DIR:
            raw.append(parent)
    except Exception:
        pass

    roots: List[str] = []
    seen: set[str] = set()
    for cand in raw:
        if cand is None:
            continue
        try:
            cand_str = str(cand)
        except Exception:
            continue
        try:
            abs_path = os.path.abspath(cand_str)
        except Exception:
            continue
        if abs_path in seen:
            continue
        if os.path.isdir(abs_path):
            seen.add(abs_path)
            roots.append(abs_path)
    return roots


def _normalize_lookup_key(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    normalized = re.sub(r"/+", "/", normalized)
    return f"{OUTPUT_DIR}|{normalized.lower()}"


def _search_basename_in_root(root: str, bn_lower: str, max_depth: int = 6) -> Optional[str]:
    if not bn_lower:
        return None
    try:
        root_path = Path(root)
    except Exception:
        return None
    if not root_path.exists():
        return None

    skip_dirs = {"disp_cache", "__pycache__", ".git", ".venv"}
    try:
        for current_root, dirnames, filenames in os.walk(root):
            try:
                depth = len(Path(current_root).relative_to(root_path).parts)
            except Exception:
                depth = 0
            if depth > max_depth:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fname in filenames:
                if fname.lower() == bn_lower:
                    return os.path.normpath(os.path.join(current_root, fname))
    except Exception as exc:
        logger.debug(f"베이스네임 검색 실패: {root} / {bn_lower}: {exc}")
    return None


def resolve_image_path(name_or_path: str) -> Optional[str]:
    """
    - 절대/상대 경로가 유효하면 그대로 사용
    - 아니면 OUTPUT_DIR 하위에서 파일명으로 검색(베이스네임 맵 사용)
    """
    if not name_or_path:
        return None
    raw = name_or_path.strip()
    if not raw:
        return None

    def _cache_and_return(cache_key: str, value: Optional[str]) -> Optional[str]:
        SEARCH_CACHE[cache_key] = value
        return value

    candidate_roots: Optional[List[str]] = None

    def _roots() -> List[str]:
        nonlocal candidate_roots
        if candidate_roots is None:
            candidate_roots = _candidate_roots()
        return candidate_roots

    # 직접 경로 우선 검사
    try:
        expanded = os.path.expanduser(raw)
    except Exception:
        expanded = raw
    if os.path.isfile(expanded):
        normed = os.path.normpath(expanded)
        return _cache_and_return(_normalize_lookup_key(raw), normed)

    rel = os.path.normpath(os.path.join(str(OUTPUT_DIR), raw))
    if os.path.isfile(rel):
        return _cache_and_return(_normalize_lookup_key(raw), rel)

    cache_key = _normalize_lookup_key(raw)
    if cache_key in SEARCH_CACHE:
        return SEARCH_CACHE[cache_key]

    bn = os.path.basename(raw).lower()

    if bn:
        mapped = BASENAME_MAP.get(bn)
        if mapped and os.path.isfile(mapped):
            return _cache_and_return(cache_key, mapped)

    # 상대 경로를 각 후보 루트와 결합해 확인
    if not os.path.isabs(raw):
        sanitized = raw.lstrip("./\\")
        for root in _roots():
            joined = os.path.normpath(os.path.join(root, sanitized))
            if os.path.isfile(joined):
                return _cache_and_return(cache_key, joined)

    if bn:
        try:
            input_map = load_input_basename_map()
            for cand in input_map.get(bn, []):
                if os.path.isfile(cand):
                    return _cache_and_return(cache_key, os.path.normpath(cand))
        except Exception:
            pass

        for root in _roots():
            root_key = f"root::{root}|{bn}"
            if root_key in SEARCH_CACHE:
                cached = SEARCH_CACHE[root_key]
                if cached:
                    return _cache_and_return(cache_key, cached)
                continue
            found = _search_basename_in_root(root, bn)
            SEARCH_CACHE[root_key] = found
            if found:
                return _cache_and_return(cache_key, found)

    return _cache_and_return(cache_key, None)


# ===== 번호/앞뒷장 유틸 =====
def _extract_first_number(s: str):
    """문자열에서 첫 번째 연속 숫자 그룹을 찾아 (숫자문자열, 정수값, start, end) 반환하거나 None 반환."""
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    ns = m.group(1)
    return ns, int(ns), m.start(1), m.end(1)

def corresponding_front_filename(fname: str) -> str:
    """파일명에서 숫자를 찾아 짝수이면 -1(앞면), 홀수면 그대로로 대응하는 앞면 파일명을 생성하여 반환."""
    rec = _extract_first_number(fname)
    if not rec:
        return fname
    ns, n, sidx, eidx = rec
    if n % 2 == 0:
        front_n = n - 1
    else:
        front_n = n
    front_ns = str(front_n).zfill(len(ns))
    return fname[:sidx] + front_ns + fname[eidx:]


# 안전한 2파일(뒷장) 판별기: 파일명이 충분히 길지 않거나 예상 포맷이 아닐 때 False 반환
def is_2file(filename: str) -> bool:
    try:
        if not filename or not isinstance(filename, str):
            return False
        # 확장자 포함 길이가 최소 5 이상이어야(예: 1.jpg) 뒤에서 5번째 문자가 존재
        return filename.lower().endswith('.jpg') and len(filename) >= 5 and filename[-5] == '2'
    except Exception:
        return False

# ===== 캐싱: 표시용(썸네일/대형) 이미지 생성 =====
@st.cache_data(show_spinner=False)
def _disp_key(src_path: str, size: int, fmt: str, quality: int) -> str:
    stat = f"{src_path}|{_file_mtime(src_path)}|{size}|{fmt}|{quality}"
    return hashlib.md5(stat.encode("utf-8")).hexdigest()[:16]

def make_display_image(src_path: str, size: int, fmt: str = "WEBP", quality: int = 95) -> str:
    """긴 변을 기준으로 size(px)까지 축소한 표시용 이미지를 캐시에 생성/재사용합니다.
    LANCZOS 리샘플링을 사용하며 포맷은 WEBP/JPEG/PNG를 지원합니다.
    quality는 JPEG/WEBP에 적용됩니다. 반환값은 캐시된 파일 경로입니다. 
    """
    fmt = fmt.upper()
    ext_map = {"WEBP": "webp", "JPEG": "jpg", "PNG": "png"}
    ext = ext_map.get(fmt, "webp")
    key = _disp_key(src_path, size, fmt, quality)
    cache_sub = os.path.join(THUMB_DIR, "disp_cache")
    os.makedirs(cache_sub, exist_ok=True)
    dst = os.path.join(cache_sub, f"{key}.{ext}")

    try:
        if not os.path.exists(dst) or _file_mtime(dst) < _file_mtime(src_path):
            img = _safe_image_open(src_path).convert("RGB")
            w, h = img.size
            if max(w, h) > size:
                if w >= h:
                    new_w = size
                    new_h = max(1, int(h * (size / w)))
                else:
                    new_h = size
                    new_w = max(1, int(w * (size / h)))
                img = img.resize((new_w, new_h), RESAMPLE)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if fmt == "PNG":
                img.save(dst, "PNG", optimize=True)
            elif fmt == "JPEG":
                img.save(dst, "JPEG", quality=quality, optimize=True, progressive=True)
            else:  # WEBP
                img.save(dst, "WEBP", quality=quality, method=6)
    except Exception as e:
        # 실패 시 원본 경로를 그대로 반환하는 대신, 사용자에게 일관된
        # 표시 결과를 제공하기 위해 플레이스홀더 이미지를 생성하여
        # 캐시(dst)에 저장하고 그 경로를 반환합니다.
        logger.debug(f"make_display_image 실패: {src_path} -> {dst}: {e}")
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            w, h = (min(1024, size), int(min(1024, size) * 0.75))
            ph = Image.new("RGB", (w, h), (240, 240, 240))
            draw = ImageDraw.Draw(ph)
            basename = os.path.basename(src_path) if src_path else "unknown"
            # 한국어 주석: 플레이스홀더 텍스트를 중앙에 표시
            txt = f"이미지 없음\n{basename}"
            try:
                # 텍스트 위치를 중앙으로 계산
                tw, th = draw.textsize(txt)
                draw.text(((w - tw) / 2, (h - th) / 2), txt, fill=(100, 100, 100))
            except Exception:
                # 일부 환경에서 textsize가 작동하지 않을 수 있으므로 단순히 왼쪽 상단에 표시
                draw.text((8, 8), txt, fill=(100, 100, 100))
            ph.save(dst, "PNG")
            return dst
        except Exception:
            # 플레이스홀더 생성도 실패하면 안전하게 원본 src_path로 폴백
            return src_path
    return dst

# ===== KPI 계산 =====
def compute_kpis(df: pd.DataFrame, img_df: pd.DataFrame) -> Dict[str, int]:
    kpis = {"총 이미지": 0, "그룹 수": 0, "공백 수": 0, "유사 후보 쌍": 0}

    # 총 이미지
    try:
        kpis["총 이미지"] = int(len(img_df)) if hasattr(img_df, '__len__') else 0
    except Exception:
        pass

    # 그룹 수 (report의 그룹ID 컬럼, '-'은 무시)
    try:
        if isinstance(df, pd.DataFrame) and '그룹ID' in df.columns:
            kpis["그룹 수"] = int(df['그룹ID'].replace('-', pd.NA).dropna().nunique())
    except Exception:
        pass

    # 공백 수: 이미지 요약 또는 리포트에서 파생 (뒷장 파일 기준)
    try:
        if isinstance(img_df, pd.DataFrame) and '빈칸여부' in img_df.columns:
            blanks_mask = img_df['빈칸여부'].astype(bool)
            if '파일' in img_df.columns:
                file_series = img_df.loc[blanks_mask, '파일'].astype(str)
                filtered = [name for name in file_series if _is_back_page(name)]
                kpis["공백 수"] = len(filtered)
            else:
                kpis["공백 수"] = int(blanks_mask.sum())
        elif isinstance(df, pd.DataFrame) and '상태' in df.columns:
            blanks_mask = df['상태'].astype(str).str.contains('공백', na=False)
            if blanks_mask.any():
                names: List[str] = []
                for col in ("파일1", "파일2"):
                    if col in df.columns:
                        names.extend(df.loc[blanks_mask, col].astype(str).tolist())
                filtered = {name for name in names if _is_back_page(name)}
                kpis["공백 수"] = len(filtered)
            else:
                kpis["공백 수"] = 0
    except Exception:
        pass

    # 유사 후보 쌍: report(df)의 '상태' 컬럼에서 '유사 후보'로 표기된 행 수 (있을 경우)
    try:
        if isinstance(df, pd.DataFrame) and '상태' in df.columns:
            kpis["유사 후보 쌍"] = int((df['상태'].astype(str) == '유사 후보').sum())
    except Exception:
        pass

    return kpis

# ===== 데이터 로딩 =====
report_cache_token = (_file_mtime(REPORT_PARQUET), _file_mtime(REPORT_CSV))
try:
    df = load_report(
        REPORT_PARQUET,
        REPORT_CSV,
        columns=REPORT_BASE_COLUMNS,
        cache_token=report_cache_token,
    )
    report_available = True
except FileNotFoundError:
    report_available = False
    df = pd.DataFrame(columns=REPORT_BASE_COLUMNS)
    st.info("리포트 파일이 아직 생성되지 않았습니다. 파이프라인 실행 후 다시 선택하세요.")

img_df = load_img_summary(IMG_SUMMARY, cache_buster=_file_mtime(IMG_SUMMARY))

# ===== KPI 카드 (디자이너 스타일) =====
# 기존 기능(숫자 계산)은 유지하되, 디자인 요구에 맞춰
# HTML 기반의 카드형 KPI 레이아웃을 추가합니다.
# - 기능(값 계산)은 변경하지 않음
# - 시각 표현만 CSS 클래스를 통해 제어
kpis = compute_kpis(df, img_df)

# KPI 카드를 HTML로 렌더(디자이너처럼 보이도록 CSS 클래스 사용)
try:
        # 디자인 목적으로 표시할 델타(증감) 값은 선택적으로 세션 상태에서 가져옵니다.
        # 실제 KPI 계산에는 영향을 주지 않습니다.
        deltas = st.session_state.get('kpi_deltas', {}) if isinstance(st.session_state.get('kpi_deltas', {}), dict) else {}
        d_tot = deltas.get('총 이미지', '')
        d_grp = deltas.get('그룹 수', '')
        d_blank = deltas.get('공백 수', '')
        d_sim = deltas.get('유사 후보 쌍', '')

        kpi_html = f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-icon">🖼️</div>
                <div class="kpi-label">총 이미지</div>
                <div class="kpi-value">{kpis.get('총 이미지', 0):,}</div>
                <div class="kpi-delta">{d_tot}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">👥</div>
                <div class="kpi-label">그룹 수</div>
                <div class="kpi-value">{kpis.get('그룹 수', 0):,}</div>
                <div class="kpi-delta">{d_grp}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">📄</div>
                <div class="kpi-label">공백 수</div>
                <div class="kpi-value">{kpis.get('공백 수', 0):,}</div>
                <div class="kpi-delta">{d_blank}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🔎</div>
                <div class="kpi-label">유사 후보 쌍</div>
                <div class="kpi-value">{kpis.get('유사 후보 쌍', 0):,}</div>
                <div class="kpi-delta">{d_sim}</div>
            </div>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)
except Exception:
        # 스트림릿 환경에 따라 HTML 삽입이 실패할 수 있으므로
        # 안전하게 기존 metric 위젯으로 폴백합니다.
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 이미지", f"{kpis['총 이미지']:,}")
        c2.metric("그룹 수", f"{kpis['그룹 수']:,}")
        c3.metric("공백 수", f"{kpis['공백 수']:,}")


# ===== 사이드바: 간결한 렌더러로 분리(가독성 개선) =====
# 사이드바 관련 UI 블록을 작은 함수로 분리하여 본문은 호출만 하도록 했습니다.

def _render_sidebar_checks():
    """사이드바 상단의 결과 파일/아티팩트 존재 여부 요약 표시"""
    try:
        missing = []
        checks = [(REPORT_PARQUET, 'report.parquet'), (REPORT_CSV, 'report.csv'), (IMG_SUMMARY, 'images_summary.csv')]
        for p, name in checks:
            if not os.path.exists(p):
                missing.append(name)
        art_txt = os.path.join(OUTPUT_DIR, 'artifacts', 'ann_backend.txt')
        # 필수가 아닌 결과물은 경고하지 않음
        if missing:
            st.sidebar.warning("결과 파일 누락: " + ", ".join(missing) + ". 먼저 파이프라인을 실행하세요.")
    except Exception:
        # UI 보조 정보 실패는 무시
        pass

_inject_theme_css(st.session_state.get('theme', DEFAULT_THEME_KEY))

group_filter = st.session_state.get("group_filter", "전체")
grid_cols = int(st.session_state.get("grid_cols", 5))
group_view_mode = st.session_state.get("group_view_mode", "그리드(다중 썸네일)")
disp_fmt = "WEBP"    # 고정 포맷

rescan_quality_profile = st.session_state.get("rescan_quality_profile", "균형")
if rescan_quality_profile == "빠름":
    rescan_thumb_px, rescan_large_px, rescan_disp_quality = 512, 1200, 92
elif rescan_quality_profile == "선명":
    rescan_thumb_px, rescan_large_px, rescan_disp_quality = 1600, 2000, 98
else:  # 균형
    rescan_thumb_px, rescan_large_px, rescan_disp_quality = 1024, 1600, 95

ok_quality_profile = st.session_state.get("ok_quality_profile", "균형")
if ok_quality_profile == "빠름":
    ok_thumb_px, ok_disp_quality = 512, 92
elif ok_quality_profile == "선명":
    ok_thumb_px, ok_disp_quality = 1600, 98
else:
    ok_thumb_px, ok_disp_quality = 1024, 95

group_page_size = 6   # 고정값
group_page = 1        # 고정값(페이지네이션은 필요시만)

# ===== 유틸: 비교용 도구 =====
def _cv2_read_unicode(path: str, flag: int) -> Optional[np.ndarray]:
    """cv2.imread는 Windows에서 유니코드 경로를 처리하지 못할 수 있으므로 안전한 대안을 제공합니다."""
    if not path:
        return None
    arr = cv2.imread(path, flag)
    if arr is not None:
        return arr
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except Exception:
        return None
    if data.size == 0:
        return None
    try:
        return cv2.imdecode(data, flag)
    except Exception:
        return None


def _read_gray_same_size(a_path: str, b_path: str) -> Tuple[np.ndarray, np.ndarray]:
    a = _cv2_read_unicode(a_path, cv2.IMREAD_GRAYSCALE)
    b = _cv2_read_unicode(b_path, cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise RuntimeError(f"이미지 로드 실패: {a_path}")
    if b is None:
        raise RuntimeError(f"이미지 로드 실패: {b_path}")
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return a, b

def _absdiff_heatmap(a: np.ndarray, b: np.ndarray, blur_size: int = 3, threshold: int = 10) -> np.ndarray:
    """두 그레이스케일 이미지의 절대차이를 heatmap으로 변환합니다.
    
    Args:
        a, b: 그레이스케일 이미지 배열
        blur_size: 가우시안 블러 크기 (홀수, 기본값 3)
        threshold: 차이 임계값 (기본값 10)
    
    Returns:
        RGB heatmap 배열
    """
    diff = cv2.absdiff(a, b)
    
    # 임계값 적용하여 노이즈 제거
    _, diff_thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # 블러 적용 (홀수만 허용)
    if blur_size > 1:
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        diff_thresh = cv2.GaussianBlur(diff_thresh, (blur_size, blur_size), 0)
    
    # 정규화 및 컬러맵 적용
    diff_norm = cv2.normalize(diff_thresh, None, 0, 255, cv2.NORM_MINMAX)
    heat = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _blend_images_rgb(a_path: str, b_path: str, alpha: float = 0.5) -> np.ndarray:
    """두 이미지를 읽어 공통 최소 크기로 리사이즈한 뒤 RGB로 블렌드하여 numpy 배열을 반환합니다.
    alpha는 첫 번째 이미지(a)의 가중치(0..1)입니다."""
    a = _cv2_read_unicode(a_path, cv2.IMREAD_COLOR)
    b = _cv2_read_unicode(b_path, cv2.IMREAD_COLOR)
    if a is None:
        raise RuntimeError(f"이미지 로드 실패: {a_path}")
    if b is None:
        raise RuntimeError(f"이미지 로드 실패: {b_path}")
    # 최소 공통 크기로 리사이즈합니다
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    out = cv2.addWeighted(a.astype('float32'), alpha, b.astype('float32'), 1.0 - alpha, 0.0)
    out = out.astype('uint8')
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _highlight_differences_rgb(a_path: str, b_path: str, color: Tuple[int, int, int] = (0, 255, 255), thresh: int = 20) -> np.ndarray:
    """절대 차이가 thresh보다 큰 영역에 색 마스크를 오버레이해 변경점을 강조한 RGB 배열을 반환합니다.
    color는 OpenCV(BGR) 형식으로 전달하되, 반환값은 RGB입니다."""
    a = _cv2_read_unicode(a_path, cv2.IMREAD_COLOR)
    b = _cv2_read_unicode(b_path, cv2.IMREAD_COLOR)
    if a is None:
        raise RuntimeError(f"이미지 로드 실패: {a_path}")
    if b is None:
        raise RuntimeError(f"이미지 로드 실패: {b_path}")
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_a, gray_b)
    # 노이즈를 줄이기 위해 약간의 블러를 적용합니다
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # color는 OpenCV의 BGR 형식을 기대합니다
    overlay = (mask3.astype('float32') / 255.0) * np.array(color, dtype='float32')
    base = cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype('float32')
    # 결합: 마스크가 있는 곳에 오버레이 색상을 베이스와 혼합합니다
    alpha = 0.6
    combined = base * (1.0 - (mask3.astype('float32') / 255.0) * alpha) + overlay * alpha
    combined = np.clip(combined, 0, 255).astype('uint8')
    return combined


def _comp_cache_key(a_path: str, b_path: str, mode: str, params: Dict) -> str:
    # 비교 결과를 캐시하기 위한 고유 키(경로, 수정시간, 모드, 파라미터 포함)를 생성합니다
    s = f"{a_path}|{_file_mtime(a_path)}|{b_path}|{_file_mtime(b_path)}|{mode}|{sorted(params.items())}"
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def _write_cached_image(arr_rgb: np.ndarray, dst: str, fmt: str = 'PNG') -> Optional[str]:
    # numpy RGB 배열을 이미지 파일로 저장해 캐시에 보관합니다. 실패 시 None 반환
    try:
        img = Image.fromarray(arr_rgb)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        img.save(dst, fmt)
        return dst
    except Exception as e:
        logger.warning(f"캐시 이미지 쓰기 실패 {dst}: {e}")
        return None


def _cached_blend_path(a_path: str, b_path: str, alpha: float = 0.5) -> Optional[str]:
    key = _comp_cache_key(a_path, b_path, 'fade', {'alpha': alpha})
    dst = os.path.join(THUMB_DIR, f"cmp_fade_{key}.png")
    if os.path.exists(dst) and _file_mtime(dst) >= max(_file_mtime(a_path), _file_mtime(b_path)):
        return dst
    arr = _blend_images_rgb(a_path, b_path, alpha=alpha)
    return _write_cached_image(arr, dst, fmt='PNG')


def _cached_heatmap_path(a_path: str, b_path: str, blur_size: int = 3, threshold: int = 10) -> Optional[str]:
    """차이 heatmap을 캐시에 생성하고 경로를 반환합니다."""
    key = _comp_cache_key(a_path, b_path, 'heatmap', {'blur': blur_size, 'thresh': threshold})
    dst = os.path.join(THUMB_DIR, f"cmp_heatmap_{key}.png")
    if os.path.exists(dst) and _file_mtime(dst) >= max(_file_mtime(a_path), _file_mtime(b_path)):
        return dst
    try:
        a_gray, b_gray = _read_gray_same_size(a_path, b_path)
        arr = _absdiff_heatmap(a_gray, b_gray, blur_size=blur_size, threshold=threshold)
        return _write_cached_image(arr, dst, fmt='PNG')
    except Exception as e:
        logger.warning(f"heatmap 캐시 생성 실패 {a_path} vs {b_path}: {e}")
        return None


def _cached_highlight_path(a_path: str, b_path: str, color: Tuple[int, int, int] = (0, 255, 255), thresh: int = 20) -> Optional[str]:
    key = _comp_cache_key(a_path, b_path, 'hl', {'color': color, 'thresh': thresh})
    dst = os.path.join(THUMB_DIR, f"cmp_hl_{key}.png")
    if os.path.exists(dst) and _file_mtime(dst) >= max(_file_mtime(a_path), _file_mtime(b_path)):
        return dst
    arr = _highlight_differences_rgb(a_path, b_path, color=color, thresh=thresh)
    return _write_cached_image(arr, dst, fmt='PNG')


# GIF 생성 지원 제거: Fade는 이제 _cached_blend_path/_blend_images_rgb의 정적 블렌드만 사용합니다

# ===== 공통: 리포트 필터링 =====
def filter_sort_report(_df: pd.DataFrame) -> pd.DataFrame:
    view = _df.copy()
    if group_filter != "전체" and "그룹ID" in view.columns:
        view = view[view["그룹ID"] == group_filter]
    # 기본 정렬: 유사도 내림차순, 그 다음 파일명
    if "유사도" in view.columns:
        view = view.sort_values(["유사도", "파일1", "파일2"], ascending=[False, True, True])
    else:
        view = view.sort_values(["파일1", "파일2"])
    return view

# ===== 세션: 비교 큐 =====
# 통합된 비교 선택 상태: 절대 경로 리스트 (최대 2개)
def toggle_compare(img_path: str):
    """비교를 위한 선택 토글 기능. 선택된 항목의 절대 경로를 $\text{gallery_selected}$에 저장합니다. (최대 2개)."""
    if not img_path:
        return
    # 가능한 경우 절대 경로로 정규화
    path = img_path
    if not os.path.isfile(path):
    # 베이스네임 맵으로 경로 해결을 시도
        resolved = resolve_image_path(path)
        if resolved and os.path.isfile(resolved):
            path = resolved
    # 이미 존재하면 선택 해제
    if path in st.session_state["gallery_selected"]:
        st.session_state["gallery_selected"] = [p for p in st.session_state["gallery_selected"] if p != path]
        return
    # 추가하되 최신 2개만 유지
    if len(st.session_state["gallery_selected"]) >= 2:
        st.session_state["gallery_selected"] = st.session_state["gallery_selected"][1:] + [path]
    else:
        st.session_state["gallery_selected"].append(path)
def refresh_image_caches():
    """파일 삭제 후 이미지 관련 캐시와 경로 맵을 새로고침합니다."""
    global BASENAME_MAP
    try:
        list_all_images.clear()
    except AttributeError:
        pass
    try:
        build_basename_map.clear()
    except AttributeError:
        pass
    try:
        SEARCH_CACHE.clear()
    except Exception:
        pass
    try:
        BASENAME_MAP = build_basename_map(str(OUTPUT_DIR), cache_buster=time.time())
    except Exception as exc:
        logger.debug(f"BASENAME_MAP 갱신 실패: {exc}")


@st.cache_data(show_spinner=False)
def load_input_basename_map() -> Dict[str, List[str]]:
    """artifacts/ordered_paths.txt가 있으면 입력 폴더 경로들을 읽어
    베이스네임(소문자) -> 원본 절대 경로 목록으로 매핑합니다.
    중복 파일명은 모두 포함합니다.
    """
    mapping: Dict[str, List[str]] = {}
    try:
        ordered_txt = os.path.join(OUTPUT_DIR, "artifacts", "ordered_paths.txt")
        if not os.path.isfile(ordered_txt):
            return mapping
        with open(ordered_txt, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                p = line.strip().strip('\"')
                if not p:
                    continue
                # 경로가 실제 존재하는 경우만 포함
                if os.path.isabs(p) and os.path.exists(p):
                    bn = os.path.basename(p).lower()
                    mapping.setdefault(bn, []).append(p)
    except Exception as exc:
        logger.debug(f"입력 경로 맵 로드 실패: {exc}")
    return mapping


def _all_output_paths_by_basename(bn_lower: str) -> List[str]:
    """OUTPUT_DIR 하위에서 주어진 베이스네임과 일치하는 모든 파일 경로를 찾아 반환합니다."""
    try:
        # images_summary 또는 파일시스템에서 전체 목록을 가져온 뒤 필터
        all_imgs = list_all_images(OUTPUT_DIR, cache_buster=time.time())
        root_abs = os.path.abspath(str(OUTPUT_DIR))
        results = []
        for p in all_imgs:
            if os.path.basename(p).lower() != bn_lower:
                continue
            try:
                if os.path.commonpath([root_abs, os.path.abspath(p)]) == root_abs:
                    results.append(p)
            except Exception:
                continue
        return results
    except Exception:
        # 폴백: 실패 시 빈 목록 반환 (기본 경로 스캔에 의존)
        return []


# (삭제) 파일 강제 삭제 헬퍼는 삭제 워크플로 제거로 더 이상 사용되지 않습니다.

# 이미지를 base64로 인코딩하거나 카드 스타일을 생성하는 헬퍼들은
# 대시보드 UI의 특정 동적 기능을 위해 존재했습니다. 소규모 파이프라인용으로는
# 이러한 커스텀 CSS/인코딩 도우미를 제거하여 의존성/복잡도를 낮춥니다.

# @st.cache_data(show_spinner=False)
# def _encode_image_base64(img_path: str) -> str:
#     with open(img_path, "rb") as fh:
#         return base64.b64encode(fh.read()).decode("utf-8")
#
# def _delete_card_css(button_key: str, img_base64: str, selected: bool, height: int, disabled: bool) -> str:
#     # (삭제) 카드 스타일 생성용 복잡한 CSS를 여기서 생성하던 코드입니다.
#     # 소규모 UI에서는 기본 Streamlit 버튼/이미지 구성만으로 충분하다고 판단하여 제거했습니다.
#     return ""
# (삭제됨) 이전의 _delete_card_css에서 생성하던 복잡한 CSS 블록을 제거했습니다.
# 소규모 대시보드에서는 Streamlit의 기본 마크업/스타일로 충분하므로
# 사용자 정의 CSS를 대폭 줄였습니다.


def render_rescan_image_card(img_path: str, caption: str, key_suffix: str, target_px: int, quality: int, card_height: int) -> None:
    """재스캔 탭에서 이미지 카드를 렌더링합니다.
    삭제 모드일 때는 클릭 가능한 선택 카드로 표시하고,
    일반 모드일 때는 기본 이미지로 표시합니다.
    """
    if not img_path or not os.path.isfile(img_path):
        st.warning(f"이미지 파일을 찾을 수 없습니다: {caption}")
        return

    display_path = make_display_image(img_path, size=target_px, fmt=disp_fmt, quality=quality)
    # 공용 래퍼 클래스: 간단한 카드 스타일 적용
    selected = img_path in st.session_state.get("gallery_selected", [])
    wrapper_class = f"thumb-card{' selected' if selected else ''}"

    st.markdown(f'<div class="{wrapper_class}" style="padding:8px; border-radius:12px;">', unsafe_allow_html=True)
    st.image(_safe_image_open(display_path), caption=caption, use_container_width=True)

    # 비교 선택 버튼 (삭제 기능 제거)
    button_label = "선택됨" if selected else "선택"
    button_key = f"cmp_rescan_card_{key_suffix}"
    if st.button(button_label, key=button_key, use_container_width=True, type=("primary" if selected else "secondary")):
        toggle_compare(img_path)
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

def on_image_click(img_path: str):
    """이미지를 클릭했을 때 비교 선택을 처리합니다."""
    toggle_compare(img_path)

# 이미지 클릭 이벤트를 처리하는 UI 요소에 on_image_click 함수를 연결합니다.
# 예를 들어, Streamlit의 st.image()를 사용할 경우:
# st.image(image_path, on_click=on_image_click, args=(image_path,))

# ===== 메인 탭 구성 (커스텀 탭 버튼으로 완전 제어) =====
tab_names = ["재스캔 필요", "정상/공백 답안", "전체 보기"]

# 탭 스타일 CSS
tab_css = """
<style>
.custom-tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 20px;
    border-bottom: 2px solid #e6e9ee;
    padding-bottom: 0;
}
.custom-tab {
    padding: 12px 24px;
    background: transparent;
    border: none;
    border-bottom: 3px solid transparent;
    cursor: pointer;
    font-size: 16px;
    font-weight: 500;
    color: #666;
    transition: all 0.3s ease;
    margin-bottom: -2px;
}
.custom-tab:hover {
    color: #1EA3A1;
    background: rgba(30, 163, 161, 0.05);
}
.custom-tab.active {
    color: #1EA3A1;
    border-bottom-color: #1EA3A1;
    font-weight: 600;
}
</style>
"""
st.markdown(tab_css, unsafe_allow_html=True)

# (기존 위치의 상단 헤더는 파일 상단으로 이동되어 중복이 제거되었습니다)

# 탭 버튼 UI
cols = st.columns(len(tab_names))
for idx, tab_name in enumerate(tab_names):
    with cols[idx]:
        is_active = st.session_state["main_tab"] == tab_name
        button_type = "primary" if is_active else "secondary"
        if st.button(
            tab_name,
            key=f"tab_btn_{idx}",
            use_container_width=True,
            type=button_type
        ):
            st.session_state["main_tab"] = tab_name
            st.rerun()

st.markdown("---")

# ===== Global: 탭 어디에서든 2장 선택 시 상단에 즉시 비교 패널 표시 =====
def _render_global_compare():
    # gallery_selected는 절대 경로 리스트로 통일되어야 함
    paths = st.session_state.get('gallery_selected', [])
    sel_paths = [p for p in paths if p and os.path.isfile(p)]
    if len(sel_paths) >= 2:
        return sel_paths[:2]
    return None


cmp_pair = _render_global_compare()

    



# Tab1(리포트 요약) 관련 UI 블록은 현재 사용되지 않아 제거했습니다.
# 필요하면 향후 탭을 다시 추가하여 활성화할 수 있습니다.


# === Tab: 재스캔 필요 ===
if st.session_state["main_tab"] == "재스캔 필요":
    # (삭제) 재스캔 탭의 파일 삭제/확인 UI는 제거되어 단순한 후보 표시 및 비교만 제공합니다.

    try:
        import imagehash
        dup_pairs = []
        hashes = {}
        # 현재는 사용자가 선택한 폴더에서 직접 스캔하지 않고 
        # 출력 폴더의 결과만을 기반으로 분석합니다
        scan_files = []  # 빈 리스트로 유지 (직접 스캔 없음)
        for f in scan_files:
            # 이 루프는 실행되지 않습니다 (scan_files가 빈 리스트이므로)
            # p = os.path.join(OUTPUT_DIR, f)  # 참조용 주석
            try:
                # 이 코드는 scan_files가 비어있어서 실행되지 않습니다
                # 안전하게 열기 (읽기 실패 파일은 건너뜀)
                # with open(p, 'rb') as fh:
                #     img = Image.open(fh).convert('L')
                #     h = imagehash.phash(img)
                # hashes[f] = h
                pass
            except Exception:
                # 읽기 실패 파일은 무시
                continue
        fl = list(hashes.keys())
        for i in range(len(fl)):
            for j in range(i+1, len(fl)):
                a, b = fl[i], fl[j]
                try:
                    d = abs(hashes[a] - hashes[b])
                except Exception:
                    continue
                # 매우 작은 해시 차이면 동일한 뒷장이 두 번 스캔된 것일 가능성 높음
                if d <= 1:
                    dup_pairs.append((a, b, int(d)))
    except Exception:
        dup_pairs = []

    # 재스캔 후보 생성: pHash 중복 쌍(input 폴더)과 리포트의 '유사 후보' 쌍을 합칩니다
    report_dups = []
    try:
        # df에는 파일명(리포트상)이 들어있음; 복원 가능한 경로로 변환
        if '상태' in df.columns:
            rpt = df[df['상태'] == '유사 후보'] if isinstance(df, pd.DataFrame) else pd.DataFrame()
            for _, row in rpt.iterrows():
                a, b = str(row.get('파일1', '')), str(row.get('파일2', ''))
                try:
                    sim = float(row.get('유사도', 0.0))
                except Exception:
                    sim = float(row.get('유사도', 0.0) or 0.0)
                pa = resolve_image_path(a) or os.path.join(OUTPUT_DIR, a)
                pb = resolve_image_path(b) or os.path.join(OUTPUT_DIR, b)
                report_dups.append((pa, pb, sim))
    except Exception:
        report_dups = []

    # 후보 병합 및 우선순위: pHash 쌍은 d가 작을수록 강한 신호이며, 리포트 쌍은 유사도(sim)가 클수록 우선
    candidates = []  # (score, a_path, b_path, source, meta) 튜플 목록
    try:
        for a, b, d in dup_pairs:
            pa = resolve_image_path(a) or os.path.join(OUTPUT_DIR, a)
            pb = resolve_image_path(b) or os.path.join(OUTPUT_DIR, b)
            # 점수: d가 작을수록 우선도 높음 -> 1/(1+d)로 반전
            score = 1.0 / (1.0 + float(d))
            candidates.append((score, pa, pb, 'phash', {'d': d}))
    except Exception:
        pass
    try:
        for pa, pb, sim in report_dups:
        # 리포트의 유사도는 이미 0..1 범위이므로 그대로 사용
            candidates.append((float(sim), pa, pb, 'report', {'sim': sim}))
    except Exception:
        pass

    # 점수 내림차순 정렬(값이 클수록 우선순위 높음)
    candidates = sorted([c for c in candidates if c[1] and c[2]], key=lambda x: x[0], reverse=True)

    # 베이스네임 쌍으로 중복 제거(순서 무시)
    seen = set()
    deduped = []
    for score, pa, pb, src, meta in candidates:
        key = tuple(sorted((os.path.basename(pa).lower(), os.path.basename(pb).lower())))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((score, pa, pb, src, meta))
    # ----- 즉시 비교 패널: 사용자가 아래 그리드에서 '↔ 비교 선택' 버튼을 클릭하면
    # rescan 탭의 상단에 바로 비교 옵션과 결과가 표시되도록 함
    # 통합된 gallery_selected(경로 리스트) 사용
    sel_exist_top = [p for p in st.session_state.get("gallery_selected", []) if p and os.path.isfile(p)]
    # 단순화된 즉시 비교: 복잡한 모드(페이드/하이라이터/Heatmap 등)를 제거하고
    # 항상 좌우 비교(또는 단일 이미지 대형 표시)만 제공합니다.
    st.markdown('<div class="float-compare">', unsafe_allow_html=True)
    # (제거됨) 재스캔 탭 내부에 중복으로 들어가던 그라데이션 선은 전역 위치에서만 렌더됩니다.
    st.markdown("### 🔍 즉시 비교 (재스캔 탭)")

    # 현재 선택된 이미지 정보 표시
    if len(sel_exist_top) == 1:
        st.info(f"📁 선택된 이미지: **{os.path.basename(sel_exist_top[0])}**")
    elif len(sel_exist_top) >= 2:
        a_name, b_name = os.path.basename(sel_exist_top[0]), os.path.basename(sel_exist_top[1])
        st.info(f"📁 비교 대상: **A**: {a_name} ↔ **B**: {b_name}")
        if len(sel_exist_top) > 2:
            st.caption(f" 추가로 {len(sel_exist_top)-2}개 이미지가 더 선택되어 있습니다. (최대 2개까지 비교)")

    # 간단한 클리어/해제 버튼
    if len(sel_exist_top) == 1:
        clear_single_col = st.columns([1, 9])
        with clear_single_col[0]:
            if st.button("🗑️ 해제", key="clear_single_top", help="선택한 이미지 해제"):
                st.session_state["gallery_selected"] = []
                st.rerun()

        bigp = make_display_image(sel_exist_top[0], size=max(1400, rescan_large_px), fmt=disp_fmt, quality=rescan_disp_quality)
        st.image(_safe_image_open(bigp), caption=os.path.basename(sel_exist_top[0]), use_container_width=True)
    elif len(sel_exist_top) >= 2:
        a_path, b_path = sel_exist_top[:2]
        # 선택 해제 버튼들
        clear_cols = st.columns([1, 1, 1, 7])
        with clear_cols[0]:
            if st.button("🗑️ A 해제", key="clear_a_top", help="첫 번째 선택 이미지 해제"):
                if a_path in st.session_state["gallery_selected"]:
                    st.session_state["gallery_selected"].remove(a_path)
                    st.rerun()
        with clear_cols[1]:
            if st.button("🗑️ B 해제", key="clear_b_top", help="두 번째 선택 이미지 해제"):
                if b_path in st.session_state["gallery_selected"]:
                    st.session_state["gallery_selected"].remove(b_path)
                    st.rerun()
        with clear_cols[2]:
            if st.button("🗑️ 전체 해제", key="clear_all_top", help="모든 선택 이미지 해제"):
                st.session_state["gallery_selected"] = []
                st.rerun()

        # 기본 좌우 비교(간단히 이미지 2장 나란히 표시)
        big_a = make_display_image(a_path, size=max(1600, rescan_large_px), fmt=disp_fmt, quality=rescan_disp_quality)
        big_b = make_display_image(b_path, size=max(1600, rescan_large_px), fmt=disp_fmt, quality=rescan_disp_quality)
        c1t, c2t = st.columns(2)
        with c1t:
            st.image(_safe_image_open(big_a), caption=f"A: {os.path.basename(a_path)}", use_container_width=True)
        with c2t:
            st.image(_safe_image_open(big_b), caption=f"B: {os.path.basename(b_path)}", use_container_width=True)

    # 비교 패널 래퍼 종료
    st.markdown('</div>', unsafe_allow_html=True)
    # 파이프라인이 grouped/ok/blank_answers 폴더를 생성하지 않으므로
    # 파일시스템의 grouped 디렉터리 전용 로직을 제거했습니다.
    # 대신 리포트(df)와 images_summary(img_df)를 기반으로 후보를 렌더링합니다.
    # 그룹이 report 기반으로만 존재하는 경우 아래 블록이 처리합니다.
    group_rows = pd.DataFrame()
    groups: List[str] = []
    if report_available and isinstance(df, pd.DataFrame):
        required_cols = {"그룹ID", "파일1", "파일2"}
        if required_cols.issubset(df.columns):
            group_rows = df[df["그룹ID"].astype(str).str.strip().ne("-")]
            try:
                groups = sorted(group_rows["그룹ID"].astype(str).unique().tolist())
            except Exception:
                groups = []

        if group_filter != "전체":
            groups = [g for g in groups if g == group_filter]

    if not groups:
        st.info("표시할 재스캔 후보 그룹이 없습니다.")
    else:
        view_mode = st.session_state.get("group_view_mode", "그리드(다중 썸네일)")
        per_row = 2 if view_mode == "대형 비교(2열)" else 4
        display_px = rescan_large_px if view_mode == "대형 비교(2열)" else rescan_thumb_px
        card_height = 320 if view_mode == "대형 비교(2열)" else 220

        for gid in groups:
            rows = group_rows[group_rows["그룹ID"].astype(str) == gid]
            if rows.empty:
                continue

            st.markdown(f"### 그룹 {gid}")

            file_candidates: set[str] = set()
            for _, row in rows.iterrows():
                for col in ("파일1", "파일2"):
                    val = row.get(col)
                    if isinstance(val, str) and val:
                        file_candidates.add(val)

            back_files = [f for f in sorted(file_candidates) if is_2file(f)]
            if not back_files:
                st.caption("표시 가능한 이미지가 없습니다.")
                continue

            # 각 뒷면 파일 단위로 앞/뒤 이미지를 묶어 뷰 모드에 맞게 렌더링합니다.
            grouped_items: List[List[Dict[str, str]]] = []
            for back_name in back_files:
                back_path = resolve_image_path(back_name)
                front_name = corresponding_front_filename(back_name)
                front_path = resolve_image_path(front_name)

                entries: List[Dict[str, str]] = []
                if front_path and os.path.exists(front_path):
                    entries.append({"kind": "앞면", "name": front_name, "path": front_path})
                if back_path and os.path.exists(back_path):
                    entries.append({"kind": "뒷면", "name": back_name, "path": back_path})

                if entries:
                    grouped_items.append(entries)

            if not grouped_items:
                st.caption("표시 가능한 이미지가 없습니다.")
                continue

            def _render_tile(entry: Dict[str, str], pair_idx: int, item_idx: int) -> None:
                key_stub = f"{gid}_{pair_idx}_{item_idx}"
                label = f"{entry['kind']}: {entry['name']}"
                path = entry["path"]

                # (삭제) 삭제 모드 관련 렌더링 제거: 항상 기본 타일/비교 선택 UI를 사용합니다.

                selected = path in st.session_state.get("gallery_selected", [])
                # 표시용 배지: 선택된 경우 A/B 순서를 캡션에 추가하여
                # 상단의 즉시 비교 패널과 문구가 섞이지 않도록 구분합니다.
                sel_list = st.session_state.get("gallery_selected", [])
                badge = ""
                try:
                    if path in sel_list:
                        pos = sel_list.index(path) + 1
                        badge = f" (선택 {'A' if pos == 1 else 'B' if pos == 2 else pos})"
                except Exception:
                    badge = ""

                disp = make_display_image(
                    path,
                    size=display_px,
                    fmt=disp_fmt,
                    quality=rescan_disp_quality,
                )
                st.image(
                    _safe_image_open(disp),
                    caption=label + badge,
                    use_container_width=True,
                )

                # 액션: 기존의 '비교 선택' 텍스트 대신 단순한 '선택' 버튼으로 라벨을 바꿔
                # 상단의 비교 패널과 혼동되지 않도록 합니다.
                button_label = "선택됨" if selected else "선택"
                button_key = f"cmp_rescan_{view_mode}_{key_stub}"
                st.markdown('<div class="thumb-action">', unsafe_allow_html=True)
                if st.button(button_label, key=button_key, use_container_width=True, type=("primary" if selected else "secondary")):
                    toggle_compare(path)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            if view_mode == "대형 비교(2열)":
                for pair_idx, bundle in enumerate(grouped_items):
                    cols = st.columns(min(per_row, len(bundle)))
                    for item_idx, entry in enumerate(bundle):
                        with cols[item_idx]:
                            _render_tile(entry, pair_idx, item_idx)
                st.markdown("")
            else:
                cols = st.columns(per_row)
                flat_entries = [
                    (entry, pair_idx, item_idx)
                    for pair_idx, bundle in enumerate(grouped_items)
                    for item_idx, entry in enumerate(bundle)
                ]
                for flat_idx, (entry, pair_idx, item_idx) in enumerate(flat_entries):
                    with cols[flat_idx % per_row]:
                        _render_tile(entry, pair_idx, item_idx)


# === Tab: 정상/공백 ===
elif st.session_state["main_tab"] == "정상/공백 답안":
    # 파일 시스템의 ok/blank_answers 하위 폴더에 의존하지 않고
    # images_summary(df)와 report(df)를 기반으로 정상/공백 뷰를 제공합니다.
    sel = st.session_state.get("ok_view_mode", "모두 보기")

    def _render_from_summary(title: str, mask: pd.Series) -> None:
        subset = img_df[mask] if isinstance(img_df, pd.DataFrame) else pd.DataFrame()
        st.subheader(title)
        if subset.empty or "파일" not in subset.columns:
            st.caption("표시할 이미지가 없습니다.")
            return
        files = [f for f in subset["파일"].astype(str).tolist() if is_2file(f) and _is_back_page(f)]
        if not files:
            st.caption("표시할 항목이 없습니다.")
            return
        cols = st.columns(grid_cols)
        for idx, f in enumerate(files):
            img_path = resolve_image_path(f)
            if not img_path or not os.path.exists(img_path):
                continue
            disp = make_display_image(img_path, size=ok_thumb_px, fmt=disp_fmt, quality=ok_disp_quality)
            with cols[idx % grid_cols]:
                st.image(_safe_image_open(disp), caption=f, use_container_width=True)

    if sel in ["모두 보기", "정상만"]:
        if "빈칸여부" in img_df.columns:
            mask = ~img_df["빈칸여부"].astype(bool)
        else:
            mask = pd.Series([False] * len(img_df), index=img_df.index)
        _render_from_summary("✅ 정상 답안", mask)

    if sel in ["모두 보기", "공백만"]:
        if "빈칸여부" in img_df.columns:
            mask = img_df["빈칸여부"].astype(bool)
        else:
            mask = pd.Series([False] * len(img_df), index=img_df.index)
        _render_from_summary("⭕ 공백 답안", mask)


# === Tab: 전체 보기 ===
elif st.session_state["main_tab"] == "전체 보기":
    quality_profile = st.session_state.get("gallery_quality_profile", "균형")
    render_mode = st.session_state.get("gallery_render_mode", "리샘플(권장)")
    grid_cols_local = max(2, int(st.session_state.get("gallery_grid_cols", 5)))
    q = st.session_state.get("gallery_search", "")
    ext_sel = st.session_state.get("gallery_exts", [])
    sort_key = st.session_state.get("gallery_sort", "파일명")

    if render_mode == "원본":
        st.caption("원본 모드: 대용량 이미지는 로딩 시간이 길어질 수 있습니다. 필요한 비교 구간에서만 사용하세요.")

    # 프로파일 → 표시 해상도/포맷/품질 파라미터 도출
    if quality_profile == "빠름":
        target_px_eff, disp_fmt_eff, disp_quality_eff = 512, "WEBP", 92
    elif quality_profile == "균형":
        target_px_eff, disp_fmt_eff, disp_quality_eff = 1024, "WEBP", 95
    else:  # 선명
        target_px_eff, disp_fmt_eff, disp_quality_eff = 1600, "WEBP", 98


    # ---------- 데이터 준비 ----------
    # # 보고서 업데이트 시 images_summary.csv 파일의 수정 시간(mtime)을 사용하여 캐시를 무효화(갱신)합니다.
    try:
        cache_buster_tuple = (
            os.path.getmtime(IMG_SUMMARY),
            len(os.listdir(os.path.dirname(IMG_SUMMARY)))
        )
    except Exception:
        cache_buster_tuple = (0, 0)
    # 결과 아티팩트가 존재하지 않으면(파이프라인 미실행 또는 잘못된 출력 폴더)
    # 이전에 디스크에 남아있는 파일들을 그대로 전체보기에서 표시하지 않도록
    # 빈 목록으로 처리합니다. 사용자에게 안내 메시지를 보여줍니다.
    artifacts_present = (
        os.path.exists(IMG_SUMMARY)
        or os.path.exists(REPORT_PARQUET)
        or os.path.exists(REPORT_CSV)
        or os.path.isdir(os.path.join(OUTPUT_DIR, "ok"))
        or os.path.isdir(os.path.join(OUTPUT_DIR, "grouped"))
    )
    if not artifacts_present:
        # 결과가 아예 없을 때는 전체보기 탭을 비워 혼동을 줄입니다.
        st.info("결과 아티팩트가 없습니다. 먼저 파이프라인을 실행하거나 올바른 출력 폴더를 지정하세요.")
        all_imgs = []
    else:
        all_imgs = list_all_images(OUTPUT_DIR, cache_buster=hash(cache_buster_tuple))
    # 검색/확장자 필터
    if q:
        all_imgs = [p for p in all_imgs if q.lower() in p.lower()]
    if ext_sel:
        all_imgs = [p for p in all_imgs if any(p.lower().endswith(e) for e in ext_sel)]

    # 정렬
    if sort_key == "파일명":
        all_imgs.sort(key=lambda p: os.path.basename(p).lower())
    elif sort_key == "수정시각(최신순)":
        all_imgs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    else:
        all_imgs.sort(key=lambda p: os.path.getmtime(p))

    total_items = len(all_imgs)
    st.caption(f"총 {total_items}개 파일")

    # 페이징/지연 로드: gallery_limit만큼만 표시. '더 보기'로 증분.
    if 'gallery_limit' not in st.session_state or st.session_state.gallery_limit <= 0:
        st.session_state.gallery_limit = 60
    # cap to total
    st.session_state.gallery_limit = min(total_items, st.session_state.gallery_limit)
    show_paths = all_imgs[:st.session_state.gallery_limit]
    st.caption(f"1–{len(show_paths)} / {total_items}")

    # ========== 이미지 그리드 섹션: 기본 그리드 동작으로 단순화 ==========
    # 레이아웃 선택 박스를 제거하고 기본 그리드만 표시합니다.
    cols = st.columns(grid_cols_local)
    for idx, path in enumerate(show_paths):
        # 표시에 사용할 이미지(리샘플 또는 원본)
        if render_mode == "원본":
            disp = path
        else:
            disp = make_display_image(path, size=target_px_eff, fmt=disp_fmt_eff, quality=disp_quality_eff)

        with cols[idx % grid_cols_local]:
            # 이미지 (클릭하면 비교 선택 토글)
            img_name = os.path.basename(path)
            try:
                st.image(
                    _safe_image_open(disp),
                    caption=img_name,
                    use_container_width=True,
                    key=f"gallery_img_{idx}",
                    on_click=on_image_click,
                    args=(path,),
                )
            except Exception:
                # 일부 Streamlit 버전에서는 st.image가 on_click을 지원하지 않을 수 있으므로
                # 실패하면 폴백으로 클릭 없는 이미지를 표시합니다.
                st.image(_safe_image_open(disp), caption=img_name, use_container_width=True)

    # ---------- 더 보기 버튼 ----------
    if st.session_state.gallery_limit < total_items:
        if st.button("더 보기"): 
            # 한 번에 60장씩 추가
            st.session_state.gallery_limit = min(total_items, st.session_state.gallery_limit + 60)
            _request_rerun()

    # 우선 비교 토글로 대체 — 모달형 미리보기 버튼 제거

    # 모달 미지원 대체 표시: 없음(직접 inline으로 대체됨)