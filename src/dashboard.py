import os
import glob
import sys
import hashlib
import re
from typing import Tuple, List, Dict, Optional

import streamlit as st
import polars as pl
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import cv2
import time
import logging

# 모듈 로거
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ===== Optional metrics/components (존재하면 사용) =====
try:
    from skimage.metrics import structural_similarity as ssim
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

try:
    import lpips
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False

# Pillow resample 상수 호환
try:
    RESAMPLE = Image.Resampling.LANCZOS
except Exception:
    RESAMPLE = Image.LANCZOS

# ====== 인자 파싱 (streamlit run ... -- --output_dir=...) ======
import argparse
def parse_streamlit_args():
    if '--' in sys.argv:
        idx = sys.argv.index('--')
        user_args = sys.argv[idx + 1:]
    else:
        user_args = []
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--output_dir', default='output')
    try:
        ns, _ = p.parse_known_args(user_args)
    except SystemExit:
        class X: output_dir = 'output'
        ns = X()
    return ns

ns = parse_streamlit_args()
OUTPUT_DIR = ns.output_dir
REPORT_PARQUET = os.path.join(OUTPUT_DIR, "report.parquet")
REPORT_CSV = os.path.join(OUTPUT_DIR, "report.csv")
IMG_SUMMARY = os.path.join(OUTPUT_DIR, "images_summary.csv")
THUMB_DIR = os.path.join(OUTPUT_DIR, "artifacts", "thumbnails")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(THUMB_DIR), exist_ok=True)
os.makedirs(THUMB_DIR, exist_ok=True)

# ===== 페이지 설정 =====
st.set_page_config(page_title="답안지 검수 대시보드", layout="wide")
st.title("📋 답안지 스캔 검수 대시보드 (Handwriting-Optimized)")

# --- Gallery state (전체 보기 탭 전용) ---
if "gallery_limit" not in st.session_state:
    st.session_state.gallery_limit = 120  # 한 번에 보여줄 개수 초기값 (증가)
if "gallery_selected" not in st.session_state:
    st.session_state.gallery_selected = []  # 비교 선택(최대 2장)


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
def load_report(report_parquet: str, report_csv: str) -> pd.DataFrame:
    if os.path.exists(report_parquet):
        return pl.read_parquet(report_parquet).to_pandas()
    if os.path.exists(report_csv):
        return pl.read_csv(report_csv).to_pandas()
    st.error("⚠️ 결과 파일이 없습니다. 먼저 main.py(파이프라인)를 실행하세요.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_img_summary(img_summary_csv: str) -> pd.DataFrame:
    if os.path.exists(img_summary_csv):
        return pd.read_csv(img_summary_csv)
    return pd.DataFrame(columns=["파일", "밀도", "빈칸여부"])

@st.cache_data(show_spinner=False)
def list_all_images(root: str, cache_buster: float = 0) -> List[str]:
    """루트 폴더 아래의 이미지 파일을 재귀적으로 나열합니다.
    cache_buster는 외부에서 캐시를 무효화할 때 사용합니다.
    """
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    all_paths = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                all_paths.append(os.path.join(dirpath, fn))
    return sorted(all_paths)

# ===== 캐싱: 베이스네임 → 경로 맵 (탐색/해결용) =====
@st.cache_data(show_spinner=False)
def build_basename_map(root: str, cache_buster: float = 0) -> Dict[str, str]:
    """
    같은 파일명이 여러 폴더에 있으면 우선순위:
    grouped/  → ok/ → blank_answers/ → 기타
    """
    # cache_buster는 파일 변경 시 호출자가 강제 재계산하도록 허용합니다
    _ = cache_buster
    imgs = list_all_images(root, cache_buster=cache_buster)
    def pri(p: str) -> int:
        low = p.replace("\\", "/").lower()
        if "/grouped/" in low: return 0
        if "/ok/" in low: return 1
        if "/blank_answers/" in low: return 2
        return 3
    best: Dict[str, str] = {}
    best_pri: Dict[str, int] = {}
    for p in imgs:
        bn = os.path.basename(p).lower()
        rank = pri(p)
        if (bn not in best) or (rank < best_pri[bn]):
            best[bn] = p
            best_pri[bn] = rank
    return best

BASENAME_MAP = build_basename_map(OUTPUT_DIR)

def resolve_image_path(name_or_path: str) -> Optional[str]:
    """
    - 절대/상대 경로가 유효하면 그대로 사용
    - 아니면 OUTPUT_DIR 하위에서 파일명으로 검색(베이스네임 맵 사용)
    """
    if not name_or_path:
        return None
    if os.path.isfile(name_or_path):
        return name_or_path
    rel = os.path.join(OUTPUT_DIR, name_or_path)
    if os.path.isfile(rel):
        return rel
    bn = os.path.basename(name_or_path).lower()
    return BASENAME_MAP.get(bn, None)


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
        logger.debug(f"make_display_image 실패: {src_path} -> {dst}: {e}")
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

    # 공백 수: output/blank_answers 폴더에 있는 이미지 파일 수를 센다 (안전하게 처리)
    try:
        blank_dir = os.path.join(OUTPUT_DIR, "blank_answers")
        blank_cnt = 0
        if os.path.isdir(blank_dir):
            exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
            for name in os.listdir(blank_dir):
                p = os.path.join(blank_dir, name)
                if os.path.isfile(p):
                    _, ext = os.path.splitext(name)
                    if ext.lower() in exts:
                        blank_cnt += 1
        kpis["공백 수"] = int(blank_cnt)
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
df = load_report(REPORT_PARQUET, REPORT_CSV)
img_df = load_img_summary(IMG_SUMMARY)

# ===== KPI 카드 =====
kpis = compute_kpis(df, img_df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("총 이미지", f"{kpis['총 이미지']:,}")
c2.metric("그룹 수", f"{kpis['그룹 수']:,}")
c3.metric("공백 수", f"{kpis['공백 수']:,}")


# ===== 사이드바: 꼭 필요한 옵션만 노출 =====
st.sidebar.header("주요 필터/설정")
# 사이드바: 결과 파일/아티팩트 존재 여부 요약
try:
    missing = []
    checks = [(REPORT_PARQUET, 'report.parquet'), (REPORT_CSV, 'report.csv'), (IMG_SUMMARY, 'images_summary.csv')]
    for p, name in checks:
        if not os.path.exists(p):
            missing.append(name)
    art_txt = os.path.join(OUTPUT_DIR, 'artifacts', 'ann_backend.txt')
    if not os.path.exists(art_txt):
        # 필수가 아닌 결과물/산출물이므로 경고는 하지 않음
        pass
    if missing:
        st.sidebar.warning("결과 파일 누락: " + ", ".join(missing) + ". 먼저 파이프라인을 실행하세요.")

except Exception:
    pass
# ===== 테마 선택: 여러 디자이너 친화적 테마 제공 =====
THEMES = {
    'Light (기본)': {
        'palette': { 'bg':'#FBFDFF','sidebar_bg':'#FFFFFF','text':'#091223','secondary':'#475569','accent':'#0B66FF','card_bg':'#FBFDFF','card_border':'#e6eef8','shadow':'0 6px 18px rgba(10,20,40,0.04)'},
    },
    'Soft Dark': {
        'palette': { 'bg':'#0f1722','sidebar_bg':'#0d1620','text':'#e6eef6','sidebar_text':'#F1F5F9','secondary':'#9fb0c3','accent':'#6fb3ff','card_bg':'#0b1a24','card_border':'#14232d','shadow':'0 6px 18px rgba(3,10,18,0.45)'} ,
    },
    'Warm Sepia': {
        'palette': { 'bg':'#f4efe6','sidebar_bg':'#efe6d9','text':'#2d2a26','secondary':'#6e5a4a','accent':'#b77936','card_bg':'#fbf6ee','card_border':'#e6dccf','shadow':'0 6px 18px rgba(30,20,10,0.08)'},
    },
    'Gentle Mint': {
        'palette': { 'bg':'#f3faf6','sidebar_bg':'#eaf7ef','text':'#082724','secondary':'#4b6b64','accent':'#39b89f','card_bg':'#ffffff','card_border':'#e6f0ec','shadow':'0 6px 18px rgba(5,30,25,0.06)'} ,
    }
}

if 'theme' not in st.session_state:
    st.session_state['theme'] = 'Light (기본)'

if 'theme' not in st.session_state:
    st.session_state['theme'] = 'Light (기본)'
def _inject_theme_css(mode: str = 'Light (기본)'):
    # mode에 따라 팔레트 선택
    theme = THEMES.get(mode, THEMES['Light (기본)'])
    pal = theme['palette']

    # 기본값 보장
    bg = pal.get('bg','#F7F9FB')
    sidebar_bg = pal.get('sidebar_bg', '#FFFFFF')
    text = pal.get('text', '#0B1726')
    sidebar_text = pal.get('sidebar_text', text)
    secondary_text = pal.get('secondary', '#41515F')
    accent = pal.get('accent', '#0B66FF')
    card_bg = pal.get('card_bg', '#FFFFFF')
    card_border = pal.get('card_border', '#e6e9ee')
    shadow = pal.get('shadow', 'none')

    css = f"""
    <style>
    .stApp {{ background-color: {bg} !important; color: {text} !important; }}
    [data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; box-shadow: none !important; color: {sidebar_text} !important; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stHeader, [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .css-1d391kg {{ color: {sidebar_text} !important; opacity: 0.98 !important; }}
    .stBlock, .stCard {{ background-color: {card_bg} !important; border: 1px solid {card_border}; border-radius: 10px; box-shadow: {shadow}; padding: 12px; }}
    .stMetric {{ color: {text} !important; }}
    /* KPI/Metric 내부 텍스트(라벨/서브텍스트)가 다크에서 안보이는 문제 해결: 강제 색상/불투명도 적용 */
    .stMetric, .stMetric * {{ color: {text} !important; opacity: 0.98 !important; }}
    .stMetric p, .stMetric span, .stMetric small {{ color: {secondary_text} !important; opacity: 0.95 !important; }}
    input, textarea, select, button {{ color: {text} !important; background-color: transparent !important; border-radius: 8px; }}
    .stApp p, .stApp span, label, .css-1v0mbdj p {{ color: {secondary_text} !important; }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{ color: {secondary_text} !important; }}
    a, .stButton>button, .css-18e3th9 a, .css-18e3th9 button {{ color: {accent} !important; }}
    .stDataFrame table {{ border-collapse: separate; border-spacing: 0 8px; }}
    img {{ border-radius: 8px; box-shadow: 0 8px 24px rgba(2,8,12,0.15); }}
    /* 검색 입력 상자 강조: 사용자 요청으로 가독성 향상용 추가 스타일입니다. */
    input[type="text"], .stTextInput>div>div>input {{
        background-color: rgba(255,255,255,0.9) !important;
        border: 1.5px solid {accent} !important;
        box-shadow: 0 4px 10px rgba(11,102,255,0.08) !important;
        padding: 10px 12px !important;
        border-radius: 10px !important;
        font-size: 14px !important;
        color: {text} !important;
    }}
    /* 사이드바 내 입력과 플레이스홀더 대비 개선 */
    [data-testid="stSidebar"] input[type="text"] {{ background-color: rgba(255,255,255,0.92) !important; }}
    input::placeholder, textarea::placeholder {{ color: rgba(0,0,0,0.38) !important; font-weight: 500 !important; }}
    </style>
    """
    try:
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        pass

# 사이드바에서 테마 선택 + 스와치 표시
with st.sidebar.expander('테마', expanded=True):
    theme_keys = list(THEMES.keys())
    # 라디오를 session_state 'theme' 키에 바인딩합니다. Streamlit은 위젯 클릭 시 자동으로 재실행하므로
    # 별도의 experimental_rerun은 필요하지 않습니다.
    default_idx = theme_keys.index(st.session_state.get('theme', theme_keys[0])) if st.session_state.get('theme') in theme_keys else 0
    st.radio('테마 선택', theme_keys, index=default_idx, key='theme')
    sel = st.session_state.get('theme', theme_keys[0])

    # 스와치: 작은 박스들로 팔레트 미리보기
    pal = THEMES[sel]['palette']
    swatch_html = '<div style="display:flex;gap:6px;margin-top:8px;align-items:center">'
    # 주요 색상들(배경/카드/텍스트/액센트)
    for k in ['bg','card_bg','text','accent']:
        if k in pal:
            swatch_html += f"<div style=\"width:36px;height:24px;border-radius:6px;background:{pal[k]};border:1px solid rgba(0,0,0,0.06)\" title=\"{k}\"></div>"
    swatch_html += '</div>'
    st.markdown(swatch_html, unsafe_allow_html=True)
    st.write(THEMES[sel].get('desc',''))

    # 라디오 클릭으로 session_state['theme']가 갱신되며 Streamlit이 재실행됩니다. 이 렌더 주기에서 바로 CSS를 주입합니다.
    try:
        _inject_theme_css(sel)
    except Exception:
        logger.debug("_inject_theme_css 즉시 적용 실패")

# 실제로 주입
_inject_theme_css(st.session_state.get('theme','Light (기본)'))

# Sidebar: 그룹화된 컨트롤 — 기본 / 고급
with st.sidebar.expander('기본', expanded=True):
    # 필수 필터/검색/그리드 설정
    min_sim = st.slider("최소 유사도", 0.0, 1.0, 0.90, 0.01, help="유사도 임계값을 조정하세요.")
    name_query = st.text_input("파일명 검색", value="", help="특정 파일명을 빠르게 찾고 싶을 때 입력")
    group_list = sorted(list(df["그룹ID"].replace('-', pd.NA).dropna().unique())) if "그룹ID" in df.columns else []
    group_filter = st.selectbox("특정 그룹만 보기(재스캔 필요)", ["전체"] + group_list)
    # 그리드 열 개수는 자주 쓰는 기본 옵션으로 노출
    grid_cols = st.slider("그리드 열 개수", 2, 8, 5, help="한 줄에 몇 장씩 볼지 선택")

with st.sidebar.expander('고급', expanded=False):
    st.markdown("고급 설정: 성능/품질 관련 옵션입니다. 기본 설정으로도 대부분의 경우 충분합니다.")
    # 재스캔 탭의 보기 모드(대형/그리드)
    group_view_mode = st.radio("재스캔 필요 보기 방식", ["대형 비교(2열)", "그리드(다중 썸네일)"], horizontal=True, index=1)
    # 표시 해상도/포맷/품질(내부 고정 파라미터) — 필요시 디버그용 노출
    grid_target_px = 768  # 고정값 (내부적으로 사용)
    disp_fmt = "WEBP"    # 고정값
    disp_quality = 95     # 고정값
    group_large_px = 1400 # 고정값
    group_page_size = 6   # 고정값
    group_page = 1        # 고정값(페이지네이션은 필요시만)
    # 간단 분석 토글(고급 사용자 전용)
    show_absdiff = st.checkbox("차이 히트맵(AbsDiff) 표시", value=False, help="선택 비교 시 차이 히트맵을 표시합니다.")
    show_ssim = st.checkbox("SSIM 맵 표시 (skimage 필요)", value=False, help="SSIM 맵을 표시하려면 skimage가 설치되어 있어야 합니다.")

# ===== 유틸: 비교용 도구 =====
def _read_gray_same_size(a_path: str, b_path: str) -> Tuple[np.ndarray, np.ndarray]:
    a = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        raise RuntimeError("이미지 로딩 실패")
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return a, b

def _absdiff_heatmap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(a, b)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _blend_images_rgb(a_path: str, b_path: str, alpha: float = 0.5) -> np.ndarray:
    """두 이미지를 읽어 공통 최소 크기로 리사이즈한 뒤 RGB로 블렌드하여 numpy 배열을 반환합니다.
    alpha는 첫 번째 이미지(a)의 가중치(0..1)입니다."""
    a = cv2.imread(a_path, cv2.IMREAD_COLOR)
    b = cv2.imread(b_path, cv2.IMREAD_COLOR)
    if a is None or b is None:
        raise RuntimeError("이미지 로드 실패")
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
    a = cv2.imread(a_path, cv2.IMREAD_COLOR)
    b = cv2.imread(b_path, cv2.IMREAD_COLOR)
    if a is None or b is None:
        raise RuntimeError("이미지 로드 실패")
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


def _cached_diff_path(a_path: str, b_path: str, blur: int = 3, thresh: int = 10) -> Optional[str]:
    key = _comp_cache_key(a_path, b_path, 'diff', {'blur': blur, 'thresh': thresh})
    dst = os.path.join(THUMB_DIR, f"cmp_diff_{key}.png")
    if os.path.exists(dst) and _file_mtime(dst) >= max(_file_mtime(a_path), _file_mtime(b_path)):
        return dst
    # 차이 히트맵 생성
    ga = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
    gb = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)
    if ga is None or gb is None:
        return None
    h = min(ga.shape[0], gb.shape[0]); w = min(ga.shape[1], gb.shape[1])
    ga = cv2.resize(ga, (w, h), interpolation=cv2.INTER_AREA)
    gb = cv2.resize(gb, (w, h), interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(ga, gb)
    diff = cv2.GaussianBlur(diff, (blur, blur), 0)
    _, diff_mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_TOZERO)
    diff_norm = cv2.normalize(diff_mask, None, 0, 255, cv2.NORM_MINMAX)
    heat = cv2.applyColorMap(diff_norm.astype('uint8'), cv2.COLORMAP_JET)
    rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return _write_cached_image(rgb, dst, fmt='PNG')


def _cached_highlight_path(a_path: str, b_path: str, color: Tuple[int, int, int] = (0, 255, 255), thresh: int = 20) -> Optional[str]:
    key = _comp_cache_key(a_path, b_path, 'hl', {'color': color, 'thresh': thresh})
    dst = os.path.join(THUMB_DIR, f"cmp_hl_{key}.png")
    if os.path.exists(dst) and _file_mtime(dst) >= max(_file_mtime(a_path), _file_mtime(b_path)):
        return dst
    arr = _highlight_differences_rgb(a_path, b_path, color=color, thresh=thresh)
    return _write_cached_image(arr, dst, fmt='PNG')


# GIF 생성 지원 제거: Fade는 이제 _cached_blend_path/_blend_images_rgb의 정적 블렌드만 사용합니다

# ===== 모달(미리보기) 지원: Streamlit 1.34+ =====
_HAS_DIALOG = hasattr(st, "dialog")
def open_preview(img_path: str, caption: str = ""):
    """모달(dialog) 기능이 있으면 모달로, 없으면 인라인으로 미리보기를 표시합니다."""
    if _HAS_DIALOG:
        @st.dialog("미리보기")
        def _d(img_path_inner: str, caption_inner: str = ""):
            st.image(_safe_image_open(img_path_inner), caption=caption_inner, use_container_width=True)
        _d(img_path, caption)
    else:
        # 모달 미지원 환경에서는 인라인으로 표시
        st.image(_safe_image_open(img_path), caption=caption, use_container_width=True)

# ===== 공통: 리포트 필터링 =====
def filter_sort_report(_df: pd.DataFrame) -> pd.DataFrame:
    view = _df.copy()
    if "유사도" in view.columns:
        view = view[view["유사도"] >= min_sim]
    if group_filter != "전체" and "그룹ID" in view.columns:
        view = view[view["그룹ID"] == group_filter]
    if name_query:
        q = name_query.lower()
        def _hit(row) -> bool:
            a = str(row.get("파일1", "")).lower()
            b = str(row.get("파일2", "")).lower()
            return (q in a) or (q in b)
        view = view[view.apply(_hit, axis=1)]
    # 기본 정렬: 유사도 내림차순, 그 다음 파일명
    if "유사도" in view.columns:
        view = view.sort_values(["유사도", "파일1", "파일2"], ascending=[False, True, True])
    else:
        view = view.sort_values(["파일1", "파일2"])
    return view

# ===== 세션: 비교 큐 =====
# 통합된 비교 선택 상태: 절대 경로 리스트 (최대 2개)
if "gallery_selected" not in st.session_state:
    st.session_state["gallery_selected"] = []
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

# ===== 탭 구성 =====
tab2, tab3, tab4 = st.tabs([ "재스캔 필요", "정상/공백 답안", "전체 보기"])

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


# === Tab2: 재스캔 필요 ===
with tab2:
  
    try:
        import imagehash
        dup_pairs = []
        hashes = {}
    # 스캔 블록이 주석 처리되어도 변수들이 존재하도록 보장
        input_dir = locals().get('input_dir', OUTPUT_DIR)
        scan_files = locals().get('scan_files', [])
        for f in scan_files:
            p = os.path.join(input_dir, f)
            try:
                # 안전하게 열기 (읽기 실패 파일은 건너뜀)
                with open(p, 'rb') as fh:
                    img = Image.open(fh).convert('L')
                    h = imagehash.phash(img)
                hashes[f] = h
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
                # report '유사도' 컬럼을 읽어 사이드바의 min_sim 이하 항목은 후보에서 제외
                try:
                    sim = float(row.get('유사도', 0.0))
                except Exception:
                    sim = float(row.get('유사도', 0.0) or 0.0)
                try:
                    # min_sim은 사이드바 위젯에서 정의되며 문자열/숫자 모두 올 수 있음
                    if float(sim) < float(locals().get('min_sim', 0)):
                        continue
                except Exception:
                    # 변환 실패 시 필터링을 적용하지 않음
                    pass
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

    # 상위 후보를 빠른 액션과 함께 표시
    if deduped:
        st.markdown("---")
        st.markdown("### ⚠️ 재스캔 권장 후보 (우선순위 순)")
    # 최대 12개의 후보를 간결하게 표시
        for idx, (score, pa, pb, src, meta) in enumerate(deduped[:12]):
            col_a, col_b, col_c = st.columns([4, 4, 2])
            an = os.path.basename(pa) if pa else 'Unknown'
            bn = os.path.basename(pb) if pb else 'Unknown'
            with col_a:
                if pa and os.path.exists(pa):
                    thumb_a = make_display_image(pa, size=300, fmt=disp_fmt, quality=80)
                    st.image(_safe_image_open(thumb_a), caption=f"A: {an}", use_container_width=True)
                else:
                    st.write(f"A: {an} (없음)")
            with col_b:
                if pb and os.path.exists(pb):
                    thumb_b = make_display_image(pb, size=300, fmt=disp_fmt, quality=80)
                    st.image(_safe_image_open(thumb_b), caption=f"B: {bn}", use_container_width=True)
                else:
                    st.write(f"B: {bn} (없음)")
            with col_c:
                st.write(f"우선도: {score:.3f}")
                # 재스캔 권고 토글 (세션에 per-pair 키로 저장)
                pair_key = f"rescan_rec_{idx}_{hashlib.md5((an+bn).encode('utf-8')).hexdigest()[:8]}"
                if pair_key not in st.session_state:
                    st.session_state[pair_key] = True
                rec = st.checkbox("재스캔 권고", value=st.session_state.get(pair_key, True), key=pair_key)
                if st.button("↔ 비교 선택", key=f"cmp_cand_{idx}"):
                    # add both to comparison (toggle behavior)
                    toggle_compare(pa)
                    toggle_compare(pb)

    # ----- 즉시 비교 패널: 사용자가 아래 그리드에서 '↔ 비교 선택' 버튼을 클릭하면
    # rescan 탭의 상단에 바로 비교 옵션과 결과가 표시되도록 함
    # 통합된 gallery_selected(경로 리스트) 사용
    sel_exist_top = [p for p in st.session_state.get("gallery_selected", []) if p and os.path.isfile(p)]
    if sel_exist_top:
        st.markdown("---")
        st.markdown("### 🔍 즉시 비교 (재스캔 탭)")
    # 공통: 모드 선택 + 도움말 옆에 배치
        colm1, colm2 = st.columns([3, 7])
        with colm1:
            # 내부 값(key)은 변경하지 않되, 사용자에게 보이는 라벨은 한국어로 제공합니다.
            cmp_mode_top = st.radio("보기 표시 (재스캔)", ["비교(좌우)","페이드(겹침)", "차이(Heatmap)", "하이라이터(오버레이)"], index=0, horizontal=True, key="cmp_mode_top")
        with colm2:
            # 선택된 비교 모드에 해당하는 설명만 표시
            # cmp_mode_top 내부값은 라디오의 label로 들어가므로 위젯의 라벨에 따라 분기합니다.
            if cmp_mode_top == "비교(좌우)":
                st.markdown(
                    "**비교(좌우)**\n"
                    "- 선택한 두 이미지를 좌우로 나란히 크게 보여줍니다. 빠르게 원본 대비를 확인할 때 사용하세요.\n"
                )
            elif cmp_mode_top == "페이드(겹침)":
                st.markdown(
                    "**페이드(겹침)**\n"
                    "- 두 이미지를 위아래로 겹쳐 보여줍니다. A 이미지의 투명도(alpha)를 조절해 미세한 변화 위치를 문맥과 함께 확인하세요.\n"
                    "- 권장: A alpha = 0.4–0.6\n"
                    "- 팁: 전체 레이아웃을 보존하므로 레이아웃 변화 식별에 유리합니다.\n"
                )
            elif cmp_mode_top == "차이(Heatmap)":
                st.markdown(
                    "**차이(Heatmap)**\n"
                    "- 그레이스케일 절대 차이를 계산해 heatmap으로 표시합니다. 픽셀 단위 변경을 강조합니다.\n"
                    "- 권장: Blur = 3, Threshold = 10\n"
                    "- 팁: 노이즈에 민감하므로 Blur/Threshold 조정으로 노이즈를 억제하세요.\n"
                )
            else:
                st.markdown(
                    "**하이라이터(오버레이)**\n"
                    "- 차이 마스크를 색상으로 원본 이미지에 오버레이합니다. 글자 추가/삭제 같은 의미 있는 변경을 빠르게 파악할 때 유용합니다.\n"
                    "- 권장: Threshold = 15, 색상 = Yellow\n"
                )

    # 모드별 파라미터 제어. '비교'는 추가 컨트롤이 없습니다.
    # 위젯이 렌더되지 않더라도 해당 변수가 존재하도록 보장(재실행 시 NameError 방지)
    # 기본값을 제공; 실제 위젯 선택 시 Streamlit이 세션 상태를 갱신합니다.
        if 'diff_blur_top' not in st.session_state:
            st.session_state['diff_blur_top'] = 3
        if 'diff_thresh_top' not in st.session_state:
            st.session_state['diff_thresh_top'] = 10
        if 'hl_color_top' not in st.session_state:
            st.session_state['hl_color_top'] = 'Yellow'
        if 'hl_thresh_top' not in st.session_state:
            st.session_state['hl_thresh_top'] = 20
    # 페이드(블렌드) 기본값
        if 'fade_alpha' not in st.session_state:
            st.session_state['fade_alpha'] = 0.5

        # 모드별 파라미터 위젯 (세션 상태를 갱신함)
        if cmp_mode_top == "비교(좌우)":
            # 단순 좌우 비교는 별도의 파라미터 없음
            pass
        elif cmp_mode_top == "차이(Heatmap)":
            diff_blur_top = st.slider("Difference: Blur 강도(odd kernel)", 1, 11, st.session_state.get('diff_blur_top', 3), 2, key="diff_blur_top")
            diff_thresh_top = st.slider("Difference: 강조 임계값", 0, 255, st.session_state.get('diff_thresh_top', 10), 1, key="diff_thresh_top")
        elif cmp_mode_top == "페이드(겹침)":
            fade_alpha = st.slider("Fade: A 이미지 알파", 0.0, 1.0, float(st.session_state.get('fade_alpha', 0.5)), 0.05, key='fade_alpha')
        elif cmp_mode_top == "하이라이터(오버레이)":
            hl_color_top = st.selectbox("하이라이터 색상", ["Yellow", "Red", "Lime", "Cyan"], index=["Yellow", "Red", "Lime", "Cyan"].index(st.session_state.get('hl_color_top', 'Yellow')), key="hl_color_top")
            hl_thresh_top = st.slider("하이라이터: 임계값", 1, 100, st.session_state.get('hl_thresh_top', 20), 1, key="hl_thresh_top")

        if len(sel_exist_top) == 1:
            bigp = make_display_image(sel_exist_top[0], size=max(1400, group_large_px), fmt=disp_fmt, quality=disp_quality)
            st.image(_safe_image_open(bigp), caption=os.path.basename(sel_exist_top[0]), use_container_width=True)
        else:
            a_path, b_path = sel_exist_top[:2]
            # 사용자가 단순 비교(좌우)를 선택하면 두 이미지를 나란히 표시; 그렇지 않으면 병합/처리된 단일 이미지를 표시
            if cmp_mode_top == "비교(좌우)":
                big_a = make_display_image(a_path, size=max(1600, group_large_px), fmt=disp_fmt, quality=disp_quality)
                big_b = make_display_image(b_path, size=max(1600, group_large_px), fmt=disp_fmt, quality=disp_quality)
                c1t, c2t = st.columns(2)
                with c1t:
                    st.image(_safe_image_open(big_a), caption=os.path.basename(a_path), use_container_width=True)
                with c2t:
                    st.image(_safe_image_open(big_b), caption=os.path.basename(b_path), use_container_width=True)
            else:
                try:
                    if cmp_mode_top == "차이(Heatmap)":
                        cached = _cached_diff_path(a_path, b_path, blur=diff_blur_top, thresh=diff_thresh_top)
                        if cached:
                            st.image(cached, caption=f"차이(임계={diff_thresh_top})", use_container_width=True)
                        else:
                            ga = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
                            gb = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)
                            h = min(ga.shape[0], gb.shape[0]); w = min(ga.shape[1], gb.shape[1])
                            ga = cv2.resize(ga, (w, h), interpolation=cv2.INTER_AREA)
                            gb = cv2.resize(gb, (w, h), interpolation=cv2.INTER_AREA)
                            diff = cv2.absdiff(ga, gb)
                            diff = cv2.GaussianBlur(diff, (diff_blur_top, diff_blur_top), 0)
                            _, diff_mask = cv2.threshold(diff, diff_thresh_top, 255, cv2.THRESH_TOZERO)
                            diff_norm = cv2.normalize(diff_mask, None, 0, 255, cv2.NORM_MINMAX)
                            heat = cv2.applyColorMap(diff_norm.astype('uint8'), cv2.COLORMAP_JET)
                            st.image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), caption=f"차이(임계={diff_thresh_top})", use_container_width=True)
                    elif cmp_mode_top == "페이드(겹침)":
                        fade_alpha_val = float(st.session_state.get('fade_alpha', 0.5))
                        blendp = _cached_blend_path(a_path, b_path, alpha=fade_alpha_val)
                        if blendp:
                            st.image(blendp, caption=f"페이드(알파={fade_alpha_val:.2f})", use_container_width=True)
                        else:
                            arr = _blend_images_rgb(a_path, b_path, alpha=fade_alpha_val)
                            st.image(arr, caption=f"페이드(알파={fade_alpha_val:.2f})", use_container_width=True)

                    else:
                        color_map = {"Yellow": (0, 255, 255), "Red": (0, 0, 255), "Lime": (0, 255, 0), "Cyan": (255, 255, 0)}
                        # session_state에서 값을 읽되, NameError 방지를 위해 기본값을 사용
                        hl_color = st.session_state.get('hl_color_top', 'Yellow')
                        hl_thresh = st.session_state.get('hl_thresh_top', 20)
                        col_bgr = color_map.get(hl_color, (0, 255, 255))
                        cached = _cached_highlight_path(a_path, b_path, color=col_bgr, thresh=hl_thresh)
                        if cached:
                            st.image(cached, caption=f"하이라이터 ({hl_color}, 임계={hl_thresh})", use_container_width=True)
                        else:
                            highlighted = _highlight_differences_rgb(a_path, b_path, color=col_bgr, thresh=hl_thresh)
                            st.image(highlighted, caption=f"하이라이터 ({hl_color}, 임계={hl_thresh})", use_container_width=True)
                except Exception as e:
                    st.info(f"비교 렌더 실패: {e}")

    grouped_dir = os.path.join(OUTPUT_DIR, "grouped")
    if os.path.isdir(grouped_dir):
        groups = sorted(os.listdir(grouped_dir))
        if group_filter != "전체":
            groups = [g for g in groups if g == group_filter]

                # 전역 is_2file 유틸 사용

        # --- 대형 비교 모드: 페이지네이션 + 두 장을 크게 나란히 ---
        if group_view_mode.startswith("대형"):
            total_groups = len(groups)
            start = max(0, (group_page - 1) * group_page_size)
            end = min(total_groups, start + group_page_size)
            st.caption(f"그룹 {start+1}–{end} / 총 {total_groups} (페이지 {group_page})")

            for gid in groups[start:end]:
                # 그룹 텍스트/캡션을 모두 표시함 (특정 그룹 숨김 제거)
                st.subheader(f"그룹: {gid}")
                files = [f for f in sorted(os.listdir(os.path.join(grouped_dir, gid))) if is_2file(f)]
                if len(files) == 0:
                    # 파일이 없으면 조용히 다음으로
                    continue
                # 명확한 재스캔 안내 — 그룹의 첫 두 장을 지목하여 재스캔 권고
                if len(files) >= 2:
                    a_name, b_name = files[0], files[1]
                    # 뒷장인 경우 앞면 파일명 제시
                    a_front = corresponding_front_filename(a_name)
                    b_front = corresponding_front_filename(b_name)
                    st.error(f"재스캔 권고: 이 그룹의 파일 A: {a_name} / B: {b_name} — 해당 뒷장이 앞면과 동일한 내용이라면, 앞면 {a_front} 및 {b_front}을 다시 스캔하세요.")
                # 대형 표시: 앞면 2장 먼저, 그 아래 뒷장 2장 표시
                pairs = []
                for f in files[:2]:  # 보통 2장이므로 2장만
                    back_path = os.path.join(grouped_dir, gid, f)
                    front_name = corresponding_front_filename(f)
                    front_path_candidate = os.path.join(grouped_dir, gid, front_name)
                    if not os.path.exists(front_path_candidate):
                        # grouped 폴더에 없으면 전역 맵에서 찾기
                        front_path_candidate = resolve_image_path(front_name) or front_path_candidate
                    pairs.append((front_path_candidate, back_path, front_name, f))

                # 두 개의 컬럼으로 앞면을 먼저 표시
                cols = st.columns(2)
                for i in range(len(pairs)):
                    front_p, back_p, front_nm, back_nm = pairs[i]
                    with cols[i]:
                        if os.path.exists(front_p):
                            disp_front = make_display_image(front_p, size=group_large_px, fmt=disp_fmt, quality=disp_quality)
                            st.image(_safe_image_open(disp_front), caption=f"앞면: {front_nm}", use_container_width=True)
                        else:
                            st.warning(f"앞면 파일을 찾을 수 없음: {front_nm}")

                # 그리고 뒷장 표시(같은 레이아웃)
                cols2 = st.columns(2)
                for i in range(len(pairs)):
                    front_p, back_p, front_nm, back_nm = pairs[i]
                    with cols2[i]:
                        if os.path.exists(back_p):
                            disp_back = make_display_image(back_p, size=group_large_px, fmt=disp_fmt, quality=disp_quality)
                            st.image(_safe_image_open(disp_back), caption=f"뒷면: {back_nm}", use_container_width=True)
                        else:
                            st.warning(f"뒷장 파일을 찾을 수 없음: {back_nm}")
                # --- 그리드 모드 복원: 사용자가 탭에서 '그리드'를 선택했을 때 표시되는 블록 ---
        else:
            sel = st.selectbox("그룹 선택", ["전체 그룹 보기"] + groups)
            targets = groups if sel == "전체 그룹 보기" else [sel]
            for gid in targets:
                st.subheader(f"그룹: {gid}")
                files = [f for f in sorted(os.listdir(os.path.join(grouped_dir, gid))) if is_2file(f)]
                if len(files) == 0:
                    continue
                if len(files) >= 2:
                    a_name, b_name = files[0], files[1]
                    a_front = corresponding_front_filename(a_name)
                    b_front = corresponding_front_filename(b_name)
                    st.error(f"재스캔 권고: 이 그룹의 파일 A: {a_name} / B: {b_name} — 해당 뒷장이 앞면과 동일한 내용이라면, 앞면 {a_front} 및 {b_front}을 다시 스캔하세요.")

                # 각 이미지를 별도 항목으로 나누어 그리드에 채움
                items = []
                for f in files:
                    back_path = os.path.join(grouped_dir, gid, f)
                    front_name = corresponding_front_filename(f)
                    front_path_candidate = os.path.join(grouped_dir, gid, front_name)
                    if not os.path.exists(front_path_candidate):
                        front_path_candidate = resolve_image_path(front_name) or front_path_candidate
                    items.append(("앞면", front_name, front_path_candidate))
                    items.append(("뒷면", f, back_path))

                # 그리드: 앞면/뒷면을 각각 별도 박스로, 한 줄에 4개 표시
                cols = st.columns(4)
                # rescan 탭 전용 선택 상태 초기화
                if "rescan_selected" not in st.session_state:
                    st.session_state["rescan_selected"] = []

                for idx, (kind, name, pth) in enumerate(items):
                    with cols[idx % 4]:
                        if os.path.exists(pth):
                            # 비교 선택 토글 (통합된 gallery_selected 사용)
                            selected = pth in st.session_state.get("gallery_selected", [])
                            label = "✔ 비교 취소" if selected else "↔ 비교 선택"
                            if st.button(label, key=f"cmp_rescan_{gid}_{idx}"):
                                toggle_compare(pth)
                            disp = make_display_image(pth, size=group_large_px, fmt=disp_fmt, quality=disp_quality)
                            st.image(_safe_image_open(disp), caption=f"{kind}: {name}", use_container_width=True)
                        else:
                            st.info(f"{kind} 파일 없음: {name}")

    else:
        st.info("그룹 결과 폴더가 없습니다. 하지만 입력 폴더 또는 리포트에서 재스캔 후보를 검사할 수 있습니다.")


# === Tab3: 정상/공백 ===
with tab3:
    ok_dir = os.path.join(OUTPUT_DIR, "ok")
    blank_dir = os.path.join(OUTPUT_DIR, "blank_answers")
    sel = st.radio("보기 옵션", ["모두 보기", "정상만", "공백만"], horizontal=True)

    # 전역 is_2file 유틸 사용

    if sel in ["모두 보기", "정상만"] and os.path.isdir(ok_dir):
        st.subheader("✅ 정상 답안")
        files = [f for f in sorted(os.listdir(ok_dir)) if is_2file(f)]
        cols = st.columns(grid_cols)
        for idx, f in enumerate(files):
            img_path = os.path.join(ok_dir, f)
            disp = make_display_image(img_path, size=grid_target_px, fmt=disp_fmt, quality=disp_quality)
            with cols[idx % grid_cols]:
                # 비교 토글 버튼으로 통일 (전역 gallery_selected 사용, 절대 경로 저장)
                img_abs = os.path.join(ok_dir, f)
                selected = img_abs in st.session_state.get("gallery_selected", [])
                label = "✔ 비교 취소" if selected else "↔ 비교 선택"
                if st.button(label, key=f"cmp_ok_{idx}"):
                    toggle_compare(img_abs)
                st.caption("(버튼: 클릭하면 비교 큐에 추가됩니다. 최대 2장)")
                caption = f + ("  ✅ 선택됨" if selected else "")
                # 작은 배지: 선택 상태가 있으면 이미지 위에 overlay 표시 (HTML 사용)
                if selected:
                    badge_html = f"<div style='position:relative;display:inline-block'>"
                    badge_html += f"<div style='position:absolute;z-index:3;right:8px;top:8px;padding:4px 6px;background:#10B981;color:white;border-radius:6px;font-size:12px;font-weight:600;'>선택됨</div>"
                    badge_html += f"</div>"
                    st.markdown(badge_html, unsafe_allow_html=True)
                st.image(_safe_image_open(disp), caption=caption, use_container_width=True)

    if sel in ["모두 보기", "공백만"] and os.path.isdir(blank_dir):
        st.subheader("⭕ 공백 답안")
        files = [f for f in sorted(os.listdir(blank_dir)) if is_2file(f)]
        cols = st.columns(grid_cols)
        for idx, f in enumerate(files):
            img_path = os.path.join(blank_dir, f)
            disp = make_display_image(img_path, size=grid_target_px, fmt=disp_fmt, quality=disp_quality)
            with cols[idx % grid_cols]:
                img_abs = os.path.join(blank_dir, f)
                selected = img_abs in st.session_state.get("gallery_selected", [])
                label = "✔ 비교 취소" if selected else "↔ 비교 선택"
                if st.button(label, key=f"cmp_blank_{idx}"):
                    toggle_compare(img_abs)
                st.caption("(버튼: 클릭하면 비교 큐에 추가됩니다. 최대 2장)")
                caption = f + ("  ✅ 선택됨" if selected else "")
                if selected:
                    badge_html = f"<div style='position:relative;display:inline-block'>"
                    badge_html += f"<div style='position:absolute;z-index:3;right:8px;top:8px;padding:4px 6px;background:#10B981;color:white;border-radius:6px;font-size:12px;font-weight:600;'>선택됨</div>"
                    badge_html += f"</div>"
                    st.markdown(badge_html, unsafe_allow_html=True)
                st.image(_safe_image_open(disp), caption=caption, use_container_width=True)

    # === Tab3: 선택된 비교 항목을 즉시 대형 비교로 표시 ===
    if st.session_state.get('gallery_selected'):
        st.markdown("---")
        st.markdown("### 🔍 선택 비교 (대형) — 탭3")
    # gallery_selected는 이미 절대 경로를 저장해야 합니다
        sel_paths = [p for p in st.session_state.get('gallery_selected', []) if p and os.path.isfile(p)]
        if len(sel_paths) == 1:
            big = make_display_image(sel_paths[0], size=max(1400, grid_target_px), fmt=disp_fmt, quality=disp_quality)
            st.image(_safe_image_open(big), caption=os.path.basename(sel_paths[0]), use_container_width=True)
        elif len(sel_paths) >= 2:
            a_path, b_path = sel_paths[:2]
            big_a = make_display_image(a_path, size=max(1600, grid_target_px), fmt=disp_fmt, quality=disp_quality)
            big_b = make_display_image(b_path, size=max(1600, grid_target_px), fmt=disp_fmt, quality=disp_quality)
            c1, c2 = st.columns(2)
            with c1:
                st.image(_safe_image_open(big_a), caption=os.path.basename(a_path), use_container_width=True)
            with c2:
                st.image(_safe_image_open(big_b), caption=os.path.basename(b_path), use_container_width=True)
        # 선택 초기화 버튼
        c1, c2 = st.columns([1, 9])
        with c1:
            if st.button("선택 초기화", key="tab3_reset"):
                st.session_state.gallery_selected = []
            st.caption("(버튼: 현재 비교 선택을 모두 초기화합니다)")

# === Tab4: 전체 보기 ===
with tab4:
    # ---------- 쉬운 화질/레이아웃 컨트롤(탭 로컬) ----------
    st.markdown("#### 표시 설정")
    colq1, colq2, colq3 = st.columns([1.3, 1.1, 1.6])
    with colq1:
        quality_profile = st.radio(
            "화질 프로파일", ["빠름", "균형", "선명"],
            index=1, horizontal=True,
            help="빠름(512px), 균형(1024px), 선명(1600px)"
        )
    with colq2:
        render_mode = st.radio(
            "렌더 방식", ["리샘플(권장)", "원본"], index=0, horizontal=True,
            help="리샘플: LANCZOS로 고화질 썸네일 생성(권장) / 원본: 브라우저 스케일(선명하지만 느릴 수 있음)"
        )
        # 원본 모드 주의 문구
        if render_mode == "원본":
            st.caption("원본 모드: 브라우저에서 원본 이미지를 직접 로드합니다. 매우 큰 이미지의 경우 메모리/네트워크 사용이 증가할 수 있으므로 소량의 선택 비교(최대 2장)에서 사용하는 것을 권장합니다.")
    with colq3:
        grid_cols_local = st.slider("그리드 열 개수", 2, 8, max(4, grid_cols), 1)

    # 프로파일 → 표시 해상도/포맷/품질 파라미터 도출
    if quality_profile == "빠름":
        target_px_eff, disp_fmt_eff, disp_quality_eff = 512, "WEBP", 92
    elif quality_profile == "균형":
        target_px_eff, disp_fmt_eff, disp_quality_eff = 1024, "WEBP", 95
    elif quality_profile == "선명":
        target_px_eff, disp_fmt_eff, disp_quality_eff = 1600, "WEBP", 98
    st.markdown("---")

    # ---------- 검색/필터 UX (간결) ----------
    row1 = st.columns([1.6, 1.2, 1.2])
    with row1[0]:
        q = st.text_input("🔎 파일명/경로 검색", value=name_query, placeholder="예: 10002, scan, .png ...")
    with row1[1]:
        ext_sel = st.multiselect("확장자", [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"], default=[])
    with row1[2]:
        sort_key = st.selectbox("정렬", ["파일명", "수정시각(최신순)", "수정시각(오래된순)"], index=0)

    # ---------- 데이터 준비 ----------
    # # 보고서 업데이트 시 images_summary.csv 파일의 수정 시간(mtime)을 사용하여 캐시를 무효화(갱신)합니다.
    try:
        cache_buster = os.path.getmtime(IMG_SUMMARY)
    except Exception:
        cache_buster = 0
    all_imgs = list_all_images(OUTPUT_DIR, cache_buster)
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

    # ---------- 그리드 렌더 ----------
    cols = st.columns(grid_cols_local)
    for idx, path in enumerate(show_paths):
        # 표시에 사용할 이미지(리샘플 or 원본)
        if render_mode == "원본":
            disp = path
        else:
            disp = make_display_image(path, size=target_px_eff, fmt=disp_fmt_eff, quality=disp_quality_eff)

        with cols[idx % grid_cols_local]:
            # 이미지
            img_name = os.path.basename(path)
            selected = path in st.session_state.get("gallery_selected", [])
            caption = img_name + ("  ✅ 선택됨" if selected else "")
            if selected:
                badge_html = f"<div style='position:relative;display:inline-block'>"
                badge_html += f"<div style='position:absolute;z-index:3;right:8px;top:8px;padding:4px 6px;background:#10B981;color:white;border-radius:6px;font-size:12px;font-weight:600;'>선택됨</div>"
                badge_html += f"</div>"
                st.markdown(badge_html, unsafe_allow_html=True)
            st.image(_safe_image_open(disp), caption=caption, use_container_width=True)
            # 동작 버튼: 비교 토글만 표시 (절대 경로 전달)
            label = "✔ 비교 취소" if selected else "↔ 비교 선택"
            if st.button(label, key=f"cmp_all_{idx}"):
                toggle_compare(path)

    # ---------- 더 보기 버튼 ----------
    if st.session_state.gallery_limit < total_items:
        if st.button("더 보기"): 
            # 한 번에 60장씩 추가
            st.session_state.gallery_limit = min(total_items, st.session_state.gallery_limit + 60)
            rerun_fn = getattr(st, 'experimental_rerun', None)
            if callable(rerun_fn):
                try:
                    rerun_fn()
                except Exception:
                    logger.debug("experimental_rerun 실패: 더 보기에서 재실행 실패")
            else:
                try:
                    st.stop()
                except Exception:
                    pass

    # ---------- 선택 비교(대형 2분할) ----------
    if st.session_state.gallery_selected:
        st.markdown("---")
        st.markdown("### 🔍 선택 비교 (대형)")
        # 파일명 → 경로 복원 (BASENAME_MAP 사용)
        # gallery_selected 변수에는 절대 경로가 포함되어야 합니다. 하지만 이전 세션의 경우 파일 이름만 저장되어 있을 수 있습니다
        sel_paths = []
        for n in st.session_state.gallery_selected:
            if os.path.isfile(n):
                sel_paths.append(n)
            else:
                # 파일 이름(basename) 매핑 시도
                p = BASENAME_MAP.get(os.path.basename(n).lower())
                if p and os.path.isfile(p):
                    sel_paths.append(p)
        if len(sel_paths) == 1:
            st.info("한 장이 선택되었습니다. 한 장을 더 선택하면 2분할 비교가 표시됩니다.")
            # 1장도 크게 보여주자 (같은 품질 파라미터로)
            big = make_display_image(sel_paths[0], size=max(1400, target_px_eff), fmt=disp_fmt_eff, quality=disp_quality_eff) \
                  if render_mode == "리샘플(권장)" else sel_paths[0]
            st.image(_safe_image_open(big), caption=os.path.basename(sel_paths[0]), use_container_width=True)

        elif len(sel_paths) >= 2:
            # 2장 나란히 대형
            a_path, b_path = sel_paths[:2]
            big_a = make_display_image(a_path, size=max(1600, target_px_eff), fmt=disp_fmt_eff, quality=disp_quality_eff) \
                    if render_mode == "리샘플(권장)" else a_path
            big_b = make_display_image(b_path, size=max(1600, target_px_eff), fmt=disp_fmt_eff, quality=disp_quality_eff) \
                    if render_mode == "리샘플(권장)" else b_path

            c1, c2 = st.columns(2)
            with c1:
                st.image(_safe_image_open(big_a), caption=os.path.basename(a_path), use_container_width=True)
            with c2:
                st.image(_safe_image_open(big_b), caption=os.path.basename(b_path), use_container_width=True)

            # 옵션에 따라 간단 분석(원하면 켜서 사용)
            if show_absdiff or show_ssim:
                st.markdown("#### 차이 분석 (선택 사항)")
                try:
                    ga = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
                    gb = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)
                    h = min(ga.shape[0], gb.shape[0]); w = min(ga.shape[1], gb.shape[1])
                    ga = cv2.resize(ga, (w, h), interpolation=cv2.INTER_AREA)
                    gb = cv2.resize(gb, (w, h), interpolation=cv2.INTER_AREA)
                    if show_absdiff:
                        diff = cv2.absdiff(ga, gb)
                        diff = cv2.GaussianBlur(diff, (3, 3), 0)
                        diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
                        heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
                        st.image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), caption="차이 히트맵(AbsDiff)", use_container_width=True)
                    if show_ssim and _HAS_SKIMAGE:
                        score, ssim_img = ssim(ga, gb, full=True, data_range=255)
                        ssim_img = (1.0 - ssim_img)
                        ssim_img = (255 * (ssim_img / (ssim_img.max() + 1e-6))).astype(np.uint8)
                        heat = cv2.applyColorMap(ssim_img, cv2.COLORMAP_INFERNO)
                        st.image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB),
                                 caption=f"SSIM 맵 (score={score:.4f})", use_container_width=True)
                except Exception as e:
                    st.info(f"분석 실패: {e}")

        # 선택 상태 관리 버튼
        cols_ctrl = st.columns([1, 1, 6])
        with cols_ctrl[0]:
            if st.button("선택 초기화"):
                st.session_state.gallery_selected = []
    # 우선 비교 토글로 대체 — 모달형 미리보기 버튼 제거

    # 모달 미지원 대체 표시: 없음(직접 inline으로 대체됨)