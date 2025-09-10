# src/dashboard.py
# - 피드백 섹션 완전 제거
# - 유사 그룹: "대형 비교" 모드 추가(두 장을 즉시 크게 나란히 표시)
# - 유사 그룹/정상·공백/전체 보기: 화질(표시 해상도), 포맷, 품질 옵션으로 선명도 향상
# - 고화질 모드는 원본을 직접 띄우지 않고, LANCZOS 기반 "큰 해상도 썸네일"을 캐싱하여
#   화질을 확보하면서도 안정적인 메모리/네트워크 사용을 보장합니다.

import os
import glob
import sys
import hashlib
import re
from typing import Tuple, List, Dict, Optional

import streamlit as st
import polars as pl
from PIL import Image
import numpy as np
import pandas as pd
import cv2

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
    p.add_argument('--input_dir', default='input_images')
    try:
        ns, _ = p.parse_known_args(user_args)
    except SystemExit:
        class X: output_dir = 'output'
        ns = X()
    return ns

ns = parse_streamlit_args()
OUTPUT_DIR = ns.output_dir
INPUT_DIR = getattr(ns, 'input_dir', 'input_images')
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
    st.session_state.gallery_limit = 60  # 한 번에 보여줄 개수 초기값
if "gallery_selected" not in st.session_state:
    st.session_state.gallery_selected = []  # 비교 선택(최대 2장)


# ===== 공용 헬퍼 =====
def _file_mtime(path: str) -> float:
    try: return os.path.getmtime(path)
    except Exception: return 0.0

def _safe_image_open(path: str) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img

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
def list_all_images(root: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    all_paths = []
    for ext in exts:
        all_paths += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return sorted(all_paths)

# ===== 캐싱: 베이스네임 → 경로 맵 (탐색/해결용) =====
@st.cache_data(show_spinner=False)
def build_basename_map(root: str) -> Dict[str, str]:
    """
    같은 파일명이 여러 폴더에 있으면 우선순위:
    grouped/  → ok/ → blank_answers/ → 기타
    """
    imgs = list_all_images(root)
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
    - 아니면 OUTPUT_DIR 하위에서 파일명으로 검색(basename map)
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
    """
    긴 변 기준으로 size(px)까지 축소한 '표시용 이미지'를 캐시에 생성/재사용.
    - LANCZOS 리샘플링
    - 포맷: WEBP/JPEG/PNG
    - quality: JPEG/WEBP에 적용
    반환: 표시용 파일 경로(캐시)
    """
    fmt = fmt.upper()
    ext_map = {"WEBP": "webp", "JPEG": "jpg", "PNG": "png"}
    ext = ext_map.get(fmt, "webp")
    key = _disp_key(src_path, size, fmt, quality)
    dst = os.path.join(THUMB_DIR, f"{key}.{ext}")

    if not os.path.exists(dst) or _file_mtime(dst) < _file_mtime(src_path):
        try:
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
        except Exception:
            return src_path
    return dst

# ===== KPI 계산 =====
def compute_kpis(df: pd.DataFrame, img_df: pd.DataFrame) -> Dict[str, int]:
    kpis = {"총 이미지": 0, "그룹 수": 0, "공백 수": 0, "유사 후보 쌍": 0}
    try: kpis["총 이미지"] = len(img_df) if len(img_df) else 0
    except Exception: pass
    try: kpis["그룹 수"] = df['그룹ID'].replace('-', pd.NA).dropna().nunique()
    except Exception: pass
    try: kpis["공백 수"] = int(img_df["빈칸여부"].sum()) if "빈칸여부" in img_df.columns else 0
    except Exception: pass
    try: kpis["유사 후보 쌍"] = int((df["상태"] == "유사 후보").sum())
    except Exception: pass
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
c4.metric("유사 후보 쌍", f"{kpis['유사 후보 쌍']:,}")


# ===== 사이드바: 꼭 필요한 옵션만 노출 =====
st.sidebar.header("주요 필터/설정")
# 캐시 새로고침: 파일/폴더 변경이 반영되지 않을 때 사용
try:
    if st.sidebar.button("새로고침 (캐시 재생성)"):
        # Clear streamlit data cache and rerun
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.experimental_rerun()
        except Exception:
            pass
except Exception:
    pass
min_sim = st.sidebar.slider("최소 유사도", 0.0, 1.0, 0.90, 0.01, help="유사도 임계값을 조정하세요.")
name_query = st.sidebar.text_input("파일명 검색", value="", help="특정 파일명을 빠르게 찾고 싶을 때 입력")
group_list = sorted(list(df["그룹ID"].replace('-', pd.NA).dropna().unique())) if "그룹ID" in df.columns else []
group_filter = st.sidebar.selectbox("특정 그룹만 보기", ["전체"] + group_list)

# 그리드 열 개수만 노출 (화질/포맷/품질 등은 고정)
grid_cols = st.sidebar.slider("그리드 열 개수", 2, 8, 5, help="한 줄에 몇 장씩 볼지 선택")

# 유사 그룹 뷰 모드(대형/그리드) — 이 컨트롤이 없으면 later code에서 NameError 발생
group_view_mode = st.sidebar.radio("유사 그룹 보기 방식", ["대형 비교(2열)", "그리드(다중 썸네일)"], horizontal=True, index=1)

# 고급 옵션(화질, 포맷, 품질, 분석 등)은 숨김/제거
grid_target_px = 768  # 고정값
disp_fmt = "WEBP"    # 고정값
disp_quality = 95     # 고정값
group_large_px = 1400 # 고정값
group_page_size = 6   # 고정값
group_page = 1        # 고정값(페이지네이션은 필요시만)
show_absdiff = False
show_ssim = False

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
    """Read two images, resize to same smallest dims, return RGB blended numpy array.
    alpha: weight for a (0..1)."""
    a = cv2.imread(a_path, cv2.IMREAD_COLOR)
    b = cv2.imread(b_path, cv2.IMREAD_COLOR)
    if a is None or b is None:
        raise RuntimeError("이미지 로드 실패")
    # resize to minimum common size
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    out = cv2.addWeighted(a.astype('float32'), alpha, b.astype('float32'), 1.0 - alpha, 0.0)
    out = out.astype('uint8')
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _highlight_differences_rgb(a_path: str, b_path: str, color: Tuple[int, int, int] = (0, 255, 255), thresh: int = 20) -> np.ndarray:
    """Highlight differences by overlaying a colored mask where absdiff > thresh.
    color is in BGR order for OpenCV but returned image is RGB."""
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
    # slight blur to reduce noise
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # color is expected in BGR
    overlay = (mask3.astype('float32') / 255.0) * np.array(color, dtype='float32')
    base = cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype('float32')
    # combine: where mask, mix overlay color with base
    alpha = 0.6
    combined = base * (1.0 - (mask3.astype('float32') / 255.0) * alpha) + overlay * alpha
    combined = np.clip(combined, 0, 255).astype('uint8')
    return combined


def _comp_cache_key(a_path: str, b_path: str, mode: str, params: Dict) -> str:
    s = f"{a_path}|{_file_mtime(a_path)}|{b_path}|{_file_mtime(b_path)}|{mode}|{sorted(params.items())}"
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def _write_cached_image(arr_rgb: np.ndarray, dst: str, fmt: str = 'PNG') -> str:
    try:
        img = Image.fromarray(arr_rgb)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        img.save(dst, fmt)
        return dst
    except Exception:
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
    # produce diff heatmap
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


def _create_fade_gif(a_path: str, b_path: str, steps: int = 20, duration_ms: int = 50) -> Optional[str]:
    key = _comp_cache_key(a_path, b_path, 'fade_gif', {'steps': steps, 'dur': duration_ms})
    dst = os.path.join(THUMB_DIR, f"cmp_fade_anim_{key}.gif")
    if os.path.exists(dst) and _file_mtime(dst) >= max(_file_mtime(a_path), _file_mtime(b_path)):
        return dst
    try:
        frames = []
        for i in range(steps + 1):
            alpha = i / steps
            arr = _blend_images_rgb(a_path, b_path, alpha=alpha)
            frames.append(Image.fromarray(arr))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        frames[0].save(dst, format='GIF', save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
        return dst
    except Exception:
        return None

# ===== 모달(미리보기) 지원: Streamlit 1.34+ =====
_HAS_DIALOG = hasattr(st, "dialog")
def open_preview(img_path: str, caption: str = ""):
    """Open a preview using dialog if available, otherwise show inline fallback."""
    if _HAS_DIALOG:
        @st.dialog("미리보기")
        def _d(img_path_inner: str, caption_inner: str = ""):
            st.image(_safe_image_open(img_path_inner), caption=caption_inner, use_container_width=True)
        _d(img_path, caption)
    else:
        # Fallback: show inline immediately
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
if "compare_list" not in st.session_state:
    st.session_state.compare_list = []
def toggle_compare(img_path: str):
    if img_path not in st.session_state.compare_list:
        st.session_state.compare_list.append(img_path)
    if len(st.session_state.compare_list) > 2:
        st.session_state.compare_list = st.session_state.compare_list[-2:]

# ===== 탭 구성 =====
tab1, tab2, tab3, tab4 = st.tabs(["리포트 요약", "재스캔 필요", "정상/공백 답안", "전체 보기"])

# ===== Global: 탭 어디에서든 2장 선택 시 상단에 즉시 비교 패널 표시 =====
def _render_global_compare():
    # 우선 gallery_selected(탭3/4)와 rescan_selected(탭2)에서 우선순위로 2장 경로 복원
    sel_names = []
    if st.session_state.get('rescan_selected'):
        sel_paths = [p for p in st.session_state.get('rescan_selected', []) if p and os.path.isfile(p)]
        if len(sel_paths) >= 2:
            return sel_paths[:2]
    if st.session_state.get('gallery_selected'):
        sel_names = st.session_state.get('gallery_selected', [])
        sel_paths = [BASENAME_MAP.get(n.lower(), None) for n in sel_names]
        sel_paths = [p for p in sel_paths if p and os.path.isfile(p)]
        if len(sel_paths) >= 2:
            return sel_paths[:2]
    return None


cmp_pair = _render_global_compare()
if cmp_pair:
    a_path, b_path = cmp_pair
    st.markdown("---")
    st.markdown("### 🔀 선택한 두 이미지 즉시 비교")
    c1g, c2g = st.columns(2)
    with c1g:
        big_a = make_display_image(a_path, size=max(1200, group_large_px), fmt=disp_fmt, quality=disp_quality)
        st.image(_safe_image_open(big_a), caption=os.path.basename(a_path), use_container_width=True)
    with c2g:
        big_b = make_display_image(b_path, size=max(1200, group_large_px), fmt=disp_fmt, quality=disp_quality)
        st.image(_safe_image_open(big_b), caption=os.path.basename(b_path), use_container_width=True)
    st.markdown("_두 장이 선택되면 여기에서 바로 비교 모드를 사용해 분석할 수 있습니다._")



# === Tab1: 리포트 요약 ===
with tab1:

    # 안전한 처리: 리포트가 비어있으면 안내
    if df is None or (hasattr(df, '__len__') and len(df) == 0):
        st.info("⚠️ 보고서가 비어 있습니다. 먼저 파이프라인을 실행하세요.")
    else:
        # 필터/정렬 적용된 뷰
        df_view = filter_sort_report(df)

        # 상단: 간단한 요약 카드/테이블
        st.subheader("요약")
        sc1, sc2, sc3 = st.columns([1.2, 1.2, 1.0])
        with sc1:
            st.write("**상태별 분포**")
            if "상태" in df_view.columns:
                st.table(df_view["상태"].value_counts().rename_axis('상태').reset_index(name='건수'))
            else:
                st.write("상태 정보 없음")
        with sc2:
            st.write("**그룹별 상위(최대 10)**")
            if "그룹ID" in df_view.columns:
                grp = df_view['그룹ID'].replace('-', pd.NA).dropna()
                if len(grp):
                    st.table(grp.value_counts().head(10).rename_axis('그룹ID').reset_index(name='건수'))
                else:
                    st.write("그룹 정보 없음")
            else:
                st.write("그룹 정보 없음")
        with sc3:
            st.write("**이미지 요약(빈칸)**")
            if isinstance(img_df, pd.DataFrame) and '빈칸여부' in img_df.columns:
                total_imgs = len(img_df)
                blank_cnt = int(img_df['빈칸여부'].sum()) if total_imgs else 0
                st.metric("공백 수", f"{blank_cnt}", delta=f"{(blank_cnt/total_imgs*100):.1f}%" if total_imgs else "")
            else:
                st.write("이미지 요약 파일이 없습니다")

        st.markdown("---")

        # 중간: 필터된 리포트 표와 다운로드
        st.subheader("필터된 리포트")
        st.dataframe(df_view, use_container_width=True, height=300)
        st.download_button("⬇ CSV 다운로드", df_view.to_csv(index=False).encode("utf-8-sig"),
                           "filtered_report.csv", "text/csv")

        # 하단: 리포트에 등장하는 파일들의 이미지 메타(밀도/빈칸여부) 병합 테이블
        st.markdown("---")
        st.subheader("파일별 메타 (리포트 연동)")
        # 파일1/파일2 컬럼을 합쳐 고유 파일 목록 생성
        files = []
        if '파일1' in df_view.columns:
            files += list(df_view['파일1'].dropna().astype(str).tolist())
        if '파일2' in df_view.columns:
            files += list(df_view['파일2'].dropna().astype(str).tolist())
        files = list(dict.fromkeys(files))
        meta_df = pd.DataFrame({'파일': files})
        if isinstance(img_df, pd.DataFrame) and '파일' in img_df.columns:
            meta_df = meta_df.merge(img_df, on='파일', how='left')
        # 기본 컬럼 정리(존재하지 않더라도 에러 방지)
        for col in ['밀도', '빈칸여부']:
            if col not in meta_df.columns:
                meta_df[col] = pd.NA
        st.dataframe(meta_df, use_container_width=True, height=240)
        # 추가 다운로드: meta
        st.download_button("⬇ 파일 메타 다운로드", meta_df.to_csv(index=False).encode("utf-8-sig"),
                           "report_files_meta.csv", "text/csv")


# === Tab2: 유사 그룹 ===
with tab2:
    # ---- Rescan(재스캔) 감지: 입력 폴더의 이미지 해시(pHash)로 거의 동일한 이미지 쌍 탐지 ----
    # 우선 CLI/streamlit 인자로 전달된 INPUT_DIR 사용, 없으면 output 경로를 기반으로 유추
    # 입력 폴더는 선택적(없는 경우에도 정상 동작)
    if os.path.isdir(INPUT_DIR):
        input_dir = INPUT_DIR
    else:
        input_dir = OUTPUT_DIR.replace("output", "input_images") if "output" in OUTPUT_DIR else "input_images"
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    # input_dir가 없으면 빈 리스트로 처리
    if os.path.isdir(input_dir):
        try:
            scan_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(exts)])
        except Exception:
            scan_files = []
    else:
        scan_files = []

    try:
        import imagehash
        dup_pairs = []
        hashes = {}
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

    # Build rescan candidates: combine pHash dup_pairs (input folder) and report '유사 후보' pairs
    report_dups = []
    try:
        # df에는 파일명(리포트상)이 들어있음; 복원 가능한 경로로 변환
        if '상태' in df.columns:
            rpt = df[df['상태'] == '유사 후보'] if isinstance(df, pd.DataFrame) else pd.DataFrame()
            for _, row in rpt.iterrows():
                a, b = str(row.get('파일1', '')), str(row.get('파일2', ''))
                pa = resolve_image_path(a) or os.path.join(OUTPUT_DIR, a)
                pb = resolve_image_path(b) or os.path.join(OUTPUT_DIR, b)
                report_dups.append((pa, pb, float(row.get('유사도', 0.0))))
    except Exception:
        report_dups = []

    # ----- 즉시 비교 패널: 사용자가 아래 그리드에서 '↔ 비교 선택' 버튼을 클릭하면
    # rescan 탭의 상단에 바로 비교 옵션과 결과가 표시되도록 함
    if "rescan_selected" not in st.session_state:
        st.session_state["rescan_selected"] = []

    sel_paths_top = st.session_state.get("rescan_selected", [])
    sel_exist_top = [p for p in sel_paths_top if p and os.path.isfile(p)]
    if sel_exist_top:
        st.markdown("---")
        st.markdown("### 🔍 즉시 비교 (재스캔 탭)")
        # 공통: 모드 선택 + 도움말 옆 배치
        colm1, colm2 = st.columns([3, 7])
        with colm1:
            cmp_mode_top = st.radio("보기 표시 (재스캔)", ["Fade", "Difference", "Highlighter"], index=0, horizontal=True, key="cmp_mode_top")
        with colm2:
            st.markdown(
                "**모드 설명 (요약 & 권장 설정)**\n"
                "- **Fade**: 두 이미지를 위아래로 겹쳐 보여줍니다. 앞(A) 이미지의 투명도(alpha)를 조절해 미세한 변화가 어느 위치에서 발생하는지 문맥과 함께 확인할 수 있습니다.\n"
                "  - 추천: A alpha = 0.4–0.6\n"
                "  - 장점: 전체 레이아웃 맥락을 유지하면서 변화 관찰 가능\n"
                "  - 단점: 색 대비가 약하거나 스캔 노이즈가 많으면 차이를 식별하기 어려울 수 있음\n"
                "- **Difference**: 그레이스케일 절대 차이를 계산해 heatmap으로 표시합니다. 픽셀 단위 변경을 강조합니다.\n"
                "  - 추천: Blur = 3, Threshold = 10\n"
                "  - 장점: 아주 작은 픽셀 변화까지 시각화 가능\n"
                "  - 단점: 스캔 노이즈(먼지, 압력 자국 등)에 민감함 — Blur/Threshold로 노이즈 제어 필요\n"
                "- **Highlighter**: Difference 마스크를 색상으로 원본 이미지에 오버레이합니다. 문서의 글자 추가/삭제 등 의미 있는 변경을 컬러로 빠르게 파악할 때 유용합니다.\n"
                "  - 추천: Threshold = 15, 색상 = Yellow\n"
                "  - 장점: 변경 영역이 직관적으로 눈에 띔\n"
                "  - 단점: 임계값과 색상 조절이 필요할 수 있음\n\n"
                "사용법: 아래 그리드에서 두 장을 선택하면 이 상단 패널에서 선택한 모드로 즉시 결과를 확인할 수 있습니다.\n"
                "- Fade 애니메이션: 'Play fade animation'을 체크하면 자동으로 alpha를 변화시키는 GIF를 재생합니다 (캐시 사용).\n"
                "- 성능: 비교 이미지는 `artifacts/thumbnails/`에 캐시되어 다음 조회 시 빠르게 로드됩니다."
            )

        if cmp_mode_top == "Fade":
            alpha_top = st.slider("Fade: 앞쪽 이미지 투명도 (A)", 0.0, 1.0, 0.5, 0.01, key="alpha_top")
            play_anim = st.checkbox("Play fade animation", key="play_fade_anim")
        elif cmp_mode_top == "Difference":
            diff_blur_top = st.slider("Difference: Blur 강도(odd kernel)", 1, 11, 3, 2, key="diff_blur_top")
            diff_thresh_top = st.slider("Difference: 강조 임계값", 0, 255, 10, 1, key="diff_thresh_top")
        else:
            hl_color_top = st.selectbox("Highlighter 색상", ["Yellow", "Red", "Lime", "Cyan"], index=0, key="hl_color_top")
            hl_thresh_top = st.slider("Highlighter: 임계값", 1, 100, 20, 1, key="hl_thresh_top")

        if len(sel_exist_top) == 1:
            bigp = make_display_image(sel_exist_top[0], size=max(1400, group_large_px), fmt=disp_fmt, quality=disp_quality)
            st.image(_safe_image_open(bigp), caption=os.path.basename(sel_exist_top[0]), use_container_width=True)
        else:
            a_path, b_path = sel_exist_top[:2]
            big_a = make_display_image(a_path, size=max(1600, group_large_px), fmt=disp_fmt, quality=disp_quality)
            big_b = make_display_image(b_path, size=max(1600, group_large_px), fmt=disp_fmt, quality=disp_quality)
            c1t, c2t = st.columns(2)
            with c1t:
                st.image(_safe_image_open(big_a), caption=os.path.basename(a_path), use_container_width=True)
            with c2t:
                try:
                    if cmp_mode_top == "Fade":
                        # use cached path if available
                        cached = _cached_blend_path(a_path, b_path, alpha=alpha_top)
                        if play_anim:
                            gif = _create_fade_gif(a_path, b_path, steps=24, duration_ms=40)
                            if gif and os.path.exists(gif):
                                st.image(gif, caption=f"Fade animation — {os.path.basename(b_path)}", use_column_width=True)
                            elif cached:
                                st.image(cached, caption=f"Fade (A alpha={alpha_top:.2f}) — {os.path.basename(b_path)}", use_container_width=True)
                            else:
                                blended = _blend_images_rgb(a_path, b_path, alpha=alpha_top)
                                st.image(blended, caption=f"Fade (A alpha={alpha_top:.2f}) — {os.path.basename(b_path)}", use_container_width=True)
                        else:
                            if cached:
                                st.image(cached, caption=f"Fade (A alpha={alpha_top:.2f}) — {os.path.basename(b_path)}", use_container_width=True)
                            else:
                                blended = _blend_images_rgb(a_path, b_path, alpha=alpha_top)
                                st.image(blended, caption=f"Fade (A alpha={alpha_top:.2f}) — {os.path.basename(b_path)}", use_container_width=True)
                    elif cmp_mode_top == "Difference":
                        cached = _cached_diff_path(a_path, b_path, blur=diff_blur_top, thresh=diff_thresh_top)
                        if cached:
                            st.image(cached, caption=f"Difference (thresh={diff_thresh_top})", use_container_width=True)
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
                            st.image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), caption=f"Difference (thresh={diff_thresh_top})", use_container_width=True)
                    else:
                        color_map = {"Yellow": (0, 255, 255), "Red": (0, 0, 255), "Lime": (0, 255, 0), "Cyan": (255, 255, 0)}
                        col_bgr = color_map.get(hl_color_top, (0, 255, 255))
                        cached = _cached_highlight_path(a_path, b_path, color=col_bgr, thresh=hl_thresh_top)
                        if cached:
                            st.image(cached, caption=f"Highlighter ({hl_color_top}, thresh={hl_thresh_top})", use_container_width=True)
                        else:
                            highlighted = _highlight_differences_rgb(a_path, b_path, color=col_bgr, thresh=hl_thresh_top)
                            st.image(highlighted, caption=f"Highlighter ({hl_color_top}, thresh={hl_thresh_top})", use_container_width=True)
                except Exception as e:
                    st.info(f"비교 렌더 실패: {e}")

    grouped_dir = os.path.join(OUTPUT_DIR, "grouped")
    if os.path.isdir(grouped_dir):
        groups = sorted(os.listdir(grouped_dir))
        if group_filter != "전체":
            groups = [g for g in groups if g == group_filter]

                # use global is_2file

        # --- 대형 비교 모드: 페이지네이션 + 두 장을 크게 나란히 ---
        if group_view_mode.startswith("대형"):
            total_groups = len(groups)
            start = max(0, (group_page - 1) * group_page_size)
            end = min(total_groups, start + group_page_size)
            st.caption(f"그룹 {start+1}–{end} / 총 {total_groups} (페이지 {group_page})")

            for gid in groups[start:end]:
                # 그룹 텍스트/캡션은 모두 표시하도록 변경 (특정 그룹 숨김 제거)
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
                            # rescan 전용 비교 선택 토글 (경로 저장)
                            selected = pth in st.session_state["rescan_selected"]
                            label = "✔ 비교 취소" if selected else "↔ 비교 선택"
                            if st.button(label, key=f"cmp_rescan_{gid}_{idx}"):
                                if selected:
                                    st.session_state["rescan_selected"] = [n for n in st.session_state["rescan_selected"] if n != pth]
                                else:
                                    if len(st.session_state["rescan_selected"]) >= 2:
                                        st.session_state["rescan_selected"] = st.session_state["rescan_selected"][1:] + [pth]
                                    else:
                                        st.session_state["rescan_selected"].append(pth)
                            disp = make_display_image(pth, size=group_large_px, fmt=disp_fmt, quality=disp_quality)
                            st.image(_safe_image_open(disp), caption=f"{kind}: {name}", use_container_width=True)
                        else:
                            st.info(f"{kind} 파일 없음: {name}")

                # 선택된 이미지 비교(대형 인라인 뷰)는 탭 상단의 "즉시 비교(재스캔 탭)"에서 제공합니다.
                # 여기서는 각 모드에 대한 간단한 설명과 사용 팁을 보여줍니다.
                sel_paths = st.session_state.get("rescan_selected", [])
                if sel_paths:
                    st.markdown("---")
                    st.markdown("#### � 비교 모드 사용 안내")
                    st.markdown("- Fade: 두 이미지를 겹쳐서 앞(A) 이미지의 투명도를 조절합니다. 작은 차이를 육안으로 직관적으로 확인할 때 유용합니다. 추천값: A alpha 0.4–0.6")
                    st.markdown("- Difference: 그레이스케일 차이를 계산해 heatmap으로 시각화합니다. 픽셀 단위의 변경을 강조할 때 좋습니다. Blur와 Threshold를 조절해 노이즈를 줄이세요. 추천값: Blur=3, Threshold=10")
                    st.markdown("- Highlighter: 차이 마스크를 색상으로 오버레이합니다. 문서 스캔의 글자 추가/삭제 같은 작은 변경을 컬러로 빠르게 식별할 때 유용합니다. 추천값: Threshold=15, 색상=Yellow")
                    st.markdown("\n사용 방법: 아래 그리드에서 '↔ 비교 선택'으로 두 장을 선택하면, 탭 상단의 '즉시 비교 (재스캔 탭)'에서 선택한 모드로 결과를 즉시 확인할 수 있습니다.")
                    # 선택 초기화 버튼(간단한 접근)
                    c1, c2 = st.columns([1, 9])
                    with c1:
                        if st.button("선택 초기화", key=f"rescan_reset_{gid}"):
                            st.session_state["rescan_selected"] = []
    else:
        st.info("그룹 결과 폴더가 없습니다. 하지만 입력 폴더 또는 리포트에서 재스캔 후보를 검사할 수 있습니다.")

    # If no grouped results, show candidates from input pHash (dup_pairs) and report '유사 후보' pairs
    if (not os.path.isdir(grouped_dir)) or (os.path.isdir(grouped_dir) and len(os.listdir(grouped_dir)) == 0):
        candidates = []
        # from input pHash duplicates
        for a, b, d in (dup_pairs if 'dup_pairs' in locals() else []):
            pa = os.path.join(input_dir, a)
            pb = os.path.join(input_dir, b)
            if os.path.exists(pa) and os.path.exists(pb):
                candidates.append((pa, pb, {'reason': f'pHash d={d}'}))
        # from report '유사 후보'
        for pa, pb, sim in (report_dups if 'report_dups' in locals() else []):
            if pa and pb and os.path.exists(pa) and os.path.exists(pb):
                candidates.append((pa, pb, {'reason': f'report sim={sim}'}))

        if candidates:
            st.subheader("재스캔 후보 (입력 폴더 / 리포트 기반)")
            for idx, (a_path, b_path, meta) in enumerate(candidates):
                cols = st.columns(2)
                with cols[0]:
                    disp_a = make_display_image(a_path, size=group_large_px, fmt=disp_fmt, quality=disp_quality)
                    st.image(_safe_image_open(disp_a), caption=os.path.basename(a_path))
                with cols[1]:
                    disp_b = make_display_image(b_path, size=group_large_px, fmt=disp_fmt, quality=disp_quality)
                    st.image(_safe_image_open(disp_b), caption=os.path.basename(b_path))
                st.markdown(f"- 이유: {meta.get('reason')}")
                st.markdown("---")
        else:
            st.info("입력 폴더 및 리포트에서 유효한 재스캔 후보가 발견되지 않았습니다.")

# === Tab3: 정상/공백 ===
with tab3:
    ok_dir = os.path.join(OUTPUT_DIR, "ok")
    blank_dir = os.path.join(OUTPUT_DIR, "blank_answers")
    sel = st.radio("보기 옵션", ["모두 보기", "정상만", "공백만"], horizontal=True)

    # use global is_2file

    if sel in ["모두 보기", "정상만"] and os.path.isdir(ok_dir):
        st.subheader("✅ 정상 답안")
        files = [f for f in sorted(os.listdir(ok_dir)) if is_2file(f)]
        cols = st.columns(grid_cols)
        for idx, f in enumerate(files):
            img_path = os.path.join(ok_dir, f)
            disp = make_display_image(img_path, size=grid_target_px, fmt=disp_fmt, quality=disp_quality)
            with cols[idx % grid_cols]:
                # 비교 토글 버튼으로 통일 (전역 gallery_selected 사용)
                selected = f in st.session_state.gallery_selected
                label = "✔ 비교 취소" if selected else "↔ 비교 선택"
                if st.button(label, key=f"cmp_ok_{idx}"):
                    if selected:
                        st.session_state.gallery_selected = [n for n in st.session_state.gallery_selected if n != f]
                    else:
                        if len(st.session_state.gallery_selected) >= 2:
                            st.session_state.gallery_selected = st.session_state.gallery_selected[1:] + [f]
                        else:
                            st.session_state.gallery_selected.append(f)
                caption = f + ("  ✅ 선택됨" if selected else "")
                st.image(_safe_image_open(disp), caption=caption, use_container_width=True)

    if sel in ["모두 보기", "공백만"] and os.path.isdir(blank_dir):
        st.subheader("⭕ 공백 답안")
        files = [f for f in sorted(os.listdir(blank_dir)) if is_2file(f)]
        cols = st.columns(grid_cols)
        for idx, f in enumerate(files):
            img_path = os.path.join(blank_dir, f)
            disp = make_display_image(img_path, size=grid_target_px, fmt=disp_fmt, quality=disp_quality)
            with cols[idx % grid_cols]:
                # 비교 토글 버튼으로 통일 (전역 gallery_selected 사용)
                selected = f in st.session_state.gallery_selected
                label = "✔ 비교 취소" if selected else "↔ 비교 선택"
                if st.button(label, key=f"cmp_blank_{idx}"):
                    if selected:
                        st.session_state.gallery_selected = [n for n in st.session_state.gallery_selected if n != f]
                    else:
                        if len(st.session_state.gallery_selected) >= 2:
                            st.session_state.gallery_selected = st.session_state.gallery_selected[1:] + [f]
                        else:
                            st.session_state.gallery_selected.append(f)
                caption = f + ("  ✅ 선택됨" if selected else "")
                st.image(_safe_image_open(disp), caption=caption, use_container_width=True)

    # === Tab3: 선택된 비교 항목을 즉시 대형 비교로 표시 ===
    if st.session_state.gallery_selected:
        st.markdown("---")
        st.markdown("### 🔍 선택 비교 (대형) — 탭3")
        sel_paths = [BASENAME_MAP.get(n.lower(), None) for n in st.session_state.gallery_selected]
        sel_paths = [p for p in sel_paths if p and os.path.isfile(p)]
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

# === Tab4: 전체 보기 ===
with tab4:
    # ---------- 쉬운 화질/레이아웃 컨트롤(탭 로컬) ----------
    st.markdown("#### 표시 설정")
    colq1, colq2, colq3 = st.columns([1.3, 1.1, 1.6])
    with colq1:
        quality_profile = st.radio(
            "화질 프로파일", ["빠름", "균형", "선명", "사용자지정"],
            index=1, horizontal=True,
            help="빠름(512px), 균형(1024px), 선명(1600px), 사용자지정(슬라이더)"
        )
    with colq2:
        render_mode = st.radio(
            "렌더 방식", ["리샘플(권장)", "원본"], index=0, horizontal=True,
            help="리샘플: LANCZOS로 고화질 썸네일 생성(권장) / 원본: 브라우저 스케일(선명하지만 느릴 수 있음)"
        )
    with colq3:
        grid_cols_local = st.slider("그리드 열 개수", 2, 8, max(4, grid_cols), 1)

    # 프로파일 → 표시 해상도/포맷/품질 파라미터 도출
    if quality_profile == "빠름":
        target_px_eff, disp_fmt_eff, disp_quality_eff = 512, "WEBP", 92
    elif quality_profile == "균형":
        target_px_eff, disp_fmt_eff, disp_quality_eff = 1024, "WEBP", 95
    elif quality_profile == "선명":
        target_px_eff, disp_fmt_eff, disp_quality_eff = 1600, "WEBP", 98
    else:
        # 사용자 지정 옵션
        st.markdown("##### 사용자 지정")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            target_px_eff = st.slider("표시 해상도(px, 긴 변)", 512, 2400, 1400, 50)
        with cc2:
            disp_fmt_eff = st.selectbox("표시 포맷", ["WEBP", "JPEG", "PNG"], index=0)
        with cc3:
            disp_quality_eff = st.slider("표시 품질(압축)", 80, 100, 95)

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
    all_imgs = list_all_images(OUTPUT_DIR)
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

    # ---------- 페이지네이션(Load more) ----------
    colp1, colp2, colp3 = st.columns([1.2, 1, 3])
    with colp1:
        page_chunk = st.slider("한 번에 더 보기", 20, 200, 80, 10)
    with colp2:
        if st.button("더 보기 ⤵"):
            st.session_state.gallery_limit = min(total_items, st.session_state.gallery_limit + page_chunk)
    with colp3:
        if st.button("처음으로 ⤴"):
            st.session_state.gallery_limit = min(total_items, page_chunk)

    # 현재 보여줄 범위
    limit = min(total_items, max(1, st.session_state.gallery_limit))
    show_paths = all_imgs[:limit]
    st.caption(f"{1}–{limit} / {total_items}")

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
            st.image(_safe_image_open(disp), caption=os.path.basename(path), use_container_width=True)
            # 동작 버튼: 비교 토글만 표시
            selected = os.path.basename(path) in st.session_state.gallery_selected
            label = "✔ 비교 취소" if selected else "↔ 비교 선택"
            if st.button(label, key=f"cmp_all_{idx}"):
                name = os.path.basename(path)
                if selected:
                    st.session_state.gallery_selected = [n for n in st.session_state.gallery_selected if n != name]
                else:
                    if len(st.session_state.gallery_selected) >= 2:
                        # 가장 오래된 선택 제거 후 추가
                        st.session_state.gallery_selected = st.session_state.gallery_selected[1:] + [name]
                    else:
                        st.session_state.gallery_selected.append(name)

    # ---------- 선택 비교(대형 2분할) ----------
    if st.session_state.gallery_selected:
        st.markdown("---")
        st.markdown("### 🔍 선택 비교 (대형)")
        # 파일명 → 경로 복원 (BASENAME_MAP 사용)
        sel_paths = [BASENAME_MAP.get(n.lower(), None) for n in st.session_state.gallery_selected]
        sel_paths = [p for p in sel_paths if p and os.path.isfile(p)]
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
                        st.image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), caption="AbsDiff Heatmap", use_container_width=True)
                    if show_ssim and _HAS_SKIMAGE:
                        score, ssim_img = ssim(ga, gb, full=True, data_range=255)
                        ssim_img = (1.0 - ssim_img)
                        ssim_img = (255 * (ssim_img / (ssim_img.max() + 1e-6))).astype(np.uint8)
                        heat = cv2.applyColorMap(ssim_img, cv2.COLORMAP_INFERNO)
                        st.image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB),
                                 caption=f"SSIM Map (score={score:.4f})", use_container_width=True)
                except Exception as e:
                    st.info(f"분석 실패: {e}")

        # 선택 상태 관리 버튼
        cols_ctrl = st.columns([1, 1, 6])
        with cols_ctrl[0]:
            if st.button("선택 초기화"):
                st.session_state.gallery_selected = []
    # 우선 비교 토글로 대체 — 모달형 미리보기 버튼 제거

    # 모달 미지원 대체 표시: 없음(직접 inline으로 대체됨)