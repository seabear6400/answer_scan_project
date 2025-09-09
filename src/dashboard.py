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
from typing import Tuple, List, Dict, Optional

import streamlit as st
import polars as pl
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import torch  # (옵션) LPIPS 계산 등에 필요

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

_HAS_IMG_CMP = False
try:
    from streamlit_image_comparison import image_comparison
    _HAS_IMG_CMP = True
except Exception:
    pass

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

# ===== 상단 도움말 =====
with st.expander("ℹ️ 빠른 사용법 / 용어 설명", expanded=False):
    st.markdown(
        """
        - **리포트 요약**: 유사도 검출 결과(파일 쌍)를 테이블로 확인하고 필터/정렬할 수 있습니다.  
        - **유사 그룹**: 파이프라인이 **2장씩 매칭**한 그룹을 '대형 비교' 또는 '그리드'로 확인합니다.  
        - **정상/공백**: 공백 감지된 답안과 정상 답안을 각각 훑어봅니다.  
        - **전체 보기**: `output/` 하위의 모든 이미지를 고화질 썸네일로 훑어봅니다.  
        """
    )

# ===== KPI 카드 =====
kpis = compute_kpis(df, img_df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("총 이미지", f"{kpis['총 이미지']:,}")
c2.metric("그룹 수", f"{kpis['그룹 수']:,}")
c3.metric("공백 수", f"{kpis['공백 수']:,}")
c4.metric("유사 후보 쌍", f"{kpis['유사 후보 쌍']:,}")


# ===== 사이드바: 꼭 필요한 옵션만 노출 =====
st.sidebar.header("주요 필터/설정")
min_sim = st.sidebar.slider("최소 유사도", 0.0, 1.0, 0.90, 0.01, help="유사도 임계값을 조정하세요.")
name_query = st.sidebar.text_input("파일명 검색", value="", help="특정 파일명을 빠르게 찾고 싶을 때 입력")
group_list = sorted(list(df["그룹ID"].replace('-', pd.NA).dropna().unique())) if "그룹ID" in df.columns else []
group_filter = st.sidebar.selectbox("특정 그룹만 보기", ["전체"] + group_list)

# 그리드 열 개수만 노출 (화질/포맷/품질 등은 고정)
grid_cols = st.sidebar.slider("그리드 열 개수", 2, 8, 5, help="한 줄에 몇 장씩 볼지 선택")

# 유사 그룹 뷰 모드(대형/그리드)만 노출
group_view_mode = st.sidebar.radio("유사 그룹 보기 방식", ["대형 비교(2열)", "그리드(다중 썸네일)"], horizontal=True, index=0)

# 고급 옵션(화질, 포맷, 품질, 분석 등)은 숨김/제거
grid_target_px = 768  # 고정값
disp_fmt = "WEBP"    # 고정값
disp_quality = 95     # 고정값
group_large_px = 1400 # 고정값
group_page_size = 6   # 고정값
group_page = 1        # 고정값(페이지네이션은 필요시만)
show_absdiff = False
show_ssim = False
show_lpips = False

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

_lpips_model = None
def _lpips_score(a_path: str, b_path: str) -> Optional[float]:
    global _lpips_model
    if not _HAS_LPIPS:
        return None
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='vgg').eval()
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor()])
    A = cv2.cvtColor(cv2.imread(a_path), cv2.COLOR_BGR2RGB)
    B = cv2.cvtColor(cv2.imread(b_path), cv2.COLOR_BGR2RGB)
    h = min(A.shape[0], B.shape[0]); w = min(A.shape[1], B.shape[1])
    A = cv2.resize(A, (w, h)); B = cv2.resize(B, (w, h))
    a = tf(Image.fromarray(A)).unsqueeze(0)
    b = tf(Image.fromarray(B)).unsqueeze(0)
    with torch.no_grad():
        d = _lpips_model(a, b).item()
    return float(d)

# ===== 모달(미리보기) 지원: Streamlit 1.34+ =====
_HAS_DIALOG = hasattr(st, "dialog")
if _HAS_DIALOG:
    @st.dialog("미리보기")
    def preview_dialog(img_path: str, caption: str = ""):
        st.image(_safe_image_open(img_path), caption=caption, use_container_width=True)
else:
    st.session_state.setdefault("legacy_preview_path", None)

def open_preview(img_path: str, caption: str = ""):
    if _HAS_DIALOG:
        preview_dialog(img_path, caption)
    else:
        st.session_state["legacy_preview_path"] = (img_path, caption)
        st.experimental_rerun()

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
tab1, tab2, tab3, tab4 = st.tabs(["리포트 요약", "유사 그룹", "정상/공백 답안", "전체 보기"])

# === Tab1: 리포트 요약 ===
with tab1:
    with st.expander("이 탭은 무엇을 하나요?", expanded=False):
        st.write("유사도 검출 결과(쌍)를 표로 보고, 필터와 정렬을 적용합니다. 필요시 CSV로 다운로드하세요.")

    df_view = filter_sort_report(df) if len(df) else df
    st.dataframe(df_view, use_container_width=True, height=480)
    st.download_button("⬇ CSV 다운로드", df_view.to_csv(index=False).encode("utf-8-sig"),
                       "filtered_report.csv", "text/csv")

    # 비교/분석 위젯
    st.markdown("### 🔍 두 파일 선택해 비교")
    names_union = sorted(set(df_view.get("파일1", [])) | set(df_view.get("파일2", [])))
    colA, colB = st.columns(2)
    with colA:
        a = st.selectbox("파일1", names_union, key="cmp_a")
    with colB:
        b = st.selectbox("파일2", names_union, key="cmp_b")

    if a and b:
        path_a = resolve_image_path(a)
        path_b = resolve_image_path(b)
        if not path_a or not os.path.isfile(path_a):
            st.error(f"파일을 찾지 못했습니다: {a} (OUTPUT_DIR 하위에서 검색 실패)")
        if not path_b or not os.path.isfile(path_b):
            st.error(f"파일을 찾지 못했습니다: {b} (OUTPUT_DIR 하위에서 검색 실패)")
        if path_a and path_b and os.path.isfile(path_a) and os.path.isfile(path_b):
            c1, c2 = st.columns(2)
            with c1:
                st.image(_safe_image_open(path_a), caption=os.path.basename(path_a), use_container_width=True)
            with c2:
                st.image(_safe_image_open(path_b), caption=os.path.basename(path_b), use_container_width=True)

            if _HAS_IMG_CMP:
                st.markdown("#### Slider 비교")
                try:
                    image_comparison(Image.open(path_a), Image.open(path_b),
                                     label1=os.path.basename(path_a), label2=os.path.basename(path_b), width=700)
                except Exception:
                    pass

            if show_absdiff or show_ssim or show_lpips:
                st.markdown("#### 차이 분석")
                try:
                    ga, gb = _read_gray_same_size(path_a, path_b)
                    if show_absdiff:
                        st.image(_absdiff_heatmap(ga, gb), caption="AbsDiff Heatmap", use_container_width=True)
                    if show_ssim and _HAS_SKIMAGE:
                        score, ssim_img = ssim(ga, gb, full=True, data_range=255)
                        ssim_img = (1.0 - ssim_img)
                        ssim_img = (255 * (ssim_img / (ssim_img.max() + 1e-6))).astype(np.uint8)
                        heat = cv2.applyColorMap(ssim_img, cv2.COLORMAP_INFERNO)
                        st.image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB),
                                 caption=f"SSIM Map (score={score:.4f})", use_container_width=True)
                    if show_lpips and _HAS_LPIPS:
                        d = _lpips_score(path_a, path_b)
                        if d is not None:
                            st.write(f"LPIPS distance: **{d:.4f}** (낮을수록 유사)")
                except Exception as e:
                    st.info(f"분석 맵 생성 실패: {e}")

# === Tab2: 유사 그룹 ===
with tab2:
    with st.expander("이 탭은 무엇을 하나요?", expanded=False):
        st.write("유사한 두 장으로 묶인 그룹을 '대형 비교' 또는 '그리드'로 확인합니다.")

    grouped_dir = os.path.join(OUTPUT_DIR, "grouped")
    if os.path.isdir(grouped_dir):
        groups = sorted(os.listdir(grouped_dir))
        if group_filter != "전체":
            groups = [g for g in groups if g == group_filter]

        def is_2file(filename):
            return filename.lower().endswith('.jpg') and filename[-5] == '2'

        # --- 대형 비교 모드: 페이지네이션 + 두 장을 크게 나란히 ---
        if group_view_mode.startswith("대형"):
            total_groups = len(groups)
            start = max(0, (group_page - 1) * group_page_size)
            end = min(total_groups, start + group_page_size)
            st.caption(f"그룹 {start+1}–{end} / 총 {total_groups} (페이지 {group_page})")

            for gid in groups[start:end]:
                st.subheader(f"그룹: {gid}")
                files = [f for f in sorted(os.listdir(os.path.join(grouped_dir, gid))) if is_2file(f)]
                if len(files) == 0:
                    st.info("이 그룹에 (2로 끝나는) 이미지가 없습니다."); continue
                # 대형 표시(긴 변 group_large_px)
                disp_paths = []
                for f in files[:2]:  # 보통 2장이므로 2장만
                    p = os.path.join(grouped_dir, gid, f)
                    disp_paths.append(make_display_image(p, size=group_large_px, fmt=disp_fmt, quality=disp_quality))

                cols = st.columns(2)
                for i in range(min(2, len(disp_paths))):
                    with cols[i]:
                        st.image(_safe_image_open(disp_paths[i]), caption=files[i], use_container_width=True)
                # 바로 미리보기 띄우기 버튼
                open_cols = st.columns(2)
                for i in range(min(2, len(files))):
                    with open_cols[i]:
                        if st.button(f"🔎 크게 보기 — {files[i]}", key=f"pv_large_{gid}_{i}"):
                            open_preview(os.path.join(grouped_dir, gid, files[i]), caption=files[i])

        # --- 그리드 모드: 여러 썸네일(해상도 설정 반영) ---
        else:
            sel = st.selectbox("그룹 선택", ["전체 그룹 보기"] + groups)
            targets = groups if sel == "전체 그룹 보기" else [sel]
            for gid in targets:
                st.subheader(f"그룹: {gid}")
                files = [f for f in sorted(os.listdir(os.path.join(grouped_dir, gid))) if is_2file(f)]
                cols = st.columns(grid_cols)
                for idx, f in enumerate(files):
                    img_path = os.path.join(grouped_dir, gid, f)
                    disp = make_display_image(img_path, size=grid_target_px, fmt=disp_fmt, quality=disp_quality)
                    with cols[idx % grid_cols]:
                        if st.button(f"🔎 {f}", key=f"pv_{gid}_{idx}"):
                            open_preview(img_path, caption=f)
                        st.image(_safe_image_open(disp), caption=f, use_container_width=True)
    else:
        st.info("그룹 결과 폴더가 없습니다. 먼저 파이프라인을 실행하세요.")

# === Tab3: 정상/공백 ===
with tab3:
    with st.expander("이 탭은 무엇을 하나요?", expanded=False):
        st.write("공백 감지된 답안과 정상 답안을 각각 훑어봅니다.")
    ok_dir = os.path.join(OUTPUT_DIR, "ok")
    blank_dir = os.path.join(OUTPUT_DIR, "blank_answers")
    sel = st.radio("보기 옵션", ["모두 보기", "정상만", "공백만"], horizontal=True)

    def is_2file(filename):
        return filename.lower().endswith('.jpg') and filename[-5] == '2'

    if sel in ["모두 보기", "정상만"] and os.path.isdir(ok_dir):
        st.subheader("✅ 정상 답안")
        files = [f for f in sorted(os.listdir(ok_dir)) if is_2file(f)]
        cols = st.columns(grid_cols)
        for idx, f in enumerate(files):
            img_path = os.path.join(ok_dir, f)
            disp = make_display_image(img_path, size=grid_target_px, fmt=disp_fmt, quality=disp_quality)
            with cols[idx % grid_cols]:
                if st.button(f"🔎 {f}", key=f"pv_ok_{idx}"):
                    open_preview(img_path, caption=f)
                st.image(_safe_image_open(disp), caption=f, use_container_width=True)

    if sel in ["모두 보기", "공백만"] and os.path.isdir(blank_dir):
        st.subheader("⭕ 공백 답안")
        files = [f for f in sorted(os.listdir(blank_dir)) if is_2file(f)]
        cols = st.columns(grid_cols)
        for idx, f in enumerate(files):
            img_path = os.path.join(blank_dir, f)
            disp = make_display_image(img_path, size=grid_target_px, fmt=disp_fmt, quality=disp_quality)
            with cols[idx % grid_cols]:
                if st.button(f"🔎 {f}", key=f"pv_blank_{idx}"):
                    open_preview(img_path, caption=f)
                st.image(_safe_image_open(disp), caption=f, use_container_width=True)

# === Tab4: 전체 보기 ===
with tab4:
    # 상단 안내는 간결하게
    with st.expander("이 탭은 무엇을 하나요?", expanded=False):
        st.write("`output/` 전체 이미지를 고화질로 훑어보고, 2장을 바로 선택해서 크게 비교할 수 있습니다.")

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
        sort_key = st.selectbox("정렬", ["파일명", "수정시각(최신순)", "수정시각(오래된순)"], index=1)

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
            # 동작 버튼 (미리보기/비교)
            cba, cbb = st.columns(2)
            with cba:
                if st.button("🔎 미리보기", key=f"pv_all_{idx}"):
                    open_preview(path, caption=os.path.basename(path))
            with cbb:
                # 비교 선택 토글 (최대 2장)
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
            if show_absdiff or show_ssim or show_lpips:
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
                    if show_lpips and _HAS_LPIPS:
                        import torchvision.transforms as T
                        tf = T.Compose([T.ToTensor()])
                        A = cv2.cvtColor(cv2.imread(a_path), cv2.COLOR_BGR2RGB)
                        B = cv2.cvtColor(cv2.imread(b_path), cv2.COLOR_BGR2RGB)
                        h2 = min(A.shape[0], B.shape[0]); w2 = min(A.shape[1], B.shape[1])
                        A = cv2.resize(A, (w2, h2)); B = cv2.resize(B, (w2, h2))
                        if '_lpips_model' not in globals() or _lpips_model is None:
                            from lpips import LPIPS
                            globals()['_lpips_model'] = LPIPS(net='vgg').eval()
                        a_t = tf(Image.fromarray(A)).unsqueeze(0)
                        b_t = tf(Image.fromarray(B)).unsqueeze(0)
                        with torch.no_grad():
                            d = _lpips_model(a_t, b_t).item()
                        st.write(f"LPIPS distance: **{d:.4f}** (낮을수록 유사)")
                except Exception as e:
                    st.info(f"분석 실패: {e}")

        # 선택 상태 관리 버튼
        cols_ctrl = st.columns([1, 1, 6])
        with cols_ctrl[0]:
            if st.button("선택 초기화"):
                st.session_state.gallery_selected = []
        with cols_ctrl[1]:
            if st.button("선택 2장 미리보기 모달"):
                if len(sel_paths) >= 1:
                    open_preview(sel_paths[0], caption=os.path.basename(sel_paths[0]))
                if len(sel_paths) >= 2:
                    open_preview(sel_paths[1], caption=os.path.basename(sel_paths[1]))

    # 모달 미지원 대체 표시
    if (not _HAS_DIALOG) and st.session_state.get("legacy_preview_path"):
        st.markdown("---")
        bp, cap = st.session_state["legacy_preview_path"]
        st.subheader(f"미리보기: {cap}")
        st.image(_safe_image_open(bp), use_container_width=True)
